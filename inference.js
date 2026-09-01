'use strict';

/* =========================================================================
   Gift of Gab browser inference runtime.

   A self-contained inference engine for the Gab 100M model — tokenizer
   and a pure-JavaScript CPU engine. No training machinery: forward pass
   only, with a persistent per-layer fp32 KV cache. The model is small
   enough that plain CPU matvecs keep chat interactive.

   The model ships in two storages:
     - F32: full float32 tensors
     - Q8:  rowwise symmetric int8 matrices + fp32 scales, norms and the
            MLP hidden bias kept in fp32 (weights stay quantized in
            memory; rows are dequantized on the fly inside the matvec)

   The per-head attention matrices are fused at export time: summing the
   per-head outputs (attn_i @ Wo_i) equals a concat plus one big output
   projection, so this runtime runs the standard fused [768, 768]
   formulation and matches the reference implementation exactly.

   The public surface used by interface.js is:
     new GabTokenizer(vocabJson)
     await GabInferenceEngine.create({ manifest, weightsBuffer, tokenizer, ... })
     for await (const item of engine.generate(messages, options)) ...
   ========================================================================= */
(function () {
  const ROLE_TOKENS = {
    user: '<|user|>',
    assistant: '<|assistant|>',
    end: '<|end|>',
    endOfText: '<|endoftext|>',
    thinkOpen: '<think>',
    thinkClose: '</think>',
  };

  const MANIFEST_FORMAT_VERSION = 2;

  function nextFrame() {
    return new Promise((resolve) => {
      if (typeof requestAnimationFrame === 'function') {
        requestAnimationFrame(() => resolve());
      } else {
        setTimeout(resolve, 0);
      }
    });
  }

  function abortError() {
    if (typeof DOMException === 'function') {
      return new DOMException('Generation stopped.', 'AbortError');
    }
    const err = new Error('Generation stopped.');
    err.name = 'AbortError';
    return err;
  }

  function throwIfAborted(signal) {
    if (signal && signal.aborted) throw abortError();
  }

  function initSessionState(engine) {
    engine.sessionIds = [];
    engine.sessionLogits = null;
    engine.sessionOpenAssistant = false;
    engine.sessionOpenThinking = false;
    engine.retainContext = Math.max(1, Math.floor(engine.maxContext / 2));
    engine.boundaryLookahead = Math.min(256, engine.retainContext);
  }

  /* =========================================================================
     GabTokenizer — byte-level BPE over a {reserved, merges} vocab.

     Ids 0-255 are the raw bytes; every other id is a merge of two earlier
     ids, learned in order. Encoding applies every merge rule in sequence to
     the UTF-8 bytes; decoding unrolls an id back down to bytes. Reserved
     tokens are chained merges, so they encode to a single id and are atomic.
     This is the same algorithm as tokenizer.py / the reference tokenizer.
     ========================================================================= */
  class GabTokenizer {
    constructor(vocabJson) {
      if (!vocabJson || !Array.isArray(vocabJson.merges)) {
        throw new Error('Unrecognized tokenizer format.');
      }

      // merges[id] = [firstId, secondId]; the first 256 are byte seeds.
      this.merges = [];
      for (let i = 0; i < 256; i++) this.merges.push([i, i]);
      for (const pair of vocabJson.merges) this.merges.push(pair);

      this.reserved = (vocabJson.reserved || []).slice();
      this.encoderUtf8 = new TextEncoder();
      this.decoderUtf8 = new TextDecoder('utf-8');
      this.textCache = new Map();   // id -> decoded text

      // Reserved tokens collapse to one id under the merges; map both ways.
      this.specials = new Map();    // text -> id
      this.specialIds = new Map();  // id -> text
      for (const text of this.reserved) {
        const ids = this.encode(text);
        if (ids.length === 1) {
          this.specials.set(text, ids[0]);
          this.specialIds.set(ids[0], text);
        }
      }
    }

    vocabSize() {
      return this.merges.length;
    }

    idFor(token) {
      return this.specials.get(token);
    }

    isSpecialId(id) {
      return this.specialIds.has(id);
    }

    encode(text) {
      if (!text) return [];
      let tokens = Array.from(this.encoderUtf8.encode(text));
      for (let rule = 256; rule < this.merges.length; rule++) {
        const [first, second] = this.merges[rule];
        tokens = this._merge(tokens, first, second, rule);
      }
      return tokens;
    }

    _merge(tokens, first, second, replacement) {
      const out = [];
      for (let i = 0; i < tokens.length; i++) {
        if (tokens[i] === first && i + 1 < tokens.length && tokens[i + 1] === second) {
          out.push(replacement);
          i++;
        } else {
          out.push(tokens[i]);
        }
      }
      return out;
    }

    tokenize(text) {
      return this.encode(text).map((id) => ({ id, text: this.decodeToken(id) }));
    }

    decode(ids) {
      let text = '';
      for (const id of ids) text += this.decodeToken(id);
      return text;
    }

    decodeToken(id) {
      const cached = this.textCache.get(id);
      if (cached !== undefined) return cached;
      if (id < 0 || id >= this.merges.length) return '';

      // Unroll the merge tree back down to raw bytes.
      const bytes = [];
      const stack = [id];
      while (stack.length > 0) {
        const cur = stack.pop();
        if (cur < 256) {
          bytes.push(cur);
        } else {
          const pair = this.merges[cur];
          stack.push(pair[1]); // second half pushed first,
          stack.push(pair[0]); // so first half pops first
        }
      }
      const text = this.decoderUtf8.decode(new Uint8Array(bytes));
      this.textCache.set(id, text);
      return text;
    }
  }

  /* =========================================================================
     Tensor views over the downloaded weight buffer.
     ========================================================================= */
  class Q8Tensor {
    constructor(buffer, entry) {
      this.name = entry.name;
      this.rows = entry.shape[0];
      this.cols = entry.shape[1];
      this.q = new Int8Array(buffer, entry.q.offset, entry.q.length);
      this.scale = new Float32Array(buffer, entry.scale.offset, entry.scale.length / 4);
    }

    embedding(row, out) {
      const base = row * this.cols;
      const s = this.scale[row];
      for (let i = 0; i < this.cols; i++) out[i] = this.q[base + i] * s;
    }

    matvec(input, out) {
      const cols = this.cols;
      const q = this.q;
      const scale = this.scale;
      for (let r = 0; r < this.rows; r++) {
        let base = r * cols;
        const end = base + cols;
        let sum0 = 0.0;
        let sum1 = 0.0;
        let sum2 = 0.0;
        let sum3 = 0.0;
        let c = 0;
        for (; base + 3 < end; base += 4, c += 4) {
          sum0 += q[base] * input[c];
          sum1 += q[base + 1] * input[c + 1];
          sum2 += q[base + 2] * input[c + 2];
          sum3 += q[base + 3] * input[c + 3];
        }
        let sum = sum0 + sum1 + sum2 + sum3;
        for (; base < end; base++, c++) sum += q[base] * input[c];
        out[r] = sum * scale[r];
      }
    }
  }

  class F32Tensor {
    constructor(buffer, entry) {
      this.name = entry.name;
      this.shape = entry.shape;
      this.rows = entry.shape[0] || 0;
      this.cols = entry.shape[1] || 0;
      this.data = new Float32Array(buffer, entry.data.offset, entry.data.length / 4);
    }

    embedding(row, out) {
      const base = row * this.cols;
      for (let i = 0; i < this.cols; i++) out[i] = this.data[base + i];
    }

    matvec(input, out) {
      const cols = this.cols;
      const data = this.data;
      for (let r = 0; r < this.rows; r++) {
        let base = r * cols;
        const end = base + cols;
        let sum0 = 0.0;
        let sum1 = 0.0;
        let sum2 = 0.0;
        let sum3 = 0.0;
        let c = 0;
        for (; base + 3 < end; base += 4, c += 4) {
          sum0 += data[base] * input[c];
          sum1 += data[base + 1] * input[c + 1];
          sum2 += data[base + 2] * input[c + 2];
          sum3 += data[base + 3] * input[c + 3];
        }
        let sum = sum0 + sum1 + sum2 + sum3;
        for (; base < end; base++, c++) sum += data[base] * input[c];
        out[r] = sum;
      }
    }
  }

  /* =========================================================================
     Math helpers (CPU backend).
     ========================================================================= */
  function erfApprox(x) {
    // Abramowitz & Stegun 7.1.26 — matches the reference and the trainers.
    const sign = x < 0 ? -1 : 1;
    x = Math.abs(x);
    const a1 = 0.254829592;
    const a2 = -0.284496736;
    const a3 = 1.421413741;
    const a4 = -1.453152027;
    const a5 = 1.061405429;
    const p = 0.3275911;
    const t = 1.0 / (1.0 + p * x);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
    return sign * y;
  }

  function geluInPlace(x) {
    const invSqrt2 = 0.7071067811865476;
    for (let i = 0; i < x.length; i++) {
      const v = x[i];
      x[i] = 0.5 * v * (1.0 + erfApprox(v * invSqrt2));
    }
  }

  function rmsNorm(input, weight, output, eps) {
    let ss = 0.0;
    for (let i = 0; i < input.length; i++) ss += input[i] * input[i];
    const inv = 1.0 / Math.sqrt(ss / input.length + eps);
    for (let i = 0; i < input.length; i++) output[i] = input[i] * inv * weight[i];
  }

  function addInPlace(a, b) {
    for (let i = 0; i < a.length; i++) a[i] += b[i];
  }

  function checkManifest(manifest) {
    if (!manifest || manifest.format_version !== MANIFEST_FORMAT_VERSION) {
      throw new Error('Model manifest is from an older runtime. Download the model again.');
    }
  }

  /* =========================================================================
     CPU engine.
     ========================================================================= */
  class GabCpuInferenceEngine {
    constructor({ manifest, weightsBuffer, tokenizer, maxContext = 512 }) {
      checkManifest(manifest);
      this.manifest = manifest;
      this.weightsBuffer = weightsBuffer;
      this.tokenizer = tokenizer;
      this.arch = manifest.architecture;
      this.maxContext = Math.min(maxContext, this.arch.max_position_embeddings);

      this.hiddenSize = this.arch.hidden_size;
      this.intermediateSize = this.arch.intermediate_size;
      this.numLayers = this.arch.num_hidden_layers;
      this.numHeads = this.arch.num_attention_heads;
      this.headDim = this.arch.head_dim;
      this.scale = 1.0 / Math.sqrt(this.headDim);

      this.eosId = tokenizer.idFor(ROLE_TOKENS.end);
      this.eotId = tokenizer.idFor(ROLE_TOKENS.endOfText);
      this.userId = tokenizer.idFor(ROLE_TOKENS.user);
      this.assistantId = tokenizer.idFor(ROLE_TOKENS.assistant);
      this.thinkOpenId = tokenizer.idFor(ROLE_TOKENS.thinkOpen);
      this.thinkCloseId = tokenizer.idFor(ROLE_TOKENS.thinkClose);
      initSessionState(this);
    }

    static async create(options) {
      const engine = new GabCpuInferenceEngine(options);
      await engine.init();
      return engine;
    }

    async init() {
      const tensors = new Map();
      for (const entry of this.manifest.tensors) {
        tensors.set(
          entry.name,
          entry.storage === 'q8_rowwise_symmetric'
            ? new Q8Tensor(this.weightsBuffer, entry)
            : new F32Tensor(this.weightsBuffer, entry)
        );
      }

      this.embed = tensors.get('model.embed_tokens.weight');
      this.finalNorm = tensors.get('model.norm.weight').data;
      this.layers = [];
      for (let i = 0; i < this.numLayers; i++) {
        this.layers.push({
          inputNorm: tensors.get(`model.layers.${i}.input_layernorm.weight`).data,
          postNorm: tensors.get(`model.layers.${i}.post_attention_layernorm.weight`).data,
          qProj: tensors.get(`model.layers.${i}.self_attn.q_proj.weight`),
          kProj: tensors.get(`model.layers.${i}.self_attn.k_proj.weight`),
          vProj: tensors.get(`model.layers.${i}.self_attn.v_proj.weight`),
          oProj: tensors.get(`model.layers.${i}.self_attn.o_proj.weight`),
          upProj: tensors.get(`model.layers.${i}.mlp.up_proj.weight`),
          upBias: tensors.get(`model.layers.${i}.mlp.up_bias`).data,
          downProj: tensors.get(`model.layers.${i}.mlp.down_proj.weight`),
          kCache: new Float32Array(this.maxContext * this.hiddenSize),
          vCache: new Float32Array(this.maxContext * this.hiddenSize),
        });
      }

      this.hidden = new Float32Array(this.hiddenSize);
      this.normed = new Float32Array(this.hiddenSize);
      this.q = new Float32Array(this.hiddenSize);
      this.k = new Float32Array(this.hiddenSize);
      this.v = new Float32Array(this.hiddenSize);
      this.attn = new Float32Array(this.hiddenSize);
      this.proj = new Float32Array(this.hiddenSize);
      this.mlp = new Float32Array(this.intermediateSize);
      this.logits = new Float32Array(this.arch.vocab_size);
      this.scores = new Float32Array(this.maxContext);
      this.ropeCos = new Float32Array(this.maxContext * (this.headDim / 2));
      this.ropeSin = new Float32Array(this.maxContext * (this.headDim / 2));
      this._initRope();
    }

    async *generate(messages, options = {}) {
      const maxNewTokens = options.maxNewTokens || 128;
      const temperature = options.temperature ?? 0.8;
      const topK = options.topK || 50;
      const wantsThinking = !!options.thinking;
      const signal = options.signal || null;
      const onRebuild = options.onRebuild || null;
      let mode = wantsThinking ? 'thinking' : 'response';
      let pendingId = null;

      try {
        throwIfAborted(signal);

        let logits = Array.isArray(options.userTokenIds)
          ? await this._prepareTurn(options.userTokenIds, wantsThinking, signal, onRebuild)
          : await this._preparePrompt(messages, wantsThinking, signal, onRebuild);

        for (let i = 0; i < maxNewTokens; i++) {
          throwIfAborted(signal);
          const nextId = this._sample(logits, temperature, topK);
          if (this._shouldStop(nextId)) {
            // The model may try to end the turn while its think trace is
            // still open — with no-think turns earlier in the context it
            // imitates their <|assistant|>...<|end|> shape and never emits
            // </think>. Since thinking was explicitly requested, force the
            // trace closed and keep going so the answer still gets
            // generated. Mode flips to 'response', so this fires at most
            // once per turn.
            if (mode === 'thinking' && this.sessionOpenThinking) {
              mode = 'response';
              logits = await this._appendToken(this.thinkCloseId, signal, onRebuild);
              await nextFrame();
              continue;
            }
            if (nextId === this.eosId) {
              await this._appendToken(nextId, signal, onRebuild);
            }
            break;
          }

          if (nextId === this.thinkCloseId) {
            mode = 'response';
            logits = await this._appendToken(nextId, signal, onRebuild);
          } else if (nextId === this.thinkOpenId || this.tokenizer.isSpecialId(nextId)) {
            logits = await this._appendToken(nextId, signal, onRebuild);
          } else {
            pendingId = nextId;
            yield { kind: mode, token: { id: nextId, text: this.tokenizer.decodeToken(nextId) } };
            logits = await this._appendToken(pendingId, signal, onRebuild);
            pendingId = null;
          }

          await nextFrame();
          throwIfAborted(signal);
        }
      } finally {
        if (pendingId !== null) {
          await this._appendToken(pendingId, null, onRebuild);
        }
        await this._ensureAssistantTurnEnded(onRebuild);
      }
    }

    async _prefillContext(ids, signal = null) {
      let logits = null;
      for (let i = 0; i < ids.length; i++) {
        throwIfAborted(signal);
        logits = await this.forwardToken(ids[i], i);
        if ((i & 1) === 1) {
          await nextFrame();
          throwIfAborted(signal);
        }
      }
      return logits;
    }

    async _preparePrompt(messages, thinking, signal, onRebuild) {
      let promptIds = this.tokenizer.encode(this._buildPrompt(messages, thinking));
      if (!promptIds.length) promptIds = [this.eosId];

      const alignedAt = this._findSessionInPrompt(promptIds);
      if (alignedAt < 0) {
        this.sessionIds = [];
        this.sessionLogits = null;
        this.sessionOpenAssistant = false;
        this.sessionOpenThinking = false;
        await this._appendTokens(promptIds, signal, onRebuild);
        return this.sessionLogits;
      }

      const suffix = promptIds.slice(alignedAt + this.sessionIds.length);
      if (suffix.length) await this._appendTokens(suffix, signal, onRebuild);
      return this.sessionLogits;
    }

    async _prepareTurn(userTokenIds, thinking, signal, onRebuild) {
      await this._closeAssistantTurn(signal, onRebuild);
      const ids = [this.userId, ...userTokenIds, this.eosId, this.assistantId];
      if (thinking) ids.push(this.thinkOpenId);
      await this._appendTokens(ids, signal, onRebuild);
      return this.sessionLogits;
    }

    async _closeAssistantTurn(signal, onRebuild) {
      if (!this.sessionOpenAssistant) return;
      const ids = [];
      if (this.sessionOpenThinking) ids.push(this.thinkCloseId);
      ids.push(this.eosId);
      await this._appendTokens(ids, signal, onRebuild);
    }

    async _ensureAssistantTurnEnded(onRebuild) {
      await this._closeAssistantTurn(null, onRebuild);
    }

    async ensureAssistantTurnEnded(onRebuild) {
      await this._ensureAssistantTurnEnded(onRebuild);
    }

    _findSessionInPrompt(promptIds) {
      const needle = this.sessionIds;
      if (!needle.length) return 0;
      if (needle.length > promptIds.length) return -1;

      const lastStart = promptIds.length - needle.length;
      for (let start = lastStart; start >= 0; start--) {
        let ok = true;
        for (let i = 0; i < needle.length; i++) {
          if (promptIds[start + i] !== needle[i]) {
            ok = false;
            break;
          }
        }
        if (ok) return start;
      }
      return -1;
    }

    async _appendTokens(ids, signal, onRebuild) {
      let logits = this.sessionLogits;
      for (let i = 0; i < ids.length; i++) {
        logits = await this._appendToken(ids[i], signal, onRebuild);
        if ((i & 1) === 1) {
          await nextFrame();
          throwIfAborted(signal);
        }
      }
      return logits;
    }

    async _appendToken(id, signal, onRebuild) {
      throwIfAborted(signal);
      if (this.sessionIds.length >= this.maxContext) {
        await this._compactSession(signal, onRebuild);
      }
      throwIfAborted(signal);
      const position = this.sessionIds.length;
      const logits = await this.forwardToken(id, position);
      this.sessionIds.push(id);
      this.sessionLogits = logits;
      this._noteSessionToken(id);
      return logits;
    }

    _noteSessionToken(id) {
      if (id === this.userId) {
        this.sessionOpenAssistant = false;
        this.sessionOpenThinking = false;
      } else if (id === this.assistantId) {
        this.sessionOpenAssistant = true;
        this.sessionOpenThinking = false;
      } else if (id === this.thinkOpenId && this.sessionOpenAssistant) {
        this.sessionOpenThinking = true;
      } else if (id === this.thinkCloseId) {
        this.sessionOpenThinking = false;
      } else if (id === this.eosId || id === this.eotId) {
        this.sessionOpenAssistant = false;
        this.sessionOpenThinking = false;
      }
    }

    async _compactSession(signal, onRebuild) {
      const before = this.sessionIds.length;
      const retained = this._retainedSessionIds();
      if (onRebuild) {
        onRebuild({ phase: 'start', dropped: before - retained.length, kept: retained.length });
      }
      try {
        this.sessionIds = [];
        this.sessionLogits = null;
        this.sessionLogits = await this._prefillContext(retained, signal);
        this.sessionIds = retained.slice();
      } finally {
        if (onRebuild) onRebuild({ phase: 'end' });
      }
    }

    _retainedSessionIds() {
      const ids = this.sessionIds;
      if (ids.length <= this.retainContext) return ids.slice();

      let start = ids.length - this.retainContext;
      const maxStart = Math.min(ids.length - 1, start + this.boundaryLookahead);
      for (let i = start; i <= maxStart; i++) {
        if (ids[i] === this.userId) {
          start = i;
          break;
        }
      }
      return ids.slice(start);
    }

    forwardToken(tokenId, position) {
      this.embed.embedding(tokenId, this.hidden);

      for (let layerIdx = 0; layerIdx < this.numLayers; layerIdx++) {
        const layer = this.layers[layerIdx];

        rmsNorm(this.hidden, layer.inputNorm, this.normed, this.arch.rms_norm_eps);
        layer.qProj.matvec(this.normed, this.q);
        layer.kProj.matvec(this.normed, this.k);
        layer.vProj.matvec(this.normed, this.v);
        this._applyRope(this.q, this.k, position);

        layer.kCache.set(this.k, position * this.hiddenSize);
        layer.vCache.set(this.v, position * this.hiddenSize);

        this._attention(layer, position, this.q, this.attn);
        layer.oProj.matvec(this.attn, this.proj);
        addInPlace(this.hidden, this.proj);

        rmsNorm(this.hidden, layer.postNorm, this.normed, this.arch.rms_norm_eps);
        layer.upProj.matvec(this.normed, this.mlp);
        addInPlace(this.mlp, layer.upBias);
        geluInPlace(this.mlp);
        layer.downProj.matvec(this.mlp, this.proj);
        addInPlace(this.hidden, this.proj);
      }

      rmsNorm(this.hidden, this.finalNorm, this.normed, this.arch.rms_norm_eps);
      this.embed.matvec(this.normed, this.logits);
      return this.logits;
    }

    _attention(layer, position, q, out) {
      out.fill(0.0);
      const seqLen = position + 1;
      const hdim = this.headDim;
      const hidden = this.hiddenSize;

      for (let h = 0; h < this.numHeads; h++) {
        const base = h * hdim;
        let maxScore = -Infinity;
        for (let t = 0; t < seqLen; t++) {
          const kBase = t * hidden + base;
          let dot = 0.0;
          for (let i = 0; i < hdim; i++) dot += q[base + i] * layer.kCache[kBase + i];
          const score = dot * this.scale;
          this.scores[t] = score;
          if (score > maxScore) maxScore = score;
        }

        let denom = 0.0;
        for (let t = 0; t < seqLen; t++) {
          const e = Math.exp(this.scores[t] - maxScore);
          this.scores[t] = e;
          denom += e;
        }
        const invDenom = denom > 0 ? 1.0 / denom : 0.0;

        for (let t = 0; t < seqLen; t++) {
          const w = this.scores[t] * invDenom;
          const vBase = t * hidden + base;
          for (let i = 0; i < hdim; i++) out[base + i] += w * layer.vCache[vBase + i];
        }
      }
    }

    // RoPE uses interleaved (adjacent) pairs: channels (2i, 2i+1) rotate
    // together, with freq_i = 1 / theta^(2i / head_dim). This differs from
    // Llama-style split-half RoPE and matches the reference exactly.
    _initRope() {
      const half = this.headDim / 2;
      for (let pos = 0; pos < this.maxContext; pos++) {
        const base = pos * half;
        for (let i = 0; i < half; i++) {
          const invFreq = 1.0 / Math.pow(this.arch.rope_theta, (2 * i) / this.headDim);
          const angle = pos * invFreq;
          this.ropeCos[base + i] = Math.cos(angle);
          this.ropeSin[base + i] = Math.sin(angle);
        }
      }
    }

    _applyRope(q, k, position) {
      const half = this.headDim / 2;
      const ropeBase = position * half;
      for (let h = 0; h < this.numHeads; h++) {
        const base = h * this.headDim;
        for (let i = 0; i < half; i++) {
          const cos = this.ropeCos[ropeBase + i];
          const sin = this.ropeSin[ropeBase + i];
          const i0 = base + 2 * i;
          const i1 = i0 + 1;

          const q0 = q[i0];
          const q1 = q[i1];
          q[i0] = q0 * cos - q1 * sin;
          q[i1] = q1 * cos + q0 * sin;

          const k0 = k[i0];
          const k1 = k[i1];
          k[i0] = k0 * cos - k1 * sin;
          k[i1] = k1 * cos + k0 * sin;
        }
      }
    }

    _buildPrompt(messages, thinking) {
      let out = '';
      let openThinking = false;
      for (const msg of messages) {
        if (msg.role === 'user') {
          if (openThinking) {
            out += ROLE_TOKENS.thinkClose + ROLE_TOKENS.end;
            openThinking = false;
          }
          out += ROLE_TOKENS.user + msg.text + ROLE_TOKENS.end;
        } else if (msg.role === 'thinking') {
          out += ROLE_TOKENS.assistant + ROLE_TOKENS.thinkOpen + msg.text;
          openThinking = true;
        } else if (msg.role === 'assistant') {
          if (openThinking) {
            out += ROLE_TOKENS.thinkClose + msg.text + ROLE_TOKENS.end;
            openThinking = false;
          } else {
            out += ROLE_TOKENS.assistant + msg.text + ROLE_TOKENS.end;
          }
        }
      }
      if (openThinking) out += ROLE_TOKENS.thinkClose + ROLE_TOKENS.end;
      out += thinking
        ? ROLE_TOKENS.assistant + ROLE_TOKENS.thinkOpen
        : ROLE_TOKENS.assistant;
      return out;
    }

    _shouldStop(id) {
      return (
        id === this.eosId ||
        id === this.eotId ||
        id === this.userId ||
        id === this.assistantId
      );
    }

    _sample(logits, temperature, topK) {
      if (!temperature || temperature <= 0) return this._argmax(logits);
      const k = Math.max(1, Math.min(topK || 1, logits.length));
      const ids = new Int32Array(k);
      const vals = new Float32Array(k);
      vals.fill(-Infinity);

      for (let id = 0; id < logits.length; id++) {
        const v = logits[id];
        if (v <= vals[k - 1]) continue;
        let j = k - 1;
        while (j > 0 && v > vals[j - 1]) {
          vals[j] = vals[j - 1];
          ids[j] = ids[j - 1];
          j--;
        }
        vals[j] = v;
        ids[j] = id;
      }

      let max = -Infinity;
      for (let i = 0; i < k; i++) {
        vals[i] = vals[i] / temperature;
        if (vals[i] > max) max = vals[i];
      }

      let total = 0.0;
      for (let i = 0; i < k; i++) {
        const p = Math.exp(vals[i] - max);
        vals[i] = p;
        total += p;
      }

      let r = Math.random() * total;
      for (let i = 0; i < k; i++) {
        r -= vals[i];
        if (r <= 0) return ids[i];
      }
      return ids[0];
    }

    _argmax(logits) {
      let best = 0;
      let bestVal = logits[0];
      for (let i = 1; i < logits.length; i++) {
        if (logits[i] > bestVal) {
          bestVal = logits[i];
          best = i;
        }
      }
      return best;
    }
  }

  class GabInferenceEngine {
    static async create(options) {
      const engine = new GabCpuInferenceEngine(options);
      engine.backend = 'cpu';
      await engine.init();
      return engine;
    }
  }

  globalThis.GabTokenizer = GabTokenizer;
  globalThis.GabInferenceEngine = GabInferenceEngine;
})();
