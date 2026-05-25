'use strict';

/* =========================================================================
   Gift of Gab browser inference runtime.

   This is a memory-oriented Q8 CPU engine:
     - weights remain rowwise int8 + fp32 scales
     - no full fp32 dequantization copy is created
     - autoregressive generation uses a persistent per-layer fp32 KV cache

   The public surface used by interface.js is:
     new GabTokenizer(tokenizerJson)
     await GabInferenceEngine.create({ manifest, weightsBuffer, tokenizer, ... })
     for await (const item of engine.generate(messages, options)) ...
   ========================================================================= */
(function () {
  const ROLE_TOKENS = {
    user: '<|user|>',
    assistant: '<|assistant|>',
    end: '<|end|>',
    thinkOpen: '<think>',
    thinkClose: '</think>',
    pad: '<|pad|>',
  };

  const PRETOKEN_PATTERN =
    /'(?:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s+$|\s*[\r\n]|\s+(?!\S)|\s/giu;

  function bytesToUnicode() {
    const bs = [];
    for (let i = 33; i <= 126; i++) bs.push(i);
    for (let i = 161; i <= 172; i++) bs.push(i);
    for (let i = 174; i <= 255; i++) bs.push(i);
    const cs = bs.slice();
    let n = 0;
    for (let b = 0; b < 256; b++) {
      if (!bs.includes(b)) {
        bs.push(b);
        cs.push(256 + n);
        n++;
      }
    }
    const enc = new Map();
    const dec = new Map();
    for (let i = 0; i < bs.length; i++) {
      const ch = String.fromCodePoint(cs[i]);
      enc.set(bs[i], ch);
      dec.set(ch, bs[i]);
    }
    return { enc, dec };
  }

  function escapeRegex(text) {
    return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }

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

  class GabTokenizer {
    constructor(tokenizerJson) {
      const model = tokenizerJson.model;
      this.vocab = model.vocab;
      this.idToToken = [];
      for (const [token, id] of Object.entries(this.vocab)) this.idToToken[id] = token;

      this.mergeRanks = new Map();
      for (let i = 0; i < model.merges.length; i++) {
        const line = model.merges[i];
        const sep = line.indexOf(' ');
        if (sep < 0) continue;
        this.mergeRanks.set(line.slice(0, sep) + '\u0000' + line.slice(sep + 1), i);
      }

      this.specials = new Map();
      this.specialIds = new Map();
      for (const item of tokenizerJson.added_tokens || []) {
        this.specials.set(item.content, item.id);
        this.specialIds.set(item.id, item.content);
        this.idToToken[item.id] = item.content;
      }

      const ordered = [...this.specials.keys()].sort((a, b) => b.length - a.length);
      this.specialRe = ordered.length
        ? new RegExp('(' + ordered.map(escapeRegex).join('|') + ')', 'g')
        : null;

      const bytes = bytesToUnicode();
      this.byteEncoder = bytes.enc;
      this.byteDecoder = bytes.dec;
      this.encoder = new TextEncoder();
      this.decoder = new TextDecoder('utf-8');
      this.cache = new Map();
    }

    idFor(token) {
      return this.specials.get(token);
    }

    isSpecialId(id) {
      return this.specialIds.has(id);
    }

    encode(text) {
      if (!text) return [];
      const out = [];
      const chunks = this.specialRe ? text.split(this.specialRe).filter(Boolean) : [text];
      for (const chunk of chunks) {
        const special = this.specials.get(chunk);
        if (special !== undefined) {
          out.push(special);
          continue;
        }
        PRETOKEN_PATTERN.lastIndex = 0;
        let match;
        while ((match = PRETOKEN_PATTERN.exec(chunk)) !== null) {
          const bytes = this.encoder.encode(match[0]);
          out.push(...this._bpe(bytes));
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
      const special = this.specialIds.get(id);
      if (special !== undefined) return special;
      const token = this.idToToken[id];
      if (token === undefined) return '';
      const bytes = [];
      for (const ch of token) {
        const b = this.byteDecoder.get(ch);
        if (b !== undefined) bytes.push(b);
      }
      if (!bytes.length) return '';
      return this.decoder.decode(new Uint8Array(bytes));
    }

    _bpe(rawBytes) {
      let cacheKey = '';
      for (let i = 0; i < rawBytes.length; i++) cacheKey += String.fromCharCode(rawBytes[i]);
      const cached = this.cache.get(cacheKey);
      if (cached) return cached;

      let symbols = new Array(rawBytes.length);
      for (let i = 0; i < rawBytes.length; i++) symbols[i] = this.byteEncoder.get(rawBytes[i]);

      if (symbols.length >= 2) {
        while (true) {
          let bestRank = Infinity;
          let bestIdx = -1;
          for (let i = 0; i < symbols.length - 1; i++) {
            const rank = this.mergeRanks.get(symbols[i] + '\u0000' + symbols[i + 1]);
            if (rank !== undefined && rank < bestRank) {
              bestRank = rank;
              bestIdx = i;
            }
          }
          if (bestIdx < 0) break;
          symbols.splice(bestIdx, 2, symbols[bestIdx] + symbols[bestIdx + 1]);
        }
      }

      const ids = symbols.map((s) => this.vocab[s]);
      if (this.cache.size < 200000) this.cache.set(cacheKey, ids);
      return ids;
    }
  }

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

  function erfApprox(x) {
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

  class GabCpuInferenceEngine {
    constructor({ manifest, weightsBuffer, tokenizer, maxContext = 512 }) {
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
      this.padId = tokenizer.idFor(ROLE_TOKENS.pad);
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
      } else if (id === this.eosId || id === this.padId) {
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

          const q0 = q[base + i];
          const q1 = q[base + i + half];
          q[base + i] = q0 * cos - q1 * sin;
          q[base + i + half] = q1 * cos + q0 * sin;

          const k0 = k[base + i];
          const k1 = k[base + i + half];
          k[base + i] = k0 * cos - k1 * sin;
          k[base + i + half] = k1 * cos + k0 * sin;
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
        id === this.padId ||
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

  const WEBGPU_COMMON_WGSL = `
struct U32Buf { data: array<u32>, }
struct F32Buf { data: array<f32>, }

fn read_i8(byte_offset: u32) -> i32 {
  let word = model.data[byte_offset >> 2u];
  let shift = (byte_offset & 3u) * 8u;
  let byte = (word >> shift) & 255u;
  return select(i32(byte), i32(byte) - 256, byte >= 128u);
}

fn read_f32(byte_offset: u32) -> f32 {
  return bitcast<f32>(model.data[byte_offset >> 2u]);
}
`;

  const WEBGPU_EMBED_WGSL = WEBGPU_COMMON_WGSL + `
struct EmbedParams {
  q_byte: u32,
  scale_byte: u32,
  token_id: u32,
  cols: u32,
}

@group(0) @binding(0) var<storage, read> model: U32Buf;
@group(0) @binding(1) var<storage, read_write> out: F32Buf;
@group(0) @binding(2) var<uniform> params: EmbedParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let col = gid.x;
  if (col >= params.cols) { return; }
  let q = f32(read_i8(params.q_byte + params.token_id * params.cols + col));
  let s = read_f32(params.scale_byte + params.token_id * 4u);
  out.data[col] = q * s;
}
`;

  const WEBGPU_RMS_WGSL = WEBGPU_COMMON_WGSL + `
struct RmsParams {
  weight_byte: u32,
  len: u32,
  pad0: u32,
  eps: f32,
}

@group(0) @binding(0) var<storage, read> model: U32Buf;
@group(0) @binding(1) var<storage, read> input: F32Buf;
@group(0) @binding(2) var<storage, read_write> output: F32Buf;
@group(0) @binding(3) var<uniform> params: RmsParams;

var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid3: vec3<u32>) {
  let lid = lid3.x;
  var ss = 0.0;
  var i = lid;
  loop {
    if (i >= params.len) { break; }
    let v = input.data[i];
    ss += v * v;
    i += 256u;
  }
  red[lid] = ss;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
    if (lid < stride) { red[lid] += red[lid + stride]; }
    workgroupBarrier();
  }
  let inv = 1.0 / sqrt(red[0] / f32(params.len) + params.eps);
  var j = lid;
  loop {
    if (j >= params.len) { break; }
    output.data[j] = input.data[j] * inv * read_f32(params.weight_byte + j * 4u);
    j += 256u;
  }
}
`;

  const WEBGPU_MATVEC_WGSL = WEBGPU_COMMON_WGSL + `
struct MatParams {
  q_byte: u32,
  scale_byte: u32,
  rows: u32,
  cols: u32,
}

@group(0) @binding(0) var<storage, read> model: U32Buf;
@group(0) @binding(1) var<storage, read> input: F32Buf;
@group(0) @binding(2) var<storage, read_write> output: F32Buf;
@group(0) @binding(3) var<uniform> params: MatParams;

var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
  let row = wid.x;
  let lid = lid3.x;
  if (row >= params.rows) { return; }
  var sum = 0.0;
  var col = lid;
  let row_base = params.q_byte + row * params.cols;
  loop {
    if (col >= params.cols) { break; }
    sum += f32(read_i8(row_base + col)) * input.data[col];
    col += 256u;
  }
  red[lid] = sum;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
    if (lid < stride) { red[lid] += red[lid + stride]; }
    workgroupBarrier();
  }
  if (lid == 0u) {
    output.data[row] = red[0] * read_f32(params.scale_byte + row * 4u);
  }
}
`;

  const WEBGPU_ROPE_WGSL = `
struct F32Buf { data: array<f32>, }
struct PosParams {
  position: u32,
  seq_len: u32,
  hidden_size: u32,
  head_dim: u32,
  num_heads: u32,
  max_context: u32,
  pad0: u32,
  rope_theta: f32,
}

@group(0) @binding(0) var<storage, read_write> q: F32Buf;
@group(0) @binding(1) var<storage, read_write> k: F32Buf;
@group(0) @binding(2) var<storage, read> v: F32Buf;
@group(0) @binding(3) var<storage, read_write> k_cache: F32Buf;
@group(0) @binding(4) var<storage, read_write> v_cache: F32Buf;
@group(0) @binding(5) var<uniform> params: PosParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.hidden_size) { return; }
  let cache_base = params.position * params.hidden_size;
  v_cache.data[cache_base + idx] = v.data[idx];

  let d = idx % params.head_dim;
  let half = params.head_dim / 2u;
  if (d >= half) { return; }
  let base = idx - d;
  let mate = base + d + half;
  let angle = f32(params.position) / pow(params.rope_theta, f32(2u * d) / f32(params.head_dim));
  let c = cos(angle);
  let s = sin(angle);

  let q0 = q.data[idx];
  let q1 = q.data[mate];
  q.data[idx] = q0 * c - q1 * s;
  q.data[mate] = q1 * c + q0 * s;

  let k0 = k.data[idx];
  let k1 = k.data[mate];
  let kr0 = k0 * c - k1 * s;
  let kr1 = k1 * c + k0 * s;
  k.data[idx] = kr0;
  k.data[mate] = kr1;
  k_cache.data[cache_base + idx] = kr0;
  k_cache.data[cache_base + mate] = kr1;
}
`;

  const WEBGPU_ATTN_SCORE_WGSL = `
struct F32Buf { data: array<f32>, }
struct PosParams {
  position: u32,
  seq_len: u32,
  hidden_size: u32,
  head_dim: u32,
  num_heads: u32,
  max_context: u32,
  pad0: u32,
  rope_theta: f32,
}

@group(0) @binding(0) var<storage, read> q: F32Buf;
@group(0) @binding(1) var<storage, read> k_cache: F32Buf;
@group(0) @binding(2) var<storage, read_write> scores: F32Buf;
@group(0) @binding(3) var<storage, read_write> head_sums: F32Buf;
@group(0) @binding(4) var<uniform> params: PosParams;

var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
  let head = wid.x;
  let lid = lid3.x;
  let head_base = head * params.head_dim;
  let score_base = head * params.max_context;
  let scale = 1.0 / sqrt(f32(params.head_dim));

  var max_score = -3.402823e38;
  var t = lid;
  loop {
    if (t >= params.seq_len) { break; }
    var dot = 0.0;
    let k_base = t * params.hidden_size + head_base;
    for (var i = 0u; i < params.head_dim; i++) {
      dot += q.data[head_base + i] * k_cache.data[k_base + i];
    }
    let score = dot * scale;
    scores.data[score_base + t] = score;
    max_score = max(max_score, score);
    t += 256u;
  }

  red[lid] = max_score;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
    if (lid < stride) { red[lid] = max(red[lid], red[lid + stride]); }
    workgroupBarrier();
  }
  let m = red[0];

  var sum = 0.0;
  t = lid;
  loop {
    if (t >= params.seq_len) { break; }
    let e = exp(scores.data[score_base + t] - m);
    scores.data[score_base + t] = e;
    sum += e;
    t += 256u;
  }
  red[lid] = sum;
  workgroupBarrier();
  for (var stride2 = 128u; stride2 > 0u; stride2 = stride2 >> 1u) {
    if (lid < stride2) { red[lid] += red[lid + stride2]; }
    workgroupBarrier();
  }
  if (lid == 0u) { head_sums.data[head] = red[0]; }
}
`;

  const WEBGPU_ATTN_VALUE_WGSL = `
struct F32Buf { data: array<f32>, }
struct PosParams {
  position: u32,
  seq_len: u32,
  hidden_size: u32,
  head_dim: u32,
  num_heads: u32,
  max_context: u32,
  pad0: u32,
  rope_theta: f32,
}

@group(0) @binding(0) var<storage, read> v_cache: F32Buf;
@group(0) @binding(1) var<storage, read> scores: F32Buf;
@group(0) @binding(2) var<storage, read> head_sums: F32Buf;
@group(0) @binding(3) var<storage, read_write> out: F32Buf;
@group(0) @binding(4) var<uniform> params: PosParams;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
  let head = wid.x;
  let dim = lid3.x;
  if (dim >= params.head_dim) { return; }
  let head_base = head * params.head_dim;
  let score_base = head * params.max_context;
  let denom = max(head_sums.data[head], 1e-20);
  var acc = 0.0;
  for (var t = 0u; t < params.seq_len; t++) {
    let w = scores.data[score_base + t] / denom;
    acc += w * v_cache.data[t * params.hidden_size + head_base + dim];
  }
  out.data[head_base + dim] = acc;
}
`;

  const WEBGPU_ADD_WGSL = `
struct F32Buf { data: array<f32>, }
struct AddParams { len: u32, pad0: u32, pad1: u32, pad2: u32, }
@group(0) @binding(0) var<storage, read_write> base: F32Buf;
@group(0) @binding(1) var<storage, read> delta: F32Buf;
@group(0) @binding(2) var<uniform> params: AddParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.len) { return; }
  base.data[i] += delta.data[i];
}
`;

  const WEBGPU_GELU_WGSL = `
struct F32Buf { data: array<f32>, }
struct GeluParams { len: u32, pad0: u32, pad1: u32, pad2: u32, }
@group(0) @binding(0) var<storage, read_write> x: F32Buf;
@group(0) @binding(1) var<uniform> params: GeluParams;

fn erf_approx(v: f32) -> f32 {
  let sign = select(-1.0, 1.0, v >= 0.0);
  let a = abs(v);
  let t = 1.0 / (1.0 + 0.3275911 * a);
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t) * exp(-(a * a));
  return sign * y;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.len) { return; }
  let v = x.data[i];
  x.data[i] = 0.5 * v * (1.0 + erf_approx(v * 0.7071067811865476));
}
`;

  class GabWebGpuInferenceEngine {
    constructor({ manifest, weightsBuffer, tokenizer, maxContext, device }) {
      this.manifest = manifest;
      this.weightsBuffer = weightsBuffer;
      this.tokenizer = tokenizer;
      this.device = device;
      this.arch = manifest.architecture;
      this.maxContext = Math.min(maxContext || this.arch.max_position_embeddings, this.arch.max_position_embeddings);
      this.hiddenSize = this.arch.hidden_size;
      this.intermediateSize = this.arch.intermediate_size;
      this.numLayers = this.arch.num_hidden_layers;
      this.numHeads = this.arch.num_attention_heads;
      this.headDim = this.arch.head_dim;
      this.vocabSize = this.arch.vocab_size;
      this.backend = 'webgpu';

      this.eosId = tokenizer.idFor(ROLE_TOKENS.end);
      this.padId = tokenizer.idFor(ROLE_TOKENS.pad);
      this.userId = tokenizer.idFor(ROLE_TOKENS.user);
      this.assistantId = tokenizer.idFor(ROLE_TOKENS.assistant);
      this.thinkOpenId = tokenizer.idFor(ROLE_TOKENS.thinkOpen);
      this.thinkCloseId = tokenizer.idFor(ROLE_TOKENS.thinkClose);
      initSessionState(this);
    }

    static async create(options) {
      if (options.manifest.format !== 'gift-of-gab-q8') return null;
      if (typeof navigator === 'undefined' || !navigator.gpu) return null;
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) return null;
      const bytes = options.weightsBuffer.byteLength;
      if (adapter.limits.maxBufferSize < bytes || adapter.limits.maxStorageBufferBindingSize < bytes) {
        return null;
      }
      const device = await adapter.requestDevice();
      const engine = new GabWebGpuInferenceEngine({ ...options, device });
      await engine.init();
      return engine;
    }

    async init() {
      const d = this.device;
      this.tensorEntries = new Map(this.manifest.tensors.map((t) => [t.name, t]));

      this.modelBuffer = this._buffer(this.weightsBuffer.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      this._writeLargeBuffer(this.modelBuffer, this.weightsBuffer);

      this.hidden = this._storage(this.hiddenSize * 4);
      this.normed = this._storage(this.hiddenSize * 4);
      this.q = this._storage(this.hiddenSize * 4);
      this.k = this._storage(this.hiddenSize * 4);
      this.v = this._storage(this.hiddenSize * 4);
      this.attn = this._storage(this.hiddenSize * 4);
      this.proj = this._storage(this.hiddenSize * 4);
      this.mlp = this._storage(this.intermediateSize * 4);
      this.logits = this._buffer(this.vocabSize * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      this.logitsRead = this._buffer(this.vocabSize * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
      this.scores = this._storage(this.numHeads * this.maxContext * 4);
      this.headSums = this._storage(this.numHeads * 4);
      this.embedParams = this._uniform(16);
      this.posParams = this._uniform(32);

      await this._createPipelines();
      this._createBindGroups();
      await d.queue.onSubmittedWorkDone();
    }

    _buffer(size, usage) {
      return this.device.createBuffer({ size: Math.ceil(size / 4) * 4, usage });
    }

    _storage(size) {
      return this._buffer(size, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    }

    _uniform(size) {
      return this._buffer(size, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    }

    _writeLargeBuffer(buffer, source) {
      const chunk = 16 * 1024 * 1024;
      for (let offset = 0; offset < source.byteLength; offset += chunk) {
        const size = Math.min(chunk, source.byteLength - offset);
        this.device.queue.writeBuffer(buffer, offset, source, offset, size);
      }
    }

    async _createPipelines() {
      const d = this.device;
      const make = (code) => d.createComputePipelineAsync({
        layout: 'auto',
        compute: { module: d.createShaderModule({ code }), entryPoint: 'main' },
      });
      [
        this.embedPipe,
        this.rmsPipe,
        this.matvecPipe,
        this.ropePipe,
        this.attnScorePipe,
        this.attnValuePipe,
        this.addPipe,
        this.geluPipe,
      ] = await Promise.all([
        make(WEBGPU_EMBED_WGSL),
        make(WEBGPU_RMS_WGSL),
        make(WEBGPU_MATVEC_WGSL),
        make(WEBGPU_ROPE_WGSL),
        make(WEBGPU_ATTN_SCORE_WGSL),
        make(WEBGPU_ATTN_VALUE_WGSL),
        make(WEBGPU_ADD_WGSL),
        make(WEBGPU_GELU_WGSL),
      ]);
    }

    _createBindGroups() {
      const d = this.device;
      const embedEntry = this._entry('model.embed_tokens.weight');
      this.embedBg = d.createBindGroup({
        layout: this.embedPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.modelBuffer } },
          { binding: 1, resource: { buffer: this.hidden } },
          { binding: 2, resource: { buffer: this.embedParams } },
        ],
      });

      const addParams = this._paramsU32([this.hiddenSize, 0, 0, 0]);
      this.addBg = d.createBindGroup({
        layout: this.addPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.hidden } },
          { binding: 1, resource: { buffer: this.proj } },
          { binding: 2, resource: { buffer: addParams } },
        ],
      });
      const geluParams = this._paramsU32([this.intermediateSize, 0, 0, 0]);
      this.geluBg = d.createBindGroup({
        layout: this.geluPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.mlp } },
          { binding: 1, resource: { buffer: geluParams } },
        ],
      });

      this.finalNormBg = this._makeRmsBindGroup(this.hidden, this.normed, this._entry('model.norm.weight'));
      this.logitsBg = this._makeMatvecBindGroup(this.normed, this.logits, embedEntry);
      this.layers = [];
      for (let i = 0; i < this.numLayers; i++) {
        const kCache = this._storage(this.maxContext * this.hiddenSize * 4);
        const vCache = this._storage(this.maxContext * this.hiddenSize * 4);
        this.layers.push({
          inputNormBg: this._makeRmsBindGroup(this.hidden, this.normed, this._entry(`model.layers.${i}.input_layernorm.weight`)),
          postNormBg: this._makeRmsBindGroup(this.hidden, this.normed, this._entry(`model.layers.${i}.post_attention_layernorm.weight`)),
          qBg: this._makeMatvecBindGroup(this.normed, this.q, this._entry(`model.layers.${i}.self_attn.q_proj.weight`)),
          kBg: this._makeMatvecBindGroup(this.normed, this.k, this._entry(`model.layers.${i}.self_attn.k_proj.weight`)),
          vBg: this._makeMatvecBindGroup(this.normed, this.v, this._entry(`model.layers.${i}.self_attn.v_proj.weight`)),
          oBg: this._makeMatvecBindGroup(this.attn, this.proj, this._entry(`model.layers.${i}.self_attn.o_proj.weight`)),
          upBg: this._makeMatvecBindGroup(this.normed, this.mlp, this._entry(`model.layers.${i}.mlp.up_proj.weight`)),
          downBg: this._makeMatvecBindGroup(this.mlp, this.proj, this._entry(`model.layers.${i}.mlp.down_proj.weight`)),
          ropeBg: this._makeRopeBindGroup(kCache, vCache),
          attnScoreBg: this._makeAttnScoreBindGroup(kCache),
          attnValueBg: this._makeAttnValueBindGroup(vCache),
        });
      }
    }

    _entry(name) {
      const entry = this.tensorEntries.get(name);
      if (!entry) throw new Error('Missing tensor: ' + name);
      return entry;
    }

    _paramsU32(values) {
      const buf = this._uniform(values.length * 4);
      this.device.queue.writeBuffer(buf, 0, new Uint32Array(values));
      return buf;
    }

    _paramsRms(weightEntry) {
      const buf = this._uniform(16);
      const raw = new ArrayBuffer(16);
      const u32 = new Uint32Array(raw);
      const f32 = new Float32Array(raw);
      u32[0] = weightEntry.data.offset;
      u32[1] = this.hiddenSize;
      u32[2] = 0;
      f32[3] = this.arch.rms_norm_eps;
      this.device.queue.writeBuffer(buf, 0, raw);
      return buf;
    }

    _makeRmsBindGroup(input, output, weightEntry) {
      return this.device.createBindGroup({
        layout: this.rmsPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.modelBuffer } },
          { binding: 1, resource: { buffer: input } },
          { binding: 2, resource: { buffer: output } },
          { binding: 3, resource: { buffer: this._paramsRms(weightEntry) } },
        ],
      });
    }

    _makeMatvecBindGroup(input, output, entry) {
      const params = this._paramsU32([entry.q.offset, entry.scale.offset, entry.shape[0], entry.shape[1]]);
      return this.device.createBindGroup({
        layout: this.matvecPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.modelBuffer } },
          { binding: 1, resource: { buffer: input } },
          { binding: 2, resource: { buffer: output } },
          { binding: 3, resource: { buffer: params } },
        ],
      });
    }

    _makeRopeBindGroup(kCache, vCache) {
      return this.device.createBindGroup({
        layout: this.ropePipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.q } },
          { binding: 1, resource: { buffer: this.k } },
          { binding: 2, resource: { buffer: this.v } },
          { binding: 3, resource: { buffer: kCache } },
          { binding: 4, resource: { buffer: vCache } },
          { binding: 5, resource: { buffer: this.posParams } },
        ],
      });
    }

    _makeAttnScoreBindGroup(kCache) {
      return this.device.createBindGroup({
        layout: this.attnScorePipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.q } },
          { binding: 1, resource: { buffer: kCache } },
          { binding: 2, resource: { buffer: this.scores } },
          { binding: 3, resource: { buffer: this.headSums } },
          { binding: 4, resource: { buffer: this.posParams } },
        ],
      });
    }

    _makeAttnValueBindGroup(vCache) {
      return this.device.createBindGroup({
        layout: this.attnValuePipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: vCache } },
          { binding: 1, resource: { buffer: this.scores } },
          { binding: 2, resource: { buffer: this.headSums } },
          { binding: 3, resource: { buffer: this.attn } },
          { binding: 4, resource: { buffer: this.posParams } },
        ],
      });
    }

    async *generate(messages, options = {}) {
      yield* GabCpuInferenceEngine.prototype.generate.call(this, messages, options);
    }

    async _prefillContext(ids, signal = null) {
      return GabCpuInferenceEngine.prototype._prefillContext.call(this, ids, signal);
    }

    async _preparePrompt(messages, thinking, signal, onRebuild) {
      return GabCpuInferenceEngine.prototype._preparePrompt.call(this, messages, thinking, signal, onRebuild);
    }

    async _prepareTurn(userTokenIds, thinking, signal, onRebuild) {
      return GabCpuInferenceEngine.prototype._prepareTurn.call(this, userTokenIds, thinking, signal, onRebuild);
    }

    async _closeAssistantTurn(signal, onRebuild) {
      return GabCpuInferenceEngine.prototype._closeAssistantTurn.call(this, signal, onRebuild);
    }

    async _ensureAssistantTurnEnded(onRebuild) {
      return GabCpuInferenceEngine.prototype._ensureAssistantTurnEnded.call(this, onRebuild);
    }

    async ensureAssistantTurnEnded(onRebuild) {
      return GabCpuInferenceEngine.prototype.ensureAssistantTurnEnded.call(this, onRebuild);
    }

    _findSessionInPrompt(promptIds) {
      return GabCpuInferenceEngine.prototype._findSessionInPrompt.call(this, promptIds);
    }

    async _appendTokens(ids, signal, onRebuild) {
      return GabCpuInferenceEngine.prototype._appendTokens.call(this, ids, signal, onRebuild);
    }

    async _appendToken(id, signal, onRebuild) {
      return GabCpuInferenceEngine.prototype._appendToken.call(this, id, signal, onRebuild);
    }

    async _compactSession(signal, onRebuild) {
      return GabCpuInferenceEngine.prototype._compactSession.call(this, signal, onRebuild);
    }

    _retainedSessionIds() {
      return GabCpuInferenceEngine.prototype._retainedSessionIds.call(this);
    }

    _noteSessionToken(id) {
      return GabCpuInferenceEngine.prototype._noteSessionToken.call(this, id);
    }

    async forwardToken(tokenId, position) {
      const embedParams = new Uint32Array([this._entry('model.embed_tokens.weight').q.offset, this._entry('model.embed_tokens.weight').scale.offset, tokenId, this.hiddenSize]);
      this.device.queue.writeBuffer(this.embedParams, 0, embedParams);

      const raw = new ArrayBuffer(32);
      const u32 = new Uint32Array(raw);
      const f32 = new Float32Array(raw);
      u32[0] = position;
      u32[1] = position + 1;
      u32[2] = this.hiddenSize;
      u32[3] = this.headDim;
      u32[4] = this.numHeads;
      u32[5] = this.maxContext;
      u32[6] = 0;
      f32[7] = this.arch.rope_theta;
      this.device.queue.writeBuffer(this.posParams, 0, raw);

      const enc = this.device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(this.embedPipe);
      pass.setBindGroup(0, this.embedBg);
      pass.dispatchWorkgroups(Math.ceil(this.hiddenSize / 256));

      for (const layer of this.layers) {
        this._dispatch(pass, this.rmsPipe, layer.inputNormBg, 1);
        this._dispatch(pass, this.matvecPipe, layer.qBg, this.hiddenSize);
        this._dispatch(pass, this.matvecPipe, layer.kBg, this.hiddenSize);
        this._dispatch(pass, this.matvecPipe, layer.vBg, this.hiddenSize);
        this._dispatch(pass, this.ropePipe, layer.ropeBg, Math.ceil(this.hiddenSize / 256));
        this._dispatch(pass, this.attnScorePipe, layer.attnScoreBg, this.numHeads);
        this._dispatch(pass, this.attnValuePipe, layer.attnValueBg, this.numHeads);
        this._dispatch(pass, this.matvecPipe, layer.oBg, this.hiddenSize);
        this._dispatch(pass, this.addPipe, this.addBg, Math.ceil(this.hiddenSize / 256));
        this._dispatch(pass, this.rmsPipe, layer.postNormBg, 1);
        this._dispatch(pass, this.matvecPipe, layer.upBg, this.intermediateSize);
        this._dispatch(pass, this.geluPipe, this.geluBg, Math.ceil(this.intermediateSize / 256));
        this._dispatch(pass, this.matvecPipe, layer.downBg, this.hiddenSize);
        this._dispatch(pass, this.addPipe, this.addBg, Math.ceil(this.hiddenSize / 256));
      }

      this._dispatch(pass, this.rmsPipe, this.finalNormBg, 1);
      this._dispatch(pass, this.matvecPipe, this.logitsBg, this.vocabSize);
      pass.end();
      enc.copyBufferToBuffer(this.logits, 0, this.logitsRead, 0, this.vocabSize * 4);
      this.device.queue.submit([enc.finish()]);

      await this.logitsRead.mapAsync(GPUMapMode.READ);
      const logits = new Float32Array(this.logitsRead.getMappedRange()).slice();
      this.logitsRead.unmap();
      return logits;
    }

    _dispatch(pass, pipeline, bindGroup, workgroups) {
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(workgroups);
    }

    _buildPrompt(messages, thinking) {
      return GabCpuInferenceEngine.prototype._buildPrompt.call(this, messages, thinking);
    }

    _shouldStop(id) {
      return GabCpuInferenceEngine.prototype._shouldStop.call(this, id);
    }

    _sample(logits, temperature, topK) {
      return GabCpuInferenceEngine.prototype._sample.call(this, logits, temperature, topK);
    }
  }

  class GabInferenceEngine {
    static async create(options) {
      try {
        const gpu = await GabWebGpuInferenceEngine.create(options);
        if (gpu) return gpu;
      } catch (err) {
        console.warn('WebGPU backend unavailable; falling back to CPU:', err);
      }
      const cpu = new GabCpuInferenceEngine(options);
      cpu.backend = 'cpu';
      await cpu.init();
      return cpu;
    }
  }

  globalThis.GabTokenizer = GabTokenizer;
  globalThis.GabInferenceEngine = GabInferenceEngine;
})();
