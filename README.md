# Gab 100M

> Try it out: [giftofgab.chat](https://giftofgab.chat)

Gab is a ~100 million parameter language model trained from scratch, along with the full suite of tools used to build it: a tokenizer, a corpus encoder, an MLX pre-trainer, a fine tuner, and a chat inference script. The canonical model definition lives in `reference.js` — a dependency-free JavaScript implementation of the tensor library, autograd, tokenizer, model, KV cache, and AdamW trainer. Everything the Python tools do is defined by matching it.

## Links

- Try it out: [giftofgab.chat](https://giftofgab.chat)
- Model: [gszauer/Gab100M](https://huggingface.co/gszauer/Gab100M)
- Pretraining data: [gszauer/Gab100MPretrain](https://huggingface.co/datasets/gszauer/Gab100MPretrain) and [Cosmopedia v2](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus)
- Fine-tuning data: [gszauer/Gab100MFinetune](https://huggingface.co/datasets/gszauer/Gab100MFinetune) and [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk)
- Blog/site: [gabormakesgames.com](https://gabormakesgames.com)

# The Model

Gab is a decoder-only transformer with pre-normalization (RMSNorm), rotary position embeddings, exact-GeLU feed-forward blocks with a hidden bias, causal self-attention, and tied input/output embeddings.

| Setting | Value |
| --- | ---: |
| Parameters | 99,753,216 |
| Transformer blocks | 13 |
| Feature dimension | 768 |
| Attention heads | 12 |
| Head dimension | 64 |
| MLP hidden dimension | 3072 (4x feature dim) |
| Context length | 2048 (training window) |
| Vocabulary size | 10,000 |
| RoPE theta | 10,000 |
| RMSNorm epsilon | 1e-5 |
| Weight format | Flat float32, headerless |

The shape is defined once in `model_config.py`, shared by the pretrainer (as CLI defaults) and inference (as the trusted shape for headerless weight files). Checkpoints carry their own config, so the finetuner reads the shape from the checkpoint it loads.

## Forward Pass

For token ids `x`, embeddings are looked up from `E`, where `E` has shape `[10000, 768]`:

```text
h = E[x]
```

There is no position embedding table, position enters only through RoPE inside attention. Each of the 13 transformer blocks applies:

```text
h = h + Attention(RMSNorm(h))
h = h + MLP(RMSNorm(h))
```

After the final block:

```text
logits = RMSNorm(h) @ E.T
```

There is no separate output head. The unembedding is the tied embedding matrix.

## RMSNorm

Each norm has one learned scale vector `g` of length 768 and no bias:

```text
RMSNorm(x) = g * x / sqrt(mean(x^2) + 1e-5)
```

## Attention

The attention layer is written per head, and the way head outputs are combined is the interesting part. Most implementations project Q, K, V with one big `[768, 768]` matrix each, run the heads, concatenate the 12 `[seq, 64]` head outputs back into `[seq, 768]`, and apply a single output projection `Wo`. Gab never concatenates. Each head owns four small matrices:

```text
Wq_i, Wk_i, Wv_i : [768, 64]    (feature_dim down to head_dim)
Wo_i             : [64, 768]    (head_dim back up to feature_dim)
```

Each head computes attention in its own 64-wide space, then projects its result all the way back up to 768 on its own, and the per-head results are **summed**:

```text
for each head i:
    q = RMSNorm(h) @ Wq_i          # [seq, 64]
    k = RMSNorm(h) @ Wk_i          # [seq, 64]
    v = RMSNorm(h) @ Wv_i          # [seq, 64]
    q, k = RoPE(q), RoPE(k)
    scores = (q @ k.T) / sqrt(64)
    attn_i = softmax(causal_mask(scores)) @ v    # [seq, 64]

Attention(h) = sum_i (attn_i @ Wo_i)             # [seq, 768]
```

This is algebraically identical to the concat formulation: concatenating head outputs and multiplying by one big `[768, 768]` output matrix is the same as multiplying each head's output by that matrix's corresponding 64-row slice and summing. Writing it as a sum of per-head `[seq, 64] @ [64, 768]` multiplies removes the concat/reshape step from both the forward and backward pass entirely — each head is an independent chain that just accumulates into the output. Total attention parameters per block are unchanged: 12 heads × 4 matrices × 49,152 values.

All projections are bias-free. Causal masking is only applied when scoring a square block (the prompt's first pass); during cached generation each new token already only sees its past, so no mask is needed.

## RoPE

Rotary position embeddings use **interleaved** (adjacent) pairs — channels `(0,1), (2,3), ... (62,63)` rotate together. This differs from Llama-style split-half RoPE, where channel `i` pairs with channel `i + 32`. For pair index `i` in a 64-wide head:

```text
freq_i = 1 / 10000^(2i / 64)
angle  = position * freq_i

[a, b] -> [a * cos(angle) - b * sin(angle), a * sin(angle) + b * cos(angle)]
```

The angles are fixed functions of position and frequency, not learned. During cached generation, new tokens are rotated by their true position (cache length + row) rather than their row index, so cached keys keep the rotation they were stored with.

## MLP

The feed-forward path is a two-matrix GeLU MLP — no gate, but with a learned per-neuron bias on the hidden layer:

```text
u = x @ Wup + b        # Wup: [768, 3072], b: [3072], starts at zero
m = GeLU(u)
MLP(x) = m @ Wdown     # Wdown: [3072, 768]
```

GeLU is the exact form, `x * Phi(x)` where `Phi` is the standard normal CDF:

```text
GeLU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
```

(The JavaScript reference inlines `erf` via the Abramowitz & Stegun 7.1.26 approximation, accurate to about 1e-7, since JS has no built-in.)

## Sampling

Generation supports temperature (0 means greedy), top-k, and top-p (nucleus) sampling, in that order: scale logits by `1/temperature`, softmax, keep the `k` most likely tokens, then keep tokens from the top until their combined probability crosses `p`, and draw from the renormalized survivors.

A KV cache holds one key tensor and one value tensor per block, per head, growing one row per generated token. Each forward pass only computes the tokens the cache hasn't seen.

## Training

The trainer is AdamW with next-token cross-entropy (log-sum-exp form, fused forward/backward):

| Hyperparameter | Value |
| --- | ---: |
| Peak learning rate | 3e-4 (pre-train), 3e-5 (fine-tune) |
| Schedule | Linear warmup, cosine decay to 3e-5 floor |
| Betas | 0.9, 0.999 |
| Epsilon | 1e-8 |
| Weight decay | 0.01 (decoupled) |
| Gradient clip | 1.0 (global L2 norm) |

The loss supports a per-token mask; fine-tuning uses it so only tokens inside assistant replies contribute to the loss.

## Weight Format

Final weights are a single headerless flat float32 buffer: every parameter tensor's data, concatenated in `parameters()` order:

1. Token embeddings `[10000, 768]`
2. For each of the 13 blocks: attention norm gamma; then for each of the 12 heads Q, K, V, O; then MLP norm gamma; `Wup`; `Wdown`; hidden bias
3. Final norm gamma

The same file loads into the JavaScript reference via `deserializeFromArrayBuffer` and into the Python tools. Because the file has no header, the shape comes from `model_config.py` (training checkpoints, by contrast, embed their own config).

# Tokenizer Design

The tokenizer is byte-level BPE with a total vocabulary of 10,000: ids 0–255 are the raw bytes, and the rest are learned merges. Reserved tokens (`<|user|>`, `<|assistant|>`, `<|end|>`, `<|endoftext|>`, `<think>`, `</think>`) are taught as chained merges before training, so they always encode to a single id and are atomic — they can never be produced by encoding ordinary text.

Before counting pairs, text is split by a custom pre-tokenizer (not the GPT-2 regex):

1. Reserved tokens always win and are atomic.
2. A run of letters forms a word, optionally carrying one leading space.
3. Digits are grouped to at most 3 per chunk.
4. Anything else (punctuation, leftover spaces) is its own chunk.

Merges never cross chunk boundaries. Decoding unrolls each id back through its merge rules to raw bytes. The serialized `{reserved, merges}` JSON and the encode/decode algorithms are byte-identical between `tokenizer.py` and `reference.js`.

# Chat Format

The model supports user, assistant, end, and thinking tokens. It does not use a system role. Turns are concatenated directly:

```text
<|user|>QUESTION<|end|><|assistant|>ANSWER<|end|>
```

Multi-turn conversations continue the same pattern:

```text
<|user|>QUESTION<|end|><|assistant|>ANSWER<|end|><|user|>QUESTION<|end|><|assistant|>ANSWER<|end|>
```

Thinking is optional and lives inside the assistant turn:

```text
<|user|>QUESTION<|end|><|assistant|><think>PRIVATE_REASONING</think>ANSWER<|end|>
```

The inference script can seed the assistant response with `<think>` so the model continues in thinking mode. Documents in the pre-training corpus are separated by `<|endoftext|>`.

# Training Data

The model was pre-trained on [gszauer/Gab100MPretrain](https://huggingface.co/datasets/gszauer/Gab100MPretrain) plus [Cosmopedia v2](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) (from the SmolLM corpus), and fine-tuned into a chat model on [gszauer/Gab100MFinetune](https://huggingface.co/datasets/gszauer/Gab100MFinetune) plus [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk). Fine-tuning conversations use the chat format above; only assistant replies contribute to the fine-tuning loss.

The encoded pre-training corpus is 33,882,142,110 tokens (about 63 GiB); the encoded fine-tuning corpus is 1,016,529,229 tokens (about 1.9 GiB).

# Training Suite

The tools below take raw text to a chatting model in four steps: train a vocabulary, encode a corpus, pre-train, fine-tune. `reference.js` is the canonical implementation; the Python tools match its tokenizer, model math, and weight layout exactly, so weights and vocabularies move freely between the two.

## Tokenizer

Use the tokenizer to create a new vocabulary

```
tokenizer.py -i "input/path" -o vocab.json -m 10000 -r "<|user|>" -r "<|assistant|>
```

* The -i argument is for input path. It is required. It points to a folder that will be recursivley scanned for txt files. Can have multiple -i's
* The -o argument is for output file. It is optional. If not provided, defaults to vocab.json
* The -m argument is for target vocabulary size. It is optional. If not provided, it defaults to 10000.
* The -r argument is for reserved token. It's optional, and can have multiple. It specifies tokens to reserve before training

Example usage: 

```
python3 tokenizer.py -i "/Users/user/Documents/Gab100M/Pretrain" -o vocab.json -m 10000 -r "<|user|>" -r "<|assistant|>" -r "<|end|>" -r "<|endoftext|>" -r "<think>" -r "</think>"
```

## Encoder

Use the encoder to pre-tokenize many text files into a single stream.

```
encoder.py -v "vocab.json" -i "input/path" -i "second/path" -o corpus.bin -s "<|endoftext|>"
```

* The -i argument is for input path. It's required. Can have more than one -i argument. It points to a folder that will be recursivley scanned for txt files.
* The -v argument is for vocab, it is required, it is the output of the tokenizer.
* The -o argument is for output. It's optional. If not provided, defaults to corpus.bin
* The -s argument is for seperator. It's optional. If provided, that's the token that will be inserted betwen files. If not provided, no seperator is used.

```
python3 encoder.py -v "vocab.json" -i "/Users/user/Documents/smolllm/cosmopedia_v2" -i "/Users/user/Documents/Gab100M/Pretrain" -o "pretrain.corpus" -s "<|endoftext|>"
```

## Pretrainer

Use the pre-trainer to train a model. It takes a vocab and an encoded corpus. It is implemented with MLX (it was trained on a Mac Studio) and uses fused/flash attention to stay within memory budget. The last sample will probably be shorter than the full context length; that's ok.

Checkpoints are time based: after a training step, if over an hour has passed since the last checkpoint, one is saved. The final training step also saves a checkpoint. Only one checkpoint exists at a time, written atomically: saved with a .tmp extension, then the old one is deleted and the new one renamed into place. The checkpoint is a single file, named after the output with a .checkpoint extension. Re-running the same command auto-resumes from it.

```
pretrainer.py -v "vocab.json" -i pretrain.corpus -o pretrained.weights
```

Arguments:
* The -v argument specifies a vocabulary. Required, only one.
* The -i argument specifies a training corpus. The output of the encoder. Required, only one.
* The -o argument specifies the output file name. Optional, defaults to pretrained.bin
* -f. Feature dimension, how many feature dims the model will have. Optional, defaults to 768
* -n. num heads, how many heads the model will have. Optional, defaults to 12.
* -r, rope base, what the rope base is. Optional, defaults to 10000
* -b, number of transformer blocks. Optional, defaults to 13
* -c, context length (training window size). optional, defaults to 2048
* -e, number of epochs. Optional, defaults to 1
* -B, batches per step. How many batches run at the same time. Default 16
* -w, warmup per step. It's a fraction. like 0.01

The vocab size isn't specified because it is inferred from the vocab file.

The defaults for -f/-n/-r/-b/-c live in model_config.py, shared with inference.py, one place that describes the model shape. The finetuner and debug tools don't need it, they read the shape from the checkpoint they load.

A few constraints:
The feature dimension divides evenly across the heads, `768 / 12 = 64`, If not, print an error.
`head_dim` must be even, so RoPE can pair its components. If not, print an error.

## Fine tuner

Use the fine tuner to turn the pre-trained model into a chat model. It trains on conversations, and only learns from the assistant's replies.

Fine-tune data is prepared with the same encoder as pre-training. Each conversation is one txt file, already formatted with the reserved tokens: `<|user|>question<|end|><|assistant|>reply<|end|>` (multi-turn just repeats; `<think>...</think>` inside an assistant turn is fine and gets learned too). Encode the folder(s) into a single corpus with a separator between files. The fine tuner splits the corpus back into individual conversations on the separator token, and computes the assistant-reply loss mask directly from the token ids — so the encoder needs no changes and the corpus format stays the same.

Like the pretrainer, this is MLX, very verbose, and uses the same checkpointing scheme (hourly + final, single `.checkpoint` file, atomic tmp-then-rename, auto-resume by re-running the same command).

```
encoder.py -v "vocab.json" -i "conversations/path" -o finetune.corpus -s "<|endoftext|>"
finetuner.py -v "vocab.json" -i finetune.corpus -c "pretrained.weights.checkpoint" -o finetuned.weights
```

* The -v argument is for vocab, it is required. Needed to look up the marker and separator token ids.
* The -i argument is the encoded fine-tune corpus, the output of the encoder. Required, only one (the encoder already merges multiple folders).
* The -c argument is the pre-trained checkpoint to start from. Required. The model shape (feature dim, heads, blocks, context, rope base) is read from the checkpoint, so none of the shape arguments exist here. The optimizer starts fresh.
* The -o argument is for output. Optional, defaults to finetuned.weights. Final weights are exported here (same JS-loadable format as the pretrainer); the training checkpoint is `<-o>.checkpoint`.
* -s, the separator token the corpus was encoded with. Optional, defaults to `<|endoftext|>`. Must match the encoder's -s, it's how conversations are told apart. The fine tuner errors out if the separator never appears in the corpus (forgot -s when encoding).
* -e, number of epochs. Optional, defaults to 1.
* -B, batch. How many conversations per training step. Optional, defaults to 16.
* -w, warmup fraction of total steps. Optional, defaults to 0.01.
* -l, peak learning rate. Optional, defaults to 3e-5 (pre-training peaked at 3e-4; fine-tuning wants roughly 10x lower so it doesn't wreck the pre-trained weights).

How training works:

* Every conversation (separator-to-separator segment) is one sample. Samples are shuffled each epoch (deterministic seed, so resume replays the same order).
* Loss masking: only tokens inside assistant replies (everything after each `<|assistant|>` up to and including its `<|end|>`) contribute to the loss. User turns and the `<|user|>`/`<|assistant|>` markers are context only. This is the same tokenMask idea the JS AdamWTrainer.crossEntropyLoss supports.
* Batches pad conversations to the longest in the batch; padding contributes no loss.
* Conversations longer than the model's context length are skipped and counted, reported at the end.
* A conversation with no assistant reply has nothing to learn from; skipped and counted too.

Example usage:

```
python3 encoder.py -v vocab.json -i "/Users/user/Documents/Gab100M/Finetune/SimpleConvoThinking-finetune" -i "/Users/user/Documents/Gab100M/Finetune/SimpleConvo-finetune" -i "/Users/user/Documents/smolllm/smoltalk_finetune" -o finetune.corpus -s "<|endoftext|>"
python3 finetuner.py -v vocab.json -i finetune.corpus -c pretrained.weights.checkpoint -o finetuned.weights
```

## Inference

Chat with the fine-tuned model. Loads the final weights (the flat MiniGPT.js-compatible file, which has no header, the model shape comes from model_config.py) and presents a chat interface. Generation uses a KV cache (ported from the MiniGPT.js KVCache): each token and each new turn only computes what the cache hasn't seen, so replies stay fast no matter how long the conversation gets. Formatting is automatic: your message is wrapped in `<|user|>...<|end|><|assistant|>` tags (plus `<think>` when think mode is on), replies stop at `<|end|>`/`<|endoftext|>` or the token limit, and the whole conversation is threaded into each turn. When the conversation outgrows the context window the oldest turns are dropped (with a printed note).

```
python3 inference.py -w finetuned.weights -t vocab.json
```

* -w the model to load: either final weights (finetuned.weights) or a training checkpoint (finetuned.weights.checkpoint), auto-detected. Optional, defaults to finetuned.weights. A checkpoint carries its own shape config; a flat weights file gets its shape from model_config.py.
* -t the tokenizer. Optional, defaults to vocab.json
* --temperature / --top-k / --top-p / --limit set the starting values for the in-chat settings.
* --no-think starts with think mode off. Think is ON by default: replies are forged with a `<think>` tag, the trace shows under a `thinking>` label, and the answer follows under `model>`. The tags themselves are never printed. The trace is buffered until the model closes it, if a reply ends without a `</think>` (the model answered directly, which mixed think/no-think training data allows), the text is shown under `model>` and the unused think tag is dropped from the conversation history.

On start it prints: `Type and press enter to chat, /help for commands.`

Replies print no stats; the only marker is `[limit reached]` when a reply was cut off by the token limit.

Commands:

* /temperature X: sampling temperature, 0 is greedy (alias /temp)
* /top-k N: keep only the N most likely tokens
* /top-p X: nucleus sampling threshold
* /limit N: max tokens per reply
* /think on|off: force a `<think>` trace in replies
* /reset: start a new conversation
* /settings: show current settings and context usage
* /quit: exit (Ctrl-D works too; Ctrl-C during a reply stops that reply)

## Web export

Use website/webexport.py to package final weights for the browser runtime in `website/`.

```
python3 website/webexport.py -w weights/finetuned.weights -v weights/vocab.json
```

* -w the flat float32 weights file. Optional, defaults to finetuned.weights.
* -v the tokenizer vocab. Optional, defaults to vocab.json.
* -o the output directory. Optional, defaults to the website folder the script lives in.

It writes two model variants, each with a tensor manifest, plus a copy of the vocab as `tokenizer.json`:

* **F32** (`model-f32.json` + `model-f32-000.bin`...): full float32 precision, about 381 MiB, sharded so every file stays under GitHub's 100 MB limit.
* **Q8** (`model-q8.json` + `model-q8.bin`): symmetric rowwise int8 quantization (`weight[row, col] = int8[row, col] * scale[row]`, one float32 scale per row), about 96 MiB in a single file. Matrices are quantized; norm gammas and the MLP hidden bias stay float32.

The per-head attention matrices are fused on export: because summing per-head outputs equals a concat plus one big output projection, the fused `[768, 768]` tensors are exact, and the browser engine runs the standard fused formulation with bit-identical float32 results.

# Browser Runtime

`website/` is the source for [giftofgab.chat](https://giftofgab.chat) — the model running entirely in the browser. `website/inference.js` is a self-contained inference engine (no training machinery): the byte-level BPE tokenizer and a pure-JavaScript CPU forward pass — the model is small enough that no GPU is needed — with a persistent float32 KV cache so each new token only computes what the cache hasn't seen. When a conversation outgrows the 2048-token window, the engine keeps the most recent half (snapped to a `<|user|>` turn boundary) and rebuilds the cache.

Downloaded model files are stored in IndexedDB, so later visits load from the device. The UI has a thinking toggle that seeds replies with `<think>`, and a raw view that shows the tokenized transcript exactly as the model sees it.
