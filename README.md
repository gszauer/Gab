# Gab

Gab is a small local language model demo that runs entirely in the browser. The model has about 100 million parameters and can be downloaded in either full F32 precision or a smaller Q8 quantized format. Nothing typed into the demo leaves the user's machine.

This repository contains the browser UI, JavaScript inference runtime, tokenizer, and model weight files used by the demo.

## Links

- Model: [gszauer/Gab100M](https://huggingface.co/gszauer/Gab100M)
- Pretraining data: [gszauer/Gab100MPretrain](https://huggingface.co/datasets/gszauer/Gab100MPretrain)
- Fine-tuning data: [gszauer/Gab100MFinetune](https://huggingface.co/datasets/gszauer/Gab100MFinetune)
- Blog/site: [gabormakesgames.com](https://gabormakesgames.com)

## Running Locally

Serve the `web` folder from a local HTTP server:

```bash
cd web
python3 -m http.server 8765 --bind 0.0.0.0
```

Then open:

```text
http://localhost:8765/
```

The app downloads the tokenizer and one model variant on first launch, then stores them in IndexedDB. Subsequent visits load from the browser cache. To switch between model variants, open the `?` menu, choose `Reset`, delete the local model, and reload.

## Model Variants

The web app offers two local model downloads:

| Variant | Approx size | Notes |
| --- | ---: | --- |
| Full F32 | 380 MiB | Higher quality, keeps the fine-tuned weights in float32. |
| Quantized Q8 | 100 MiB | Smaller and usually better for phones, but lower precision can make the model appear less capable. |

Only one variant is cached at a time.

## Chat Format

The model supports user, assistant, end, and thinking tokens. It does not use a system role.

Turns are concatenated directly:

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

The browser UI can optionally seed the assistant response with `<think>` so the model continues in thinking mode.

## Architecture

The Gift Of Gab is a decoder-only transformer with pre-normalization, rotary position embeddings, exact GeLU feed-forward blocks, causal self-attention, and tied input/output embeddings.

| Setting | Value |
| --- | ---: |
| Parameters | 99,711,744 |
| Layers | 12 |
| Hidden size | 768 |
| MLP size | 3456 |
| Attention heads | 12 |
| Head dim | 64 |
| Context length | 4096 active tokens, rolling window in the browser runtime |
| Vocabulary size | 10,000 |
| RoPE theta | 100,000 |
| RMSNorm epsilon | 1e-5 |
| Weight formats | Full F32 or rowwise Q8 |

### Forward Pass

For token ids `x`, embeddings are looked up from `E`, where `E` has shape
`[10000, 768]`:

```text
h = E[x]
```

Each of the 12 transformer blocks applies:

```text
h = h + Attention(RMSNorm(h))
h = h + MLP(RMSNorm(h))
```

After the final block:

```text
logits = RMSNorm(h) @ E.T
```

There is no separate output head. The output projection uses the tied embedding matrix.

### RMSNorm

Each norm has one learned scale vector `g` of length 768 and no bias:

```text
RMSNorm(x) = g * x / sqrt(mean(x^2) + 1e-5)
```

### Attention

Each block projects normalized hidden states with bias-free matrices:

```text
q = x @ Wq.T
k = x @ Wk.T
v = x @ Wv.T
```

Each projection has shape `[768, 768]`. The result is reshaped to `[12 heads, seq, 64]`. RoPE is applied to `q` and `k` before causal attention:

```text
scores = (q @ k.T) / sqrt(64)
attn = softmax(causal_mask(scores)) @ v
Attention(x) = concat(attn_heads) @ Wo.T
```

`Wo` is also `[768, 768]` and bias-free.

### RoPE

Rotary position embeddings use split-half rotation, matching Llama-style RoPE. For each pair index `i` in a 64-wide head:

```text
freq_i = 1 / theta^(2i / 64)
angle = position * freq_i
```

With `theta = 100000`, the first 32 channels and last 32 channels form rotation pairs:

```text
[a, b] -> [a * cos(angle) - b * sin(angle), b * cos(angle) + a * sin(angle)]
```

### MLP

The feed-forward path is a two-matrix GeLU MLP with no gate and no biases:

```text
u = x @ Wup.T
m = GeLU(u)
MLP(x) = m @ Wdown.T
```

`Wup` has shape `[3456, 768]`; `Wdown` has shape `[768, 3456]`.

GeLU uses the exact form:

```text
GeLU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
```

## Tokenizer

The tokenizer is byte-level BPE with 10,000 ids. Special tokens occupy ids 0-16, including:

```text
<|end|>
<|user|>
<|assistant|>
<think>
</think>
```

Normal text is split by the GPT-style regex pre-tokenizer, converted through the GPT-2 byte-to-unicode map, then BPE merges are applied.

## Browser Runtime

Inference runs in JavaScript inside the browser. The runtime uses WebGPU where available, keeps the KV cache in float32, and stores downloaded model files in IndexedDB.

The Q8 model uses symmetric rowwise int8 weights with float32 scales:

```text
weight[row, col] = int8[row, col] * scale[row]
```

The F32 model keeps the original fine-tuned float32 weights, split across multiple files so each file stays under GitHub's upload limit.

## Notes

This is a research project running a small local model. It can speak English and hold short conversations, but it is still a compact experimental model. Expect quirks, especially from the Q8 variant.

The model is dumb by LLM standards.
