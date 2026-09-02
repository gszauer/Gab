#!/usr/bin/env python3
"""pretrainer.py — MLX pre-trainer for the MiniGPT model (port of MiniGPT.js).

Trains a decoder-only transformer that mirrors MiniGPT.js exactly (verified: the
forward pass matches to float32 epsilon): token embeddings tied to the
unembedding, per-block RMSNorm -> multi-head attention with interleaved RoPE
(causal) -> residual -> RMSNorm -> GELU MLP -> residual, a final RMSNorm, and an
AdamW trainer with warmup + cosine decay, global-norm gradient clipping, and
mean cross-entropy loss.

Runs on Apple Silicon via MLX. Attention uses fused (flash) scaled-dot-product
attention so the score matrix is never materialized. Training is batched: many
context windows per step. The corpus is chunked into windows of `context`+1
tokens; the final window may be shorter than the context length, which is fine.

Checkpointing: a single file `<output>.checkpoint` with model weights, optimizer
state, and step/config metadata. Saved when >1h has passed since the last (checked
after each step) and always on the final step. Written atomically (temp ->
os.replace). If the checkpoint exists at startup, training resumes from it.

Usage:
    python pretrainer.py -v vocab.json -i corpus.bin [-o pretrained.bin]
        [-f 768] [-n 12] [-r 10000] [-b 13] [-c 1024] [-e 1]
        [--batch 16] [-w 0.01]
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_config
from tokenizer import Tokenizer, _format_duration

# AdamW hyperparameters, from the MiniGPT.js AdamWTrainer defaults.
MAX_LR = 3e-4
MIN_LR = 3e-5
BETA1, BETA2 = 0.9, 0.999
EPS = 1e-8
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
RMS_EPS = 1e-5

CHECKPOINT_INTERVAL_SEC = 3600  # save a checkpoint after this long since the last


# --- Model (mirrors MiniGPT.js, with a leading batch dimension) ------------


def rope(x, base):
    """Interleaved RoPE over the last dim (even), position offset 0 — as in JS.

    x: (..., seq, head_dim). Rotates each adjacent component pair (2p, 2p+1) by
    angle = position * base**(-2p/head_dim).
    """
    *lead, seq, dim = x.shape
    half = dim // 2
    inv_freq = mx.exp(-math.log(base) * (2.0 * mx.arange(half, dtype=mx.float32) / dim))
    pos = mx.arange(seq, dtype=mx.float32)
    ang = pos[:, None] * inv_freq[None, :]           # (seq, half)
    cos, sin = mx.cos(ang), mx.sin(ang)
    xp = x.reshape(*lead, seq, half, 2)
    a, b = xp[..., 0], xp[..., 1]
    ra = a * cos - b * sin
    rb = a * sin + b * cos
    return mx.stack([ra, rb], axis=-1).reshape(*lead, seq, dim)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = mx.ones((dim,))

    def __call__(self, x):
        rms = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + RMS_EPS)
        return x * rms * self.gamma


class Block(nn.Module):
    def __init__(self, dim, heads, rope_base):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.rope_base = rope_base
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.attn_norm = RMSNorm(dim)
        # Packed per-head projections (equivalent to JS per-head Q/K/V and summed
        # O). Init scales match JS: Q/K/V use 1/sqrt(dim), O uses 1/sqrt(head_dim).
        sq = 1.0 / math.sqrt(dim)
        so = 1.0 / math.sqrt(self.head_dim)
        self.Wq = mx.random.uniform(-sq, sq, (dim, dim))
        self.Wk = mx.random.uniform(-sq, sq, (dim, dim))
        self.Wv = mx.random.uniform(-sq, sq, (dim, dim))
        self.Wo = mx.random.uniform(-so, so, (dim, dim))

        self.mlp_norm = RMSNorm(dim)
        hidden = dim * 4
        su = 1.0 / math.sqrt(dim)
        sd = 1.0 / math.sqrt(hidden)
        self.Wup = mx.random.uniform(-su, su, (dim, hidden))
        self.Wdown = mx.random.uniform(-sd, sd, (hidden, dim))
        self.bias = mx.zeros((hidden,))

    def __call__(self, x):                            # x: (B, S, dim)
        B, S, _ = x.shape
        h = self.attn_norm(x)
        q = (h @ self.Wq).reshape(B, S, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        k = (h @ self.Wk).reshape(B, S, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        v = (h @ self.Wv).reshape(B, S, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        q = rope(q, self.rope_base)
        k = rope(k, self.rope_base)
        # Fused (flash) causal attention over (B, heads, S, head_dim).
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        o = o.transpose(0, 2, 1, 3).reshape(B, S, self.heads * self.head_dim)
        x = x + o @ self.Wo

        m = self.mlp_norm(x)
        m = nn.gelu(m @ self.Wup + self.bias) @ self.Wdown
        return x + m


class MiniGPT(nn.Module):
    def __init__(self, vocab, dim, heads, rope_base, blocks):
        super().__init__()
        s = 1.0 / math.sqrt(vocab)
        self.tok_emb = mx.random.uniform(-s, s, (vocab, dim))
        self.blocks = [Block(dim, heads, rope_base) for _ in range(blocks)]
        self.final_norm = RMSNorm(dim)

    def __call__(self, ids):                          # ids: (B, S) int
        x = self.tok_emb[ids]
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return x @ self.tok_emb.T                      # tied unembedding


def loss_fn(model, inp, tgt):                          # inp, tgt: (B, S)
    logits = model(inp)                                # (B, S, vocab)
    return mx.mean(nn.losses.cross_entropy(logits, tgt, reduction="none"))


def learning_rate(step, total_steps, warmup_steps):
    """Warmup then cosine decay, matching MiniGPT.js AdamWTrainer.learningRate()."""
    if step < warmup_steps:
        return MAX_LR * (step / warmup_steps)
    if step >= total_steps:
        return MIN_LR
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LR + (MAX_LR - MIN_LR) * cosine


# --- Data batching ----------------------------------------------------------


def count_batches(n_tokens, ctx, batch):
    """Number of optimizer steps in one epoch (full batches + a short tail)."""
    n_full_windows = (n_tokens - 1) // ctx
    n_batches = (n_full_windows + batch - 1) // batch if n_full_windows > 0 else 0
    if n_tokens - n_full_windows * ctx >= 2:
        n_batches += 1
    return n_batches


def iter_batches(data, n_tokens, ctx, batch, start_batch=0):
    """Yield (inp, tgt) numpy int32 arrays for one epoch, starting at `start_batch`.

    Full windows (each `ctx`+1 tokens, stride `ctx`) are grouped `batch` at a
    time into (rows, ctx) arrays via a single contiguous slice + reshape; the
    final (partial) window is one (1, short) batch. Targets across batches are
    contiguous and non-overlapping, so every token is predicted exactly once.
    """
    n_full_windows = (n_tokens - 1) // ctx
    n_full_batches = (n_full_windows + batch - 1) // batch if n_full_windows > 0 else 0
    for k in range(start_batch, n_full_batches):
        b0 = k * batch
        rows = min(batch, n_full_windows - b0)
        s = b0 * ctx
        inp = np.asarray(data[s: s + rows * ctx], dtype=np.int32).reshape(rows, ctx)
        tgt = np.asarray(data[s + 1: s + 1 + rows * ctx], dtype=np.int32).reshape(rows, ctx)
        yield inp, tgt
    rem_start = n_full_windows * ctx
    if (n_tokens - rem_start) >= 2 and start_batch <= n_full_batches:
        w = np.asarray(data[rem_start:n_tokens], dtype=np.int32)
        yield w[None, :-1], w[None, 1:]


def sample_max_id(data, n_tokens, sample=3_000_000):
    """Max token id over a sample (3 contiguous chunks) — a cheap vocab sanity
    check that avoids scanning an 80 GB corpus end to end."""
    if n_tokens <= sample:
        return int(np.asarray(data).max())
    k = sample // 3
    chunks = [data[0:k], data[n_tokens // 2: n_tokens // 2 + k], data[n_tokens - k:n_tokens]]
    return max(int(np.asarray(c).max()) for c in chunks)


# --- Checkpointing ----------------------------------------------------------


def _array_leaves(tree, prefix):
    return {prefix + k: v for k, v in tree_flatten(tree) if isinstance(v, mx.array)}


def save_checkpoint(path, model, optimizer, step, config):
    """Write a single checkpoint file atomically (temp -> os.replace)."""
    arrays = {}
    arrays.update(_array_leaves(model.parameters(), "model."))
    arrays.update(_array_leaves(optimizer.state, "opt."))
    metadata = {"step": str(step), "config": json.dumps(config)}

    tmp = path + ".tmp.safetensors"   # save_safetensors requires a .safetensors name
    mx.save_safetensors(tmp, arrays, metadata=metadata)
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, path)             # atomic: drops the old, renames the new


def load_checkpoint(path):
    arrays, metadata = mx.load(path, format="safetensors", return_metadata=True)
    model_items, opt_items = [], []
    for key, val in arrays.items():
        if key.startswith("model."):
            model_items.append((key[len("model."):], val))
        elif key.startswith("opt."):
            opt_items.append((key[len("opt."):], val))
    return model_items, opt_items, int(metadata["step"]), json.loads(metadata["config"])


def export_weights(path, model, heads):
    """Write final weights as a flat little-endian float32 buffer in MiniGPT.js
    `parameters()` order, loadable directly by JS `deserializeFromArrayBuffer`.

    The packed (dim, dim) Q/K/V/O matrices are unpacked into JS's per-head
    layout (learnedQ/K/V are (dim, head_dim) column slices; learnedO is a
    (head_dim, dim) row slice). Verified identical to a JS forward pass to
    float32 epsilon. Written atomically (temp -> os.replace).
    """
    dim = model.tok_emb.shape[1]
    head_dim = dim // heads
    parts = []

    def add(a):
        parts.append(np.array(a, dtype=np.float32).reshape(-1))

    add(model.tok_emb)                                   # tokenEmbeddings (vocab, dim)
    for blk in model.blocks:
        add(blk.attn_norm.gamma)                         # attentionNorm.gamma
        wq, wk, wv, wo = (np.array(blk.Wq), np.array(blk.Wk),
                          np.array(blk.Wv), np.array(blk.Wo))
        for h in range(heads):                           # per head: Q, K, V, O
            add(wq[:, h * head_dim:(h + 1) * head_dim])  # learnedQ (dim, head_dim)
            add(wk[:, h * head_dim:(h + 1) * head_dim])
            add(wv[:, h * head_dim:(h + 1) * head_dim])
            add(wo[h * head_dim:(h + 1) * head_dim, :])  # learnedO (head_dim, dim)
        add(blk.mlp_norm.gamma)                          # mlpNorm.gamma
        add(blk.Wup)                                     # learnedUp (dim, hidden)
        add(blk.Wdown)                                   # learnedDown (hidden, dim)
        add(blk.bias)                                    # bias (1, hidden)
    add(model.final_norm.gamma)                          # finalNorm.gamma

    buf = np.concatenate(parts).astype("<f4")
    tmp = path + ".tmp"
    buf.tofile(tmp)
    os.replace(tmp, path)


# --- Training ---------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="MLX pre-trainer for the MiniGPT model.")
    parser.add_argument("-v", required=True, metavar="VOCAB", help="tokenizer JSON (vocab)")
    parser.add_argument("-i", required=True, metavar="CORPUS", help="encoded corpus (.bin)")
    parser.add_argument("-o", default="pretrained.bin", metavar="FILE",
                        help="output/base name (default: %(default)s)")
    parser.add_argument("-f", type=int, default=model_config.FEATURE_DIM, metavar="DIM",
                        help="feature dimension (default: %(default)s, from model_config.py)")
    parser.add_argument("-n", type=int, default=model_config.NUM_HEADS, metavar="HEADS",
                        help="number of heads (default: %(default)s, from model_config.py)")
    parser.add_argument("-r", type=int, default=model_config.ROPE_BASE, metavar="BASE",
                        help="RoPE base (default: %(default)s, from model_config.py)")
    parser.add_argument("-b", type=int, default=model_config.NUM_BLOCKS, metavar="BLOCKS",
                        help="transformer blocks (default: %(default)s, from model_config.py)")
    parser.add_argument("-c", type=int, default=model_config.CONTEXT_LENGTH, metavar="CTX",
                        help="context length (default: %(default)s, from model_config.py)")
    parser.add_argument("-e", type=int, default=1, metavar="EPOCHS", help="epochs")
    parser.add_argument("--batch", "-B", type=int, default=16, metavar="N",
                        help="context windows per optimizer step (default: %(default)s; "
                             "throughput saturates around here, larger just costs memory)")
    parser.add_argument("-w", type=float, default=0.01, metavar="FRAC",
                        help="warmup as a fraction of total steps (default: %(default)s)")
    args = parser.parse_args()

    if not os.path.isfile(args.v):
        sys.exit(f"Error: vocab file not found: {args.v}")
    if not os.path.isfile(args.i):
        sys.exit(f"Error: corpus file not found: {args.i}")
    if args.f % args.n != 0:
        sys.exit(f"Error: feature dimension {args.f} is not divisible by {args.n} heads.")
    head_dim = args.f // args.n
    if head_dim % 2 != 0:
        sys.exit(f"Error: head_dim {head_dim} (= {args.f}/{args.n}) must be even for RoPE.")
    if args.batch < 1:
        sys.exit("Error: --batch must be at least 1.")
    if not (0.0 <= args.w < 1.0):
        sys.exit("Error: -w (warmup fraction) must be in [0, 1).")

    tk = Tokenizer()
    with open(args.v, "r", encoding="utf-8") as f:
        tk.deserialize_from_json(f.read())
    vocab_size = tk.vocab_size()

    data = np.memmap(args.i, dtype="<u2", mode="r")
    n_tokens = data.shape[0]
    if n_tokens < 2:
        sys.exit("Error: corpus has fewer than 2 tokens.")
    max_id = sample_max_id(data, n_tokens)
    if max_id >= vocab_size:
        sys.exit(f"Error: corpus contains token id {max_id} (sampled) but vocab size is "
                 f"only {vocab_size}; vocab and corpus do not match.")

    ctx = args.c
    batch = args.batch
    nbpe = count_batches(n_tokens, ctx, batch)
    total_steps = nbpe * args.e
    warmup_steps = max(1, round(args.w * total_steps))

    checkpoint_path = args.o + ".checkpoint"

    # Build model + optimizer, resuming from a checkpoint if present.
    start_step = 0
    if os.path.exists(checkpoint_path):
        model_items, opt_items, start_step, cfg = load_checkpoint(checkpoint_path)
        for key, cli, name in [("feature_dim", args.f, "-f"), ("num_heads", args.n, "-n"),
                               ("rope_base", args.r, "-r"), ("num_blocks", args.b, "-b"),
                               ("context", args.c, "-c"), ("batch", args.batch, "--batch"),
                               ("vocab_size", vocab_size, "vocab")]:
            if cfg.get(key) != cli:
                print(f"  note: {name} from checkpoint ({cfg.get(key)}) overrides CLI ({cli})")
        model = MiniGPT(cfg["vocab_size"], cfg["feature_dim"], cfg["num_heads"],
                        cfg["rope_base"], cfg["num_blocks"])
        model.update(tree_unflatten(model_items))
        optimizer = optim.AdamW(learning_rate=MIN_LR, betas=[BETA1, BETA2],
                                eps=EPS, weight_decay=WEIGHT_DECAY)
        optimizer.state = tree_unflatten(opt_items)
        args.f, args.n, args.r = cfg["feature_dim"], cfg["num_heads"], cfg["rope_base"]
        args.b, ctx, batch = cfg["num_blocks"], cfg["context"], cfg["batch"]
        head_dim = args.f // args.n
        nbpe = count_batches(n_tokens, ctx, batch)
        total_steps = cfg.get("total_steps", nbpe * args.e)
        warmup_steps = cfg.get("warmup_steps", warmup_steps)
        print(f"Resuming from {checkpoint_path} at step {start_step:,}/{total_steps:,}.")
    else:
        model = MiniGPT(vocab_size, args.f, args.n, args.r, args.b)
        optimizer = optim.AdamW(learning_rate=MIN_LR, betas=[BETA1, BETA2],
                                eps=EPS, weight_decay=WEIGHT_DECAY)

    config = {"vocab_size": vocab_size, "feature_dim": args.f, "num_heads": args.n,
              "rope_base": args.r, "num_blocks": args.b, "context": ctx, "batch": batch,
              "epochs": args.e, "total_steps": total_steps, "warmup_steps": warmup_steps}

    mx.eval(model.parameters())
    n_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"Model: {n_params / 1e6:.1f}M params | vocab {vocab_size} | dim {args.f} | "
          f"{args.n} heads (head_dim {head_dim}) | {args.b} blocks | rope {args.r} | ctx {ctx}")
    print(f"Corpus: {n_tokens:,} tokens | batch {batch} | {nbpe:,} steps/epoch x {args.e} "
          f"= {total_steps:,} steps | warmup {warmup_steps:,} ({args.w:.1%})")
    print(f"Checkpoint: {checkpoint_path} (every {CHECKPOINT_INTERVAL_SEC // 60} min + final)")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    step = start_step
    start_epoch = start_step // nbpe if nbpe else 0
    start_batch = start_step % nbpe if nbpe else 0

    t_start = time.monotonic()
    last_log = [0.0]
    last_checkpoint = [time.monotonic()]
    tokens_done = [0]
    running_loss = [float("nan")]
    cur_lr = [0.0]
    gnorm_val = [0.0]

    def report(force=False):
        now = time.monotonic()
        if not force and now - last_log[0] < 1.0:
            return
        last_log[0] = now
        elapsed = now - t_start
        rate = (step - start_step) / elapsed if elapsed > 0 else 0
        eta = (total_steps - step) / rate if rate > 0 else None
        tok_s = tokens_done[0] / elapsed if elapsed > 0 else 0
        ck_in = max(0, CHECKPOINT_INTERVAL_SEC - (now - last_checkpoint[0]))
        print(f"\r  step {step:,}/{total_steps:,} ({step / total_steps * 100:.1f}%) "
              f"| loss {running_loss[0]:.3f} | lr {cur_lr[0]:.2e} | gnorm {gnorm_val[0]:.2f} "
              f"| {tok_s:,.0f} tok/s | ETA {_format_duration(eta)} | ckpt in {_format_duration(ck_in)}   ",
              end="", flush=True)

    print("Training...")
    for epoch in range(start_epoch, args.e):
        sb = start_batch if epoch == start_epoch else 0
        for inp_np, tgt_np in iter_batches(data, n_tokens, ctx, batch, sb):
            inp = mx.array(inp_np)
            tgt = mx.array(tgt_np)

            loss, grads = loss_and_grad(model, inp, tgt)
            grads, gnorm = optim.clip_grad_norm(grads, GRAD_CLIP)

            step += 1
            lr = learning_rate(step, total_steps, warmup_steps)
            optimizer.learning_rate = lr
            optimizer.update(model, grads)
            mx.eval(loss, model.parameters(), optimizer.state)

            tokens_done[0] += inp.size
            running_loss[0] = loss.item()
            cur_lr[0] = lr
            gnorm_val[0] = gnorm.item()

            is_final = (step >= total_steps)
            report(force=is_final)

            now = time.monotonic()
            if is_final or (now - last_checkpoint[0] >= CHECKPOINT_INTERVAL_SEC):
                print()
                which = "final" if is_final else "hourly"
                print(f"  saving {which} checkpoint at step {step:,}...", flush=True)
                save_checkpoint(checkpoint_path, model, optimizer, step, config)
                last_checkpoint[0] = time.monotonic()
                print(f"  saved {checkpoint_path}")
    print()

    if step == start_step or not os.path.exists(checkpoint_path):
        print("  saving final checkpoint...", flush=True)
        save_checkpoint(checkpoint_path, model, optimizer, step, config)
        print(f"  saved {checkpoint_path}")

    # Export the final weights to -o in JS-loadable (MiniGPT.js) float32 format.
    print(f"  exporting weights to {args.o}...", flush=True)
    mx.eval(model.parameters())
    export_weights(args.o, model, config["num_heads"])
    print(f"  saved {args.o} ({os.path.getsize(args.o) / 1e6:.1f} MB)")

    print(f"Done. {step:,} steps, final loss {running_loss[0]:.3f}.")
    print(f"  weights (JS-loadable) -> {args.o}")
    print(f"  resume checkpoint     -> {checkpoint_path}")


if __name__ == "__main__":
    main()
