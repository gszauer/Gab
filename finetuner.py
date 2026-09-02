#!/usr/bin/env python3
"""finetuner.py — MLX fine-tuner: turn the pre-trained MiniGPT into a chat model.

Takes an encoder-produced corpus of conversations (one conversation per source
file, separated by the separator token) and a pre-trained checkpoint, and trains
on the conversations — learning ONLY from the assistant's replies.

Conversations are recovered by splitting the token stream on the separator id.
The loss mask is computed directly from the token ids: a token is learnable iff
the nearest control token (<|user|>, <|assistant|>, <|end|>, separator) strictly
before it is <|assistant|> — i.e. assistant reply tokens, including the closing
<|end|> (so the model learns to stop), and any <think>...</think> inside the
reply. User turns, markers, and padding contribute no loss. This is the same
tokenMask idea MiniGPT.js AdamWTrainer.crossEntropyLoss supports.

Model shape comes from the -c checkpoint; there are no shape arguments. The
optimizer starts fresh. Checkpointing matches the pretrainer: a single
`<-o>.checkpoint`, saved hourly + on the final step, written atomically, with
auto-resume by re-running the same command. Final weights are exported to -o in
the JS-loadable MiniGPT.js format.

Usage:
    python3 finetuner.py -v vocab.json -i finetune.corpus \\
        -c pretrained.weights.checkpoint [-o finetuned.weights]
        [-s "<|endoftext|>"] [-e 1] [-B 16] [-w 0.01] [-l 3e-5]
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
from tokenizer import Tokenizer, _format_duration, _format_bytes
from pretrainer import (
    MiniGPT, save_checkpoint, load_checkpoint, export_weights, sample_max_id,
    CHECKPOINT_INTERVAL_SEC, BETA1, BETA2, EPS, WEIGHT_DECAY, GRAD_CLIP,
)

# Reserved marker strings (must exist in the vocab as single tokens).
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
END_TOKEN = "<|end|>"

_MASK_CHUNK = 1 << 26  # tokens per chunk for the vectorized mask pass (~64M)


def masked_loss(model, inp, tgt, mask):
    """Mean cross-entropy over unmasked targets only (mask is 0/1 float)."""
    logits = model(inp)
    ce = nn.losses.cross_entropy(logits, tgt, reduction="none")  # (B, T)
    return (ce * mask).sum() / mask.sum()


def learning_rate(step, total_steps, warmup_steps, peak_lr):
    """Warmup then cosine decay, like the pretrainer, at fine-tune scale.
    Decays to peak/10, mirroring the pretrainer's 3e-4 -> 3e-5 ratio."""
    min_lr = peak_lr / 10.0
    if warmup_steps > 0 and step < warmup_steps:
        return peak_lr * (step / warmup_steps)
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (peak_lr - min_lr) * cosine


def single_token_id(tokenizer, text, what):
    """Encode a marker string; it must be exactly one token id."""
    ids = tokenizer.encode(text)
    if len(ids) != 1:
        sys.exit(f"Error: {what} {text!r} encodes to {len(ids)} tokens, expected 1. "
                 f"Was it reserved (-r) when the tokenizer was trained?")
    return ids[0]


def find_token_positions(data, n_tokens, token_id):
    """Positions of `token_id` in the corpus, scanned in chunks."""
    found = []
    for s in range(0, n_tokens, _MASK_CHUNK):
        chunk = np.asarray(data[s:s + _MASK_CHUNK])
        hits = np.where(chunk == token_id)[0]
        if hits.size:
            found.append(hits.astype(np.int64) + s)
    return np.concatenate(found) if found else np.empty(0, dtype=np.int64)


def assistant_token_mask(data, n_tokens, assistant_id, other_ctrl_ids):
    """Boolean mask over the whole corpus: True where the nearest control token
    strictly BEFORE the position is <|assistant|>. That marks assistant reply
    tokens including the closing <|end|>; the markers themselves, user text,
    and everything after a non-assistant control stay False. Vectorized, in
    chunks, carrying the control state across chunk boundaries."""
    mask = np.zeros(n_tokens, dtype=bool)
    others = np.asarray(sorted(other_ctrl_ids), dtype=np.uint16)
    carry = -1  # control state entering the next chunk: 1 = assistant, -1 = other/none
    for s in range(0, n_tokens, _MASK_CHUNK):
        chunk = np.asarray(data[s:s + _MASK_CHUNK])
        m = chunk.shape[0]
        code = np.zeros(m, dtype=np.int8)
        code[np.isin(chunk, others)] = -1
        code[chunk == assistant_id] = 1
        pos = np.where(code != 0, np.arange(1, m + 1, dtype=np.int64), 0)
        run = np.maximum.accumulate(pos)        # last control at-or-before i (1-based; 0 = none)
        prev = np.empty(m, dtype=np.int64)
        prev[0] = 0
        prev[1:] = run[:-1]                     # last control strictly before i
        vals = np.where(prev > 0, code[np.clip(prev - 1, 0, m - 1)], carry)
        mask[s:s + m] = (vals == 1)
        if m and run[-1] > 0:
            carry = int(code[run[-1] - 1])
    return mask


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a pre-trained MiniGPT on conversations (assistant replies only).")
    parser.add_argument("-v", required=True, metavar="VOCAB", help="tokenizer JSON (vocab)")
    parser.add_argument("-i", required=True, metavar="CORPUS",
                        help="encoded fine-tune corpus (encoder output)")
    parser.add_argument("-c", required=True, metavar="CKPT",
                        help="pre-trained checkpoint to start from")
    parser.add_argument("-o", default="finetuned.weights", metavar="FILE",
                        help="output weights (default: %(default)s); checkpoint at <-o>.checkpoint")
    parser.add_argument("-s", default="<|endoftext|>", metavar="TOKEN",
                        help="separator token the corpus was encoded with (default: %(default)s)")
    parser.add_argument("-e", type=int, default=1, metavar="EPOCHS", help="epochs")
    parser.add_argument("--batch", "-B", type=int, default=16, metavar="N",
                        help="conversations per training step (default: %(default)s)")
    parser.add_argument("-w", type=float, default=0.01, metavar="FRAC",
                        help="warmup as a fraction of total steps (default: %(default)s)")
    parser.add_argument("-l", type=float, default=3e-5, metavar="LR",
                        help="peak learning rate (default: %(default)s)")
    args = parser.parse_args()

    if not os.path.isfile(args.v):
        sys.exit(f"Error: vocab file not found: {args.v}")
    if not os.path.isfile(args.i):
        sys.exit(f"Error: corpus file not found: {args.i}")
    if not os.path.isfile(args.c):
        sys.exit(f"Error: checkpoint not found: {args.c}")
    if args.batch < 1:
        sys.exit("Error: -B/--batch must be at least 1.")
    if not (0.0 <= args.w < 1.0):
        sys.exit("Error: -w (warmup fraction) must be in [0, 1).")
    if args.l <= 0:
        sys.exit("Error: -l (peak learning rate) must be positive.")

    # --- Vocab + marker ids --------------------------------------------------
    tk = Tokenizer()
    with open(args.v, "r", encoding="utf-8") as f:
        tk.deserialize_from_json(f.read())
    vocab_size = tk.vocab_size()

    sep_id = single_token_id(tk, args.s, "separator")
    user_id = single_token_id(tk, USER_TOKEN, "marker")
    assistant_id = single_token_id(tk, ASSISTANT_TOKEN, "marker")
    end_id = single_token_id(tk, END_TOKEN, "marker")
    print(f"Loaded vocab: {vocab_size} tokens | {USER_TOKEN}={user_id} "
          f"{ASSISTANT_TOKEN}={assistant_id} {END_TOKEN}={end_id} | "
          f"separator {args.s!r}={sep_id}")

    # --- Corpus --------------------------------------------------------------
    data = np.memmap(args.i, dtype="<u2", mode="r")
    n_tokens = data.shape[0]
    if n_tokens < 2:
        sys.exit("Error: corpus has fewer than 2 tokens.")
    max_id = sample_max_id(data, n_tokens)
    if max_id >= vocab_size:
        sys.exit(f"Error: corpus contains token id {max_id} (sampled) but vocab size "
                 f"is only {vocab_size}; vocab and corpus do not match.")
    print(f"Corpus: {n_tokens:,} tokens ({_format_bytes(n_tokens * 2)})")

    # Split into conversations on the separator.
    t0 = time.monotonic()
    sep_pos = find_token_positions(data, n_tokens, sep_id)
    if sep_pos.size == 0:
        sys.exit(f"Error: separator {args.s!r} (id {sep_id}) never appears in the "
                 f"corpus. Was it encoded with -s {args.s!r}?")
    starts = np.concatenate([[0], sep_pos + 1])
    ends = np.concatenate([sep_pos, [n_tokens]])
    print(f"  {sep_pos.size:,} separator(s) -> {starts.size:,} conversation(s) "
          f"({time.monotonic() - t0:.1f}s)")

    # Assistant-reply mask over the whole corpus.
    t0 = time.monotonic()
    mask = assistant_token_mask(data, n_tokens, assistant_id,
                                other_ctrl_ids=[user_id, end_id, sep_id])
    learnable_per_conv = np.add.reduceat(mask.astype(np.int64), starts)
    print(f"  masked assistant replies: {int(mask.sum()):,} learnable of "
          f"{n_tokens:,} tokens ({mask.sum() / n_tokens * 100:.1f}%) "
          f"({time.monotonic() - t0:.1f}s)")

    # --- Model from the pre-trained checkpoint (or resume) -------------------
    checkpoint_path = args.o + ".checkpoint"
    start_step = 0
    resumed = False
    if os.path.exists(checkpoint_path):
        model_items, opt_items, start_step, cfg = load_checkpoint(checkpoint_path)
        resumed = True
        print(f"Resuming from {checkpoint_path} at step {start_step:,} "
              f"(-c {args.c} is ignored on resume).")
        for key, cli, name in [("batch", args.batch, "-B"), ("epochs", args.e, "-e"),
                               ("peak_lr", args.l, "-l")]:
            if cfg.get(key) != cli:
                print(f"  note: {name} from checkpoint ({cfg.get(key)}) overrides CLI ({cli})")
    else:
        model_items, _discard_opt, base_step, cfg = load_checkpoint(args.c)
        opt_items = None  # fresh optimizer for fine-tuning
        print(f"Base model: {args.c} (pre-trained to step {base_step:,})")

    if cfg["vocab_size"] != vocab_size:
        sys.exit(f"Error: model vocab {cfg['vocab_size']} != tokenizer vocab "
                 f"{vocab_size}; wrong vocab for this checkpoint.")

    ctx = cfg["context"]
    model = MiniGPT(cfg["vocab_size"], cfg["feature_dim"], cfg["num_heads"],
                    cfg["rope_base"], cfg["num_blocks"])
    model.update(tree_unflatten(model_items))
    optimizer = optim.AdamW(learning_rate=args.l, betas=[BETA1, BETA2],
                            eps=EPS, weight_decay=WEIGHT_DECAY)
    if opt_items:
        optimizer.state = tree_unflatten(opt_items)
    mx.eval(model.parameters())

    # --- Select trainable conversations --------------------------------------
    lengths = ends - starts
    too_long = lengths > (ctx + 1)          # a sample of L tokens trains L-1 targets
    no_assistant = (learnable_per_conv == 0) & ~too_long
    keep = ~too_long & ~no_assistant & (lengths >= 2)
    no_assistant |= (lengths < 2) & ~too_long
    k_starts = starts[keep]
    k_ends = ends[keep]
    n_kept = k_starts.size
    if n_kept == 0:
        sys.exit("Error: no trainable conversations "
                 "(all skipped as too long or missing an assistant reply).")
    print(f"  conversations: {n_kept:,} kept | {int(too_long.sum()):,} skipped too long "
          f"(> {ctx + 1:,} tokens) | {int(no_assistant.sum()):,} skipped, no assistant reply")

    # --- Training plan -------------------------------------------------------
    if resumed:
        batch, epochs, peak_lr = cfg["batch"], cfg["epochs"], cfg["peak_lr"]
        total_steps, warmup_steps = cfg["total_steps"], cfg["warmup_steps"]
        if cfg.get("n_kept") != n_kept:
            print(f"  warning: kept-conversation count changed since the checkpoint "
                  f"({cfg.get('n_kept'):,} -> {n_kept:,}); did the corpus change?")
        steps_per_epoch = (n_kept + batch - 1) // batch
    else:
        batch, epochs, peak_lr = args.batch, args.e, args.l
        steps_per_epoch = (n_kept + batch - 1) // batch
        total_steps = steps_per_epoch * epochs
        warmup_steps = max(1, round(args.w * total_steps))

    config = {"vocab_size": vocab_size, "feature_dim": cfg["feature_dim"],
              "num_heads": cfg["num_heads"], "rope_base": cfg["rope_base"],
              "num_blocks": cfg["num_blocks"], "context": ctx,
              "batch": batch, "epochs": epochs, "total_steps": total_steps,
              "warmup_steps": warmup_steps, "peak_lr": peak_lr,
              "n_kept": n_kept, "kind": "finetune", "base_checkpoint": args.c}

    n_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"Model: {n_params / 1e6:.1f}M params | dim {cfg['feature_dim']} | "
          f"{cfg['num_heads']} heads | {cfg['num_blocks']} blocks | ctx {ctx}")
    print(f"Fine-tuning: batch {batch} | {steps_per_epoch:,} steps/epoch x {epochs} "
          f"= {total_steps:,} steps | warmup {warmup_steps:,} | peak lr {peak_lr:.1e}")
    print(f"Checkpoint: {checkpoint_path} (every {CHECKPOINT_INTERVAL_SEC // 60} min + final)")

    loss_and_grad = nn.value_and_grad(model, masked_loss)

    step = start_step
    start_epoch = min(step // steps_per_epoch, epochs - 1)
    start_batch = step % steps_per_epoch if step < total_steps else steps_per_epoch

    t_start = time.monotonic()
    last_log = [0.0]
    last_checkpoint = [time.monotonic()]
    tokens_done = [0]
    running_loss = [float("nan")]
    cur_lr = [0.0]
    gnorm_val = [0.0]

    def report(epoch, force=False):
        now = time.monotonic()
        if not force and now - last_log[0] < 1.0:
            return
        last_log[0] = now
        elapsed = now - t_start
        rate = (step - start_step) / elapsed if elapsed > 0 else 0
        eta = (total_steps - step) / rate if rate > 0 else None
        tok_s = tokens_done[0] / elapsed if elapsed > 0 else 0
        ck_in = max(0, CHECKPOINT_INTERVAL_SEC - (now - last_checkpoint[0]))
        print(f"\r  epoch {epoch + 1}/{epochs} | step {step:,}/{total_steps:,} "
              f"({step / total_steps * 100:.1f}%) | loss {running_loss[0]:.3f} "
              f"| lr {cur_lr[0]:.2e} | gnorm {gnorm_val[0]:.2f} | {tok_s:,.0f} tok/s "
              f"| ETA {_format_duration(eta)} | ckpt in {_format_duration(ck_in)}   ",
              end="", flush=True)

    print("Training...")
    for epoch in range(start_epoch, epochs):
        # Deterministic per-epoch shuffle: resume replays the same order.
        order = np.random.default_rng(epoch).permutation(n_kept)
        sb = start_batch if epoch == start_epoch else 0
        for k in range(sb, steps_per_epoch):
            idxs = order[k * batch:(k + 1) * batch]
            b_starts = k_starts[idxs]
            b_ends = k_ends[idxs]
            T = int((b_ends - b_starts).max()) - 1
            rows = idxs.size
            inp_np = np.zeros((rows, T), dtype=np.int32)
            tgt_np = np.zeros((rows, T), dtype=np.int32)
            m_np = np.zeros((rows, T), dtype=np.float32)
            for r in range(rows):
                s, e = int(b_starts[r]), int(b_ends[r])
                L = e - s
                seg = np.asarray(data[s:e], dtype=np.int32)
                inp_np[r, :L - 1] = seg[:-1]
                tgt_np[r, :L - 1] = seg[1:]
                m_np[r, :L - 1] = mask[s + 1:e]

            loss, grads = loss_and_grad(model, mx.array(inp_np), mx.array(tgt_np),
                                        mx.array(m_np))
            grads, gnorm = optim.clip_grad_norm(grads, GRAD_CLIP)

            step += 1
            lr = learning_rate(step, total_steps, warmup_steps, peak_lr)
            optimizer.learning_rate = lr
            optimizer.update(model, grads)
            mx.eval(loss, model.parameters(), optimizer.state)

            tokens_done[0] += inp_np.size
            running_loss[0] = loss.item()
            cur_lr[0] = lr
            gnorm_val[0] = gnorm.item()

            is_final = (step >= total_steps)
            report(epoch, force=is_final)

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

    print(f"  exporting weights to {args.o}...", flush=True)
    mx.eval(model.parameters())
    export_weights(args.o, model, config["num_heads"])
    print(f"  saved {args.o} ({os.path.getsize(args.o) / 1e6:.1f} MB)")

    loss_str = ("n/a (no new steps; run was already complete)"
                if step == start_step else f"{running_loss[0]:.3f}")
    print(f"Done. {step:,} steps, final loss {loss_str}.")
    print(f"  weights (JS-loadable) -> {args.o}")
    print(f"  resume checkpoint     -> {checkpoint_path}")


if __name__ == "__main__":
    main()
