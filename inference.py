#!/usr/bin/env python3
"""inference.py — chat with the fine-tuned MiniGPT.

Loads either final weights (the flat float32 MiniGPT.js-compatible file exported
by the finetuner/pretrainer) or a training .checkpoint — auto-detected — plus
the tokenizer, and presents an interactive chat. A flat weights file is
headerless, so its shape comes from model_config.py; a checkpoint carries its
own config and needs nothing else.
Formatting is handled automatically: your message is wrapped as
<|user|>...<|end|><|assistant|> (plus <think> when think mode is on), the reply
streams until <|end|>/<|endoftext|> or the token limit, and the conversation
history is threaded into every turn. When the conversation outgrows the model's
context window, the oldest turns are dropped (a note is printed).

The .weights file is headerless, so the model shape comes from model_config.py
— the same place the pretrainer gets its defaults.

Usage:
    python3 inference.py [-w finetuned.weights] [-t vocab.json]
        [--temperature 1.0] [--top-k 50] [--top-p 0.9] [--limit 256] [--no-think]

Think mode is ON by default: replies are forged with a <think> tag and the
trace streams under a "thinking>" label, then the answer under "model>".

In-chat commands: /help /temperature /top-k /top-p /limit /think /reset
/settings /quit
"""

import argparse
import codecs
import math
import os
import sys

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_config
from tokenizer import Tokenizer
from pretrainer import MiniGPT, load_checkpoint
from pretrainer_debug import token_to_bytes, sample_token
from finetuner import single_token_id, USER_TOKEN, ASSISTANT_TOKEN, END_TOKEN
from finetuner_debug import ENDOFTEXT_TOKEN, THINK_TOKEN


def is_safetensors(path):
    """Checkpoints are safetensors: a little-endian u64 header length followed
    by a JSON header ('{'). Flat .weights files are raw float32s, so this sniff
    tells the two apart without relying on the file name."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(9)
    if len(head) < 9:
        return False
    header_len = int.from_bytes(head[:8], "little")
    return 0 < header_len < size and head[8:9] == b"{"


def expected_param_count(vocab, dim, blocks):
    """Total float32 values in a MiniGPT .weights file of this shape."""
    hidden = dim * 4
    per_block = (dim                # attention norm gamma
                 + 4 * dim * dim    # Wq, Wk, Wv, Wo
                 + dim              # mlp norm gamma
                 + dim * hidden     # up
                 + hidden * dim     # down
                 + hidden)          # bias
    return vocab * dim + blocks * per_block + dim  # + final norm gamma


def import_weights(path, model, heads):
    """Load a flat little-endian float32 buffer written by export_weights (the
    MiniGPT.js `parameters()` order) into the packed MLX model. Exact inverse of
    the export: per-head Q/K/V column slices and O row slices are repacked into
    the (dim, dim) matrices."""
    dim = model.tok_emb.shape[1]
    vocab = model.tok_emb.shape[0]
    head_dim = dim // heads
    hidden = dim * 4

    buf = np.fromfile(path, dtype="<f4")
    expected = expected_param_count(vocab, dim, len(model.blocks))
    if buf.size != expected:
        sys.exit(f"Error: {path} holds {buf.size:,} float32 values but the "
                 f"model_config shape (vocab {vocab}, dim {dim}, "
                 f"{len(model.blocks)} blocks) needs {expected:,}. "
                 f"Wrong weights file or wrong model_config.py.")

    off = 0

    def take(*shape):
        nonlocal off
        n = int(np.prod(shape))
        out = buf[off:off + n].reshape(shape)
        off += n
        return out

    model.tok_emb = mx.array(take(vocab, dim))
    for blk in model.blocks:
        blk.attn_norm.gamma = mx.array(take(dim))
        wq = np.empty((dim, dim), np.float32)
        wk = np.empty((dim, dim), np.float32)
        wv = np.empty((dim, dim), np.float32)
        wo = np.empty((dim, dim), np.float32)
        for h in range(heads):
            cols = slice(h * head_dim, (h + 1) * head_dim)
            wq[:, cols] = take(dim, head_dim)
            wk[:, cols] = take(dim, head_dim)
            wv[:, cols] = take(dim, head_dim)
            wo[cols, :] = take(head_dim, dim)
        blk.Wq, blk.Wk, blk.Wv, blk.Wo = (mx.array(wq), mx.array(wk),
                                          mx.array(wv), mx.array(wo))
        blk.mlp_norm.gamma = mx.array(take(dim))
        blk.Wup = mx.array(take(dim, hidden))
        blk.Wdown = mx.array(take(hidden, dim))
        blk.bias = mx.array(take(hidden))
    model.final_norm.gamma = mx.array(take(dim))
    assert off == buf.size  # guaranteed by the size check above
    mx.eval(model.parameters())


class KVCache:
    """Per-block key/value history for incremental generation — a port of the
    MiniGPT.js KVCache. keys[b] / values[b] are (heads, cached_tokens, head_dim);
    the packed head dimension replaces the JS per-head tensor lists."""

    def __init__(self, num_blocks):
        self.keys = [None] * num_blocks
        self.values = [None] * num_blocks

    @property
    def length(self):
        first = self.keys[0]
        return 0 if first is None else first.shape[1]


def rope_offset(x, base, offset):
    """Interleaved RoPE with a position offset (JS ropeRotated positionOffset):
    row i of x is rotated as absolute position offset + i. x: (heads, S, head_dim)."""
    *lead, seq, dim = x.shape
    half = dim // 2
    inv_freq = mx.exp(-math.log(base) * (2.0 * mx.arange(half, dtype=mx.float32) / dim))
    pos = mx.arange(seq, dtype=mx.float32) + offset
    ang = pos[:, None] * inv_freq[None, :]
    cos, sin = mx.cos(ang), mx.sin(ang)
    xp = x.reshape(*lead, seq, half, 2)
    a, b = xp[..., 0], xp[..., 1]
    return mx.stack([a * cos - b * sin, a * sin + b * cos], axis=-1).reshape(*lead, seq, dim)


def cached_forward(model, new_ids, cache):
    """Forward only `new_ids` (list of ints) through the model, reading and
    growing `cache`. Returns logits (len(new_ids), vocab) for the new tokens.

    Mirrors MiniGPT.js forward(tokenIdArray, cache): new tokens are RoPE-rotated
    by their true position (cache.length offset), new K/V rows are appended to
    the per-block cache, and attention runs over the full history. MLX's fused
    causal mask is bottom-right aligned (verified), so it is exactly right for
    every case: square prefill, single-token steps, and multi-token
    continuations of an existing cache.
    """
    offset = cache.length
    x = model.tok_emb[mx.array(new_ids, dtype=mx.int32)]        # (S, dim)
    seq = x.shape[0]
    for li, blk in enumerate(model.blocks):
        h = blk.attn_norm(x)
        q = (h @ blk.Wq).reshape(seq, blk.heads, blk.head_dim).transpose(1, 0, 2)
        k = (h @ blk.Wk).reshape(seq, blk.heads, blk.head_dim).transpose(1, 0, 2)
        v = (h @ blk.Wv).reshape(seq, blk.heads, blk.head_dim).transpose(1, 0, 2)
        q = rope_offset(q, blk.rope_base, offset)
        k = rope_offset(k, blk.rope_base, offset)
        if cache.keys[li] is not None:
            k = mx.concatenate([cache.keys[li], k], axis=1)
            v = mx.concatenate([cache.values[li], v], axis=1)
        cache.keys[li] = k
        cache.values[li] = v
        o = mx.fast.scaled_dot_product_attention(
            q[None], k[None], v[None], scale=blk.scale, mask="causal")[0]
        o = o.transpose(1, 0, 2).reshape(seq, blk.heads * blk.head_dim)
        x = x + o @ blk.Wo
        m = blk.mlp_norm(x)
        x = x + nn.gelu(m @ blk.Wup + blk.bias) @ blk.Wdown
    x = model.final_norm(x)
    return x @ model.tok_emb.T


HELP = """Commands:
  /temperature X   sampling temperature (0 = greedy)        [current: {temperature}]
  /top-k N         keep the N most likely tokens            [current: {top_k}]
  /top-p X         nucleus sampling threshold               [current: {top_p}]
  /limit N         max tokens per reply                     [current: {limit}]
  /think on|off    force a <think> trace in replies         [current: {think}]
  /reset           start a new conversation
  /settings        show current settings and context usage
  /help            this text
  /quit            exit (also Ctrl-D)"""


def main():
    parser = argparse.ArgumentParser(description="Chat with the fine-tuned MiniGPT.")
    parser.add_argument("-w", default="finetuned.weights", metavar="WEIGHTS",
                        help="weights file to load (default: %(default)s)")
    parser.add_argument("-t", default="vocab.json", metavar="VOCAB",
                        help="tokenizer JSON (default: %(default)s)")
    parser.add_argument("--temperature", type=float, default=1.0, metavar="T")
    parser.add_argument("--top-k", type=int, default=50, metavar="K")
    parser.add_argument("--top-p", type=float, default=0.9, metavar="P")
    parser.add_argument("--limit", type=int, default=256, metavar="N",
                        help="max tokens per reply (default: %(default)s)")
    parser.add_argument("--think", dest="think", action="store_true",
                        help="start with think mode on (the default)")
    parser.add_argument("--no-think", dest="think", action="store_false",
                        help="start with think mode off")
    parser.set_defaults(think=True)
    args = parser.parse_args()

    if not os.path.isfile(args.w):
        sys.exit(f"Error: weights file not found: {args.w}")
    if not os.path.isfile(args.t):
        sys.exit(f"Error: tokenizer not found: {args.t}")

    tokenizer = Tokenizer()
    with open(args.t, "r", encoding="utf-8") as f:
        tokenizer.deserialize_from_json(f.read())
    vocab_size = tokenizer.vocab_size()

    user_id = single_token_id(tokenizer, USER_TOKEN, "marker")
    assistant_id = single_token_id(tokenizer, ASSISTANT_TOKEN, "marker")
    end_id = single_token_id(tokenizer, END_TOKEN, "marker")
    eot_id = single_token_id(tokenizer, ENDOFTEXT_TOKEN, "marker")
    think_id = single_token_id(tokenizer, THINK_TOKEN, "marker")
    think_end_id = single_token_id(tokenizer, "</think>", "marker")
    think_ids = [think_id]

    if is_safetensors(args.w):
        # A checkpoint: self-describing, shape comes from its stored config.
        model_items, _opt, step, cfg = load_checkpoint(args.w)
        if cfg["vocab_size"] != vocab_size:
            sys.exit(f"Error: model vocab {cfg['vocab_size']} != tokenizer vocab "
                     f"{vocab_size}; wrong tokenizer for this checkpoint.")
        dim, heads = cfg["feature_dim"], cfg["num_heads"]
        blocks, ctx = cfg["num_blocks"], cfg["context"]
        model = MiniGPT(vocab_size, dim, heads, cfg["rope_base"], blocks)
        model.update(tree_unflatten(model_items))
        mx.eval(model.parameters())
        source = f"checkpoint, step {step:,} {cfg.get('kind', 'pretrain')}, shape from checkpoint"
    else:
        # Flat final weights: headerless, shape comes from model_config.py.
        dim, heads = model_config.FEATURE_DIM, model_config.NUM_HEADS
        blocks, ctx = model_config.NUM_BLOCKS, model_config.CONTEXT_LENGTH
        model = MiniGPT(vocab_size, dim, heads, model_config.ROPE_BASE, blocks)
        import_weights(args.w, model, heads)
        source = "final weights, shape from model_config.py"
    n_params = expected_param_count(vocab_size, dim, blocks)

    settings = {"temperature": args.temperature, "top_k": args.top_k,
                "top_p": args.top_p, "limit": args.limit, "think": args.think}
    history = []          # list of per-turn token-id lists, in order
    gen = {"cache": KVCache(blocks)}   # KV cache; always a prefix of the conversation

    print(f"Loaded {args.w} ({n_params / 1e6:.1f}M params | dim {dim} | "
          f"{heads} heads | {blocks} blocks | ctx {ctx}) [{source}]")
    print("Type and press enter to chat, /help for commands.")

    def history_len():
        return sum(len(t) for t in history)

    def generate_reply():
        """Build the prompt from history, stream the reply, return its ids."""
        # Trim oldest turns until the prompt plus the reply budget fits.
        budget = min(settings["limit"], ctx // 2)
        trimmed = False
        while len(history) > 1 and history_len() + 1 + len(think_ids) + budget > ctx:
            dropped = 0
            history.pop(0)                       # oldest user turn
            dropped += 1
            if history:                          # its assistant reply, if any
                history.pop(0)
                dropped += 1
            trimmed = True
            print(f"[context full — dropped the oldest {dropped} turn(s)]")

        prompt_ids = [tid for turn in history for tid in turn]
        prompt_ids.append(assistant_id)
        if settings["think"]:
            prompt_ids.extend(think_ids)

        # Trimming shifts every position, so the cached rotations are stale.
        if trimmed or gen["cache"].length > len(prompt_ids):
            gen["cache"] = KVCache(blocks)
        cache = gen["cache"]

        # Feed only what the cache hasn't seen: the whole prompt on the first
        # turn, just the latest turn(s) afterwards, one token per step during
        # generation. This is the JS predictNextToken slice(cache.length).
        feed = prompt_ids[cache.length:]

        generated = []
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        # Thinking is delineated in the display. The trace is BUFFERED, not
        # streamed: the model only reveals whether it is actually thinking when
        # it emits </think>. If it does, the trace prints as one "thinking>"
        # block and the answer streams live under "model>". If the reply ends
        # without ever closing the trace (the model answered directly — common
        # with mixed think/no-think training data), the buffered text was the
        # answer all along and prints under "model>" instead.
        think_buf = []
        buffering = settings["think"]
        saw_think_end = False
        if not buffering:
            sys.stdout.write("model> ")
            sys.stdout.flush()
        stop = "limit"
        try:
            for _ in range(settings["limit"]):
                logits = cached_forward(model, feed, cache)
                last = logits[-1]
                mx.eval(last)
                next_id = sample_token(last, settings["temperature"],
                                       settings["top_k"], settings["top_p"])
                if next_id == end_id or next_id == eot_id:
                    stop = "end"
                    break
                generated.append(next_id)
                feed = [next_id]
                if buffering:
                    if next_id == think_end_id:
                        saw_think_end = True
                        buffering = False
                        trace = tokenizer.decode(think_buf).strip()
                        if trace:
                            print(f"thinking> {trace}")
                        sys.stdout.write("model> ")
                        sys.stdout.flush()
                    elif next_id != think_id:
                        think_buf.append(next_id)
                else:
                    if next_id in (think_id, think_end_id):
                        continue  # tags are never printed
                    sys.stdout.write(decoder.decode(token_to_bytes(tokenizer, next_id)))
                    sys.stdout.flush()
        except KeyboardInterrupt:
            stop = "interrupted"

        if buffering:
            # Never saw </think>: the buffered text is the actual reply.
            print(f"model> {tokenizer.decode(think_buf).strip()}")
        else:
            sys.stdout.write(decoder.decode(b"", final=True))
            print()
        if stop == "limit":
            print("[limit reached]")

        # Store the assistant turn closed with <|end|>, whatever stopped it.
        # If the forged <think> was never honored, store the turn WITHOUT it so
        # the conversation history stays well-formed — and reset the cache,
        # whose prefix contains the forged tag we are dropping.
        if settings["think"] and not saw_think_end:
            history.append([assistant_id] + generated + [end_id])
            gen["cache"] = KVCache(blocks)
        else:
            turn = [assistant_id] + (list(think_ids) if settings["think"] else [])
            history.append(turn + generated + [end_id])

    def set_number(key, raw, cast, lo, hi):
        try:
            val = cast(raw)
        except (ValueError, TypeError):
            print(f"usage: /{key} <number>")
            return
        if not (lo <= val <= hi):
            print(f"/{key}: value must be between {lo} and {hi}")
            return
        settings[key.replace('-', '_')] = val
        print(f"{key} = {val}")

    while True:
        try:
            line = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if not line:
            continue

        if line.startswith("/"):
            parts = line.split()
            cmd, rest = parts[0].lower(), parts[1] if len(parts) > 1 else None
            if cmd in ("/quit", "/exit"):
                print("bye")
                break
            elif cmd == "/help":
                print(HELP.format(**{**settings,
                                     "think": "on" if settings["think"] else "off"}))
            elif cmd in ("/temperature", "/temp"):
                set_number("temperature", rest, float, 0.0, 10.0)
            elif cmd == "/top-k":
                set_number("top-k", rest, int, 1, vocab_size)
            elif cmd == "/top-p":
                set_number("top-p", rest, float, 0.0, 1.0)
            elif cmd == "/limit":
                set_number("limit", rest, int, 1, 100000)
            elif cmd == "/think":
                if rest in ("on", "off"):
                    settings["think"] = (rest == "on")
                    print(f"think mode {rest}")
                else:
                    print("usage: /think on|off")
            elif cmd == "/reset":
                history.clear()
                gen["cache"] = KVCache(blocks)
                print("conversation cleared")
            elif cmd == "/settings":
                print(f"temperature {settings['temperature']} | top-k {settings['top_k']} "
                      f"| top-p {settings['top_p']} | limit {settings['limit']} "
                      f"| think {'on' if settings['think'] else 'off'}")
                print(f"context: {history_len():,}/{ctx:,} tokens "
                      f"({len(history)} turns)")
            else:
                print(f"unknown command {cmd} — /help for the list")
            continue

        history.append([user_id] + tokenizer.encode(line) + [end_id])
        generate_reply()


if __name__ == "__main__":
    main()
