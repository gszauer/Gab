"""webexport.py — package final weights for the browser runtime.

Reads a flat .weights file (the headerless float32 MiniGPT format, in
reference.js parameters() order) and writes the files the website loads:

    model-f32.json + model-f32-000.bin ... (sharded, each under GitHub's limit)
    model-q8.json  + model-q8.bin          (rowwise symmetric int8, one file)
    tokenizer.json                         (a copy of the vocab)

The per-head attention matrices are fused on export. Summing per-head
outputs (attn_i @ Wo_i) equals concatenating the heads and applying one
big output projection, so the fused [768, 768] tensors are exact — the
browser engine runs the standard fused formulation and produces
bit-identical float32 results.

Tensor layout in the manifests is HF-style [out_features, in_features]
row-major, so the runtime's matvec is out[r] = weight_row_r . input.

Usage (from the repo root):
    python3 website/webexport.py -w weights/finetuned.weights -v weights/vocab.json
"""

import argparse
import hashlib
import json
import os
import shutil
import sys

import numpy as np

# This script lives in website/; model_config.py lives in the repo root.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))

from model_config import (FEATURE_DIM, NUM_HEADS, ROPE_BASE, NUM_BLOCKS,
                          CONTEXT_LENGTH)

ALIGNMENT = 16
SHARD_SIZE = 90 * 1024 * 1024  # 94,371,840 bytes — well under GitHub's 100 MB cap


def load_flat_weights(path, vocab_size):
    dim = FEATURE_DIM
    heads = NUM_HEADS
    head_dim = dim // heads
    hidden = dim * 4

    flat = np.fromfile(path, dtype="<f4")
    expected = vocab_size * dim
    per_block = (dim                                  # attn norm gamma
                 + heads * (3 * dim * head_dim + head_dim * dim)  # Q,K,V,O
                 + dim                                # mlp norm gamma
                 + dim * hidden + hidden * dim        # Wup, Wdown
                 + hidden)                            # bias
    expected += NUM_BLOCKS * per_block + dim          # final norm gamma
    if flat.size != expected:
        sys.exit(f"Error: {path} holds {flat.size} floats, expected {expected}. "
                 f"Wrong file or wrong model_config.py shape.")

    pos = 0

    def take(*shape):
        nonlocal pos
        n = int(np.prod(shape))
        out = flat[pos:pos + n].reshape(shape)
        pos += n
        return out

    model = {"embed": take(vocab_size, dim), "blocks": []}
    for _ in range(NUM_BLOCKS):
        block = {"attn_norm": take(dim)}
        q_heads, k_heads, v_heads, o_heads = [], [], [], []
        for _ in range(heads):
            q_heads.append(take(dim, head_dim))
            k_heads.append(take(dim, head_dim))
            v_heads.append(take(dim, head_dim))
            o_heads.append(take(head_dim, dim))
        # Fuse: concat head columns, then transpose to [out, in].
        # q = x @ Wq_h puts head h at output slice [h*64, (h+1)*64], which is
        # exactly hstack; o_proj is the stacked Wo rows, transposed.
        block["q_proj"] = np.ascontiguousarray(np.hstack(q_heads).T)
        block["k_proj"] = np.ascontiguousarray(np.hstack(k_heads).T)
        block["v_proj"] = np.ascontiguousarray(np.hstack(v_heads).T)
        block["o_proj"] = np.ascontiguousarray(np.vstack(o_heads).T)
        block["mlp_norm"] = take(dim)
        block["up_proj"] = np.ascontiguousarray(take(dim, hidden).T)
        block["down_proj"] = np.ascontiguousarray(take(hidden, dim).T)
        block["up_bias"] = take(hidden)
        model["blocks"].append(block)
    model["final_norm"] = take(dim)
    assert pos == flat.size
    return model


def named_tensors(model):
    yield "model.embed_tokens.weight", model["embed"]
    for i, block in enumerate(model["blocks"]):
        p = f"model.layers.{i}."
        yield p + "input_layernorm.weight", block["attn_norm"]
        yield p + "self_attn.q_proj.weight", block["q_proj"]
        yield p + "self_attn.k_proj.weight", block["k_proj"]
        yield p + "self_attn.v_proj.weight", block["v_proj"]
        yield p + "self_attn.o_proj.weight", block["o_proj"]
        yield p + "post_attention_layernorm.weight", block["mlp_norm"]
        yield p + "mlp.up_proj.weight", block["up_proj"]
        yield p + "mlp.up_bias", block["up_bias"]
        yield p + "mlp.down_proj.weight", block["down_proj"]
    yield "model.norm.weight", model["final_norm"]


def architecture(vocab_size):
    return {
        "model_type": "gift_of_gab",
        "vocab_size": vocab_size,
        "max_position_embeddings": CONTEXT_LENGTH,
        "hidden_size": FEATURE_DIM,
        "intermediate_size": FEATURE_DIM * 4,
        "num_hidden_layers": NUM_BLOCKS,
        "num_attention_heads": NUM_HEADS,
        "head_dim": FEATURE_DIM // NUM_HEADS,
        "rope_theta": float(ROPE_BASE),
        "rope_style": "interleaved",
        "rms_norm_eps": 1e-5,
        "hidden_act": "gelu",
        "attention_bias": False,
        "mlp_bias": True,
        "tie_word_embeddings": True,
    }


class BlobWriter:
    def __init__(self):
        self.parts = []
        self.size = 0

    def add(self, arr):
        pad = (-self.size) % ALIGNMENT
        if pad:
            self.parts.append(b"\0" * pad)
            self.size += pad
        raw = arr.tobytes()
        offset = self.size
        self.parts.append(raw)
        self.size += len(raw)
        return offset, len(raw)

    def bytes(self):
        return b"".join(self.parts)


def quantize_rowwise(mat):
    """Symmetric rowwise int8: q = round(w / scale), scale = rowmax/127."""
    absmax = np.abs(mat).max(axis=1)
    scale = np.where(absmax > 0, absmax / 127.0, 1.0).astype("<f4")
    q = np.clip(np.rint(mat / scale[:, None]), -127, 127).astype(np.int8)
    return q, scale


def export_f32(model, vocab_size, out_dir):
    blob = BlobWriter()
    tensors = []
    for name, arr in named_tensors(model):
        offset, length = blob.add(arr.astype("<f4"))
        tensors.append({
            "name": name,
            "shape": list(arr.shape),
            "storage": "float32",
            "data": {"dtype": "float32", "offset": offset, "length": length},
        })

    raw = blob.bytes()
    shards = []
    for i in range(0, len(raw), SHARD_SIZE):
        chunk = raw[i:i + SHARD_SIZE]
        fname = f"model-f32-{len(shards):03d}.bin"
        with open(os.path.join(out_dir, fname), "wb") as f:
            f.write(chunk)
        shards.append({
            "file": fname,
            "length": len(chunk),
            "sha256": hashlib.sha256(chunk).hexdigest(),
        })

    manifest = {
        "format": "gift-of-gab-f32",
        "format_version": 2,
        "endianness": "little",
        "alignment": ALIGNMENT,
        "byte_length": len(raw),
        "architecture": architecture(vocab_size),
        "storage": {
            "scheme": "float32_full_precision",
            "dtype": "float32",
            "sharded": True,
            "shard_size": SHARD_SIZE,
        },
        "files": {"weights_shards": shards},
        "tensors": tensors,
    }
    with open(os.path.join(out_dir, "model-f32.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    return len(raw), len(shards)


def export_q8(model, vocab_size, out_dir):
    blob = BlobWriter()
    tensors = []
    for name, arr in named_tensors(model):
        if arr.ndim == 2:  # matrices are quantized; norms and biases stay f32
            q, scale = quantize_rowwise(arr)
            q_off, q_len = blob.add(q)
            s_off, s_len = blob.add(scale)
            tensors.append({
                "name": name,
                "shape": list(arr.shape),
                "storage": "q8_rowwise_symmetric",
                "q": {"dtype": "int8", "offset": q_off, "length": q_len},
                "scale": {"dtype": "float32", "offset": s_off, "length": s_len},
            })
        else:
            offset, length = blob.add(arr.astype("<f4"))
            tensors.append({
                "name": name,
                "shape": list(arr.shape),
                "storage": "float32",
                "data": {"dtype": "float32", "offset": offset, "length": length},
            })

    raw = blob.bytes()
    fname = "model-q8.bin"
    with open(os.path.join(out_dir, fname), "wb") as f:
        f.write(raw)

    manifest = {
        "format": "gift-of-gab-q8",
        "format_version": 2,
        "endianness": "little",
        "alignment": ALIGNMENT,
        "byte_length": len(raw),
        "architecture": architecture(vocab_size),
        "storage": {
            "scheme": "q8_rowwise_symmetric",
            "sharded": False,
        },
        "files": {
            "weights": {
                "file": fname,
                "length": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            },
        },
        "tensors": tensors,
    }
    with open(os.path.join(out_dir, "model-q8.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    return len(raw)


def main():
    parser = argparse.ArgumentParser(description="Export weights for the browser runtime.")
    parser.add_argument("-w", default="finetuned.weights", metavar="WEIGHTS",
                        help="flat float32 weights file (default: finetuned.weights)")
    parser.add_argument("-v", default="vocab.json", metavar="VOCAB",
                        help="tokenizer vocab JSON (default: vocab.json)")
    parser.add_argument("-o", default=_SCRIPT_DIR, metavar="DIR",
                        help="output directory (default: this script's folder)")
    args = parser.parse_args()

    for path in (args.w, args.v):
        if not os.path.isfile(path):
            sys.exit(f"Error: file not found: {path}")
    os.makedirs(args.o, exist_ok=True)

    vocab = json.load(open(args.v))
    vocab_size = 256 + len(vocab["merges"])
    print(f"Vocab: {vocab_size} tokens, {len(vocab['reserved'])} reserved")

    print(f"Loading {args.w}...")
    model = load_flat_weights(args.w, vocab_size)

    print("Exporting F32...")
    f32_bytes, shard_count = export_f32(model, vocab_size, args.o)
    print(f"  {f32_bytes:,} bytes ({f32_bytes / (1 << 20):.1f} MiB) in {shard_count} shards")

    print("Exporting Q8...")
    q8_bytes = export_q8(model, vocab_size, args.o)
    print(f"  {q8_bytes:,} bytes ({q8_bytes / (1 << 20):.1f} MiB)")
    if q8_bytes >= 100 * (1 << 20):
        sys.exit("Error: Q8 export is 100 MiB or more; it must stay under 100 MiB.")

    shutil.copyfile(args.v, os.path.join(args.o, "tokenizer.json"))
    print("Copied vocab to tokenizer.json")
    print("Done.")


if __name__ == "__main__":
    main()
