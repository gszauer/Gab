#!/usr/bin/env python3
"""encoder.py — encode a tree of text files into one token stream.

Reads every .txt file under the -i folder(s), tokenizes each with the vocab
produced by tokenizer.py, and writes all token ids to a flat binary file
(little-endian uint16). An optional separator token is inserted between files.

Encoding matches the tokenizer: each file is pre-tokenized with `_split` and each
pre-token encoded, exactly as during training — and pre-tokens are cached, so a
repeated word is encoded once. For all non-astral text this is identical to
`Tokenizer.encode` of the whole file; astral characters (emoji, ...) follow the
training convention (each becomes U+FFFD), which is what the vocab was built on.

Output format: raw little-endian uint16 per token, no header. Load with e.g.
    import numpy as np;  ids = np.fromfile("corpus.bin", dtype="<u2")
Requires vocab size <= 65536 so every id fits in uint16.

Usage:
    python encoder.py -v vocab.json -i input/path [-i more/paths ...] \\
                      [-o corpus.bin] [-s "<|endoftext|>"]

    -v  tokenizer JSON (vocab) to encode with (required)
    -i  folder scanned recursively for .txt files (required, repeatable)
    -o  output binary file (optional, default: corpus.bin)
    -s  separator token string inserted between files (optional; none if omitted)
"""

import argparse
import array
import codecs
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenizer import (
    Tokenizer, find_files, TEXT_EXTENSIONS,
    _format_bytes, _format_duration, _CHUNK_BYTES,
)

# uint16 can hold ids 0..65535, so the vocabulary must be at most this big.
_MAX_VOCAB = 65536
# Flush the write buffer to disk every ~1M tokens (2 MiB) to bound memory.
_FLUSH_AT = 1 << 20


def build_pair_to_id(tokenizer):
    """Map each merge pair (a, b) -> its token id (rank). Lowest id wins on the
    off chance a pair repeats, matching how `encode` applies rules in id order."""
    pair_to_id = {}
    for token_id in range(256, len(tokenizer.merges)):
        a, b = tokenizer.merges[token_id]
        pair_to_id.setdefault((a, b), token_id)
    return pair_to_id


def make_encode_piece(tokenizer, pair_to_id):
    """Return encode_piece(text) -> list[int]: a fast per-pre-token encoder that
    is byte-identical to `tokenizer.encode(text)`.

    Repeatedly merges the lowest-rank adjacent pair present. Because a merge into
    id r only creates pairs of rank > r, processing lowest-rank-first reproduces
    `encode`'s "apply every rule in id order" exactly — but only touches pairs
    that are actually present, so short pre-tokens encode in a handful of passes.
    """
    merges = tokenizer.merges
    _merge = tokenizer._merge
    to_bytes = tokenizer._encode_utf8_like_textencoder

    def encode_piece(text):
        syms = list(to_bytes(text))
        while len(syms) >= 2:
            best = None
            for i in range(len(syms) - 1):
                rank = pair_to_id.get((syms[i], syms[i + 1]))
                if rank is not None and (best is None or rank < best):
                    best = rank
            if best is None:
                break
            a, b = merges[best]
            syms = _merge(syms, a, b, best)
        return syms

    return encode_piece


def encode_one_file(path, tokenizer, encode_cached, out_buf, sep_ids,
                    prepend_sep, on_bytes, flush_fn):
    """Stream one file, appending its token ids to `out_buf`.

    Reads in chunks and carries the last pre-token across chunk boundaries so the
    split is identical to processing the whole file at once. If `prepend_sep`,
    the separator is written just before this file's first token. Returns
    (ok, n_tokens); ok is False if the file isn't valid UTF-8 (any valid prefix
    is still written).
    """
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    carry = ""
    state = {"started": False, "ntok": 0}

    def emit(ids):
        if not ids:
            return
        if prepend_sep and sep_ids and not state["started"]:
            out_buf.extend(sep_ids)
            state["ntok"] += len(sep_ids)  # count the separator too
        state["started"] = True
        out_buf.extend(ids)
        state["ntok"] += len(ids)
        if len(out_buf) >= _FLUSH_AT:
            flush_fn()

    try:
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(_CHUNK_BYTES)
                if not chunk:
                    break
                on_bytes(len(chunk))
                try:
                    piece = decoder.decode(chunk)
                except UnicodeDecodeError:
                    return False, state["ntok"]
                if not piece:
                    continue
                pretokens = tokenizer._split(carry + piece)
                for pt in pretokens[:-1]:
                    emit(encode_cached(pt))
                carry = pretokens[-1]
            try:
                tail = decoder.decode(b"", final=True)
            except UnicodeDecodeError:
                return False, state["ntok"]
    except OSError:
        return False, state["ntok"]

    for pt in tokenizer._split(carry + tail):
        emit(encode_cached(pt))
    return True, state["ntok"]


def main():
    parser = argparse.ArgumentParser(
        description="Encode text files into one uint16 token stream using a tokenizer vocab."
    )
    parser.add_argument("-v", required=True, metavar="VOCAB",
                        help="tokenizer JSON (vocab) to encode with")
    parser.add_argument("-i", action="append", required=True, metavar="FOLDER",
                        help="folder scanned recursively for .txt files; repeatable")
    parser.add_argument("-o", default="corpus.bin", metavar="FILE",
                        help="output binary file (default: %(default)s)")
    parser.add_argument("-s", metavar="TOKEN",
                        help="separator token inserted between files (default: none)")
    args = parser.parse_args()

    if not os.path.isfile(args.v):
        sys.exit(f"Error: vocab file not found: {args.v}")
    for folder in args.i:
        if not os.path.isdir(folder):
            sys.exit(f"Error: input path is not a folder: {folder}")

    # Load the vocabulary.
    tokenizer = Tokenizer()
    try:
        with open(args.v, "r", encoding="utf-8") as f:
            tokenizer.deserialize_from_json(f.read())
    except (json.JSONDecodeError, KeyError, TypeError) as err:
        sys.exit(f"Error: {args.v} is not a valid tokenizer JSON ({err})")
    if tokenizer.vocab_size() > _MAX_VOCAB:
        sys.exit(f"Error: vocab size {tokenizer.vocab_size()} exceeds {_MAX_VOCAB}; "
                 f"ids do not fit in uint16.")
    print(f"Loaded vocab: {tokenizer.vocab_size()} tokens, "
          f"{len(tokenizer.reserved)} reserved.")

    pair_to_id = build_pair_to_id(tokenizer)
    encode_piece = make_encode_piece(tokenizer, pair_to_id)
    cache = {}

    def encode_cached(pretoken):
        ids = cache.get(pretoken)
        if ids is None:
            ids = encode_piece(pretoken)
            cache[pretoken] = ids
        return ids

    # Encode the separator (if any) the same way as file content.
    sep_ids = []
    if args.s is not None:
        for pt in tokenizer._split(args.s):
            sep_ids += encode_piece(pt)
        print(f"Separator {args.s!r} -> {len(sep_ids)} token(s): {sep_ids}")
        if len(sep_ids) != 1:
            print("  (note: encodes to multiple tokens — reserve it in the tokenizer "
                  "with -r for a single-token separator)")

    # Resolve + de-duplicate input folders (drop nested/duplicate paths).
    folders = []
    for folder in args.i:
        rp = os.path.realpath(folder)
        if any(rp == r or rp.startswith(r + os.sep) for r in folders):
            continue
        folders = [r for r in folders if not r.startswith(rp + os.sep)]
        folders.append(rp)

    where = folders[0] if len(folders) == 1 else f"{len(folders)} folders"
    print(f"Scanning {where} for {{.txt}} files...", flush=True)
    disc = {"last": 0.0}

    def on_discover(n_files, n_bytes):
        now = time.monotonic()
        if now - disc["last"] < 1.0:
            return
        disc["last"] = now
        print(f"\r  discovered {n_files:,} files, {_format_bytes(n_bytes)}...    ",
              end="", flush=True)

    paths, total_bytes = find_files(folders, TEXT_EXTENSIONS, on_discover)
    if not paths:
        sys.exit(f"Error: no .txt files found under {', '.join(folders)}")
    print(f"\r  found {len(paths):,} files, {_format_bytes(total_bytes)} total."
          f"                    ")

    # Encode.
    print(f"Encoding to {args.o} (uint16 little-endian)...")
    out_buf = array.array("H")
    enc = {"bytes": 0, "last": 0.0, "t0": time.monotonic(),
           "files": 0, "skipped": 0, "tokens": 0}

    with open(args.o, "wb") as out:
        def flush():
            if sys.byteorder == "big":
                out_buf.byteswap()
            out_buf.tofile(out)
            del out_buf[:]

        def report(force=False):
            now = time.monotonic()
            if not force and now - enc["last"] < 1.0:
                return
            enc["last"] = now
            el = now - enc["t0"]
            frac = enc["bytes"] / total_bytes if total_bytes else 1.0
            eta = (el / frac - el) if frac > 0 else None
            print(f"\r  encoding {enc['files'] + enc['skipped']:,}/{len(paths):,} files, "
                  f"{_format_bytes(enc['bytes'])}/{_format_bytes(total_bytes)} "
                  f"({frac * 100:.1f}%), {enc['tokens']:,} tokens  "
                  f"ETA {_format_duration(eta)}    ", end="", flush=True)

        def on_bytes(delta):
            enc["bytes"] += delta
            report()

        for path in paths:
            prepend = enc["tokens"] > 0
            ok, ntok = encode_one_file(path, tokenizer, encode_cached, out_buf,
                                       sep_ids, prepend, on_bytes, flush)
            enc["tokens"] += ntok
            if ok:
                enc["files"] += 1
            else:
                enc["skipped"] += 1
                print(f"\r  skipped (not UTF-8 / unreadable): {path}    ")
        flush()
        report(force=True)
    print()

    out_size = os.path.getsize(args.o)
    print(f"Encoded {enc['files']:,} file(s) (skipped {enc['skipped']:,}); "
          f"{len(cache):,} unique pre-tokens.")
    print(f"Wrote {out_size // 2:,} tokens to {args.o} ({_format_bytes(out_size)}).")


if __name__ == "__main__":
    main()
