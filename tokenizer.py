#!/usr/bin/env python3
"""tokenizer.py — Python port of the Tokenizer from MiniGPT.js (Chapter 4).

Trains a byte-pair-encoding tokenizer on a folder of text files and writes a
JSON file that JavaScript's `Tokenizer.deserializeFromJSON` loads unchanged.

Compatibility surface: the serialized {reserved, merges} and the encode/decode
algorithms are byte-identical to MiniGPT.js. `encode`/`decode` apply merges to
the raw UTF-8 byte stream and do NOT use `_split`, so the pre-tokenizer only
shapes which merges are *learned* — a vocab trained here always loads and
round-trips identically in the JS version.

Training is tuned for large prose corpora (built for ~10 GiB):
  * Pre-tokens are de-duplicated (each unique word counted once, weighted),
    so memory and work scale with the number of *distinct* pre-tokens, not the
    corpus size.
  * BPE uses an incremental pair-count index + lazy heap, so each merge only
    touches the words that contained the merged pair instead of rescanning the
    whole corpus.
  * Progress with an ETA is printed for both the scan and the merge phases.

UTF-16 fidelity: `_split` iterates UTF-16 code units like JS; an astral char is
two lone surrogates, each its own chunk, and `encode` maps a lone surrogate to
U+FFFD (matching `TextEncoder`). `encode` combines valid surrogate pairs.

Usage:
    python tokenizer.py -i <folder> [-o <vocab.json>] [-m <target_vocab>] [-r <token> ...]

    -i  folder scanned recursively for training files (required)
    -o  output JSON file (optional, default: vocab.json)
    -m  target vocabulary size (optional, default: VOCAB_SIZE below)
    -r  reserved token, applied before training; repeatable (optional)
"""

import argparse
import array
import heapq
import json
import os
import sys
import time

# Default target vocabulary size: training stops once the vocab reaches this
# many tokens (the first 256 are the byte tokens, so learned merges = this - 256).
VOCAB_SIZE = 10000

# Only files with these extensions are treated as training text (case-insensitive).
TEXT_EXTENSIONS = {".txt"}

# Rule 4 run caps: the longest run of whitespace / punctuation that can live in a
# single pre-token (and therefore the longest such token BPE can learn). A run
# longer than the cap is split into back-to-back chunks of at most this length.
MAX_WHITESPACE_RUN = 4
MAX_PUNCT_RUN = 4

# ASCII whitespace code units grouped by Rule 4a.
_WHITESPACE = frozenset((0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20))


class Tokenizer:
    """Byte-pair encoding tokenizer, faithful to the MiniGPT.js implementation.

    Token ids 0-255 are the raw bytes (seeded placeholders, never serialized).
    Every id from 256 up is a learned merge of two earlier token ids.
    """

    def __init__(self):
        self.merges = [[i, i] for i in range(256)]  # merges[id] = [first, second]
        self.reserved = []                          # atomic strings

    def vocab_size(self):
        return len(self.merges)

    # --- UTF-16 / UTF-8 helpers (matching JS string + TextEncoder semantics) --

    @staticmethod
    def _to_code_units(text):
        """`text` as a list of UTF-16 code units, exactly how JS indexes a string."""
        units = array.array("H")
        units.frombytes(text.encode("utf-16-le", "surrogatepass"))
        if sys.byteorder == "big":
            units.byteswap()
        return units.tolist()

    @staticmethod
    def _encode_utf8_like_textencoder(text):
        """UTF-8 bytes for `text`, matching JS `new TextEncoder().encode(text)`:
        valid surrogate pairs combine; a lone surrogate becomes U+FFFD."""
        try:
            return text.encode("utf-8")  # fast path: no lone surrogates
        except UnicodeEncodeError:
            pass
        out = bytearray()
        i, n = 0, len(text)
        while i < n:
            cp = ord(text[i])
            if 0xD800 <= cp <= 0xDBFF and i + 1 < n and 0xDC00 <= ord(text[i + 1]) <= 0xDFFF:
                combined = 0x10000 + ((cp - 0xD800) << 10) + (ord(text[i + 1]) - 0xDC00)
                out += chr(combined).encode("utf-8")
                i += 2
            elif 0xD800 <= cp <= 0xDFFF:
                out += b"\xef\xbf\xbd"
                i += 1
            else:
                out += text[i].encode("utf-8")
                i += 1
        return bytes(out)

    @staticmethod
    def _utf16_length(s):
        return sum(2 if ord(c) > 0xFFFF else 1 for c in s)

    # --- Encoding -----------------------------------------------------------

    def encode(self, text):
        """string -> list[int]. UTF-8 bytes (TextEncoder rules), every merge applied in order."""
        tokens = list(self._encode_utf8_like_textencoder(text))
        for rule in range(256, len(self.merges)):
            first, second = self.merges[rule]
            tokens = self._merge(tokens, first, second, rule)
        return tokens

    @staticmethod
    def _merge(list_of_tokens, first_token, second_token, replacement_token):
        """Replace every adjacent (first, second) pair with replacement, left to right."""
        result = []
        i = 0
        n = len(list_of_tokens)
        while i < n:
            if (
                list_of_tokens[i] == first_token
                and i + 1 < n
                and list_of_tokens[i + 1] == second_token
            ):
                result.append(replacement_token)
                i += 2
                continue
            result.append(list_of_tokens[i])
            i += 1
        return result

    def decode(self, ids):
        """list[int] -> string. Expand each id back to bytes via its merge tree."""
        out_bytes = []
        for tok in ids:
            stack = [tok]
            while stack:
                current = stack.pop()
                if current < 256:
                    out_bytes.append(current)
                else:
                    pair = self.merges[current]
                    stack.append(pair[1])
                    stack.append(pair[0])
        return bytes(out_bytes).decode("utf-8", errors="replace")

    # --- Chunking (pre-tokenization) ---------------------------------------

    @staticmethod
    def _is_letter_cu(cu):
        if cu is None:
            return False
        if cu < 0x80:  # ASCII fast path (same result as the general rule)
            return 0x41 <= cu <= 0x5A or 0x61 <= cu <= 0x7A
        ch = chr(cu)
        return ch.lower() != ch.upper()

    @staticmethod
    def _is_digit_cu(cu):
        return cu is not None and 0x30 <= cu <= 0x39

    @staticmethod
    def _is_whitespace_cu(cu):
        return cu in _WHITESPACE

    def _split(self, input_text):
        """Split text into chunks over UTF-16 code units.

        Rules (in order): reserved keyword; word (optional one leading space +
        letters); digits (<=3); whitespace run (<=MAX_WHITESPACE_RUN, but a space
        that leads a word is left for the word); a lone surrogate alone; a run of
        other/punctuation characters (<=MAX_PUNCT_RUN).
        """
        self.reserved.sort(key=lambda s: -self._utf16_length(s))
        reserved_units = [self._to_code_units(kw) for kw in self.reserved]

        units = self._to_code_units(input_text)
        n = len(units)

        def match_keyword(pos):
            for idx, ku in enumerate(reserved_units):
                klen = len(ku)
                if klen and units[pos:pos + klen] == ku:
                    return idx, klen
            return -1, 0

        chunks = []
        i = 0
        while i < n:
            # Rule 1: reserved keywords are atomic and always win.
            kidx, klen = match_keyword(i)
            if kidx != -1:
                chunks.append(self.reserved[kidx])
                i += klen
                continue

            cu = units[i]
            nxt = units[i + 1] if i + 1 < n else None

            # Rule 2: a word, optionally carrying ONE leading space.
            if self._is_letter_cu(cu) or (cu == 0x20 and self._is_letter_cu(nxt)):
                start = i
                i += 1
                while i < n and self._is_letter_cu(units[i]):
                    i += 1
                chunks.append("".join(chr(u) for u in units[start:i]))
                continue

            # Rule 3: digits, grouped to at most 3.
            if self._is_digit_cu(cu):
                start = i
                count = 0
                while count < 3 and i < n and self._is_digit_cu(units[i]):
                    i += 1
                    count += 1
                chunks.append("".join(chr(u) for u in units[start:i]))
                continue

            # Rule 4a: a run of whitespace, capped, leaving a word-leading space.
            # A run never swallows the start of a reserved keyword: the JS rule 4
            # was one char per chunk with the keyword check at every position, so
            # grouping must stop where a keyword begins to keep tokens atomic.
            if self._is_whitespace_cu(cu):
                start = i
                count = 0
                while (count < MAX_WHITESPACE_RUN and i < n
                       and self._is_whitespace_cu(units[i])
                       and not (units[i] == 0x20 and i + 1 < n
                                and self._is_letter_cu(units[i + 1]))
                       and not (i > start and match_keyword(i)[0] != -1)):
                    i += 1
                    count += 1
                if i == start:  # defensive: never stall
                    chunks.append(chr(cu))
                    i += 1
                else:
                    chunks.append("".join(chr(u) for u in units[start:i]))
                continue

            # Rule 4b: a lone surrogate stands alone (encodes to U+FFFD).
            if 0xD800 <= cu <= 0xDFFF:
                chunks.append(chr(cu))
                i += 1
                continue

            # Rule 4c: a run of other/punctuation characters, capped. Stops where
            # a reserved keyword begins (see rule 4a) so e.g. ".<|end|>" splits
            # into "." + the atomic reserved token instead of shredding the tag.
            start = i
            count = 0
            while (count < MAX_PUNCT_RUN and i < n
                   and not self._is_letter_cu(units[i])
                   and not self._is_digit_cu(units[i])
                   and not self._is_whitespace_cu(units[i])
                   and not (0xD800 <= units[i] <= 0xDFFF)
                   and not (i > start and match_keyword(i)[0] != -1)):
                i += 1
                count += 1
            chunks.append("".join(chr(u) for u in units[start:i]))

        return chunks

    # --- Reserved tokens ----------------------------------------------------

    def reserve(self, text):
        """Reserve `text` as an atomic token before training (JS `reserve`)."""
        tokens = self.encode(text)
        if not tokens:
            self.reserved.append(text)
            return None
        token_id = tokens[0]
        for i in range(1, len(tokens)):
            self.merges.append([token_id, tokens[i]])
            token_id = len(self.merges) - 1
        self.reserved.append(text)
        return token_id

    # --- Training -----------------------------------------------------------

    def add_document_counts(self, text, counter):
        """Split one document and add each pre-token's occurrence to `counter`."""
        for piece in self._split(text):
            counter[piece] = counter.get(piece, 0) + 1

    def train(self, text, target_vocab_size):
        """JS-compatible convenience: learn merges from a single string."""
        self.train_documents([text], target_vocab_size)

    def train_documents(self, documents, target_vocab_size):
        """Learn merges from an iterable of documents (no progress reporting)."""
        counter = {}
        for text in documents:
            self.add_document_counts(text, counter)
        return self.train_from_counts(counter, target_vocab_size)

    def train_from_counts(self, counter, target_vocab_size, progress=None):
        """Run BPE over de-duplicated pre-token counts.

        `counter` maps each unique pre-token string to its occurrence count. Each
        is encoded once; the BPE loop then merges the most frequent adjacent pair
        (ties broken by the smaller (a, b) token-id pair — deterministic), using
        an incremental pair index so a merge only revisits the affected words.
        Returns the number of merges learned. `progress(phase, done, total)` is
        called through each long phase ("encoding", "indexing", "merging") so no
        step runs silently on a large corpus.
        """
        def report(phase, done, total):
            if progress is not None:
                progress(phase, done, total)

        # Encode each unique pre-token once; keep those with a mergeable pair.
        items = list(counter.items())
        n_items = len(items)
        words_symbols = []
        words_count = []
        for k in range(n_items):
            piece, cnt = items[k]
            syms = self.encode(piece)
            if len(syms) >= 2:
                words_symbols.append(syms)
                words_count.append(cnt)
            if (k & 0x7FFF) == 0:
                report("encoding", k, n_items)
        report("encoding", n_items, n_items)

        # Initial weighted pair counts and the words each pair lives in.
        pair_count = {}
        pair_words = {}
        n_words = len(words_symbols)
        for w in range(n_words):
            seq = words_symbols[w]
            cnt = words_count[w]
            for pair in zip(seq, seq[1:]):
                pair_count[pair] = pair_count.get(pair, 0) + cnt
                bucket = pair_words.get(pair)
                if bucket is None:
                    bucket = set()
                    pair_words[pair] = bucket
                bucket.add(w)
            if (w & 0x7FFF) == 0:
                report("indexing", w, n_words)
        report("indexing", n_words, n_words)

        # Lazy max-heap keyed by (-count, a, b): highest count first, then the
        # smallest token-id pair. Stale entries are filtered on pop.
        heap = [(-c, a, b) for (a, b), c in pair_count.items()]
        heapq.heapify(heap)

        target_merges = max(0, target_vocab_size - len(self.merges))
        done = 0

        while len(self.merges) < target_vocab_size:
            best = None
            best_count = 0
            while heap:
                neg_c, a, b = heapq.heappop(heap)
                pair = (a, b)
                c = -neg_c
                if pair_count.get(pair, 0) != c:
                    continue  # stale entry
                best = pair
                best_count = c
                break
            if best is None or best_count < 2:
                break

            A, B = best
            new_id = len(self.merges)
            self.merges.append([A, B])

            touched = set()
            for w in list(pair_words.get(best, ())):
                seq = words_symbols[w]
                cnt = words_count[w]

                for pair in zip(seq, seq[1:]):
                    pair_count[pair] = pair_count.get(pair, 0) - cnt
                    touched.add(pair)
                for pair in set(zip(seq, seq[1:])):
                    bucket = pair_words.get(pair)
                    if bucket is not None:
                        bucket.discard(w)

                newseq = self._merge(seq, A, B, new_id)
                words_symbols[w] = newseq

                for pair in zip(newseq, newseq[1:]):
                    pair_count[pair] = pair_count.get(pair, 0) + cnt
                    touched.add(pair)
                for pair in set(zip(newseq, newseq[1:])):
                    pair_words.setdefault(pair, set()).add(w)

            # Reconcile touched pairs: drop empties, push fresh counts.
            for pair in touched:
                c = pair_count.get(pair, 0)
                if c <= 0:
                    pair_count.pop(pair, None)
                    pair_words.pop(pair, None)
                else:
                    heapq.heappush(heap, (-c, pair[0], pair[1]))

            done += 1
            report("merging", done, target_merges)

        return done

    # --- Serialization ------------------------------------------------------

    def serialize_to_json(self):
        """JSON string matching JS `serializeToJSON` (compact, ids 256+)."""
        return json.dumps(
            {"reserved": self.reserved, "merges": self.merges[256:]},
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def deserialize_from_json(self, json_str):
        data = json.loads(json_str)
        self.merges = [[i, i] for i in range(256)]
        self.reserved = data["reserved"]
        for pair in data["merges"]:
            self.merges.append(pair)


# --- File scanning ----------------------------------------------------------


def find_files(folders, extensions, on_progress=None):
    """Recursively find training-file paths under each of `folders` and total size.

    `folders` is an iterable of directory paths (assumed already de-duplicated /
    non-overlapping). Returns (paths, total_bytes). `on_progress(n_files, n_bytes)`
    is called as files are discovered so a large tree (millions of files) isn't
    silent.
    """
    paths = []
    total_bytes = 0
    for folder in folders:
        for dirpath, dirnames, filenames in os.walk(folder):
            dirnames.sort()
            for name in sorted(filenames):
                if os.path.splitext(name)[1].lower() in extensions:
                    p = os.path.join(dirpath, name)
                    try:
                        total_bytes += os.path.getsize(p)
                    except OSError:
                        continue
                    paths.append(p)
                    if on_progress is not None:
                        on_progress(len(paths), total_bytes)
    return paths, total_bytes


# Bytes read per chunk when streaming a file. Bounds peak memory regardless of
# file size, so a single multi-GiB file is fine.
_CHUNK_BYTES = 4 * 1024 * 1024


def scan_file_into_counter(tokenizer, path, counter, on_bytes):
    """Stream a UTF-8 file in chunks, adding each pre-token's count to `counter`.

    A pre-token is never split across a chunk boundary: the last pre-token of a
    chunk is carried forward and re-joined with the next chunk, so the emitted
    stream is identical to splitting the whole file at once. `on_bytes(delta)`
    reports raw bytes consumed (for progress). Returns True if fully read, or
    False if the file isn't valid UTF-8 (any valid prefix already counted).
    """
    import codecs
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    carry = ""
    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(_CHUNK_BYTES)
                if not chunk:
                    break
                on_bytes(len(chunk))
                try:
                    piece = decoder.decode(chunk)
                except UnicodeDecodeError:
                    return False
                if not piece:
                    continue
                pretokens = tokenizer._split(carry + piece)
                # The final pre-token may continue into the next chunk — carry it.
                for p in pretokens[:-1]:
                    counter[p] = counter.get(p, 0) + 1
                carry = pretokens[-1]
            try:
                tail = decoder.decode(b"", final=True)  # flush; raises on truncated char
            except UnicodeDecodeError:
                return False
    except OSError:
        return False
    for p in tokenizer._split(carry + tail):
        counter[p] = counter.get(p, 0) + 1
    return True


# --- Progress formatting ----------------------------------------------------


def _format_duration(seconds):
    if seconds is None or seconds != seconds or seconds < 0:  # None/NaN/neg
        return "?"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _format_bytes(num):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if num < 1024 or unit == "TiB":
            return f"{num:.1f} {unit}" if unit != "B" else f"{int(num)} B"
        num /= 1024


# --- CLI --------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Train a MiniGPT-compatible BPE tokenizer on a folder of text files."
    )
    parser.add_argument("-i", action="append", required=True, metavar="FOLDER",
                        help="folder to scan recursively for training files; repeatable")
    parser.add_argument("-o", default="vocab.json", metavar="FILE",
                        help="output JSON file (default: %(default)s)")
    parser.add_argument("-m", type=int, default=VOCAB_SIZE, metavar="N",
                        help="target vocabulary size — training stops at N tokens, "
                             "including the 256 byte tokens (default: %(default)s)")
    parser.add_argument("-r", action="append", default=[], metavar="TOKEN",
                        help="reserved token, applied before training; repeatable")
    args = parser.parse_args()

    for folder in args.i:
        if not os.path.isdir(folder):
            sys.exit(f"Error: input path is not a folder: {folder}")
    if args.m < 256:
        sys.exit("Error: -m must be at least 256 (the 256 byte tokens are always present)")
    if any(tok == "" for tok in args.r):
        sys.exit("Error: -r reserved token must not be empty")

    # Resolve input folders and drop any that duplicate or nest inside another,
    # so overlapping -i paths don't scan the same files twice.
    folders = []
    for folder in args.i:
        rp = os.path.realpath(folder)
        if any(rp == r or rp.startswith(r + os.sep) for r in folders):
            continue  # already covered by an ancestor folder
        folders = [r for r in folders if not r.startswith(rp + os.sep)]  # drop children
        folders.append(rp)

    ext_str = ", ".join(sorted(TEXT_EXTENSIONS))
    where = folders[0] if len(folders) == 1 else f"{len(folders)} folders"
    print(f"Scanning {where} for {{{ext_str}}} files...", flush=True)
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
        sys.exit(f"Error: no files with extension(s) {{{ext_str}}} found under "
                 f"{', '.join(folders)}")
    print(f"\r  found {len(paths):,} files, {_format_bytes(total_bytes)} total."
          f"                    ")

    tokenizer = Tokenizer()
    for tok in args.r:
        tokenizer.reserve(tok)
    if args.r:
        print(f"Reserved {len(args.r)} token(s): {', '.join(repr(t) for t in args.r)}")

    # --- Phase 1: scan + de-duplicate pre-tokens -----------------------------
    counter = {}
    stats = {"read": 0, "skipped": 0}
    progress = {"bytes": 0}
    t0 = time.monotonic()
    last_print = [0.0]

    def report_scan(force=False):
        now = time.monotonic()
        if not force and now - last_print[0] < 2.0:
            return
        last_print[0] = now
        elapsed = now - t0
        frac = progress["bytes"] / total_bytes if total_bytes else 1.0
        eta = (elapsed / frac - elapsed) if frac > 0 else None
        print(f"\r  scanning {_format_bytes(progress['bytes'])}/{_format_bytes(total_bytes)} "
              f"({frac * 100:.1f}%)  {len(counter):,} unique  ETA {_format_duration(eta)}    ",
              end="", flush=True)

    def on_bytes(delta):
        progress["bytes"] += delta
        report_scan()

    for path in paths:
        if scan_file_into_counter(tokenizer, path, counter, on_bytes):
            stats["read"] += 1
        else:
            stats["skipped"] += 1
            print(f"\r  skipped (not UTF-8 / unreadable): {path}    ")
    report_scan(force=True)
    print()

    if stats["read"] == 0:
        sys.exit("Error: no readable UTF-8 text found.")
    print(f"Scanned {stats['read']} file(s) (skipped {stats['skipped']}); "
          f"{len(counter)} unique pre-tokens.")

    # --- Phase 2: BPE merges -------------------------------------------------
    print(f"Building vocabulary toward {args.m} tokens ({args.m - 256} merges)...")
    phase_state = {"name": None, "t0": 0.0, "last": 0.0}

    def on_progress(phase, done, total):
        now = time.monotonic()
        if phase != phase_state["name"]:
            if phase_state["name"] is not None:
                print()  # finalize the previous phase's line
            phase_state["name"] = phase
            phase_state["t0"] = now
            phase_state["last"] = 0.0
        if now - phase_state["last"] < 1.0 and done < total:
            return
        phase_state["last"] = now
        elapsed = now - phase_state["t0"]
        frac = done / total if total else 1.0
        eta = (elapsed / frac - elapsed) if frac > 0 else None
        print(f"\r  {phase} {done:,}/{total:,} ({frac * 100:.1f}%)  "
              f"ETA {_format_duration(eta)}    ", end="", flush=True)

    tokenizer.train_from_counts(counter, args.m, progress=on_progress)
    print()

    # --- Report --------------------------------------------------------------
    learned = tokenizer.vocab_size() - 256
    requested = args.m - 256
    if tokenizer.vocab_size() < args.m:
        print(f"Learned {learned} merges of {requested} requested — stopped early: "
              f"no remaining adjacent pair occurs 2+ times.")
    else:
        print(f"Learned {learned} merges — reached the target vocabulary.")
    print(f"Final vocabulary: {tokenizer.vocab_size()} tokens.")

    with open(args.o, "w", encoding="utf-8") as f:
        f.write(tokenizer.serialize_to_json())
    print(f"Saved tokenizer to {args.o}")


if __name__ == "__main__":
    main()
