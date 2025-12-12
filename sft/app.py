"""
GabGPT SFT Trainer - Supervised Fine-Tuning for GPT-2 style transformers

Trains on conversation data with loss masking (only trains on assistant responses).
Requires a pre-trained checkpoint (.pt) to start.
"""

import os
import re
import gc
import time
import struct
import threading
import traceback
from pathlib import Path
from typing import Optional, Generator, Tuple, List, Dict
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial

import torch
import psutil
import numpy as np
import gradio as gr
from datasets import load_dataset

from gab import GabGPT, SFTDataset, sft_collate_fn, get_learning_rate


# ============================================================================
# CONFIGURATION
# ============================================================================

TMP_DIR = Path("/tmp/gabgpt_sft")
LOG_FILE = TMP_DIR / "logs.txt"
CACHE_DIR = Path("/data") if os.path.exists("/data") else TMP_DIR / "cache"
CACHE_FILE = CACHE_DIR / "cache.gab"
MAX_LOG_SIZE = 1024 * 1024 * 1024  # 1 GB rolling buffer
ENCODER_DIR = TMP_DIR / "encoder"
ENCODER_BINARY = ENCODER_DIR / "encoder"
TOKENS_FILE = TMP_DIR / "tokens.bin"
DATASET_FILE = TMP_DIR / "dataset.txt"

# Streaming chunk size for large datasets
CHUNK_SIZE_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB per chunk

TMP_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# EMBEDDED C ENCODER - Streaming BPE encoder for large datasets
# ============================================================================

ENCODER_C = r'''/*
 * encoder.c - Streaming BPE Encoder for Gab
 *
 * Processes null-delimited samples from input file, encodes each with BPE,
 * writes tokens to binary output. Memory usage is O(max_sample_size).
 *
 * Optimization: Uses token presence bitmap to skip merges that cannot apply.
 * Most merges are skipped because their tokens don't exist in the sample.
 *
 * Usage: ./encoder vocab.bpe input.txt output.tokens
 *
 * Input format: UTF-8 text samples separated by null bytes (\0)
 * Output format: TOK1 header + uint32_t token count + uint32_t tokens[]
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

/* Maximum sample size (1MB = ~250K chars, plenty for any story) */
#define MAX_SAMPLE_SIZE (1 << 20)

/* Maximum vocabulary size for bitmap (64K tokens = 8KB bitmap) */
#define MAX_VOCAB 65536
#define BITMAP_SIZE (MAX_VOCAB / 8)

/* Merge rule */
typedef struct {
    uint32_t t1;
    uint32_t t2;
    uint32_t merged;
} Merge;

static Merge *g_merges = NULL;
static uint32_t g_num_merges = 0;

/* Per-sample encoding buffers (ping-pong) */
static uint32_t *g_buf_a = NULL;
static uint32_t *g_buf_b = NULL;

/* Sample read buffer */
static uint8_t *g_sample_buf = NULL;

/* Token presence bitmap - tracks which tokens exist in current sample */
static uint8_t g_token_present[BITMAP_SIZE];

/* ========================================================================== */
/* Bitmap operations                                                          */
/* ========================================================================== */

static inline void bitmap_clear(void) {
    memset(g_token_present, 0, BITMAP_SIZE);
}

static inline void bitmap_set(uint32_t token) {
    if (token < MAX_VOCAB) {
        g_token_present[token >> 3] |= (1 << (token & 7));
    }
}

static inline int bitmap_test(uint32_t token) {
    if (token >= MAX_VOCAB) return 0;
    return (g_token_present[token >> 3] >> (token & 7)) & 1;
}

/* ========================================================================== */

static int load_bpe(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open vocab file '%s'\n", path);
        return -1;
    }

    uint32_t magic, version, next_token;
    if (fread(&magic, 4, 1, f) != 1 ||
        fread(&version, 4, 1, f) != 1 ||
        fread(&next_token, 4, 1, f) != 1) {
        fprintf(stderr, "Error: Failed to read .bpe header\n");
        fclose(f);
        return -1;
    }

    if (magic != 0x42504531) {
        fprintf(stderr, "Error: Invalid .bpe file (bad magic: 0x%08X)\n", magic);
        fclose(f);
        return -1;
    }

    if (fread(&g_num_merges, 4, 1, f) != 1) {
        fprintf(stderr, "Error: Failed to read merge count\n");
        fclose(f);
        return -1;
    }

    g_merges = (Merge *)malloc(g_num_merges * sizeof(Merge));
    if (!g_merges) {
        fprintf(stderr, "Error: Failed to allocate merge table\n");
        fclose(f);
        return -1;
    }

    for (uint32_t i = 0; i < g_num_merges; i++) {
        if (fread(&g_merges[i].t1, 4, 1, f) != 1 ||
            fread(&g_merges[i].t2, 4, 1, f) != 1 ||
            fread(&g_merges[i].merged, 4, 1, f) != 1) {
            fprintf(stderr, "Error: Failed to read merge %u\n", i);
            fclose(f);
            return -1;
        }
    }

    fclose(f);
    return 0;
}

static size_t apply_merge(const uint32_t *src, size_t count,
                          uint32_t *dst,
                          uint32_t t1, uint32_t t2, uint32_t merged) {
    size_t dst_count = 0;
    size_t i = 0;

    while (i < count) {
        if (i < count - 1 && src[i] == t1 && src[i + 1] == t2) {
            dst[dst_count++] = merged;
            i += 2;
        } else {
            dst[dst_count++] = src[i];
            i++;
        }
    }

    return dst_count;
}

static size_t encode_sample(const uint8_t *text, size_t len, uint32_t *output) {
    if (len == 0) return 0;

    /* Initialize with raw bytes */
    for (size_t i = 0; i < len; i++) {
        g_buf_a[i] = text[i];
    }

    /* Build bitmap of which tokens are present */
    bitmap_clear();
    for (size_t i = 0; i < len; i++) {
        bitmap_set(text[i]);
    }

    uint32_t *current = g_buf_a;
    uint32_t *other = g_buf_b;
    size_t count = len;

    /* Apply merges in order, skipping those that can't apply */
    for (uint32_t m = 0; m < g_num_merges; m++) {
        uint32_t t1 = g_merges[m].t1;
        uint32_t t2 = g_merges[m].t2;
        uint32_t merged = g_merges[m].merged;

        /* Skip if either token doesn't exist in current sequence */
        if (!bitmap_test(t1) || !bitmap_test(t2)) {
            continue;
        }

        /* Apply merge */
        size_t new_count = apply_merge(current, count, other, t1, t2, merged);

        /* Only swap if merge actually changed something */
        if (new_count < count) {
            uint32_t *tmp = current;
            current = other;
            other = tmp;
            count = new_count;

            /* Mark new token as present for future merges */
            bitmap_set(merged);
        }
    }

    /* Copy to output */
    memcpy(output, current, count * sizeof(uint32_t));
    return count;
}

int main(int argc, char *argv[]) {
    /* Unbuffered output for progress */
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    if (argc != 4) {
        fprintf(stderr, "Usage: %s vocab.bpe input.txt output.tokens\n", argv[0]);
        return 1;
    }

    const char *vocab_path = argv[1];
    const char *input_path = argv[2];
    const char *output_path = argv[3];

    /* Load vocabulary */
    printf("Loading vocabulary from %s...\n", vocab_path);
    if (load_bpe(vocab_path) != 0) return 1;
    printf("Loaded %u merges\n", g_num_merges);

    /* Allocate buffers */
    g_buf_a = (uint32_t *)malloc(MAX_SAMPLE_SIZE * sizeof(uint32_t));
    g_buf_b = (uint32_t *)malloc(MAX_SAMPLE_SIZE * sizeof(uint32_t));
    g_sample_buf = (uint8_t *)malloc(MAX_SAMPLE_SIZE);

    if (!g_buf_a || !g_buf_b || !g_sample_buf) {
        fprintf(stderr, "Error: Failed to allocate buffers\n");
        return 1;
    }

    /* Open input file */
    FILE *fin = fopen(input_path, "rb");
    if (!fin) {
        fprintf(stderr, "Error: Cannot open input file '%s'\n", input_path);
        return 1;
    }

    /* Get input file size for progress */
    fseek(fin, 0, SEEK_END);
    size_t file_size = ftell(fin);
    fseek(fin, 0, SEEK_SET);
    printf("Input file: %.2f MB\n", file_size / (1024.0 * 1024.0));

    /* Open output file */
    FILE *fout = fopen(output_path, "wb");
    if (!fout) {
        fprintf(stderr, "Error: Cannot create output file '%s'\n", output_path);
        fclose(fin);
        return 1;
    }

    /* Write placeholder header (will update at end) */
    uint32_t magic = 0x544F4B31;  /* 'TOK1' */
    uint64_t total_tokens = 0;
    fwrite(&magic, 4, 1, fout);
    fwrite(&total_tokens, 8, 1, fout);

    /* Process samples */
    printf("Encoding samples...\n");
    clock_t start = clock();

    size_t sample_count = 0;
    size_t bytes_read = 0;
    size_t sample_len = 0;
    int c;
    int last_pct = -1;

    /* Temporary buffer for encoded tokens */
    uint32_t *encoded = (uint32_t *)malloc(MAX_SAMPLE_SIZE * sizeof(uint32_t));
    if (!encoded) {
        fprintf(stderr, "Error: Failed to allocate output buffer\n");
        return 1;
    }

    while ((c = fgetc(fin)) != EOF) {
        bytes_read++;

        if (c == '\0') {
            /* End of sample - encode it */
            if (sample_len > 0) {
                size_t num_tokens = encode_sample(g_sample_buf, sample_len, encoded);
                fwrite(encoded, sizeof(uint32_t), num_tokens, fout);
                total_tokens += num_tokens;
            }
            sample_count++;
            sample_len = 0;

            /* Progress */
            int pct = (int)((bytes_read * 100) / file_size);
            if (pct != last_pct && (pct % 5 == 0 || pct == 1)) {
                double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
                double rate = bytes_read / (1024.0 * 1024.0) / elapsed;
                printf("[%3d%%] %zu samples, %zu tokens (%.1f MB/s)\n",
                       pct, sample_count, (size_t)total_tokens, rate);
                last_pct = pct;
            }
        } else {
            /* Accumulate sample */
            if (sample_len < MAX_SAMPLE_SIZE - 1) {
                g_sample_buf[sample_len++] = (uint8_t)c;
            }
        }
    }

    /* Handle last sample if file doesn't end with \0 */
    if (sample_len > 0) {
        size_t num_tokens = encode_sample(g_sample_buf, sample_len, encoded);
        fwrite(encoded, sizeof(uint32_t), num_tokens, fout);
        total_tokens += num_tokens;
        sample_count++;
    }

    /* Update header with actual token count */
    fseek(fout, 4, SEEK_SET);
    fwrite(&total_tokens, 8, 1, fout);

    fclose(fin);
    fclose(fout);

    double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
    printf("\nEncoding complete!\n");
    printf("  Samples: %zu\n", sample_count);
    printf("  Tokens:  %zu\n", (size_t)total_tokens);
    printf("  Time:    %.2f seconds\n", elapsed);
    printf("  Rate:    %.1f samples/sec\n", sample_count / elapsed);
    printf("  Compression: %.2fx\n", (double)file_size / (total_tokens * 4));
    printf("Saved to %s\n", output_path);

    /* Cleanup */
    free(g_merges);
    free(g_buf_a);
    free(g_buf_b);
    free(g_sample_buf);
    free(encoded);

    return 0;
}
'''


def compile_encoder() -> Tuple[bool, str]:
    """Compile the embedded C encoder. Returns (success, message)."""
    import subprocess

    ENCODER_DIR.mkdir(parents=True, exist_ok=True)
    src_path = ENCODER_DIR / "encoder.c"

    # Write C source
    with open(src_path, 'w') as f:
        f.write(ENCODER_C)

    # Compile
    result = subprocess.run(
        ["gcc", "-O3", "-o", str(ENCODER_BINARY), str(src_path)],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        return False, f"Compilation failed: {result.stderr}"

    return True, "Encoder compiled successfully"


def run_encoder(vocab_path: str, input_path: str, output_path: str,
                log_fn=None) -> Tuple[bool, str]:
    """Run the C encoder with streaming output. Returns (success, message)."""
    import subprocess

    if not ENCODER_BINARY.exists():
        return False, "Encoder not compiled"

    # Use Popen to stream output in real-time
    process = subprocess.Popen(
        [str(ENCODER_BINARY), vocab_path, input_path, output_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1  # Line buffered
    )

    output_lines = []
    for line in process.stdout:
        line = line.rstrip()
        output_lines.append(line)
        if log_fn:
            log_fn(f"[encoder] {line}")

    process.wait()

    if process.returncode != 0:
        return False, f"Encoder failed with code {process.returncode}"

    return True, "\n".join(output_lines)


def load_tokens_file(path: str) -> Optional[np.ndarray]:
    """Load tokens from binary file. Returns memory-mapped array or None on error."""
    with open(path, 'rb') as f:
        magic = struct.unpack('<I', f.read(4))[0]
        if magic != 0x544F4B31:  # 'TOK1'
            return None

        num_tokens = struct.unpack('<Q', f.read(8))[0]

    # Memory-map the token data (skip 12-byte header)
    return np.memmap(path, dtype=np.int32, mode='r', offset=12, shape=(num_tokens,))


def write_chunk_to_file(conversations: List[str], output_path: str) -> Tuple[int, int]:
    """
    Write formatted conversations to a null-delimited file.
    Returns (num_samples, bytes_written).
    """
    bytes_written = 0

    with open(output_path, 'wb') as f:
        for text in conversations:
            text_bytes = text.encode('utf-8')
            f.write(text_bytes)
            f.write(b'\0')
            bytes_written += len(text_bytes) + 1

    return len(conversations), bytes_written


def stream_sft_dataset_chunks(dataset_name: str, split: str, conv_column: str,
                               allow_think: bool = True,
                               max_samples: int = 0,
                               chunk_size_bytes: int = CHUNK_SIZE_BYTES,
                               log_fn=None):
    """
    Stream SFT dataset in chunks of approximately chunk_size_bytes.
    Uses HuggingFace streaming mode - never loads full dataset.

    Yields: (chunk_conversations, chunk_samples, chunk_bytes, total_bytes_so_far)
    """
    if log_fn:
        log_fn(f"Connecting to HuggingFace dataset: {dataset_name}...")

    dataset = load_dataset(dataset_name, split=split, streaming=True)

    if log_fn:
        log_fn(f"Dataset connected, starting to stream...")

    chunk_texts = []
    chunk_bytes = 0
    total_bytes = 0
    total_samples = 0
    chunk_num = 0
    last_progress_mb = 0

    for sample in dataset:
        # Check max_samples limit
        if max_samples > 0 and total_samples >= max_samples:
            break

        conversation = sample.get(conv_column, [])
        if not conversation:
            continue

        # Format conversation to training format
        formatted = format_conversation(conversation, allow_think=allow_think)
        if not formatted:
            continue

        text_bytes = len(formatted.encode('utf-8'))
        chunk_texts.append(formatted)
        chunk_bytes += text_bytes
        total_bytes += text_bytes
        total_samples += 1

        # Log progress every 250MB while accumulating chunk
        current_mb = chunk_bytes // (250 * 1024 * 1024)
        if current_mb > last_progress_mb:
            last_progress_mb = current_mb
            if log_fn:
                log_fn(f"Accumulating chunk: {chunk_bytes / (1024**3):.2f}/{chunk_size_bytes / (1024**3):.1f} GB ({total_samples:,} samples)")

        if chunk_bytes >= chunk_size_bytes:
            chunk_num += 1
            if log_fn:
                log_fn(f"Chunk {chunk_num} ready: {len(chunk_texts):,} samples, {chunk_bytes / (1024**3):.2f} GB")

            yield (chunk_texts, len(chunk_texts), chunk_bytes, total_bytes)

            chunk_texts = []
            chunk_bytes = 0
            last_progress_mb = 0

    # Final partial chunk
    if chunk_texts:
        chunk_num += 1
        if log_fn:
            log_fn(f"Final chunk {chunk_num}: {len(chunk_texts):,} samples, {chunk_bytes / (1024**3):.2f} GB")
        yield (chunk_texts, len(chunk_texts), chunk_bytes, total_bytes)


# ============================================================================
# BPE TOKENIZER (minimal - only for get_special_token_id lookups)
# The C encoder handles bulk tokenization. This is only used to compute
# special token IDs by encoding small strings.
# ============================================================================

class BPETokenizer:
    """
    Minimal BPE Tokenizer for special token ID lookups.
    Bulk tokenization is done by the C encoder for performance.
    """

    def __init__(self):
        self.merges: List[Tuple[int, int, int]] = []  # (token1, token2, merged_token)
        self.vocab_size: int = 256  # Start with byte tokens

    def load(self, path: str) -> 'BPETokenizer':
        """Load tokenizer from .bpe file"""
        with open(path, 'rb') as f:
            magic = struct.unpack('<I', f.read(4))[0]
            if magic != 0x42504531:  # 'BPE1'
                raise ValueError(f"Invalid .bpe file (bad magic: 0x{magic:08X})")

            version = struct.unpack('<I', f.read(4))[0]
            self.vocab_size = struct.unpack('<I', f.read(4))[0]
            num_merges = struct.unpack('<I', f.read(4))[0]

            self.merges = []
            for _ in range(num_merges):
                t1 = struct.unpack('<I', f.read(4))[0]
                t2 = struct.unpack('<I', f.read(4))[0]
                merged = struct.unpack('<I', f.read(4))[0]
                self.merges.append((t1, t2, merged))

            # Skip vocabulary section (token_id -> byte sequence mappings)
            try:
                num_vocab = struct.unpack('<I', f.read(4))[0]
                for _ in range(num_vocab):
                    token_id = struct.unpack('<I', f.read(4))[0]
                    byte_len = struct.unpack('<I', f.read(4))[0]
                    f.read(byte_len)
            except struct.error:
                pass

        return self

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        if not text:
            return []

        tokens = list(text.encode('utf-8'))

        for t1, t2, merged in self.merges:
            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == t1 and tokens[i + 1] == t2:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def get_special_token_id(self, name: str) -> Optional[int]:
        """Get token ID for a special token by encoding it"""
        tokens = self.encode(name)
        return tokens[0] if len(tokens) == 1 else None


# Global tokenizer instance
tokenizer: Optional[BPETokenizer] = None


# ============================================================================
# CONVERSATION FORMATTING
# ============================================================================

def format_conversation(messages: List[Dict], allow_think: bool = True) -> str:
    """
    Convert conversation messages to training format.

    Input format:
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "<think>...</think>\n\nResponse"}]

    Output format (with allow_think=True):
        <|user|>
        User content
        <|think|>
        Thinking content (if present)
        <|assistant|>
        Response content<|end|>

    Output format (with allow_think=False):
        <|user|>
        User content
        <|assistant|>
        Response content<|end|>
    """
    output = ""

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "user":
            output += f"<|user|>\n{content}\n"

        elif role == "assistant":
            # Check for <think>...</think> tags
            think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)

            if think_match:
                thinking = think_match.group(1).strip()
                # Get response after </think>
                response = content[think_match.end():].strip()

                if allow_think:
                    output += f"<|think|>\n{thinking}\n<|assistant|>\n{response}"
                else:
                    # Skip thinking, only include response
                    output += f"<|assistant|>\n{response}"
            else:
                # No thinking, straight to response
                output += f"<|assistant|>\n{content}"

    output += "<|end|>"
    return output


def process_chunk(chunk_texts: List[str], max_seq_length: int,
                  special_token_ids: dict, log_fn=None) -> List[dict]:
    """
    Process a chunk of formatted conversations using the C encoder.

    Args:
        chunk_texts: List of formatted conversation strings
        max_seq_length: Maximum sequence length
        special_token_ids: Dict mapping special token names to their IDs
        log_fn: Optional logging function

    Returns:
        List of dicts with 'input_ids' and 'loss_mask'
    """
    global state

    if state.tokenizer_path is None:
        raise ValueError("Tokenizer not loaded")

    # Get special token IDs for loss masking
    end_token = special_token_ids.get('<|end|>')
    user_tokens = {v for k, v in special_token_ids.items()
                   if v is not None and ('user' in k.lower() or 'system' in k.lower() or 'result' in k.lower())}
    assistant_tokens = {v for k, v in special_token_ids.items()
                        if v is not None and ('assistant' in k.lower() or 'think' in k.lower() or 'call' in k.lower())}
    end_tokens = {v for k, v in special_token_ids.items()
                  if v is not None and 'end' in k.lower()}
    pad_tokens = {v for k, v in special_token_ids.items()
                  if v is not None and 'pad' in k.lower()}

    # Step 1: Write chunk to null-delimited temp file
    if log_fn:
        log_fn(f"Writing {len(chunk_texts)} conversations to temp file...")
    num_written, bytes_written = write_chunk_to_file(chunk_texts, str(DATASET_FILE))

    # Step 2: Run C encoder
    if log_fn:
        log_fn(f"Encoding with C encoder...")
    success, output = run_encoder(
        state.tokenizer_path,
        str(DATASET_FILE),
        str(TOKENS_FILE),
        log_fn=log_fn
    )

    if not success:
        raise RuntimeError(f"C encoder failed: {output}")

    # Step 3: Load tokens from TOK1 file
    all_tokens = load_tokens_file(str(TOKENS_FILE))
    if all_tokens is None:
        raise RuntimeError("Failed to load token file")

    if log_fn:
        log_fn(f"Loaded {len(all_tokens):,} tokens from C encoder")

    # Step 4: Split tokens by end_token into individual conversations
    examples = []
    skipped = 0
    conv_start = 0
    conv_idx = 0

    for i, token in enumerate(all_tokens):
        if token == end_token:
            # Found end of conversation
            conv_tokens = list(all_tokens[conv_start:i + 1])  # Include end token

            # Truncate if too long
            if len(conv_tokens) > max_seq_length:
                conv_tokens = conv_tokens[:max_seq_length]
                # Ensure we end with end_token
                if conv_tokens[-1] != end_token:
                    conv_tokens[-1] = end_token

            # Create loss mask
            loss_mask = []
            in_trainable_region = False

            for tid in conv_tokens:
                if tid in user_tokens:
                    in_trainable_region = False
                    loss_mask.append(0)
                elif tid in assistant_tokens:
                    in_trainable_region = True
                    loss_mask.append(1)
                elif tid in end_tokens:
                    loss_mask.append(1)
                elif tid in pad_tokens:
                    loss_mask.append(0)
                else:
                    loss_mask.append(1 if in_trainable_region else 0)

            # Skip if no trainable tokens
            if sum(loss_mask) == 0:
                skipped += 1
            else:
                examples.append({
                    'input_ids': conv_tokens,
                    'loss_mask': loss_mask,
                })

            conv_start = i + 1
            conv_idx += 1

    # Clean up temp files
    if DATASET_FILE.exists():
        DATASET_FILE.unlink()
    if TOKENS_FILE.exists():
        # Don't delete yet - memmap might still be in use
        pass

    if log_fn:
        log_fn(f"Chunk processed: {len(examples)} examples ({skipped} skipped)")

    return examples


# ============================================================================
# LOGGING SYSTEM
# ============================================================================

class RollingFileLogger:
    """Thread-safe rolling file logger with size limit"""

    def __init__(self, filepath: Path, max_size: int = MAX_LOG_SIZE):
        self.filepath = filepath
        self.max_size = max_size
        self.lock = threading.Lock()
        self._ensure_file()

    def _ensure_file(self):
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        if not self.filepath.exists():
            self.filepath.touch()

    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        formatted = f"[{timestamp}] [{level}] {message}\n"

        with self.lock:
            try:
                if self.filepath.exists():
                    size = self.filepath.stat().st_size
                    if size > self.max_size:
                        with open(self.filepath, 'r') as f:
                            content = f.read()
                        with open(self.filepath, 'w') as f:
                            f.write(content[len(content)//2:])

                with open(self.filepath, 'a') as f:
                    f.write(formatted)
            except Exception as e:
                print(f"Logging error: {e}")

    def read_latest(self, lines: int = 500) -> str:
        with self.lock:
            try:
                if not self.filepath.exists():
                    return ""
                with open(self.filepath, 'r') as f:
                    all_lines = f.readlines()
                return ''.join(all_lines[-lines:])
            except Exception:
                return ""

    def clear(self):
        with self.lock:
            try:
                with open(self.filepath, 'w') as f:
                    f.write("")
            except Exception:
                pass


logger = RollingFileLogger(LOG_FILE)


def log(message: str, level: str = "INFO"):
    logger.log(message, level)
    print(f"[{level}] {message}")


def get_memory_stats() -> str:
    """Get current CPU and GPU memory usage."""
    ram = psutil.virtual_memory()
    ram_used_gb = ram.used / (1024 ** 3)
    ram_total_gb = ram.total / (1024 ** 3)

    if torch.cuda.is_available():
        gpu_used_gb = torch.cuda.memory_reserved() / (1024 ** 3)
        gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return f"RAM: {ram_used_gb:.1f}/{ram_total_gb:.1f}GB | GPU: {gpu_used_gb:.1f}/{gpu_total_gb:.1f}GB"
    else:
        return f"RAM: {ram_used_gb:.1f}/{ram_total_gb:.1f}GB | GPU: N/A"


# ============================================================================
# TRAINING STATE
# ============================================================================

@dataclass
class TrainingState:
    """Global training state"""
    model: Optional[GabGPT] = None
    optimizer: Optional[torch.optim.Adam] = None

    # Tokenizer
    tokenizer_loaded: bool = False
    tokenizer_path: Optional[str] = None

    # Checkpoint info
    checkpoint_loaded: bool = False
    model_config: dict = field(default_factory=dict)

    # Training progress
    is_training: bool = False
    is_paused: bool = False
    current_epoch: int = 0
    current_step: int = 0
    total_steps: int = 0
    current_loss: float = 0.0
    avg_loss: float = 0.0
    status: str = "Idle"

    # Streaming/chunked training progress
    chunks_completed: int = 0
    samples_seen: int = 0
    bytes_seen: int = 0

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


state = TrainingState()


# ============================================================================
# CACHE SYSTEM
# ============================================================================

def has_persistent_storage() -> bool:
    return os.path.exists("/data") and os.access("/data", os.W_OK)


def save_to_cache() -> str:
    if state.model is None:
        return "No model to cache"

    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        state.model.save_gab(str(CACHE_FILE))
        gc.collect()

        if has_persistent_storage():
            log(f"Model cached to persistent storage: {CACHE_FILE}", "INFO")
            return f"Saved to cache: {CACHE_FILE}"
        else:
            log(f"WARNING: No persistent storage! Cached to temp: {CACHE_FILE}", "WARNING")
            return f"WARNING: No persistent storage! Saved to temp: {CACHE_FILE}"
    except Exception as e:
        log(f"Failed to save cache: {e}", "ERROR")
        return f"Error: {e}"


def clear_cache() -> str:
    try:
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()
            log("Cache cleared", "INFO")
            return "Cache cleared"
        return "No cache to clear"
    except Exception as e:
        log(f"Failed to clear cache: {e}", "ERROR")
        return f"Error: {e}"


def download_cache() -> Optional[str]:
    if CACHE_FILE.exists():
        return str(CACHE_FILE)
    return None


def check_cache_status() -> str:
    if CACHE_FILE.exists():
        size_bytes = CACHE_FILE.stat().st_size
        if size_bytes >= 1e9:
            size_str = f"{size_bytes / 1e9:.2f} GB"
        elif size_bytes >= 1e6:
            size_str = f"{size_bytes / 1e6:.2f} MB"
        else:
            size_str = f"{size_bytes / 1e3:.2f} KB"
        return f"Cache exists: {size_str}"
    return "No cache file"


# ============================================================================
# CHECKPOINT & TOKENIZER HANDLING
# ============================================================================

def import_checkpoint(file) -> Tuple[str, str]:
    """Import a .pt checkpoint file"""
    global state

    if file is None:
        return "No file selected", ""

    try:
        filepath = file.name if hasattr(file, 'name') else file
        log(f"Importing checkpoint: {filepath}", "INFO")

        checkpoint = torch.load(filepath, map_location=state.device, weights_only=False)
        config = checkpoint.get('config', {})

        # Create model from config
        state.model = GabGPT(
            vocab_size=config['vocab_size'],
            embedding_dim=config['embedding_dim'],
            num_heads=config['num_heads'],
            num_blocks=config['num_blocks'],
            max_seq_length=config['max_seq_length']
        ).to(state.device)

        state.model.load_state_dict(checkpoint['model_state_dict'])

        # Create optimizer and load state
        state.optimizer = torch.optim.Adam(state.model.parameters())
        state.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Store config
        state.model_config = config
        state.checkpoint_loaded = True
        state.current_step = checkpoint.get('step', 0)

        # Format config info for display
        config_str = (
            f"Vocab: {config['vocab_size']} | "
            f"Embed: {config['embedding_dim']} | "
            f"Heads: {config['num_heads']} | "
            f"Blocks: {config['num_blocks']} | "
            f"MaxSeq: {config['max_seq_length']}"
        )

        log(f"Checkpoint loaded: {config_str}", "INFO")
        log(f"Resuming from step {state.current_step}", "INFO")

        return "Checkpoint loaded!", config_str

    except Exception as e:
        log(f"Failed to import checkpoint: {e}\n{traceback.format_exc()}", "ERROR")
        return f"Error: {e}", ""


def import_tokenizer(file) -> Tuple[str, int]:
    """Import a .bpe tokenizer file"""
    global tokenizer, state

    if file is None:
        return "No file selected", 0

    try:
        filepath = file.name if hasattr(file, 'name') else file
        log(f"Importing tokenizer: {filepath}", "INFO")

        tokenizer = BPETokenizer().load(filepath)
        state.tokenizer_loaded = True
        state.tokenizer_path = filepath

        # Log special tokens
        if tokenizer.special_tokens:
            log(f"Special tokens: {list(tokenizer.special_tokens.keys())}", "INFO")

        log(f"Tokenizer loaded! Vocab size: {tokenizer.vocab_size}", "INFO")
        return f"Loaded! Vocab: {tokenizer.vocab_size}", tokenizer.vocab_size

    except Exception as e:
        log(f"Failed to import tokenizer: {e}", "ERROR")
        return f"Error: {e}", 0


def export_checkpoint() -> Optional[str]:
    """Export current training state as .pt file"""
    if state.model is None or state.optimizer is None:
        log("No model/optimizer to export", "WARNING")
        return None

    try:
        filepath = TMP_DIR / f"sft_checkpoint_{int(time.time())}.pt"
        state.model.save_checkpoint(
            str(filepath),
            state.optimizer,
            state.current_step,
            state.current_loss,
            epoch=state.current_epoch,
        )
        log(f"Checkpoint exported: {filepath}", "INFO")
        return str(filepath)
    except Exception as e:
        log(f"Failed to export checkpoint: {e}", "ERROR")
        return None


# ============================================================================
# TRAINING LOOP
# ============================================================================

def training_worker(
    # Dataset config
    dataset_name: str,
    dataset_split: str,
    conv_column: str,
    max_samples: int,
    allow_think: bool,
    special_tokens_str: str,
    # Training config
    num_epochs: int,
    batch_size: int,
    max_seq_length: int,
    learning_rate: float,
    warmup_ratio: float,
    log_every: int,
):
    """
    SFT Training worker that runs in a background thread.
    Streams chunks, tokenizes each with C encoder, trains, then discards.
    Updates state object directly - UI polls for updates.
    """
    global state, tokenizer

    try:
        # ================================================================
        # PHASE 1: Validation
        # ================================================================
        state.status = "Validating..."
        state.is_training = True
        log("=" * 60, "INFO")
        log("STARTING CHUNKED STREAMING SFT", "INFO")
        log("=" * 60, "INFO")
        log(f"Dataset: {dataset_name} (split: {dataset_split})", "INFO")
        log(f"Chunk size: {CHUNK_SIZE_BYTES / (1024**3):.1f} GB", "INFO")

        # Clean up any leftover files from previous runs
        for old_file in [DATASET_FILE, TOKENS_FILE]:
            if old_file.exists():
                try:
                    old_file.unlink()
                except Exception:
                    pass

        if not state.checkpoint_loaded or state.model is None:
            log("ERROR: No checkpoint loaded! Please import a .pt file first.", "ERROR")
            state.status = "Error: No checkpoint"
            state.is_training = False
            return

        if not state.tokenizer_loaded or tokenizer is None:
            log("ERROR: No tokenizer loaded! Please import a .bpe file first.", "ERROR")
            state.status = "Error: No tokenizer"
            state.is_training = False
            return

        # ================================================================
        # PHASE 2: Compile C Encoder
        # ================================================================
        state.status = "Compiling encoder..."
        log("Compiling C encoder...", "INFO")

        success, msg = compile_encoder()
        if not success:
            log(f"ERROR: {msg}", "ERROR")
            state.status = "Error: Encoder compilation failed"
            state.is_training = False
            return
        log(msg, "INFO")

        # ================================================================
        # PHASE 3: Compute Special Token IDs
        # ================================================================
        state.status = "Computing special tokens..."
        log("Computing special token IDs...", "INFO")

        special_tokens = [t.strip() for t in special_tokens_str.split(',') if t.strip()]
        special_token_ids = {}
        for tok in special_tokens:
            token_id = tokenizer.get_special_token_id(tok)
            if token_id is not None:
                special_token_ids[tok] = token_id
                log(f"  {tok} -> {token_id}", "INFO")

        pad_token_id = special_token_ids.get('<|pad|>')
        if pad_token_id is None:
            log("ERROR: Tokenizer missing <|pad|> token!", "ERROR")
            state.status = "Error: Missing pad token"
            state.is_training = False
            return

        end_token_id = special_token_ids.get('<|end|>')
        if end_token_id is None:
            log("ERROR: Tokenizer missing <|end|> token!", "ERROR")
            state.status = "Error: Missing end token"
            state.is_training = False
            return

        # ================================================================
        # PHASE 4: Setup Training
        # ================================================================
        state.status = "Setting up training..."

        # Use model's max_seq_length if not overridden
        actual_max_seq = min(max_seq_length, state.model_config.get('max_seq_length', max_seq_length))
        log(f"Max sequence length: {actual_max_seq}", "INFO")

        # Update learning rate
        for param_group in state.optimizer.param_groups:
            param_group['lr'] = learning_rate
        log(f"Learning rate: {learning_rate}", "INFO")
        log(f"Device: {state.device}", "INFO")
        log(f"Memory at start: {get_memory_stats()}", "INFO")
        log("=" * 60, "INFO")

        # ================================================================
        # PHASE 5: Chunked Streaming Training
        # ================================================================
        global_step = state.current_step
        chunk_losses = []

        # For warmup, estimate based on first chunk (will be updated)
        warmup_steps = 1000  # Initial estimate, gets refined

        log("Starting chunked streaming...", "INFO")
        state.status = "Streaming..."

        chunk_generator = stream_sft_dataset_chunks(
            dataset_name, dataset_split, conv_column,
            allow_think=allow_think,
            max_samples=max_samples if max_samples > 0 else 0,
            chunk_size_bytes=CHUNK_SIZE_BYTES,
            log_fn=lambda m: log(m, "INFO")
        )

        for chunk_idx, (chunk_texts, chunk_samples, chunk_bytes, total_bytes_so_far) in enumerate(chunk_generator):
            # Check if we should stop
            if not state.is_training:
                log("Training stopped by user", "WARNING")
                return

            log(f"{'='*60}", "INFO")
            log(f"CHUNK {chunk_idx + 1} | {chunk_samples:,} samples | {chunk_bytes / (1024**3):.2f} GB", "INFO")
            log(f"Total streamed so far: {total_bytes_so_far / (1024**3):.2f} GB", "INFO")

            # Step 5a: Process chunk with C encoder
            state.status = f"Processing chunk {chunk_idx + 1}..."

            examples = process_chunk(
                chunk_texts, actual_max_seq, special_token_ids,
                log_fn=lambda m: log(m, "INFO")
            )

            # Free chunk texts from memory
            del chunk_texts
            gc.collect()

            if not examples:
                log(f"WARNING: Chunk {chunk_idx + 1} produced no valid examples, skipping", "WARNING")
                continue

            # Step 5b: Create dataloader for this chunk
            dataset = SFTDataset(examples)
            collate_fn = partial(sft_collate_fn, pad_token_id=pad_token_id)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
            )

            batches_in_chunk = len(dataloader)
            state.total_steps = global_step + (batches_in_chunk * num_epochs)
            log(f"Dataloader: {batches_in_chunk} batches (batch={batch_size})", "INFO")

            # Update warmup estimate based on actual data
            if chunk_idx == 0:
                estimated_total_steps = batches_in_chunk * num_epochs * 100  # Assume ~100 chunks
                warmup_steps = int(estimated_total_steps * warmup_ratio)
                log(f"Warmup steps (estimated): {warmup_steps}", "INFO")

            # Step 5c: Train on this chunk for num_epochs
            state.status = "Training..."
            chunk_epoch_losses = []

            for epoch in range(num_epochs):
                state.current_epoch = epoch
                epoch_losses = []
                log(f"  Epoch {epoch + 1}/{num_epochs} on chunk {chunk_idx + 1}", "INFO")

                for batch_idx, batch in enumerate(dataloader):
                    # Check pause
                    while state.is_paused:
                        state.status = "Paused"
                        time.sleep(0.5)
                        if not state.is_training:
                            log("Training stopped while paused", "WARNING")
                            return

                    if not state.is_training:
                        log("Training stopped by user", "WARNING")
                        return

                    state.status = "Training..."

                    # Move batch to device
                    input_tokens = batch['input_tokens'].to(state.device)
                    target_tokens = batch['target_tokens'].to(state.device)
                    attention_mask = batch['attention_mask'].to(state.device)
                    loss_mask = batch['loss_mask'].to(state.device)

                    # Update learning rate
                    lr = get_learning_rate(global_step, warmup_steps, warmup_steps * 10, learning_rate)
                    for param_group in state.optimizer.param_groups:
                        param_group['lr'] = lr

                    # Training step with loss masking
                    loss = state.model.train_step(
                        input_tokens, target_tokens, state.optimizer,
                        attention_mask=attention_mask,
                        loss_mask=loss_mask
                    )

                    epoch_losses.append(loss)
                    chunk_losses.append(loss)
                    state.current_loss = loss
                    state.avg_loss = sum(chunk_losses[-100:]) / len(chunk_losses[-100:])
                    state.current_step = global_step

                    # Logging
                    if global_step % log_every == 0 or batch_idx == 0 or batch_idx == batches_in_chunk - 1:
                        mem_stats = get_memory_stats()
                        log(f"Step {global_step + 1}/{state.total_steps} | Loss: {loss:.6f} | LR: {lr:.6f} | {mem_stats}", "INFO")

                    global_step += 1

                # End of epoch on this chunk
                epoch_avg = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
                chunk_epoch_losses.append(epoch_avg)
                log(f"  Epoch {epoch + 1} done | Avg loss: {epoch_avg:.6f}", "INFO")

            # Step 5d: Chunk complete - cleanup and update state
            chunk_avg = sum(chunk_epoch_losses) / len(chunk_epoch_losses) if chunk_epoch_losses else 0
            log(f"CHUNK {chunk_idx + 1} COMPLETE | Avg loss: {chunk_avg:.6f}", "INFO")

            state.chunks_completed = chunk_idx + 1
            state.samples_seen += chunk_samples
            state.bytes_seen += chunk_bytes

            log(f"Progress: {state.chunks_completed} chunks, {state.samples_seen:,} samples, {state.bytes_seen / (1024**3):.2f} GB", "INFO")

            # Cleanup to free memory before next chunk
            del examples
            del dataset
            del dataloader
            gc.collect()

            if TOKENS_FILE.exists():
                try:
                    TOKENS_FILE.unlink()
                except Exception:
                    pass

            log(f"Memory after cleanup: {get_memory_stats()}", "INFO")

        # ================================================================
        # PHASE 6: Training Complete
        # ================================================================
        state.status = "Complete!"
        state.is_training = False

        log("=" * 60, "INFO")
        log("CHUNKED STREAMING SFT COMPLETE!", "INFO")
        log(f"Total chunks: {state.chunks_completed}", "INFO")
        log(f"Total samples: {state.samples_seen:,}", "INFO")
        log(f"Total data: {state.bytes_seen / (1024**3):.2f} GB", "INFO")
        log(f"Total steps: {global_step}", "INFO")
        log("=" * 60, "INFO")

        # Auto-save to cache
        log("Auto-saving to cache...", "INFO")
        result = save_to_cache()
        log(result, "INFO")

    except Exception as e:
        state.status = f"Error: {e}"
        state.is_training = False
        log(f"Training error: {e}\n{traceback.format_exc()}", "ERROR")


def start_training(*args):
    """Start training in a background thread - returns immediately"""
    global state

    # Don't start if already training
    if state.is_training:
        log("Training already in progress", "WARNING")
        return "Already training..."

    state.is_training = True
    state.is_paused = False
    state.chunks_completed = 0
    state.samples_seen = 0
    state.bytes_seen = 0

    # Launch training in background thread
    thread = threading.Thread(target=training_worker, args=args, daemon=True)
    thread.start()

    log("Training thread started", "INFO")
    return "Starting..."


def pause_training():
    global state
    if state.is_training:
        state.is_paused = True
        log("Training paused by user", "WARNING")
    return "Paused"


def resume_training():
    global state
    if state.is_paused:
        state.is_paused = False
        log("Training resumed by user", "INFO")
    return "Training..."


def get_training_status():
    """Get current training status for UI polling"""
    # Format epoch display with chunk info
    if state.chunks_completed > 0 or state.is_training:
        epoch_str = f"C{state.chunks_completed + 1}E{state.current_epoch + 1}"
    else:
        epoch_str = "-"

    # Format step display
    if state.total_steps > 0:
        step_str = f"{state.current_step + 1}/{state.total_steps}"
    elif state.current_step > 0:
        step_str = str(state.current_step + 1)
    else:
        step_str = "-"

    return (
        state.status,
        epoch_str,
        step_str,
        f"{state.current_loss:.6f}" if state.current_loss > 0 else "-",
        f"{state.avg_loss:.6f}" if state.avg_loss > 0 else "-",
        logger.read_latest(),
    )


def refresh_ui():
    """Refresh UI with current state including cache status"""
    status_tuple = get_training_status()
    return status_tuple + (check_cache_status(),)


# ============================================================================
# GRADIO UI
# ============================================================================

def create_ui():
    with gr.Blocks(title="GabGPT SFT Trainer") as app:
        with gr.Row():
            # ================================================================
            # LEFT COLUMN: Configuration
            # ================================================================
            with gr.Column(scale=1):
                # Checkpoint (Required)
                with gr.Accordion("Checkpoint (Required)", open=True):
                    checkpoint_file = gr.File(
                        label="Import .pt Checkpoint",
                        file_types=[".pt", ".pth"]
                    )
                    checkpoint_status = gr.Textbox(
                        label="Status",
                        value="No checkpoint loaded",
                        interactive=False
                    )
                    model_info = gr.Textbox(
                        label="Model Config",
                        value="",
                        interactive=False
                    )

                # Tokenizer
                with gr.Accordion("Tokenizer", open=True):
                    tokenizer_file = gr.File(
                        label="Import .bpe Tokenizer",
                        file_types=[".bpe"]
                    )
                    tokenizer_status = gr.Textbox(
                        label="Status",
                        value="No tokenizer loaded",
                        interactive=False
                    )
                    special_tokens_input = gr.Textbox(
                        label="Special Tokens (comma-separated)",
                        value="<|im_start|>, <|im_end|>, <|end|>, <|pad|>, <|endoftext|>, <|user|>, <|assistant|>, <|system|>, <|tool|>, <|tool_call|>, <|tool_result|>, <|call|>, <|result|>, <|think|>",
                        info="Tokens used for loss masking"
                    )

                # Dataset Configuration
                with gr.Accordion("Dataset Configuration", open=True):
                    dataset_name = gr.Textbox(
                        label="Dataset Name",
                        value="gszauer/TechTalk",
                        info="HuggingFace dataset name"
                    )
                    with gr.Row():
                        dataset_split = gr.Textbox(
                            label="Split",
                            value="train"
                        )
                        conv_column = gr.Textbox(
                            label="Conversations Column",
                            value="conversations"
                        )
                    with gr.Row():
                        max_samples = gr.Number(
                            label="Max Samples (0 = all)",
                            value=0,
                            precision=0
                        )
                        allow_think = gr.Checkbox(
                            label="Allow Thinking",
                            value=True,
                            info="Include <|think|> sections in training"
                        )

                # Training Configuration
                with gr.Accordion("Training Configuration", open=True):
                    with gr.Row():
                        num_epochs = gr.Number(
                            label="Epochs",
                            value=1,
                            precision=0
                        )
                        batch_size = gr.Number(
                            label="Batch Size",
                            value=4,
                            precision=0
                        )
                    with gr.Row():
                        max_seq_length = gr.Number(
                            label="Max Sequence Length",
                            value=512,
                            precision=0
                        )
                        learning_rate = gr.Number(
                            label="Learning Rate",
                            value=0.0001
                        )
                    with gr.Row():
                        warmup_ratio = gr.Number(
                            label="Warmup Ratio",
                            value=0.1
                        )
                        log_every = gr.Number(
                            label="Log Every N Steps",
                            value=10,
                            precision=0
                        )

            # ================================================================
            # RIGHT COLUMN: Training & Monitoring
            # ================================================================
            with gr.Column(scale=1):
                # Training Controls
                with gr.Accordion("Training Controls", open=True):
                    status_display = gr.Textbox(
                        label="Status",
                        value="Idle",
                        interactive=False
                    )

                    with gr.Row():
                        start_btn = gr.Button("Start Training", variant="primary")
                        pause_btn = gr.Button("Pause")
                        resume_btn = gr.Button("Resume")
                        refresh_btn = gr.Button("Refresh")

                    with gr.Row():
                        epoch_display = gr.Textbox(label="Epoch", value="-", interactive=False)
                        step_display = gr.Textbox(label="Step", value="-", interactive=False)
                        loss_display = gr.Textbox(label="Loss", value="-", interactive=False)
                        avg_loss_display = gr.Textbox(label="Avg Loss", value="-", interactive=False)

                # Training Log
                with gr.Accordion("Training Log", open=True):
                    log_display = gr.Textbox(
                        label="Logs",
                        value="",
                        lines=15,
                        max_lines=20,
                        interactive=False
                    )
                    with gr.Row():
                        clear_log_btn = gr.Button("Clear Log", size="sm")

                # Export
                with gr.Accordion("Export", open=True):
                    with gr.Row():
                        export_pt_btn = gr.Button("Export .pt Checkpoint")
                        export_pt_file = gr.File(label="Download .pt", interactive=False)

                # Cache
                with gr.Accordion("Cache (.gab)", open=True):
                    storage_info = "Persistent storage: " + ("AVAILABLE" if has_persistent_storage() else "NOT AVAILABLE")
                    gr.Markdown(f"*{storage_info}*")

                    with gr.Row():
                        save_cache_btn = gr.Button("Save to Cache")
                        clear_cache_btn = gr.Button("Clear Cache")
                        download_cache_btn = gr.Button("Download Cache")

                    cache_status = gr.Textbox(
                        label="Cache Status",
                        value=check_cache_status(),
                        interactive=False
                    )
                    cache_file = gr.File(label="Download .gab", interactive=False)

        # ================================================================
        # EVENT HANDLERS
        # ================================================================

        # Checkpoint import
        checkpoint_file.change(
            fn=import_checkpoint,
            inputs=[checkpoint_file],
            outputs=[checkpoint_status, model_info]
        )

        # Tokenizer import
        tokenizer_file.change(
            fn=import_tokenizer,
            inputs=[tokenizer_file],
            outputs=[tokenizer_status, gr.State()]
        )

        # Training controls
        training_inputs = [
            dataset_name, dataset_split, conv_column, max_samples, allow_think, special_tokens_input,
            num_epochs, batch_size, max_seq_length, learning_rate, warmup_ratio, log_every
        ]

        training_outputs = [
            status_display, epoch_display, step_display,
            loss_display, avg_loss_display, log_display
        ]

        # Start button launches training in background thread
        start_btn.click(
            fn=start_training,
            inputs=training_inputs,
            outputs=[status_display]
        )

        pause_btn.click(fn=pause_training, outputs=[status_display])
        resume_btn.click(fn=resume_training, outputs=[status_display])

        # Manual refresh button - user polls for updates
        refresh_btn.click(
            fn=get_training_status,
            outputs=training_outputs
        )

        def clear_log():
            logger.clear()
            log("Log cleared", "INFO")
            return logger.read_latest()

        clear_log_btn.click(fn=clear_log, outputs=[log_display])

        # Export handlers
        export_pt_btn.click(fn=export_checkpoint, outputs=[export_pt_file])

        # Cache handlers
        save_cache_btn.click(fn=save_to_cache, outputs=[cache_status])
        clear_cache_btn.click(fn=clear_cache, outputs=[cache_status])
        download_cache_btn.click(fn=download_cache, outputs=[cache_file])

    return app


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    log("=" * 60, "INFO")
    log("GabGPT SFT Trainer Starting", "INFO")
    log(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}", "INFO")
    log(f"Persistent storage: {'Available' if has_persistent_storage() else 'Not available'}", "INFO")
    log("=" * 60, "INFO")

    app = create_ui()
    app.queue()
    app.launch(allowed_paths=["/data"])
