"""
GabGPT Trainer - Gradio Frontend for gab.py
A HuggingFace Spaces compatible trainer for GPT-2 style transformers.
"""

import os
import math
import time
import threading
import traceback
from pathlib import Path
from typing import Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

import torch
import psutil
import numpy as np
import gradio as gr
from datasets import load_dataset

from gab import (
    GabGPT,
    create_dataloader, get_learning_rate,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Default paths
TMP_DIR = Path("/tmp/gabgpt")
LOG_FILE = TMP_DIR / "logs.txt"
CACHE_DIR = Path("/data") if os.path.exists("/data") else TMP_DIR / "cache"
CACHE_FILE = CACHE_DIR / "cache.gab"
MAX_LOG_SIZE = 1024 * 1024 * 1024  # 1024 MiB rolling buffer
ENCODER_DIR = TMP_DIR / "encoder"
ENCODER_BINARY = ENCODER_DIR / "encoder"
TOKENS_FILE = TMP_DIR / "tokens.bin"
DATASET_FILE = TMP_DIR / "dataset.txt"

# Streaming chunk size for large datasets
CHUNK_SIZE_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB per chunk

# Ensure directories exist
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
    import struct

    with open(path, 'rb') as f:
        magic = struct.unpack('<I', f.read(4))[0]
        if magic != 0x544F4B31:  # 'TOK1'
            return None

        num_tokens = struct.unpack('<Q', f.read(8))[0]

    # Memory-map the token data (skip 12-byte header)
    # Use int32 to match TextDataset expectations and avoid copy
    # Token IDs are always < 65536 (uint16 range) so this is safe
    return np.memmap(path, dtype=np.int32, mode='r', offset=12, shape=(num_tokens,))


def stream_dataset_to_file(dataset, text_column: str, output_path: str,
                           log_fn=None) -> Tuple[int, int]:
    """
    Stream dataset samples to a null-delimited file.
    Returns (num_samples, bytes_written).
    """
    num_samples = 0
    bytes_written = 0
    total = len(dataset)

    with open(output_path, 'wb') as f:
        for i, example in enumerate(dataset):
            text = example.get(text_column, "")
            if text:
                # Write UTF-8 text followed by null terminator
                text_bytes = text.encode('utf-8')
                f.write(text_bytes)
                f.write(b'\0')
                bytes_written += len(text_bytes) + 1
                num_samples += 1

            # Progress logging
            if log_fn and (i + 1) % 100000 == 0:
                pct = ((i + 1) / total) * 100
                log_fn(f"Writing dataset: {pct:.0f}% ({i + 1:,}/{total:,} samples, {bytes_written / (1024**3):.2f} GB)")

    return num_samples, bytes_written


def stream_dataset_chunks(dataset_name: str, split: str, text_column: str,
                          chunk_size_bytes: int = CHUNK_SIZE_BYTES,
                          skip_samples: int = 0,
                          start_chunk_num: int = 1,
                          log_fn=None):
    """
    Stream dataset in chunks of approximately chunk_size_bytes.
    Uses HuggingFace streaming mode - never loads full dataset.

    Args:
        skip_samples: Number of samples to skip using HuggingFace .skip() method.
                      This avoids loading skipped samples into memory.
        start_chunk_num: What chunk number to start counting from (for logging).

    Yields: (chunk_texts, chunk_samples, chunk_bytes, total_bytes_so_far)
    """
    if log_fn:
        log_fn(f"Connecting to HuggingFace dataset: {dataset_name}...")

    dataset = load_dataset(dataset_name, split=split, streaming=True)

    # Use HuggingFace's .skip() to efficiently skip samples without loading them
    if skip_samples > 0:
        if log_fn:
            log_fn(f"Skipping {skip_samples:,} samples using HuggingFace .skip()...")
            log_fn(f"Memory before skip: {get_detailed_memory_stats()}")
        dataset = dataset.skip(skip_samples)
        if log_fn:
            log_fn(f"Skip complete. Memory after skip: {get_detailed_memory_stats()}")

    if log_fn:
        log_fn(f"Dataset connected, starting to stream...")

    chunk_texts = []
    chunk_bytes = 0
    total_bytes = 0
    total_samples = 0
    chunk_num = start_chunk_num - 1  # Will be incremented before yield
    last_progress_mb = 0

    for sample in dataset:
        text = sample.get(text_column, "")
        if not text:
            continue

        text_bytes = len(text.encode('utf-8'))
        chunk_texts.append(text)
        chunk_bytes += text_bytes
        total_bytes += text_bytes
        total_samples += 1

        # Log progress every 250MB while accumulating chunk
        current_mb = chunk_bytes // (250 * 1024 * 1024)
        if current_mb > last_progress_mb:
            last_progress_mb = current_mb
            if log_fn:
                log_fn(f"Accumulating chunk: {chunk_bytes / (1024**3):.2f}/{chunk_size_bytes / (1024**3):.1f} GB ({total_samples + skip_samples:,} samples)")

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


def write_chunk_to_file(texts: list, output_path: str) -> Tuple[int, int]:
    """
    Write a list of texts to a null-delimited file.
    Returns (num_samples, bytes_written).
    """
    bytes_written = 0

    with open(output_path, 'wb') as f:
        for text in texts:
            text_bytes = text.encode('utf-8')
            f.write(text_bytes)
            f.write(b'\0')
            bytes_written += len(text_bytes) + 1

    return len(texts), bytes_written


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
        """Create log file if it doesn't exist"""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        if not self.filepath.exists():
            self.filepath.touch()

    def log(self, message: str, level: str = "INFO"):
        """Write a log message with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        formatted = f"[{timestamp}] [{level}] {message}\n"

        with self.lock:
            try:
                # Check file size and truncate if needed
                if self.filepath.exists():
                    size = self.filepath.stat().st_size
                    if size > self.max_size:
                        # Keep last half of the file
                        with open(self.filepath, 'r') as f:
                            content = f.read()
                        with open(self.filepath, 'w') as f:
                            f.write(content[len(content)//2:])

                # Append new log
                with open(self.filepath, 'a') as f:
                    f.write(formatted)
            except Exception as e:
                print(f"Logging error: {e}")

    def read_latest(self, lines: int = 500) -> str:
        """Read the latest N lines from the log"""
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
        """Clear the log file"""
        with self.lock:
            try:
                with open(self.filepath, 'w') as f:
                    f.write("")
            except Exception:
                pass


# Global logger
logger = RollingFileLogger(LOG_FILE)


def log(message: str, level: str = "INFO"):
    """Global logging function"""
    logger.log(message, level)
    print(f"[{level}] {message}")


def get_memory_stats() -> str:
    """Get current CPU and GPU memory usage."""
    # CPU RAM
    ram = psutil.virtual_memory()
    ram_used_gb = ram.used / (1024 ** 3)
    ram_total_gb = ram.total / (1024 ** 3)

    # GPU VRAM - use memory_reserved() to show actual GPU memory claimed by PyTorch
    if torch.cuda.is_available():
        gpu_used_gb = torch.cuda.memory_reserved() / (1024 ** 3)
        gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return f"RAM: {ram_used_gb:.1f}/{ram_total_gb:.1f}GB | GPU: {gpu_used_gb:.1f}/{gpu_total_gb:.1f}GB"
    else:
        return f"RAM: {ram_used_gb:.1f}/{ram_total_gb:.1f}GB | GPU: N/A"


def get_detailed_memory_stats() -> str:
    """Get detailed memory stats including process RSS and available memory."""
    import os
    ram = psutil.virtual_memory()
    process = psutil.Process(os.getpid())
    rss = process.memory_info().rss  # Resident Set Size - actual memory used by this process

    rss_gb = rss / (1024 ** 3)
    available_gb = ram.available / (1024 ** 3)
    used_gb = ram.used / (1024 ** 3)
    total_gb = ram.total / (1024 ** 3)

    # GPU stats
    gpu_str = ""
    if torch.cuda.is_available():
        gpu_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
        gpu_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        gpu_str = f" | GPU alloc: {gpu_alloc:.2f}GB, reserved: {gpu_reserved:.2f}GB"

    return f"Process RSS: {rss_gb:.2f}GB | RAM used: {used_gb:.1f}GB | available: {available_gb:.1f}GB / {total_gb:.1f}GB{gpu_str}"


def get_attention_backend() -> str:
    """Detect which attention backend is being used."""
    if not torch.cuda.is_available():
        return "CPU (no Flash Attention)"

    # Check Flash Attention availability
    if hasattr(torch.backends.cuda, 'flash_sdp_enabled'):
        flash_available = torch.backends.cuda.flash_sdp_enabled()
        mem_efficient = torch.backends.cuda.mem_efficient_sdp_enabled()
        math_sdp = torch.backends.cuda.math_sdp_enabled()

        backends = []
        if flash_available:
            backends.append("Flash Attention")
        if mem_efficient:
            backends.append("Memory-Efficient")
        if math_sdp:
            backends.append("Math (fallback)")

        if backends:
            return " > ".join(backends) + " (priority order)"
        return "Unknown"
    else:
        return "PyTorch <2.0 (no Flash Attention)"


# ============================================================================
# TRAINING STATE
# ============================================================================

@dataclass
class TrainingState:
    """Global training state"""
    model: Optional[GabGPT] = None
    tokenizer: bool = False  # Flag: tokenizer loaded
    tokenizer_path: Optional[str] = None  # Path to .bpe file for C encoder
    optimizer: Optional[torch.optim.Adam] = None

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

    # Config from loaded checkpoint
    loaded_config: Optional[dict] = None
    config_locked: bool = False

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


state = TrainingState()


# ============================================================================
# PARAMETER ESTIMATION
# ============================================================================

def estimate_parameters(vocab_size: int, embedding_dim: int, num_heads: int,
                       num_blocks: int, max_seq_length: int) -> int:
    """Estimate total parameter count for the model"""
    hidden_dim = embedding_dim * 4

    params = 0
    # Token embeddings
    params += vocab_size * embedding_dim
    # Position embeddings
    params += max_seq_length * embedding_dim

    # Per transformer block
    for _ in range(num_blocks):
        # LayerNorm1: gamma + beta
        params += 2 * embedding_dim
        # Attention: Q, K, V, O matrices
        params += 4 * embedding_dim * embedding_dim
        # LayerNorm2: gamma + beta
        params += 2 * embedding_dim
        # MLP: dense1 (embed -> hidden) + dense2 (hidden -> embed)
        params += hidden_dim * embedding_dim + hidden_dim  # dense1 weights + bias
        params += embedding_dim * hidden_dim + embedding_dim  # dense2 weights + bias

    # Final LayerNorm
    params += 2 * embedding_dim
    # Output layer
    params += embedding_dim * vocab_size + vocab_size

    return params


def format_params(count: int) -> str:
    """Format parameter count for display"""
    if count >= 1e9:
        return f"~{count/1e9:.1f}B"
    elif count >= 1e6:
        return f"~{count/1e6:.1f}M"
    elif count >= 1e3:
        return f"~{count/1e3:.0f}K"
    return f"~{count}"


def get_preset(name: str) -> dict:
    """Get preset configurations"""
    presets = {
        "~300K": {
            "embedding_dim": 64,
            "num_heads": 4,
            "num_blocks": 4,
            "max_seq_length": 128,
        },
        "~25M": {
            "embedding_dim": 512,
            "num_heads": 8,
            "num_blocks": 6,
            "max_seq_length": 2048,
        },
        "~1B": {
            "embedding_dim": 1536,
            "num_heads": 16,
            "num_blocks": 24,
            "max_seq_length": 1024,
        }
    }
    return presets.get(name, {})


# ============================================================================
# CACHE SYSTEM
# ============================================================================

def has_persistent_storage() -> bool:
    """Check if HuggingFace persistent storage is available"""
    return os.path.exists("/data") and os.access("/data", os.W_OK)


def save_to_cache() -> str:
    """Save current model to cache as .gab file"""
    if state.model is None:
        return "No model to cache"

    try:
        import gc
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        state.model.save_gab(str(CACHE_FILE))
        gc.collect()  # Free serialization buffers

        if has_persistent_storage():
            log(f"Model cached to persistent storage: {CACHE_FILE}", "INFO")
            return f"Saved to cache: {CACHE_FILE}"
        else:
            log(f"WARNING: No persistent storage! Cached to temp: {CACHE_FILE}", "WARNING")
            return f"WARNING: No persistent storage attached! Saved to temp: {CACHE_FILE}"
    except Exception as e:
        log(f"Failed to save cache: {e}", "ERROR")
        return f"Error: {e}"


def clear_cache() -> str:
    """Clear the cache file"""
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
    """Return cache file path for download"""
    if CACHE_FILE.exists():
        return str(CACHE_FILE)
    return None


# ============================================================================
# CHECKPOINT HANDLING
# ============================================================================

def export_checkpoint() -> Optional[str]:
    """Export current training state as .pt file"""
    if state.model is None or state.optimizer is None:
        log("No model/optimizer to export", "WARNING")
        return None

    try:
        filepath = TMP_DIR / f"checkpoint_{int(time.time())}.pt"
        state.model.save_checkpoint(
            str(filepath),
            state.optimizer,
            state.current_step,
            state.current_loss,
            epoch=state.current_epoch,
            total_steps=state.total_steps,
            # Chunk resumption data
            chunks_completed=state.chunks_completed,
            samples_seen=state.samples_seen,
        )
        log(f"Checkpoint exported: {filepath}", "INFO")
        log(f"  Chunks completed: {state.chunks_completed}, Samples seen: {state.samples_seen:,}", "INFO")
        return str(filepath)
    except Exception as e:
        log(f"Failed to export checkpoint: {e}", "ERROR")
        return None


def import_checkpoint(file) -> Tuple[str, dict]:
    """Import a .pt checkpoint file and return status + config"""
    if file is None:
        return "No file selected", {}

    try:
        filepath = file.name if hasattr(file, 'name') else file
        log(f"Importing checkpoint: {filepath}", "INFO")

        checkpoint = torch.load(filepath, map_location=state.device)
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

        # Reset training state - step/epoch don't carry over since we're likely
        # seeking to a different chunk. Optimizer momentum is what matters.
        state.current_step = 0
        state.current_epoch = 0
        state.total_steps = 0
        state.current_loss = checkpoint.get('loss', 0.0)  # Keep for reference

        # Restore chunk resumption data - this IS needed for proper skip
        state.chunks_completed = checkpoint.get('chunks_completed', 0)
        state.samples_seen = checkpoint.get('samples_seen', 0)

        # Lock config
        state.loaded_config = config
        state.config_locked = True

        log(f"Checkpoint loaded successfully (step reset, optimizer state preserved)", "INFO")
        log(f"  Vocab size: {config['vocab_size']}", "INFO")
        log(f"  Embedding dim: {config['embedding_dim']}", "INFO")
        log(f"  Num heads: {config['num_heads']}", "INFO")
        log(f"  Num blocks: {config['num_blocks']}", "INFO")
        log(f"  Max seq length: {config['max_seq_length']}", "INFO")
        log(f"  Chunks completed: {state.chunks_completed}", "INFO")
        log(f"  Samples seen: {state.samples_seen:,}", "INFO")

        return "Checkpoint loaded! Config locked.", config

    except Exception as e:
        log(f"Failed to import checkpoint: {e}\n{traceback.format_exc()}", "ERROR")
        return f"Error: {e}", {}


# ============================================================================
# TOKENIZER HANDLING
# ============================================================================

def import_tokenizer(file) -> Tuple[str, int]:
    """Import a .bpe tokenizer file - lightweight, just reads header"""
    if file is None:
        return "No file selected", 5256

    try:
        import struct
        filepath = file.name if hasattr(file, 'name') else file
        log(f"Importing tokenizer: {filepath}", "INFO")

        # Just read header for vocab size (C encoder handles the actual tokenization)
        with open(filepath, 'rb') as f:
            magic, version = struct.unpack('<II', f.read(8))
            if magic != 0x42504531:
                raise ValueError("Invalid .bpe file (bad magic)")
            vocab_size = struct.unpack('<I', f.read(4))[0]
            num_merges = struct.unpack('<I', f.read(4))[0]

        state.tokenizer = True  # Flag for validation
        state.tokenizer_path = filepath

        log(f"Tokenizer loaded! Vocab size: {vocab_size}, Merges: {num_merges}", "INFO")
        return f"Tokenizer loaded! Vocab: {vocab_size}", vocab_size

    except Exception as e:
        log(f"Failed to import tokenizer: {e}", "ERROR")
        return f"Error: {e}", 5256


# ============================================================================
# TRAINING LOOP
# ============================================================================

def training_worker(
    # Dataset config
    dataset_name: str,
    dataset_split: str,
    text_column: str,
    max_samples: int,
    # Model config
    vocab_size: int,
    embedding_dim: int,
    num_heads: int,
    num_blocks: int,
    max_seq_length: int,
    # Training config
    num_epochs: int,
    batch_size: int,
    seq_length: int,
    learning_rate: float,
    warmup_ratio: float,
    log_every: int,
    start_from_chunk: int = 1,
):
    """
    Training worker that runs in a background thread.
    Streams 2GB chunks, tokenizes each, trains, then discards.
    Updates state object directly - UI polls for updates.
    """
    global state
    import gc

    try:
        # ================================================================
        # PHASE 1: Setup (compiler, model, optimizer)
        # ================================================================
        state.status = "Initializing..."
        state.is_training = True
        log("=" * 60, "INFO")
        log("STARTING CHUNKED STREAMING TRAINING", "INFO")
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

        # Check tokenizer
        if not state.tokenizer or state.tokenizer_path is None:
            log("ERROR: No tokenizer loaded! Please import a tokenizer first.", "ERROR")
            state.status = "Error: No tokenizer"
            state.is_training = False
            return

        # Compile C encoder once
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
        # PHASE 2: Create Model
        # ================================================================
        state.status = "Creating model..."

        # Use loaded config if available, otherwise use UI values
        if state.config_locked and state.loaded_config:
            log("Using config from loaded checkpoint", "INFO")
            actual_vocab = state.loaded_config['vocab_size']
            actual_embed = state.loaded_config['embedding_dim']
            actual_heads = state.loaded_config['num_heads']
            actual_blocks = state.loaded_config['num_blocks']
            actual_maxseq = state.loaded_config['max_seq_length']
        else:
            actual_vocab = vocab_size
            actual_embed = embedding_dim
            actual_heads = num_heads
            actual_blocks = num_blocks
            actual_maxseq = max_seq_length

        if state.model is None:
            log(f"Creating GabGPT(vocab={actual_vocab}, embed={actual_embed}, heads={actual_heads}, blocks={actual_blocks}, maxseq={actual_maxseq})", "INFO")
            state.model = GabGPT(
                actual_vocab, actual_embed, actual_heads, actual_blocks, actual_maxseq
            ).to(state.device)
            param_count = estimate_parameters(actual_vocab, actual_embed, actual_heads, actual_blocks, actual_maxseq)
            log(f"Model created with {format_params(param_count)} parameters", "INFO")
        else:
            log("Using model from loaded checkpoint", "INFO")

        # ================================================================
        # PHASE 3: Create Optimizer
        # ================================================================
        state.status = "Preparing optimizer..."

        if state.optimizer is None:
            state.optimizer = torch.optim.Adam(state.model.parameters(), lr=learning_rate)
            log(f"Created Adam optimizer (lr={learning_rate})", "INFO")
        else:
            for param_group in state.optimizer.param_groups:
                param_group['lr'] = learning_rate
            log(f"Using optimizer from checkpoint, updated lr={learning_rate}", "INFO")

        log(f"Device: {state.device}", "INFO")
        log(f"Attention backend: {get_attention_backend()}", "INFO")
        log(f"Memory at start: {get_memory_stats()}", "INFO")
        log("=" * 60, "INFO")

        # ================================================================
        # PHASE 4: Chunked Streaming Training
        # ================================================================
        global_step = state.current_step
        chunk_losses = []

        # For warmup, estimate based on first chunk (will be updated)
        warmup_steps = 1000  # Initial estimate, gets refined

        log("Starting chunked streaming...", "INFO")
        start_from_chunk = int(start_from_chunk) if start_from_chunk else 1

        # Calculate samples to skip using HuggingFace .skip() method
        # This avoids loading skipped data into memory (fixes memory leak on resume)
        skip_samples = 0
        if start_from_chunk > 1:
            if state.samples_seen > 0:
                # We have checkpoint data - use it for efficient skip
                skip_samples = state.samples_seen
                log(f"Resuming from chunk {start_from_chunk} (skipping {skip_samples:,} samples from checkpoint)", "INFO")
            else:
                # No checkpoint data - warn user
                log(f"WARNING: No checkpoint loaded! Cannot efficiently skip to chunk {start_from_chunk}.", "WARNING")
                log(f"Please load a checkpoint first, or start from chunk 1.", "WARNING")
                log(f"Starting from chunk 1 instead.", "WARNING")
                start_from_chunk = 1

        log(f"Memory before streaming: {get_detailed_memory_stats()}", "INFO")
        state.status = "Streaming..."

        chunk_generator = stream_dataset_chunks(
            dataset_name, dataset_split, text_column,
            chunk_size_bytes=CHUNK_SIZE_BYTES,
            skip_samples=skip_samples,
            start_chunk_num=start_from_chunk,
            log_fn=lambda m: log(m, "INFO")
        )

        for chunk_idx, (chunk_texts, chunk_samples, chunk_bytes, total_bytes_so_far) in enumerate(chunk_generator):
            # Check if we should stop
            if not state.is_training:
                log("Training stopped by user", "WARNING")
                return

            # Chunk numbering now starts from start_from_chunk
            actual_chunk_num = start_from_chunk + chunk_idx

            log(f"{'='*60}", "INFO")
            log(f"CHUNK {actual_chunk_num} | {chunk_samples:,} samples | {chunk_bytes / (1024**3):.2f} GB", "INFO")
            log(f"Total streamed so far: {total_bytes_so_far / (1024**3):.2f} GB", "INFO")
            log(f"Memory at chunk start: {get_detailed_memory_stats()}", "INFO")

            # Step 4a: Write chunk to temp file
            state.status = f"Writing chunk {actual_chunk_num}..."

            num_written, bytes_written = write_chunk_to_file(chunk_texts, str(DATASET_FILE))
            log(f"Chunk written to temp file: {bytes_written / (1024**3):.2f} GB", "INFO")

            # Free chunk texts from memory
            del chunk_texts
            gc.collect()

            # Step 4b: Encode chunk
            state.status = f"Encoding chunk {actual_chunk_num}..."

            success, output = run_encoder(
                state.tokenizer_path,
                str(DATASET_FILE),
                str(TOKENS_FILE),
                log_fn=lambda m: log(m, "INFO")
            )

            if not success:
                log(f"ERROR: {output}", "ERROR")
                state.status = "Error: Encoding failed"
                state.is_training = False
                return

            # Step 4c: Load tokens
            all_tokens = load_tokens_file(str(TOKENS_FILE))
            if all_tokens is None:
                log("ERROR: Failed to load token file", "ERROR")
                state.status = "Error: Token load failed"
                state.is_training = False
                return

            log(f"Chunk tokenized: {len(all_tokens):,} tokens ({len(all_tokens) / chunk_samples:.1f} avg per sample)", "INFO")

            # Clean up temp dataset file
            if DATASET_FILE.exists():
                DATASET_FILE.unlink()

            if len(all_tokens) < seq_length + 1:
                log(f"WARNING: Chunk has too few tokens ({len(all_tokens)}), skipping", "WARNING")
                del all_tokens
                gc.collect()
                continue

            # Step 4d: Create dataloader for this chunk
            dataloader = create_dataloader(all_tokens, batch_size, seq_length, shuffle=True)
            batches_in_chunk = len(dataloader)
            state.total_steps = global_step + (batches_in_chunk * num_epochs)
            log(f"Dataloader: {batches_in_chunk} batches (batch={batch_size}, seq={seq_length})", "INFO")

            # Update warmup estimate based on actual data
            if chunk_idx == 0:
                # Rough estimate: assume similar chunk sizes
                estimated_total_steps = batches_in_chunk * num_epochs * 100  # Assume ~100 chunks for large dataset
                warmup_steps = int(estimated_total_steps * warmup_ratio)
                log(f"Warmup steps (estimated): {warmup_steps}", "INFO")

            # Step 4e: Train on this chunk for num_epochs
            state.status = "Training..."
            chunk_epoch_losses = []

            for epoch in range(num_epochs):
                state.current_epoch = epoch
                epoch_losses = []
                log(f"  Epoch {epoch + 1}/{num_epochs} on chunk {actual_chunk_num}", "INFO")

                for batch_idx, (input_tokens, target_tokens) in enumerate(dataloader):
                    # Check pause - just sleep and check flags
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

                    # Move to device
                    input_tokens = input_tokens.to(state.device)
                    target_tokens = target_tokens.to(state.device)

                    # Get learning rate with schedule
                    lr = get_learning_rate(global_step, warmup_steps, warmup_steps * 10, learning_rate)
                    for param_group in state.optimizer.param_groups:
                        param_group['lr'] = lr

                    # Training step
                    loss = state.model.train_step(input_tokens, target_tokens, state.optimizer)

                    epoch_losses.append(loss)
                    chunk_losses.append(loss)
                    state.current_loss = loss
                    state.avg_loss = sum(chunk_losses[-100:]) / len(chunk_losses[-100:])  # Rolling avg
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

            # Step 4f: Chunk complete - cleanup and update state
            chunk_avg = sum(chunk_epoch_losses) / len(chunk_epoch_losses) if chunk_epoch_losses else 0
            log(f"CHUNK {actual_chunk_num} COMPLETE | Avg loss: {chunk_avg:.6f}", "INFO")

            state.chunks_completed = actual_chunk_num
            state.samples_seen += chunk_samples
            state.bytes_seen += chunk_bytes

            log(f"Progress: {state.chunks_completed} chunks, {state.samples_seen:,} samples, {state.bytes_seen / (1024**3):.2f} GB", "INFO")

            # Auto-save to cache after each chunk
            log("Auto-saving to cache after chunk...", "INFO")
            log(f"BEFORE save_to_cache: {get_detailed_memory_stats()}", "INFO")
            result = save_to_cache()
            log(result, "INFO")
            log(f"AFTER save_to_cache: {get_detailed_memory_stats()}", "INFO")

            # === CRITICAL MEMORY CLEANUP ===
            # Must delete in correct order: dataloader holds reference to dataset which holds memmap

            # 1. Delete dataloader first (releases reference to TextDataset)
            del dataloader

            # 1.5 Explicitly close memmap backing file before dropping reference
            if isinstance(all_tokens, np.memmap) and getattr(all_tokens, "_mmap", None):
                log("Closing token memmap before deletion", "INFO")
                all_tokens._mmap.close()
                all_tokens._mmap = None

            # 2. Delete the memmap reference
            del all_tokens

            # 3. Force garbage collection to release memory
            gc.collect()

            # 4. Clear CUDA memory cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 5. Delete the token file (now safe since memmap is garbage collected)
            if TOKENS_FILE.exists():
                try:
                    TOKENS_FILE.unlink()
                except Exception as e:
                    log(f"Warning: Could not delete token file: {e}", "WARNING")

            # 6. Clear accumulated losses to prevent unbounded growth
            chunk_losses.clear()

            # Force another gc pass after all cleanup
            gc.collect()

            log(f"Memory after cleanup: {get_memory_stats()}", "INFO")
            log(f"AFTER full cleanup: {get_detailed_memory_stats()}", "INFO")

        # ================================================================
        # PHASE 5: Training Complete
        # ================================================================
        state.status = "Complete!"
        state.is_training = False

        log("=" * 60, "INFO")
        log("CHUNKED STREAMING TRAINING COMPLETE!", "INFO")
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

    # Launch training in background thread
    thread = threading.Thread(target=training_worker, args=args, daemon=True)
    thread.start()

    log("Training thread started", "INFO")
    return "Starting..."


def stop_training():
    """Stop training"""
    global state
    if state.is_training:
        state.is_training = False
        log("Training stop requested by user", "WARNING")
        return "Stopping..."
    return "Not training"


def pause_training():
    """Pause training"""
    global state
    if state.is_training:
        state.is_paused = True
        log("Training paused by user", "WARNING")
    return "Paused"


def resume_training():
    """Resume training"""
    global state
    if state.is_paused:
        state.is_paused = False
        log("Training resumed by user", "INFO")
    return "Training..."


def check_cache_status() -> str:
    """Check if cache file exists and return status"""
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


def get_training_status():
    """Get current training status for UI polling"""
    # Format epoch display
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
    """Create the Gradio interface"""

    with gr.Blocks(title="GabGPT Trainer") as app:
        with gr.Row():
            # ================================================================
            # LEFT COLUMN: Configuration
            # ================================================================
            with gr.Column(scale=1):
                # Dataset Configuration
                with gr.Accordion("Dataset Configuration", open=True):
                    dataset_name = gr.Textbox(
                        label="Dataset Name",
                        value="roneneldan/TinyStories",
                        info="HuggingFace dataset name"
                    )
                    with gr.Row():
                        dataset_split = gr.Textbox(
                            label="Split",
                            value="train"
                        )
                        text_column = gr.Textbox(
                            label="Text Column",
                            value="text"
                        )
                    max_samples = gr.Number(
                        label="Max Samples (0 = all)",
                        value=0,
                        precision=0
                    )

                # Tokenizer
                with gr.Accordion("Tokenizer", open=True):
                    tokenizer_file = gr.File(
                        label="Import Tokenizer (.bpe)",
                        file_types=[".bpe"]
                    )
                    tokenizer_status = gr.Textbox(
                        label="Status",
                        value="No tokenizer loaded",
                        interactive=False
                    )
                    vocab_size = gr.Number(
                        label="Vocab Size",
                        value=5256,
                        precision=0,
                        info="From tokenizer or manual override"
                    )

                # Model Architecture
                with gr.Accordion("Model Architecture", open=True):
                    with gr.Row():
                        embedding_dim = gr.Number(
                            label="Embedding Dim",
                            value=512,
                            precision=0
                        )
                        num_heads = gr.Number(
                            label="Attention Heads",
                            value=8,
                            precision=0
                        )
                    with gr.Row():
                        num_blocks = gr.Number(
                            label="Transformer Blocks",
                            value=6,
                            precision=0
                        )
                        max_seq_length = gr.Number(
                            label="Max Sequence Length",
                            value=2048,
                            precision=0
                        )

                    param_estimate = gr.Textbox(
                        label="Estimated Parameters",
                        value="~25M",
                        interactive=False
                    )

                    with gr.Row():
                        preset_300k = gr.Button("~300K", size="sm")
                        preset_25m = gr.Button("~25M", size="sm")
                        preset_1b = gr.Button("~1B", size="sm")

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
                        seq_length = gr.Number(
                            label="Sequence Length",
                            value=2048,
                            precision=0
                        )
                        learning_rate = gr.Number(
                            label="Learning Rate",
                            value=0.0003
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
                    start_from_chunk = gr.Number(
                        label="Start From Chunk",
                        value=1,
                        precision=0,
                        info="Skip to this chunk (1 = start from beginning)"
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
                        refresh_btn = gr.Button("🔄 Refresh")

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
                        lines=20,
                        max_lines=30,
                        interactive=False
                    )
                    clear_log_btn = gr.Button("Clear Log", size="sm")

                # Checkpoints
                with gr.Accordion("Checkpoints (.pt)", open=True):
                    gr.Markdown("*Full training state for resuming*")
                    with gr.Row():
                        export_pt_btn = gr.Button("Export .pt")
                        export_pt_file = gr.File(label="Download", interactive=False)

                    import_pt_file = gr.File(
                        label="Import .pt Checkpoint",
                        file_types=[".pt", ".pth"]
                    )
                    import_pt_status = gr.Textbox(
                        label="Import Status",
                        value="",
                        interactive=False
                    )

                # Cache
                with gr.Accordion("Cache (.gab)", open=True):
                    storage_status = "Persistent storage: " + ("AVAILABLE" if has_persistent_storage() else "NOT AVAILABLE (using temp)")
                    gr.Markdown(f"*{storage_status}*")

                    with gr.Row():
                        save_cache_btn = gr.Button("Save to Cache")
                        clear_cache_btn = gr.Button("Clear Cache")
                        download_cache_btn = gr.Button("Download Cache")

                    cache_status = gr.Textbox(
                        label="Cache Status",
                        value="",
                        interactive=False
                    )
                    cache_file = gr.File(label="Download", interactive=False)

        # ================================================================
        # EVENT HANDLERS
        # ================================================================

        # Tokenizer import
        def handle_tokenizer_import(file):
            status, vocab = import_tokenizer(file)
            return status, vocab

        tokenizer_file.change(
            fn=handle_tokenizer_import,
            inputs=[tokenizer_file],
            outputs=[tokenizer_status, vocab_size]
        )

        # Parameter estimation
        def update_param_estimate(vocab, embed, heads, blocks, maxseq):
            try:
                vocab = int(vocab) if vocab else 5256
                embed = int(embed) if embed else 512
                heads = int(heads) if heads else 8
                blocks = int(blocks) if blocks else 6
                maxseq = int(maxseq) if maxseq else 512

                if embed % heads != 0:
                    return "Error: Embedding dim must be divisible by heads!"

                params = estimate_parameters(vocab, embed, heads, blocks, maxseq)
                return format_params(params)
            except:
                return "~?"

        for input_component in [vocab_size, embedding_dim, num_heads, num_blocks, max_seq_length]:
            input_component.change(
                fn=update_param_estimate,
                inputs=[vocab_size, embedding_dim, num_heads, num_blocks, max_seq_length],
                outputs=[param_estimate]
            )

        # Presets
        def apply_preset_300k():
            p = get_preset("~300K")
            return p["embedding_dim"], p["num_heads"], p["num_blocks"], p["max_seq_length"]

        def apply_preset_25m():
            p = get_preset("~25M")
            return p["embedding_dim"], p["num_heads"], p["num_blocks"], p["max_seq_length"]

        def apply_preset_1b():
            p = get_preset("~1B")
            return p["embedding_dim"], p["num_heads"], p["num_blocks"], p["max_seq_length"]

        preset_300k.click(
            fn=apply_preset_300k,
            outputs=[embedding_dim, num_heads, num_blocks, max_seq_length]
        )
        preset_25m.click(
            fn=apply_preset_25m,
            outputs=[embedding_dim, num_heads, num_blocks, max_seq_length]
        )
        preset_1b.click(
            fn=apply_preset_1b,
            outputs=[embedding_dim, num_heads, num_blocks, max_seq_length]
        )

        # Training controls
        training_inputs = [
            dataset_name, dataset_split, text_column, max_samples,
            vocab_size, embedding_dim, num_heads, num_blocks, max_seq_length,
            num_epochs, batch_size, seq_length, learning_rate, warmup_ratio, log_every,
            start_from_chunk
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

        pause_btn.click(
            fn=pause_training,
            outputs=[status_display]
        )

        resume_btn.click(
            fn=resume_training,
            outputs=[status_display]
        )

        # Manual refresh button
        refresh_btn.click(
            fn=get_training_status,
            outputs=training_outputs
        )

        def clear_log():
            logger.clear()
            log("Log cleared", "INFO")
            return logger.read_latest()

        clear_log_btn.click(
            fn=clear_log,
            outputs=[log_display]
        )

        # Checkpoint handlers
        export_pt_btn.click(
            fn=export_checkpoint,
            outputs=[export_pt_file]
        )

        def handle_checkpoint_import(file):
            if file is None:
                return "", embedding_dim.value, num_heads.value, num_blocks.value, max_seq_length.value, vocab_size.value

            status, config = import_checkpoint(file)

            if config:
                return (
                    status,
                    config.get('embedding_dim', embedding_dim.value),
                    config.get('num_heads', num_heads.value),
                    config.get('num_blocks', num_blocks.value),
                    config.get('max_seq_length', max_seq_length.value),
                    config.get('vocab_size', vocab_size.value)
                )
            return status, embedding_dim.value, num_heads.value, num_blocks.value, max_seq_length.value, vocab_size.value

        import_pt_file.change(
            fn=handle_checkpoint_import,
            inputs=[import_pt_file],
            outputs=[import_pt_status, embedding_dim, num_heads, num_blocks, max_seq_length, vocab_size]
        )

        # Cache handlers
        save_cache_btn.click(
            fn=save_to_cache,
            outputs=[cache_status]
        )

        clear_cache_btn.click(
            fn=clear_cache,
            outputs=[cache_status]
        )

        download_cache_btn.click(
            fn=download_cache,
            outputs=[cache_file]
        )

    return app


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    log("=" * 60, "INFO")
    log("GabGPT Trainer Starting", "INFO")
    log(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}", "INFO")
    log(f"Persistent storage: {'Available' if has_persistent_storage() else 'Not available'}", "INFO")
    log("=" * 60, "INFO")

    app = create_ui()
    app.queue()
    app.launch(allowed_paths=["/data"])
