import gradio as gr
import subprocess
import os
import shlex
import shutil

# HuggingFace Spaces persistent storage path
STORAGE_PATH = "/data"
SAVED_MODEL_PATH = os.path.join(STORAGE_PATH, "vocab.bpe")

# Fixed working directory for training (survives disconnects)
WORK_DIR = "/tmp/gab_tokenizer"
LOG_FILE = os.path.join(WORK_DIR, "log.txt")
OUTPUT_FILE = os.path.join(WORK_DIR, "input.bpe")

# Global training state
training_state = {
    "process": None,
    "log_handle": None,
}

# Common ChatML tokens
CHATML_TOKENS = '<|im_start|> <|im_end|> <|end|> <|pad|> <|endoftext|> <|user|> <|assistant|> <|system|> <|tool|> <|tool_call|> <|tool_result|> <|call|> <|result|> <|think|>'

TOKENIZER_C = r'''/*
 * tokenizer.c - Fast BPE Tokenizer for Gab
 *
 * Compile:   gcc -O3 -march=native -o tokenizer tokenizer.c
 * Usage:     ./tokenizer <input.txt> <num_merges> [reserved_token1] [reserved_token2] ...
 *
 * Example:   ./tokenizer training_data.txt 5000 "<|endoftext|>" "<|pad|>"
 *
 * Reserved tokens are "pre-trained" by merging their bytes before processing
 * the main dataset. This ensures they become atomic tokens with low IDs.
 *
 * Buffers are allocated once at startup and freed at exit. No heap allocations
 * occur during the BPE training loop. Tokens are stored as u16 (max 65,536 tokens).
 *
 * Memory usage scales with input file size (roughly 4x input + 128MB hash table).
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

/* ============================================================================
 * CONFIGURATION
 * ============================================================================ */

/* Hash table size for pair counting - must be power of 2 */
/* Larger = fewer collisions but more memory. 16M entries = 128MB */
#define HASH_SIZE (1 << 24)
#define HASH_MASK (HASH_SIZE - 1)

/* Maximum vocabulary size (u16 limit) */
#define MAX_VOCAB 65536

/* Maximum merges (vocab - 256 base bytes) */
#define MAX_MERGES (MAX_VOCAB - 256)

/* ============================================================================
 * DYNAMICALLY ALLOCATED BUFFERS - Allocated once at startup
 * ============================================================================ */

/* Two token buffers for ping-pong compaction (allocated based on file size) */
static uint16_t *g_buffer_a = NULL;
static uint16_t *g_buffer_b = NULL;
static size_t g_buffer_capacity = 0;

/* Hash table for pair counting */
/* Entry: key (high 32 bits) + count (low 32 bits), 0 = empty */
typedef struct {
    uint32_t pair;   /* (token1 << 16) | token2, 0xFFFFFFFF = empty */
    uint32_t count;
} PairEntry;

static PairEntry *g_hash = NULL;

/* Merge records */
typedef struct {
    uint16_t t1;
    uint16_t t2;
    uint16_t merged;
} Merge;

static Merge g_merges[MAX_MERGES];
static uint32_t g_merge_count = 0;

/* Vocabulary: for each token, store its byte sequence */
/* We use a flat buffer + offset/length arrays for cache efficiency */
static uint8_t  *g_vocab_data = NULL;            /* Byte sequences (allocated at startup) */
static size_t   g_vocab_data_capacity = 0;
static uint32_t g_vocab_offset[MAX_VOCAB];       /* Offset into g_vocab_data */
static uint16_t g_vocab_length[MAX_VOCAB];       /* Length of byte sequence */
static uint32_t g_vocab_data_used = 0;
static uint16_t g_next_token = 256;

/* ============================================================================
 * HASH TABLE OPERATIONS
 * ============================================================================ */

static inline uint32_t hash_pair(uint32_t pair) {
    /* FNV-1a inspired mixing */
    uint32_t h = pair;
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h & HASH_MASK;
}

static void hash_clear(void) {
    for (size_t i = 0; i < HASH_SIZE; i++) {
        g_hash[i].pair = 0xFFFFFFFF;
        g_hash[i].count = 0;
    }
}

static inline void hash_increment(uint32_t pair) {
    uint32_t idx = hash_pair(pair);

    /* Linear probing */
    while (1) {
        if (g_hash[idx].pair == pair) {
            g_hash[idx].count++;
            return;
        }
        if (g_hash[idx].pair == 0xFFFFFFFF) {
            g_hash[idx].pair = pair;
            g_hash[idx].count = 1;
            return;
        }
        idx = (idx + 1) & HASH_MASK;
    }
}

static void find_best_pair(uint32_t *out_pair, uint32_t *out_count) {
    uint32_t best_pair = 0xFFFFFFFF;
    uint32_t best_count = 0;

    for (size_t i = 0; i < HASH_SIZE; i++) {
        if (g_hash[i].pair != 0xFFFFFFFF && g_hash[i].count > best_count) {
            best_count = g_hash[i].count;
            best_pair = g_hash[i].pair;
        }
    }

    *out_pair = best_pair;
    *out_count = best_count;
}

/* ============================================================================
 * VOCABULARY OPERATIONS
 * ============================================================================ */

static void vocab_init(void) {
    /* Initialize base vocabulary (single bytes 0-255) */
    for (int i = 0; i < 256; i++) {
        g_vocab_offset[i] = g_vocab_data_used;
        g_vocab_length[i] = 1;
        g_vocab_data[g_vocab_data_used++] = (uint8_t)i;
    }
    g_next_token = 256;
}

static uint16_t vocab_add_merge(uint16_t t1, uint16_t t2) {
    if (g_next_token >= MAX_VOCAB) {
        fprintf(stderr, "Error: Vocabulary full (max %d tokens)\n", MAX_VOCAB);
        exit(1);
    }

    uint16_t new_token = g_next_token++;

    /* New token's bytes = t1's bytes + t2's bytes */
    uint32_t len1 = g_vocab_length[t1];
    uint32_t len2 = g_vocab_length[t2];
    uint32_t new_len = len1 + len2;

    g_vocab_offset[new_token] = g_vocab_data_used;
    g_vocab_length[new_token] = (uint16_t)new_len;

    /* Copy byte sequences */
    memcpy(&g_vocab_data[g_vocab_data_used],
           &g_vocab_data[g_vocab_offset[t1]], len1);
    memcpy(&g_vocab_data[g_vocab_data_used + len1],
           &g_vocab_data[g_vocab_offset[t2]], len2);

    g_vocab_data_used += new_len;

    return new_token;
}

static uint16_t find_existing_merge(uint16_t t1, uint16_t t2) {
    /* Check if (t1, t2) was already merged - returns merged token or 0xFFFF if not found */
    for (uint32_t i = 0; i < g_merge_count; i++) {
        if (g_merges[i].t1 == t1 && g_merges[i].t2 == t2) {
            return g_merges[i].merged;
        }
    }
    return 0xFFFF;
}

static uint16_t add_reserved_token(const char *str) {
    /*
     * Pre-train a reserved token by merging its bytes left-to-right.
     * This ensures the entire string becomes a single token.
     * Reuses existing merges if the same (t1, t2) pair was already merged.
     * Returns the final token ID for this reserved token.
     */
    size_t len = strlen(str);
    if (len == 0) return 0;
    if (len == 1) return (uint16_t)(unsigned char)str[0];

    /* Start with first byte */
    uint16_t current = (uint16_t)(unsigned char)str[0];

    /* Merge each subsequent byte */
    for (size_t i = 1; i < len; i++) {
        uint16_t next_byte = (uint16_t)(unsigned char)str[i];

        /* Check if this merge already exists */
        uint16_t existing = find_existing_merge(current, next_byte);
        uint16_t merged;

        if (existing != 0xFFFF) {
            /* Reuse existing merge */
            merged = existing;
        } else {
            /* Create new merge */
            merged = vocab_add_merge(current, next_byte);

            /* Record the merge */
            g_merges[g_merge_count].t1 = current;
            g_merges[g_merge_count].t2 = next_byte;
            g_merges[g_merge_count].merged = merged;
            g_merge_count++;
        }

        current = merged;
    }

    return current;
}

/* ============================================================================
 * BPE CORE OPERATIONS
 * ============================================================================ */

static void count_pairs(const uint16_t *tokens, size_t count) {
    hash_clear();

    if (count < 2) return;

    for (size_t i = 0; i < count - 1; i++) {
        uint32_t pair = ((uint32_t)tokens[i] << 16) | tokens[i + 1];
        hash_increment(pair);
    }
}

static size_t apply_merge(const uint16_t *src, size_t src_count,
                          uint16_t *dst,
                          uint16_t t1, uint16_t t2, uint16_t merged) {
    size_t dst_count = 0;
    size_t i = 0;

    while (i < src_count) {
        if (i < src_count - 1 && src[i] == t1 && src[i + 1] == t2) {
            dst[dst_count++] = merged;
            i += 2;
        } else {
            dst[dst_count++] = src[i];
            i++;
        }
    }

    return dst_count;
}

/* ============================================================================
 * MEMORY MANAGEMENT - Allocate once at startup, free at exit
 * ============================================================================ */

static size_t get_file_size(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror("Error opening input file");
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fclose(f);

    if (size <= 0) {
        fprintf(stderr, "Error: Empty or invalid file\n");
        exit(1);
    }
    return (size_t)size;
}

static void allocate_buffers(size_t file_size) {
    g_buffer_capacity = file_size;

    /* Token buffers: one u16 per byte of input */
    g_buffer_a = (uint16_t *)malloc(file_size * sizeof(uint16_t));
    g_buffer_b = (uint16_t *)malloc(file_size * sizeof(uint16_t));

    /* Hash table for pair counting */
    g_hash = (PairEntry *)malloc(HASH_SIZE * sizeof(PairEntry));

    /* Vocab data buffer: needs 256 bytes for base vocab + 2x input for merges */
    g_vocab_data_capacity = 256 + file_size * 2;
    g_vocab_data = (uint8_t *)malloc(g_vocab_data_capacity);

    if (!g_buffer_a || !g_buffer_b || !g_hash || !g_vocab_data) {
        fprintf(stderr, "Error: Failed to allocate memory for %zu byte input\n", file_size);
        fprintf(stderr, "Required: ~%.1f MB\n",
                (file_size * 4 + HASH_SIZE * sizeof(PairEntry) + g_vocab_data_capacity) / (1024.0 * 1024.0));
        exit(1);
    }
}

static void free_buffers(void) {
    free(g_buffer_a);
    free(g_buffer_b);
    free(g_hash);
    free(g_vocab_data);
    g_buffer_a = g_buffer_b = NULL;
    g_hash = NULL;
    g_vocab_data = NULL;
}

/* ============================================================================
 * FILE I/O
 * ============================================================================ */

static size_t load_file(const char *path, uint16_t *tokens, size_t capacity) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror("Error opening input file");
        exit(1);
    }

    /* Read file in chunks and convert bytes to u16 tokens */
    uint8_t read_buffer[1024 * 1024]; /* 1MB read buffer */
    size_t total_tokens = 0;

    while (total_tokens < capacity) {
        size_t max_read = capacity - total_tokens;
        if (max_read > sizeof(read_buffer)) max_read = sizeof(read_buffer);

        size_t bytes_read = fread(read_buffer, 1, max_read, f);
        if (bytes_read == 0) break;

        /* Convert bytes to u16 tokens */
        for (size_t i = 0; i < bytes_read; i++) {
            tokens[total_tokens++] = read_buffer[i];
        }
    }

    fclose(f);
    return total_tokens;
}

static void write_bpe(const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) {
        perror("Error creating output file");
        exit(1);
    }

    /* Write header */
    uint32_t magic = 0x42504531;  /* 'BPE1' */
    uint32_t version = 1;
    uint32_t next_token = g_next_token;
    uint32_t num_merges = g_merge_count;

    fwrite(&magic, 4, 1, f);
    fwrite(&version, 4, 1, f);
    fwrite(&next_token, 4, 1, f);
    fwrite(&num_merges, 4, 1, f);

    /* Write merges */
    for (uint32_t i = 0; i < g_merge_count; i++) {
        uint32_t t1 = g_merges[i].t1;
        uint32_t t2 = g_merges[i].t2;
        uint32_t merged = g_merges[i].merged;
        fwrite(&t1, 4, 1, f);
        fwrite(&t2, 4, 1, f);
        fwrite(&merged, 4, 1, f);
    }

    /* Write vocabulary */
    uint32_t vocab_size = g_next_token;
    fwrite(&vocab_size, 4, 1, f);

    for (uint32_t i = 0; i < vocab_size; i++) {
        uint32_t token_id = i;
        uint32_t length = g_vocab_length[i];
        fwrite(&token_id, 4, 1, f);
        fwrite(&length, 4, 1, f);
        fwrite(&g_vocab_data[g_vocab_offset[i]], 1, length, f);
    }

    fclose(f);
}

/* ============================================================================
 * PROGRESS DISPLAY
 * ============================================================================ */

static void print_progress(uint32_t merge_num, uint32_t total_merges,
                          uint32_t best_count, size_t token_count,
                          double elapsed) {
    double pct = (100.0 * merge_num) / total_merges;
    double merges_per_sec = merge_num / elapsed;
    double eta = (total_merges - merge_num) / merges_per_sec;

    printf("[%5.1f%%] Merge %5u/%u | Freq: %6u | Tokens: %10zu | "
           "%.1f merges/s | ETA: %.0fs\n",
           pct, merge_num, total_merges, best_count, token_count,
           merges_per_sec, eta);
    fflush(stdout);
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

static void derive_output_path(const char *input_path, char *output_path, size_t output_size) {
    /* Derive output path from input: "foo.txt" -> "foo.bpe" */
    strncpy(output_path, input_path, output_size - 1);
    output_path[output_size - 1] = '\0';

    char *dot = strrchr(output_path, '.');
    char *slash = strrchr(output_path, '/');

    /* Only replace extension if dot is after last slash (or no slash) */
    if (dot && (!slash || dot > slash)) {
        strcpy(dot, ".bpe");
    } else {
        /* No extension, just append .bpe */
        strncat(output_path, ".bpe", output_size - strlen(output_path) - 1);
    }
}

int main(int argc, char *argv[]) {
    /* Disable output buffering for real-time streaming */
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input.txt> <num_merges> [reserved_token1] [reserved_token2] ...\n", argv[0]);
        fprintf(stderr, "\nFast BPE tokenizer for Gab.\n");
        fprintf(stderr, "  input.txt      - Training text file (UTF-8)\n");
        fprintf(stderr, "  num_merges     - Number of BPE merges\n");
        fprintf(stderr, "  reserved_token - Optional reserved tokens (pre-trained first)\n");
        fprintf(stderr, "\nOutput file is derived from input: input.txt -> input.bpe\n");
        fprintf(stderr, "\nExample: %s data.txt 5000 \"<|endoftext|>\" \"<|pad|>\"\n", argv[0]);
        return 1;
    }

    const char *input_path = argv[1];
    char output_path[4096];
    derive_output_path(input_path, output_path, sizeof(output_path));

    uint32_t num_merges = (uint32_t)atoi(argv[2]);
    int num_reserved = argc - 3;

    if (num_merges > MAX_MERGES) {
        fprintf(stderr, "Error: num_merges too large (max %d)\n", MAX_MERGES);
        return 1;
    }

    printf("=== Gab BPE Tokenizer ===\n");
    printf("Input:  %s\n", input_path);
    printf("Output: %s\n", output_path);
    printf("Merges: %u\n", num_merges);
    if (num_reserved > 0) {
        printf("Reserved tokens: %d\n", num_reserved);
    }
    printf("\n");

    /* Get file size and allocate buffers first (before vocab operations) */
    size_t file_size = get_file_size(input_path);
    printf("Allocating buffers for %.2f MB input...\n", file_size / (1024.0 * 1024.0));
    allocate_buffers(file_size);

    /* Initialize vocabulary with base bytes */
    vocab_init();

    /* Pre-train reserved tokens */
    if (num_reserved > 0) {
        printf("Pre-training %d reserved token(s)...\n", num_reserved);

        /* Store token IDs for summary */
        uint16_t reserved_ids[256];
        int num_stored = (num_reserved < 256) ? num_reserved : 256;

        /* Track merges before/after each token to show new vs reused */
        uint32_t merges_before = g_merge_count;

        for (int i = 0; i < num_reserved; i++) {
            const char *token_str = argv[3 + i];
            size_t len = strlen(token_str);
            uint32_t merges_before_token = g_merge_count;

            uint16_t token_id = add_reserved_token(token_str);

            uint32_t new_merges = g_merge_count - merges_before_token;
            uint32_t total_merges = (len > 1) ? (uint32_t)(len - 1) : 0;
            uint32_t reused_merges = total_merges - new_merges;

            if (i < 256) {
                reserved_ids[i] = token_id;
            }

            printf("  [%d] \"%s\" -> token %u (%zu bytes, %u new + %u reused merges)\n",
                   i + 1, token_str, token_id, len, new_merges, reused_merges);
        }

        uint32_t total_new_merges = g_merge_count - merges_before;
        printf("\nReserved tokens created %u new merges (reused existing where possible)\n", total_new_merges);

        /* Print clean summary for easy reference */
        printf("\n=== Reserved Token IDs ===\n");
        for (int i = 0; i < num_stored; i++) {
            printf("%s - %u\n", argv[3 + i], reserved_ids[i]);
        }
        printf("==========================\n\n");
    }

    /* Load input file */
    printf("Loading input file...\n");
    clock_t start_load = clock();

    size_t token_count = load_file(input_path, g_buffer_a, g_buffer_capacity);

    double load_time = (double)(clock() - start_load) / CLOCKS_PER_SEC;
    printf("Loaded %zu bytes (%.2f MB) in %.2f seconds\n\n",
           token_count, token_count / (1024.0 * 1024.0), load_time);

    /* Ping-pong buffer pointers */
    uint16_t *current = g_buffer_a;
    uint16_t *other = g_buffer_b;
    size_t current_count = token_count;

    /* BPE training loop - subtract merges already used by reserved tokens */
    uint32_t remaining_merges = (g_merge_count < num_merges) ? (num_merges - g_merge_count) : 0;
    printf("Training BPE with %u data merges (+ %u reserved)...\n", remaining_merges, g_merge_count);
    clock_t start_train = clock();

    for (uint32_t m = 0; m < remaining_merges; m++) {
        /* Count all pairs */
        count_pairs(current, current_count);

        /* Find most frequent pair */
        uint32_t best_pair, best_count;
        find_best_pair(&best_pair, &best_count);

        /* Stop if no pairs with count > 1 */
        if (best_pair == 0xFFFFFFFF || best_count < 2) {
            printf("\nStopping early at merge %u (no frequent pairs)\n", m);
            break;
        }

        /* Extract tokens from pair */
        uint16_t t1 = (uint16_t)(best_pair >> 16);
        uint16_t t2 = (uint16_t)(best_pair & 0xFFFF);

        /* Create new merged token */
        uint16_t merged = vocab_add_merge(t1, t2);

        /* Record merge */
        g_merges[g_merge_count].t1 = t1;
        g_merges[g_merge_count].t2 = t2;
        g_merges[g_merge_count].merged = merged;
        g_merge_count++;

        /* Apply merge: copy from current to other, replacing pairs */
        size_t new_count = apply_merge(current, current_count, other, t1, t2, merged);

        /* Swap buffers */
        uint16_t *tmp = current;
        current = other;
        other = tmp;
        current_count = new_count;

        /* Progress every 10 merges or on first/last */
        if (m == 0 || (m + 1) % 10 == 0 || m == remaining_merges - 1) {
            double elapsed = (double)(clock() - start_train) / CLOCKS_PER_SEC;
            print_progress(m + 1, remaining_merges, best_count, current_count, elapsed);
        }
    }

    double train_time = (double)(clock() - start_train) / CLOCKS_PER_SEC;
    printf("\nTraining complete in %.2f seconds\n", train_time);
    printf("Final vocabulary size: %u tokens\n", g_next_token);
    printf("Final token count: %zu (%.2fx compression)\n",
           current_count, (double)token_count / current_count);

    /* Write output file */
    printf("\nWriting output file...\n");
    write_bpe(output_path);
    printf("Saved to %s\n", output_path);

    /* Cleanup */
    free_buffers();

    printf("\n=== Done ===\n");
    return 0;
}
'''


def has_storage():
    """Check if persistent storage is available"""
    return os.path.isdir(STORAGE_PATH) and os.access(STORAGE_PATH, os.W_OK)


def get_storage_status():
    """Get current storage status message"""
    if not has_storage():
        return "No storage attached"
    if os.path.exists(SAVED_MODEL_PATH):
        size = os.path.getsize(SAVED_MODEL_PATH)
        return f"Saved model: {size:,} bytes"
    return "Storage available (no model saved)"


def add_chatml_tokens(current_tokens):
    """Add ChatML tokens to the reserved tokens field"""
    if current_tokens and current_tokens.strip():
        return current_tokens.strip() + " " + CHATML_TOKENS
    return CHATML_TOKENS


def download_from_storage():
    """Download model from persistent storage"""
    if has_storage() and os.path.exists(SAVED_MODEL_PATH):
        return SAVED_MODEL_PATH, get_storage_status()
    return None, "No model in storage"


def clear_storage():
    """Clear model from persistent storage"""
    if has_storage() and os.path.exists(SAVED_MODEL_PATH):
        os.remove(SAVED_MODEL_PATH)
        return get_storage_status()
    return get_storage_status()


def train_tokenizer(training_file, num_merges, reserved_tokens):
    """Compile embedded code and start tokenizer training in background"""
    if training_file is None:
        return "Error: No training file uploaded", None, get_storage_status()

    # Kill any existing training process
    if training_state["process"] is not None:
        try:
            training_state["process"].kill()
        except Exception:
            pass
    if training_state["log_handle"] is not None:
        try:
            training_state["log_handle"].close()
        except Exception:
            pass
    training_state["process"] = None
    training_state["log_handle"] = None

    # Clean/create working directory
    if os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR, ignore_errors=True)
    os.makedirs(WORK_DIR, exist_ok=True)

    src = os.path.join(WORK_DIR, "tokenizer.c")
    binary = os.path.join(WORK_DIR, "tokenizer")
    input_path = os.path.join(WORK_DIR, "input.txt")

    # Write C source
    with open(src, "w") as f:
        f.write(TOKENIZER_C)

    # Compile
    compile_result = subprocess.run(
        ["gcc", "-O3", "-o", binary, src],
        capture_output=True,
        text=True
    )

    if compile_result.returncode != 0:
        return f"Compilation failed!\n\n{compile_result.stderr}", None, get_storage_status()

    # Copy input file
    with open(training_file, "rb") as f_in:
        with open(input_path, "wb") as f_out:
            f_out.write(f_in.read())

    # Build command
    cmd = [binary, input_path, str(int(num_merges))]
    if reserved_tokens and reserved_tokens.strip():
        tokens = shlex.split(reserved_tokens)
        cmd.extend(tokens)

    # Start training with stdout redirected to log file
    training_state["log_handle"] = open(LOG_FILE, 'w')
    training_state["process"] = subprocess.Popen(
        cmd,
        stdout=training_state["log_handle"],
        stderr=subprocess.STDOUT
    )

    return "Training started... Click 'Refresh' to see progress.", None, get_storage_status()


def refresh_ui():
    """Read current training output from log file and check for completion"""
    output = ""
    output_file = None

    # Read log file if it exists
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            output = f.read()

    # Check if training is complete (output file exists)
    if os.path.exists(OUTPUT_FILE):
        # Save to persistent storage if available
        if has_storage():
            shutil.copy(OUTPUT_FILE, SAVED_MODEL_PATH)
            output += f"\n\nSaved to persistent storage: {SAVED_MODEL_PATH}"

        output_file = OUTPUT_FILE
        output += f"\n\nTraining complete! Output file ready for download."

    elif training_state["process"] is not None:
        # Check process status
        ret = training_state["process"].poll()
        if ret is None:
            output += "\n\n[Training in progress... click Refresh for updates]"
        elif ret != 0:
            output += f"\n\n[Training failed with exit code {ret}]"

    if not output:
        output = "No training in progress. Upload a file and click Train."

    return output, output_file, get_storage_status()


with gr.Blocks(title="Gab BPE Tokenizer") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            training_file = gr.File(
                label="Training text file",
                file_types=[".txt", ".text", ".md", ".csv", ".json"]
            )

            num_merges = gr.Number(
                label="Number of merges",
                value=5000,
                minimum=1,
                maximum=65000,
                precision=0
            )
            reserved_tokens = gr.Textbox(
                label="Reserved tokens (space-separated, quote if needed)",
                placeholder='<|endoftext|> <|pad|> "<|im_start|>"',
                lines=2
            )
            with gr.Group():
                gr.Markdown("**Reserve common tokens**")
                chatml_btn = gr.Button("+ ChatML Tokens", size="sm")
            train_btn = gr.Button("Train Tokenizer", variant="primary")
            refresh_btn = gr.Button("Refresh UI", size="sm")

            with gr.Accordion("Storage", open=False):
                storage_status = gr.Textbox(
                    label="Storage Status",
                    value=get_storage_status(),
                    interactive=False
                )
                with gr.Row():
                    storage_download_btn = gr.Button("Download from Storage", size="sm")
                    storage_clear_btn = gr.Button("Clear Storage", size="sm", variant="stop")
                storage_file = gr.File(label="Downloaded from storage")

        with gr.Column(scale=1):
            output_text = gr.Textbox(label="Output", lines=20, max_lines=40)
            output_file = gr.File(label="Download trained vocabulary (.bpe)")

    # Event handlers
    chatml_btn.click(
        fn=add_chatml_tokens,
        inputs=[reserved_tokens],
        outputs=[reserved_tokens]
    )

    train_btn.click(
        fn=train_tokenizer,
        inputs=[training_file, num_merges, reserved_tokens],
        outputs=[output_text, output_file, storage_status]
    )

    storage_download_btn.click(
        fn=download_from_storage,
        inputs=[],
        outputs=[storage_file, storage_status]
    )

    storage_clear_btn.click(
        fn=clear_storage,
        inputs=[],
        outputs=[storage_status]
    )

    refresh_btn.click(
        fn=refresh_ui,
        inputs=[],
        outputs=[output_text, output_file, storage_status]
    )

demo.launch(allowed_paths=["/data"])
