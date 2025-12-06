"""
GabGPT - Minimal LLM Training with Gradio UI
"""

import numpy as np
import json
import struct
import threading
import time
import io
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any, Callable
from collections import OrderedDict
import gradio as gr

# Optional HuggingFace datasets
try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False


# =============================================================================
# TOKENIZER (BPE)
# =============================================================================

class Tokenizer:
    """Byte-Pair Encoding tokenizer with serialization support."""

    __slots__ = ('merges', 'vocabulary', 'next_token_id', '_merge_cache')

    def __init__(self):
        self.merges: Dict[Tuple[int, int], int] = OrderedDict()
        self.vocabulary: Dict[int, bytes] = {}
        self.next_token_id = 256
        self._merge_cache: Dict[Tuple[int, int], int] = {}

        # Initialize with byte tokens
        for i in range(256):
            self.vocabulary[i] = bytes([i])

    def _make_merge(self, token1: int, token2: int) -> int:
        """Create or retrieve a merge for a token pair."""
        key = (token1, token2)
        if key in self.merges:
            return self.merges[key]

        new_token_id = self.next_token_id
        self.next_token_id += 1
        self.merges[key] = new_token_id

        # Combine bytes
        self.vocabulary[new_token_id] = self.vocabulary[token1] + self.vocabulary[token2]
        return new_token_id

    def _apply_merge(self, tokens: np.ndarray, token1: int, token2: int, merged_id: int) -> np.ndarray:
        """Apply a single merge operation efficiently using NumPy."""
        if len(tokens) < 2:
            return tokens

        # Find all positions where the pair occurs
        pair_mask = (tokens[:-1] == token1) & (tokens[1:] == token2)
        if not np.any(pair_mask):
            return tokens

        # Handle overlapping pairs (e.g., "aaa" with pair "aa" should only merge first two)
        pair_indices = np.where(pair_mask)[0]
        if len(pair_indices) > 1:
            # Remove overlapping: if i and i+1 are both marked, keep only i
            valid = np.empty(len(pair_indices), dtype=bool)
            valid[0] = True
            valid[1:] = np.diff(pair_indices) > 1
            pair_indices = pair_indices[valid]

        # Build keep mask (inverse of delete mask for efficiency)
        keep_mask = np.ones(len(tokens), dtype=bool)
        keep_mask[pair_indices + 1] = False

        # Build result: replace merged positions and filter
        result = np.empty(np.count_nonzero(keep_mask), dtype=np.int32)
        tokens_copy = tokens.copy()
        tokens_copy[pair_indices] = merged_id
        result[:] = tokens_copy[keep_mask]

        return result

    def _find_most_frequent_pair(self, tokens: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find the most frequent adjacent pair in tokens using NumPy."""
        if len(tokens) < 2:
            return None

        # Combine adjacent tokens into single 64-bit keys: (t1 << 32) | t2
        # This avoids creating tuples and allows numpy to count efficiently
        left = tokens[:-1].astype(np.int64)
        right = tokens[1:].astype(np.int64)
        pair_keys = (left << 32) | right

        # Count unique pairs
        unique_keys, counts = np.unique(pair_keys, return_counts=True)

        if len(unique_keys) == 0:
            return None

        # Find max
        max_idx = np.argmax(counts)
        if counts[max_idx] <= 1:
            return None

        # Decode the pair back from the 64-bit key
        best_key = unique_keys[max_idx]
        t1 = int(best_key >> 32)
        t2 = int(best_key & 0xFFFFFFFF)
        return (t1, t2)

    def train(self, text: str, num_merges: int, callback: Optional[Callable[[int, int], None]] = None,
              log_callback: Optional[Callable[[str], None]] = None) -> None:
        """Train the tokenizer on text with the given number of merges."""
        tokens = np.frombuffer(text.encode('utf-8'), dtype=np.uint8).astype(np.int32)

        for merge_num in range(num_merges):
            pair = self._find_most_frequent_pair(tokens)
            if pair is None:
                if log_callback:
                    log_callback(f"Tokenizer training stopped early at merge {merge_num} (no more pairs)")
                break

            new_token_id = self._make_merge(pair[0], pair[1])
            tokens = self._apply_merge(tokens, pair[0], pair[1], new_token_id)

            if callback and merge_num % 50 == 0:
                callback(merge_num, num_merges)

            # Heartbeat log every 100 merges
            if log_callback and (merge_num + 1) % 100 == 0:
                log_callback(f"  Tokenizer progress: {merge_num + 1}/{num_merges} merges, vocab size: {self.vocab_size}")

    def reserve_token(self, token_string: str) -> int:
        """Reserve a special token, returning its ID."""
        token_bytes = token_string.encode('utf-8')

        if len(token_bytes) < 2:
            return token_bytes[0] if token_bytes else 0

        current_token = token_bytes[0]
        for i in range(1, len(token_bytes)):
            current_token = self._make_merge(current_token, token_bytes[i])

        return current_token

    def encode(self, text: str, log_callback: Optional[Callable[[str], None]] = None) -> np.ndarray:
        """Encode text to token IDs.

        For large texts, processes in chunks to reduce peak memory usage.
        """
        text_bytes = text.encode('utf-8')
        text_len = len(text_bytes)

        # For small texts, use simple approach
        if text_len < 1_000_000:  # < 1MB
            return self._encode_simple(text_bytes, log_callback)

        # For large texts, process in chunks
        if log_callback:
            log_callback(f"  Large text detected ({text_len:,} bytes), using chunked encoding")

        chunk_size = 500_000  # 500KB chunks
        all_tokens = []

        for chunk_start in range(0, text_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, text_len)
            chunk_bytes = text_bytes[chunk_start:chunk_end]

            chunk_tokens = self._encode_simple(chunk_bytes, None)
            all_tokens.append(chunk_tokens)

            if log_callback and (chunk_start // chunk_size) % 10 == 0:
                progress = chunk_end / text_len * 100
                log_callback(f"  Encoding progress: {progress:.1f}% ({chunk_end:,}/{text_len:,} bytes)")

        if log_callback:
            log_callback(f"  Concatenating {len(all_tokens)} chunks...")

        result = np.concatenate(all_tokens)

        if log_callback:
            log_callback(f"  Encoding complete: {len(result):,} tokens")

        return result

    def _encode_simple(self, text_bytes: bytes, log_callback: Optional[Callable[[str], None]] = None) -> np.ndarray:
        """Encode bytes to token IDs (simple non-chunked version)."""
        tokens = np.frombuffer(text_bytes, dtype=np.uint8).astype(np.int32)

        if log_callback:
            log_callback(f"  Encoding: initial token count = {len(tokens):,}")

        num_merges = len(self.merges)
        for i, ((token1, token2), merged_token) in enumerate(self.merges.items()):
            tokens = self._apply_merge(tokens, token1, token2, merged_token)

            # Log progress: first 10 merges, then every 100th
            if log_callback:
                if i < 10 or (i + 1) % 100 == 0:
                    log_callback(f"  Encoding merge {i + 1}/{num_merges}: tokens = {len(tokens):,}")

        if log_callback:
            log_callback(f"  Encoding complete: final token count = {len(tokens):,}")

        return tokens

    def decode(self, tokens: np.ndarray) -> str:
        """Decode token IDs back to text."""
        byte_list = []
        for token_id in tokens:
            if token_id in self.vocabulary:
                byte_list.append(self.vocabulary[token_id])

        return b''.join(byte_list).decode('utf-8', errors='replace')

    @property
    def vocab_size(self) -> int:
        return len(self.vocabulary)

    def serialize(self) -> bytes:
        """Serialize tokenizer to bytes."""
        buffer = io.BytesIO()

        # Magic + version
        buffer.write(struct.pack('<II', 0x42504531, 1))
        buffer.write(struct.pack('<I', self.next_token_id))

        # Merges
        buffer.write(struct.pack('<I', len(self.merges)))
        for (t1, t2), merged in self.merges.items():
            buffer.write(struct.pack('<III', t1, t2, merged))

        # Vocabulary
        buffer.write(struct.pack('<I', len(self.vocabulary)))
        for token_id, token_bytes in self.vocabulary.items():
            buffer.write(struct.pack('<II', token_id, len(token_bytes)))
            buffer.write(token_bytes)

        return buffer.getvalue()

    @classmethod
    def deserialize(cls, data: bytes) -> 'Tokenizer':
        """Deserialize tokenizer from bytes."""
        buffer = io.BytesIO(data)

        magic, version = struct.unpack('<II', buffer.read(8))
        if magic != 0x42504531:
            raise ValueError("Invalid tokenizer file format")
        if version != 1:
            raise ValueError(f"Unsupported tokenizer version: {version}")

        tokenizer = cls()
        tokenizer.next_token_id = struct.unpack('<I', buffer.read(4))[0]

        # Load merges
        num_merges = struct.unpack('<I', buffer.read(4))[0]
        tokenizer.merges.clear()
        for _ in range(num_merges):
            t1, t2, merged = struct.unpack('<III', buffer.read(12))
            tokenizer.merges[(t1, t2)] = merged

        # Load vocabulary
        num_vocab = struct.unpack('<I', buffer.read(4))[0]
        tokenizer.vocabulary.clear()
        for _ in range(num_vocab):
            token_id, length = struct.unpack('<II', buffer.read(8))
            token_bytes = buffer.read(length)
            tokenizer.vocabulary[token_id] = token_bytes

        return tokenizer


# =============================================================================
# NEURAL NETWORK LAYERS (NumPy)
# =============================================================================

class DenseLayer:
    """Dense (fully connected) layer with efficient NumPy operations."""

    __slots__ = ('weights', 'bias', '_cached_input', '_id')

    _instance_counter = 0

    def __init__(self, input_dim: int, output_dim: int):
        scale = np.sqrt(2.0 / input_dim)
        self.weights = np.random.randn(input_dim, output_dim).astype(np.float32) * scale
        self.bias = np.zeros(output_dim, dtype=np.float32)
        self._cached_input: Optional[np.ndarray] = None
        self._id = f"dense_{DenseLayer._instance_counter}"
        DenseLayer._instance_counter += 1

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: y = x @ W + b"""
        self._cached_input = x
        return x @ self.weights + self.bias

    def backward(self, grad_output: np.ndarray, lr: float, optimizer: Optional['AdamOptimizer'] = None) -> np.ndarray:
        """Backward pass with weight update."""
        # Gradient w.r.t. input
        grad_input = grad_output @ self.weights.T

        # Gradient w.r.t. weights and bias
        # Reshape for batch matmul: (batch, seq, in) and (batch, seq, out)
        if self._cached_input.ndim == 3:
            # (batch, seq, in).T @ (batch, seq, out) -> sum over batch,seq
            grad_weights = np.einsum('bsi,bso->io', self._cached_input, grad_output)
            grad_bias = grad_output.sum(axis=(0, 1))
        else:
            grad_weights = self._cached_input.T @ grad_output
            grad_bias = grad_output.sum(axis=0)

        # Update
        if optimizer is not None:
            optimizer.update(f"{self._id}_w", self.weights, grad_weights, lr)
            optimizer.update(f"{self._id}_b", self.bias, grad_bias, lr)
        else:
            self.weights -= lr * grad_weights
            self.bias -= lr * grad_bias

        return grad_input


class Activation:
    """Activation functions with derivatives."""

    __slots__ = ('kind', '_cached_input')

    def __init__(self, kind: str = 'gelu'):
        self.kind = kind
        self._cached_input: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._cached_input = x

        if self.kind == 'relu':
            return np.maximum(0, x)
        elif self.kind == 'gelu':
            return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
        elif self.kind == 'tanh':
            return np.tanh(x)
        elif self.kind == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        return x

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        x = self._cached_input

        if self.kind == 'relu':
            return grad_output * (x > 0).astype(np.float32)
        elif self.kind == 'gelu':
            c = np.sqrt(2 / np.pi)
            x3 = x ** 3
            inner = c * (x + 0.044715 * x3)
            tanh_inner = np.tanh(inner)
            sech2 = 1 - tanh_inner ** 2
            inner_deriv = c * (1 + 3 * 0.044715 * x ** 2)
            return grad_output * (0.5 * (1 + tanh_inner) + 0.5 * x * sech2 * inner_deriv)
        elif self.kind == 'tanh':
            return grad_output * (1 - np.tanh(x) ** 2)
        elif self.kind == 'sigmoid':
            sig = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
            return grad_output * sig * (1 - sig)
        return grad_output


class LayerNorm:
    """Layer Normalization."""

    __slots__ = ('gamma', 'beta', 'eps', '_cached_input', '_cached_mean', '_cached_var', '_cached_norm', '_id')

    _instance_counter = 0

    def __init__(self, dim: int, eps: float = 1e-5):
        self.gamma = np.ones(dim, dtype=np.float32)
        self.beta = np.zeros(dim, dtype=np.float32)
        self.eps = eps
        self._cached_input = None
        self._cached_mean = None
        self._cached_var = None
        self._cached_norm = None
        self._id = f"ln_{LayerNorm._instance_counter}"
        LayerNorm._instance_counter += 1

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._cached_input = x
        self._cached_mean = x.mean(axis=-1, keepdims=True)
        self._cached_var = x.var(axis=-1, keepdims=True)
        self._cached_norm = (x - self._cached_mean) / np.sqrt(self._cached_var + self.eps)
        return self.gamma * self._cached_norm + self.beta

    def backward(self, grad_output: np.ndarray, lr: float, optimizer: Optional['AdamOptimizer'] = None) -> np.ndarray:
        N = self._cached_input.shape[-1]

        # Gradients for gamma and beta
        grad_gamma = (grad_output * self._cached_norm).sum(axis=tuple(range(grad_output.ndim - 1)))
        grad_beta = grad_output.sum(axis=tuple(range(grad_output.ndim - 1)))

        # Gradient for input
        std_inv = 1 / np.sqrt(self._cached_var + self.eps)
        grad_norm = grad_output * self.gamma

        grad_var = (grad_norm * (self._cached_input - self._cached_mean) * -0.5 * std_inv**3).sum(axis=-1, keepdims=True)
        grad_mean = (grad_norm * -std_inv).sum(axis=-1, keepdims=True) + grad_var * -2 * (self._cached_input - self._cached_mean).mean(axis=-1, keepdims=True)

        grad_input = grad_norm * std_inv + grad_var * 2 * (self._cached_input - self._cached_mean) / N + grad_mean / N

        # Update
        if optimizer is not None:
            optimizer.update(f"{self._id}_gamma", self.gamma, grad_gamma, lr)
            optimizer.update(f"{self._id}_beta", self.beta, grad_beta, lr)
        else:
            self.gamma -= lr * grad_gamma
            self.beta -= lr * grad_beta

        return grad_input


class Embedding:
    """Token embedding layer."""

    __slots__ = ('weights', '_cached_tokens', '_id')

    _instance_counter = 0

    def __init__(self, vocab_size: int, embed_dim: int):
        scale = np.sqrt(1.0 / embed_dim)
        self.weights = np.random.randn(vocab_size, embed_dim).astype(np.float32) * scale
        self._cached_tokens: Optional[np.ndarray] = None
        self._id = f"embed_{Embedding._instance_counter}"
        Embedding._instance_counter += 1

    def forward(self, tokens: np.ndarray) -> np.ndarray:
        """tokens: (batch, seq) -> (batch, seq, embed_dim)"""
        self._cached_tokens = tokens
        return self.weights[tokens]

    def backward(self, grad_output: np.ndarray, lr: float, optimizer: Optional['AdamOptimizer'] = None) -> None:
        """Update embeddings for accessed tokens only."""
        if optimizer is not None:
            # For Adam: compute full gradient, then apply Adam update
            grad = np.zeros_like(self.weights)
            np.add.at(grad, self._cached_tokens, grad_output)
            optimizer.update(self._id, self.weights, grad, lr)
        else:
            # Sparse update: only update embeddings that were used
            np.add.at(self.weights, self._cached_tokens, -lr * grad_output)


class PositionalEmbedding:
    """Combined token + positional embeddings."""

    __slots__ = ('token_embed', 'pos_weights', 'max_seq_len', 'embed_dim', '_cached_seq_len', '_id')

    _instance_counter = 0

    def __init__(self, vocab_size: int, embed_dim: int, max_seq_len: int):
        self.token_embed = Embedding(vocab_size, embed_dim)
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        scale = np.sqrt(1.0 / embed_dim)
        self.pos_weights = np.random.randn(max_seq_len, embed_dim).astype(np.float32) * scale
        self._cached_seq_len = 0
        self._id = f"pos_embed_{PositionalEmbedding._instance_counter}"
        PositionalEmbedding._instance_counter += 1

    def forward(self, tokens: np.ndarray) -> np.ndarray:
        """tokens: (batch, seq) -> (batch, seq, embed_dim)"""
        seq_len = tokens.shape[1]
        self._cached_seq_len = seq_len

        token_emb = self.token_embed.forward(tokens)
        pos_emb = self.pos_weights[:seq_len]  # (seq, embed_dim)

        return token_emb + pos_emb

    def backward(self, grad_output: np.ndarray, lr: float, optimizer: Optional['AdamOptimizer'] = None) -> None:
        """Update both embeddings."""
        # Position embedding gradient: sum over batch
        grad_pos = grad_output.sum(axis=0)  # (seq, embed_dim)

        if optimizer is not None:
            # Create full gradient for position embeddings
            full_grad_pos = np.zeros_like(self.pos_weights)
            full_grad_pos[:self._cached_seq_len] = grad_pos
            optimizer.update(self._id, self.pos_weights, full_grad_pos, lr)
        else:
            self.pos_weights[:self._cached_seq_len] -= lr * grad_pos

        # Token embedding
        self.token_embed.backward(grad_output, lr, optimizer)


class MultiHeadAttention:
    """Multi-head self-attention with causal masking."""

    __slots__ = ('embed_dim', 'num_heads', 'head_dim', 'scale',
                 'q_proj', 'k_proj', 'v_proj', 'out_proj',
                 '_cached_q', '_cached_k', '_cached_v', '_cached_attn', '_cached_input', '_id')

    _instance_counter = 0

    def __init__(self, embed_dim: int, num_heads: int):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / np.sqrt(self.head_dim)

        # Linear projections as weight matrices
        scale = np.sqrt(2.0 / embed_dim)
        self.q_proj = np.random.randn(embed_dim, embed_dim).astype(np.float32) * scale
        self.k_proj = np.random.randn(embed_dim, embed_dim).astype(np.float32) * scale
        self.v_proj = np.random.randn(embed_dim, embed_dim).astype(np.float32) * scale
        self.out_proj = np.random.randn(embed_dim, embed_dim).astype(np.float32) * scale

        self._cached_q = None
        self._cached_k = None
        self._cached_v = None
        self._cached_attn = None
        self._cached_input = None
        self._id = f"mha_{MultiHeadAttention._instance_counter}"
        MultiHeadAttention._instance_counter += 1

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        x: (batch, seq, embed_dim)
        Returns: (batch, seq, embed_dim)
        """
        batch, seq_len, _ = x.shape
        self._cached_input = x

        # Project to Q, K, V
        q = x @ self.q_proj  # (batch, seq, embed_dim)
        k = x @ self.k_proj
        v = x @ self.v_proj

        # Reshape to (batch, num_heads, seq, head_dim)
        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        self._cached_q = q
        self._cached_k = k
        self._cached_v = v

        # Attention scores: (batch, heads, seq, seq)
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        # Causal mask
        causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=np.float32) * -1e9, k=1)
        scores = scores + causal_mask

        # Additional padding mask if provided
        if mask is not None:
            scores = scores + mask

        # Softmax
        scores_max = scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attn = exp_scores / (exp_scores.sum(axis=-1, keepdims=True) + 1e-10)
        self._cached_attn = attn

        # Apply attention to values
        out = attn @ v  # (batch, heads, seq, head_dim)

        # Reshape back: (batch, seq, embed_dim)
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.embed_dim)

        # Output projection
        return out @ self.out_proj

    def backward(self, grad_output: np.ndarray, lr: float, optimizer: Optional['AdamOptimizer'] = None) -> np.ndarray:
        """Backward pass through attention."""
        batch, seq_len, _ = grad_output.shape

        # Gradient through output projection
        grad_out_proj = self._cached_input.reshape(-1, self.embed_dim).T @ grad_output.reshape(-1, self.embed_dim)
        grad_pre_out = grad_output @ self.out_proj.T

        # Reshape gradient: (batch, seq, embed_dim) -> (batch, heads, seq, head_dim)
        grad_pre_out = grad_pre_out.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Gradient through attention @ V
        grad_attn = grad_pre_out @ self._cached_v.transpose(0, 1, 3, 2)  # (batch, heads, seq, seq)
        grad_v = self._cached_attn.transpose(0, 1, 3, 2) @ grad_pre_out  # (batch, heads, seq, head_dim)

        # Gradient through softmax
        grad_scores = self._cached_attn * (grad_attn - (grad_attn * self._cached_attn).sum(axis=-1, keepdims=True))
        grad_scores = grad_scores * self.scale

        # Gradient through Q @ K.T
        grad_q = grad_scores @ self._cached_k  # (batch, heads, seq, head_dim)
        grad_k = grad_scores.transpose(0, 1, 3, 2) @ self._cached_q  # (batch, heads, seq, head_dim)

        # Reshape back to (batch, seq, embed_dim)
        grad_q = grad_q.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.embed_dim)
        grad_k = grad_k.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.embed_dim)
        grad_v = grad_v.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.embed_dim)

        # Gradient through projections
        grad_input = grad_q @ self.q_proj.T + grad_k @ self.k_proj.T + grad_v @ self.v_proj.T

        # Update projection weights
        x_flat = self._cached_input.reshape(-1, self.embed_dim)
        grad_q_proj = x_flat.T @ grad_q.reshape(-1, self.embed_dim)
        grad_k_proj = x_flat.T @ grad_k.reshape(-1, self.embed_dim)
        grad_v_proj = x_flat.T @ grad_v.reshape(-1, self.embed_dim)

        if optimizer is not None:
            optimizer.update(f"{self._id}_q", self.q_proj, grad_q_proj, lr)
            optimizer.update(f"{self._id}_k", self.k_proj, grad_k_proj, lr)
            optimizer.update(f"{self._id}_v", self.v_proj, grad_v_proj, lr)
            optimizer.update(f"{self._id}_o", self.out_proj, grad_out_proj, lr)
        else:
            self.q_proj -= lr * grad_q_proj
            self.k_proj -= lr * grad_k_proj
            self.v_proj -= lr * grad_v_proj
            self.out_proj -= lr * grad_out_proj

        return grad_input


class MLPBlock:
    """Feed-forward MLP block with GELU activation."""

    __slots__ = ('dense1', 'activation', 'dense2')

    def __init__(self, embed_dim: int, expansion: int = 4):
        hidden_dim = embed_dim * expansion
        self.dense1 = DenseLayer(embed_dim, hidden_dim)
        self.activation = Activation('gelu')
        self.dense2 = DenseLayer(hidden_dim, embed_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = self.dense1.forward(x)
        h = self.activation.forward(h)
        return self.dense2.forward(h)

    def backward(self, grad_output: np.ndarray, lr: float, optimizer: Optional['AdamOptimizer'] = None) -> np.ndarray:
        grad = self.dense2.backward(grad_output, lr, optimizer)
        grad = self.activation.backward(grad)
        return self.dense1.backward(grad, lr, optimizer)


class TransformerBlock:
    """Single transformer block with pre-norm architecture."""

    __slots__ = ('attention', 'mlp', 'norm1', 'norm2', '_cached_x', '_cached_attn_out')

    def __init__(self, embed_dim: int, num_heads: int):
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.mlp = MLPBlock(embed_dim)
        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)
        self._cached_x = None
        self._cached_attn_out = None

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        self._cached_x = x

        # Self-attention with residual
        normed = self.norm1.forward(x)
        attn_out = self.attention.forward(normed, mask)
        x = x + attn_out
        self._cached_attn_out = x

        # MLP with residual
        normed = self.norm2.forward(x)
        mlp_out = self.mlp.forward(normed)
        return x + mlp_out

    def backward(self, grad_output: np.ndarray, lr: float, optimizer: Optional['AdamOptimizer'] = None) -> np.ndarray:
        # MLP branch
        grad_mlp = self.mlp.backward(grad_output, lr, optimizer)
        grad_norm2 = self.norm2.backward(grad_mlp, lr, optimizer)
        grad_residual1 = grad_output + grad_norm2

        # Attention branch
        grad_attn = self.attention.backward(grad_residual1, lr, optimizer)
        grad_norm1 = self.norm1.backward(grad_attn, lr, optimizer)

        return grad_residual1 + grad_norm1


class OutputLayer:
    """Output projection with softmax for language modeling."""

    __slots__ = ('weights', 'bias', 'vocab_size', '_cached_input', '_cached_probs', '_id')

    def __init__(self, embed_dim: int, vocab_size: int):
        scale = np.sqrt(2.0 / embed_dim)
        self.weights = np.random.randn(embed_dim, vocab_size).astype(np.float32) * scale
        self.bias = np.zeros(vocab_size, dtype=np.float32)
        self.vocab_size = vocab_size
        self._cached_input = None
        self._cached_probs = None
        self._id = "output"

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return softmax probabilities."""
        self._cached_input = x

        logits = x @ self.weights + self.bias

        # Numerically stable softmax
        logits_max = logits.max(axis=-1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / (exp_logits.sum(axis=-1, keepdims=True) + 1e-10)
        self._cached_probs = probs

        return probs

    def backward(self, targets: np.ndarray, lr: float, optimizer: Optional['AdamOptimizer'] = None) -> np.ndarray:
        """Backward pass with cross-entropy loss gradient."""
        # Gradient of cross-entropy + softmax: probs - one_hot(targets)
        grad = self._cached_probs.copy()
        batch, seq_len, _ = grad.shape

        # Subtract 1 from correct class
        for b in range(batch):
            for t in range(seq_len):
                grad[b, t, targets[b, t]] -= 1

        # Average over batch and sequence
        grad = grad / (batch * seq_len)

        # Gradient through linear
        grad_input = grad @ self.weights.T

        # Update weights
        x_flat = self._cached_input.reshape(-1, self._cached_input.shape[-1])
        grad_flat = grad.reshape(-1, self.vocab_size)

        grad_weights = x_flat.T @ grad_flat
        grad_bias = grad_flat.sum(axis=0)

        if optimizer is not None:
            optimizer.update(f"{self._id}_w", self.weights, grad_weights, lr)
            optimizer.update(f"{self._id}_b", self.bias, grad_bias, lr)
        else:
            self.weights -= lr * grad_weights
            self.bias -= lr * grad_bias

        return grad_input

    def compute_loss(self, probs: np.ndarray, targets: np.ndarray) -> float:
        """Compute cross-entropy loss."""
        batch, seq_len, _ = probs.shape
        total_loss = 0.0

        for b in range(batch):
            for t in range(seq_len):
                target_prob = probs[b, t, targets[b, t]]
                total_loss -= np.log(target_prob + 1e-10)

        return float(total_loss / (batch * seq_len))


# =============================================================================
# GABGPT MODEL
# =============================================================================

class GabGPT:
    """Minimal GPT-style language model."""

    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int,
                 num_blocks: int, max_seq_len: int):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.max_seq_len = max_seq_len

        self.embedding = PositionalEmbedding(vocab_size, embed_dim, max_seq_len)
        self.blocks = [TransformerBlock(embed_dim, num_heads) for _ in range(num_blocks)]
        self.final_norm = LayerNorm(embed_dim)
        self.output = OutputLayer(embed_dim, vocab_size)

    def forward(self, tokens: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        tokens: (batch, seq) of token IDs
        Returns: (batch, seq, vocab_size) probabilities
        """
        x = self.embedding.forward(tokens)

        for block in self.blocks:
            x = block.forward(x, mask)

        x = self.final_norm.forward(x)
        return self.output.forward(x)

    def backward(self, targets: np.ndarray, lr: float, optimizer: Optional['AdamOptimizer'] = None) -> None:
        """Backward pass through entire model."""
        grad = self.output.backward(targets, lr, optimizer)
        grad = self.final_norm.backward(grad, lr, optimizer)

        for block in reversed(self.blocks):
            grad = block.backward(grad, lr, optimizer)

        self.embedding.backward(grad, lr, optimizer)

    def train_step(self, inputs: np.ndarray, targets: np.ndarray, lr: float, optimizer: Optional['AdamOptimizer'] = None) -> float:
        """Single training step, returns loss."""
        probs = self.forward(inputs)
        loss = self.output.compute_loss(probs, targets)
        if optimizer is not None:
            optimizer.step()  # Increment timestep for Adam
        self.backward(targets, lr, optimizer)
        return loss

    def generate(self, prompt_tokens: np.ndarray, max_length: int,
                 temperature: float = 1.0) -> np.ndarray:
        """Generate tokens from a prompt."""
        tokens = list(prompt_tokens)

        for _ in range(max_length):
            # Use only last max_seq_len tokens
            context = np.array([tokens[-self.max_seq_len:]], dtype=np.int32)
            probs = self.forward(context)
            next_probs = probs[0, -1]

            # Temperature sampling
            if temperature != 1.0:
                logits = np.log(next_probs + 1e-10) / temperature
                exp_logits = np.exp(logits - logits.max())
                next_probs = exp_logits / exp_logits.sum()

            # Sample
            next_token = np.random.choice(len(next_probs), p=next_probs)
            tokens.append(next_token)

        return np.array(tokens, dtype=np.int32)

    def serialize(self) -> bytes:
        """Serialize model to bytes in GAB1 format (compatible with JS version)."""
        buffer = io.BytesIO()

        hidden_dim = self.embed_dim * 4

        # Helper to write uint32 little-endian
        def write_u32(val):
            buffer.write(struct.pack('<I', int(val)))

        # Helper to write float32 little-endian
        def write_f32(val):
            buffer.write(struct.pack('<f', float(val)))

        # Helper to write 1D array
        def write_1d(arr):
            for v in arr.flat if hasattr(arr, 'flat') else arr:
                write_f32(v)

        # Helper to write 2D array (row by row)
        def write_2d(arr):
            for row in arr:
                for v in row:
                    write_f32(v)

        # Write header
        write_u32(0x47414231)  # 'GAB1' magic
        write_u32(1)  # version
        write_u32(self.vocab_size)
        write_u32(self.embed_dim)
        write_u32(self.num_heads)
        write_u32(self.num_blocks)
        write_u32(self.max_seq_len)

        # Write token embeddings: (vocab_size, embed_dim)
        write_2d(self.embedding.token_embed.weights)

        # Write position embeddings: (max_seq_len, embed_dim)
        write_2d(self.embedding.pos_weights)

        # Write transformer blocks
        for block in self.blocks:
            # layerNorm1: gamma, beta
            write_1d(block.norm1.gamma)
            write_1d(block.norm1.beta)

            # attention weights: Q, K, V, O (each embed_dim x embed_dim)
            write_2d(block.attention.q_proj)
            write_2d(block.attention.k_proj)
            write_2d(block.attention.v_proj)
            write_2d(block.attention.out_proj)

            # layerNorm2: gamma, beta
            write_1d(block.norm2.gamma)
            write_1d(block.norm2.beta)

            # MLP dense1: JS stores as (hiddenDim neurons, each with embeddingDim weights + 1 bias)
            # My Python stores weights as (embed_dim, hidden_dim), bias as (hidden_dim,)
            # JS neuron[n].weights[i] corresponds to Python weights[i, n]
            # So we need to write transposed: for each output neuron, write its input weights then bias
            for n in range(hidden_dim):
                # Write weights for neuron n (column n of weight matrix)
                for i in range(self.embed_dim):
                    write_f32(block.mlp.dense1.weights[i, n])
                # Write bias for neuron n
                write_f32(block.mlp.dense1.bias[n])

            # MLP dense2: (embeddingDim neurons, each with hiddenDim weights + 1 bias)
            for n in range(self.embed_dim):
                for i in range(hidden_dim):
                    write_f32(block.mlp.dense2.weights[i, n])
                write_f32(block.mlp.dense2.bias[n])

        # Write finalNorm: gamma, beta
        write_1d(self.final_norm.gamma)
        write_1d(self.final_norm.beta)

        # Write output layer: weights (embed_dim x vocab_size), bias (vocab_size)
        write_2d(self.output.weights)
        write_1d(self.output.bias)

        return buffer.getvalue()

    @classmethod
    def deserialize(cls, data: bytes) -> 'GabGPT':
        """Deserialize model from GAB1 format bytes."""
        buffer = io.BytesIO(data)

        # Helper to read uint32 little-endian
        def read_u32():
            return struct.unpack('<I', buffer.read(4))[0]

        # Helper to read float32 little-endian
        def read_f32():
            return struct.unpack('<f', buffer.read(4))[0]

        # Helper to read 1D array
        def read_1d(length):
            return np.array([read_f32() for _ in range(length)], dtype=np.float32)

        # Helper to read 2D array
        def read_2d(rows, cols):
            arr = np.zeros((rows, cols), dtype=np.float32)
            for i in range(rows):
                for j in range(cols):
                    arr[i, j] = read_f32()
            return arr

        # Read header
        magic = read_u32()
        if magic != 0x47414231:
            raise ValueError(f"Invalid GabGPT file format (magic: {hex(magic)})")

        version = read_u32()
        if version != 1:
            raise ValueError(f"Unsupported GabGPT version: {version}")

        vocab_size = read_u32()
        embed_dim = read_u32()
        num_heads = read_u32()
        num_blocks = read_u32()
        max_seq_len = read_u32()

        hidden_dim = embed_dim * 4

        # Create model
        model = cls(vocab_size, embed_dim, num_heads, num_blocks, max_seq_len)

        # Read token embeddings
        model.embedding.token_embed.weights = read_2d(vocab_size, embed_dim)

        # Read position embeddings
        model.embedding.pos_weights = read_2d(max_seq_len, embed_dim)

        # Read transformer blocks
        for block in model.blocks:
            # layerNorm1
            block.norm1.gamma = read_1d(embed_dim)
            block.norm1.beta = read_1d(embed_dim)

            # attention weights
            block.attention.q_proj = read_2d(embed_dim, embed_dim)
            block.attention.k_proj = read_2d(embed_dim, embed_dim)
            block.attention.v_proj = read_2d(embed_dim, embed_dim)
            block.attention.out_proj = read_2d(embed_dim, embed_dim)

            # layerNorm2
            block.norm2.gamma = read_1d(embed_dim)
            block.norm2.beta = read_1d(embed_dim)

            # MLP dense1: read neuron-by-neuron, reconstruct weight matrix
            weights1 = np.zeros((embed_dim, hidden_dim), dtype=np.float32)
            bias1 = np.zeros(hidden_dim, dtype=np.float32)
            for n in range(hidden_dim):
                for i in range(embed_dim):
                    weights1[i, n] = read_f32()
                bias1[n] = read_f32()
            block.mlp.dense1.weights = weights1
            block.mlp.dense1.bias = bias1

            # MLP dense2
            weights2 = np.zeros((hidden_dim, embed_dim), dtype=np.float32)
            bias2 = np.zeros(embed_dim, dtype=np.float32)
            for n in range(embed_dim):
                for i in range(hidden_dim):
                    weights2[i, n] = read_f32()
                bias2[n] = read_f32()
            block.mlp.dense2.weights = weights2
            block.mlp.dense2.bias = bias2

        # Read finalNorm
        model.final_norm.gamma = read_1d(embed_dim)
        model.final_norm.beta = read_1d(embed_dim)

        # Read output layer
        model.output.weights = read_2d(embed_dim, vocab_size)
        model.output.bias = read_1d(vocab_size)

        return model


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def create_batches(tokens: np.ndarray, batch_size: int, seq_length: int,
                   log_callback: Optional[Callable[[str], None]] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create training batches from token sequence (for pre-training)."""
    batches = []
    num_sequences = (len(tokens) - 1) // seq_length
    total_batches = (num_sequences + batch_size - 1) // batch_size

    if log_callback:
        log_callback(f"  Creating batches: {num_sequences:,} sequences -> ~{total_batches:,} batches")

    for batch_idx, batch_start in enumerate(range(0, num_sequences, batch_size)):
        batch_end = min(batch_start + batch_size, num_sequences)
        inputs = []
        targets = []

        for i in range(batch_start, batch_end):
            start = i * seq_length
            inputs.append(tokens[start:start + seq_length])
            targets.append(tokens[start + 1:start + seq_length + 1])

        batches.append((
            np.array(inputs, dtype=np.int32),
            np.array(targets, dtype=np.int32)
        ))

        # Log progress every 1000 batches
        if log_callback and (batch_idx + 1) % 1000 == 0:
            log_callback(f"  Batch creation progress: {batch_idx + 1:,}/{total_batches:,}")

    if log_callback:
        log_callback(f"  Batch creation complete: {len(batches):,} batches")

    return batches


class StreamingBatchGenerator:
    """Memory-efficient batch generator that streams data from HuggingFace datasets.

    Instead of loading all data into memory, this generator:
    1. Streams text from the dataset
    2. Tokenizes on-the-fly with a pre-trained tokenizer
    3. Maintains a token buffer and yields batches as they become available
    4. Supports infinite iteration over the dataset (for multiple epochs)
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer: 'Tokenizer',
        batch_size: int,
        seq_length: int,
        split: str = "train",
        text_column: str = "text",
        max_samples: int = 0,
        is_sft: bool = False,
        pad_token_id: int = 0,
    ):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.split = split
        self.text_column = text_column
        self.max_samples = max_samples
        self.is_sft = is_sft
        self.pad_token_id = pad_token_id

        # For pre-training: continuous token buffer
        self._token_buffer: List[int] = []

        # For SFT: conversation buffer
        self._conv_buffer: List[np.ndarray] = []

        # Track total batches yielded (for progress)
        self.batches_yielded = 0
        self._estimated_total_batches: Optional[int] = None

    def _get_data_stream(self):
        """Get the appropriate data stream based on mode."""
        if self.is_sft:
            return stream_hf_dataset_sft(self.dataset_name, self.split, self.max_samples)
        else:
            return stream_hf_dataset_text(self.dataset_name, self.split, self.text_column, self.max_samples)

    def _pretrain_batch_generator(self):
        """Generate batches for pre-training (continuous token stream)."""
        for text in self._get_data_stream():
            # Tokenize this chunk
            tokens = self.tokenizer.encode(text)
            self._token_buffer.extend(tokens.tolist())

            # Yield batches when we have enough tokens
            tokens_needed = (self.batch_size * self.seq_length) + 1  # +1 for targets

            while len(self._token_buffer) >= tokens_needed:
                inputs = []
                targets = []

                for _ in range(self.batch_size):
                    start = 0
                    inp = self._token_buffer[start:start + self.seq_length]
                    tgt = self._token_buffer[start + 1:start + self.seq_length + 1]
                    inputs.append(inp)
                    targets.append(tgt)
                    # Remove used tokens
                    self._token_buffer = self._token_buffer[self.seq_length:]

                self.batches_yielded += 1
                yield (
                    np.array(inputs, dtype=np.int32),
                    np.array(targets, dtype=np.int32)
                )

        # Yield any remaining tokens as a final (possibly smaller) batch
        if len(self._token_buffer) > self.seq_length + 1:
            num_seqs = (len(self._token_buffer) - 1) // self.seq_length
            if num_seqs > 0:
                inputs = []
                targets = []
                for i in range(min(num_seqs, self.batch_size)):
                    start = i * self.seq_length
                    inputs.append(self._token_buffer[start:start + self.seq_length])
                    targets.append(self._token_buffer[start + 1:start + self.seq_length + 1])

                if inputs:
                    self.batches_yielded += 1
                    yield (
                        np.array(inputs, dtype=np.int32),
                        np.array(targets, dtype=np.int32)
                    )

    def _sft_batch_generator(self):
        """Generate batches for SFT (conversation-aware with padding)."""
        for conv_text in self._get_data_stream():
            tokens = self.tokenizer.encode(conv_text)
            if len(tokens) > 1:  # Need at least 2 tokens
                self._conv_buffer.append(tokens)

            # Yield batch when we have enough conversations
            while len(self._conv_buffer) >= self.batch_size:
                batch_convs = self._conv_buffer[:self.batch_size]
                self._conv_buffer = self._conv_buffer[self.batch_size:]

                # Find max length (capped at seq_length + 1)
                max_len = min(max(len(t) for t in batch_convs), self.seq_length + 1)

                inputs = []
                targets = []

                for tokens in batch_convs:
                    tokens = tokens[:max_len]
                    inp = tokens[:-1]
                    tgt = tokens[1:]

                    # Pad to max_len - 1
                    pad_len = max_len - 1 - len(inp)
                    if pad_len > 0:
                        inp = np.concatenate([inp, np.full(pad_len, self.pad_token_id, dtype=np.int32)])
                        tgt = np.concatenate([tgt, np.full(pad_len, self.pad_token_id, dtype=np.int32)])

                    inputs.append(inp)
                    targets.append(tgt)

                self.batches_yielded += 1
                yield (
                    np.array(inputs, dtype=np.int32),
                    np.array(targets, dtype=np.int32)
                )

        # Yield remaining conversations
        if self._conv_buffer:
            max_len = min(max(len(t) for t in self._conv_buffer), self.seq_length + 1)
            inputs = []
            targets = []

            for tokens in self._conv_buffer:
                tokens = tokens[:max_len]
                inp = tokens[:-1]
                tgt = tokens[1:]

                pad_len = max_len - 1 - len(inp)
                if pad_len > 0:
                    inp = np.concatenate([inp, np.full(pad_len, self.pad_token_id, dtype=np.int32)])
                    tgt = np.concatenate([tgt, np.full(pad_len, self.pad_token_id, dtype=np.int32)])

                inputs.append(inp)
                targets.append(tgt)

            if inputs:
                self.batches_yielded += 1
                yield (
                    np.array(inputs, dtype=np.int32),
                    np.array(targets, dtype=np.int32)
                )
            self._conv_buffer = []

    def __iter__(self):
        """Iterate over batches for one epoch."""
        self._token_buffer = []
        self._conv_buffer = []

        if self.is_sft:
            yield from self._sft_batch_generator()
        else:
            yield from self._pretrain_batch_generator()

    def reset(self):
        """Reset the generator for a new epoch."""
        self._token_buffer = []
        self._conv_buffer = []
        self.batches_yielded = 0

    def estimate_total_batches(self, sample_size: int = 100) -> int:
        """Estimate total batches by sampling the dataset.

        This provides a rough estimate for progress tracking without loading all data.
        """
        if self._estimated_total_batches is not None:
            return self._estimated_total_batches

        # Sample first N items to estimate average tokens per item
        total_tokens = 0
        sample_count = 0

        for text in self._get_data_stream():
            tokens = self.tokenizer.encode(text)
            total_tokens += len(tokens)
            sample_count += 1
            if sample_count >= sample_size:
                break

        if sample_count == 0:
            return 0

        avg_tokens = total_tokens / sample_count

        # Estimate total items (use max_samples if set, otherwise assume 10x sample)
        if self.max_samples > 0:
            estimated_items = self.max_samples
        else:
            estimated_items = sample_count * 10  # Rough estimate

        estimated_total_tokens = avg_tokens * estimated_items
        tokens_per_batch = self.batch_size * self.seq_length

        self._estimated_total_batches = max(1, int(estimated_total_tokens / tokens_per_batch))
        return self._estimated_total_batches


def create_sft_batches(
    conversations: List[str],
    tokenizer: 'Tokenizer',
    batch_size: int,
    seq_length: int,
    pad_token_id: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create training batches for SFT with padding."""
    batches = []

    # Tokenize all conversations
    tokenized = []
    for conv in conversations:
        tokens = tokenizer.encode(conv)
        if len(tokens) > 1:  # Need at least 2 tokens for input/target
            tokenized.append(tokens)

    # Sort by length for efficient batching (similar lengths together)
    tokenized.sort(key=len)

    # Create batches
    for batch_start in range(0, len(tokenized), batch_size):
        batch_end = min(batch_start + batch_size, len(tokenized))
        batch_tokens = tokenized[batch_start:batch_end]

        # Find max length in this batch (capped at seq_length)
        max_len = min(max(len(t) for t in batch_tokens), seq_length + 1)

        inputs = []
        targets = []

        for tokens in batch_tokens:
            # Truncate if needed
            tokens = tokens[:max_len]

            # Create input (all but last) and target (all but first)
            inp = tokens[:-1]
            tgt = tokens[1:]

            # Pad to max_len - 1
            pad_len = max_len - 1 - len(inp)
            if pad_len > 0:
                inp = np.concatenate([inp, np.full(pad_len, pad_token_id, dtype=np.int32)])
                tgt = np.concatenate([tgt, np.full(pad_len, pad_token_id, dtype=np.int32)])

            inputs.append(inp)
            targets.append(tgt)

        if inputs:
            batches.append((
                np.array(inputs, dtype=np.int32),
                np.array(targets, dtype=np.int32)
            ))

    return batches


def load_hf_dataset_text(dataset_name: str, split: str = "train", text_column: str = "text", max_samples: int = 0) -> str:
    """Load a HuggingFace dataset and return concatenated text for pre-training."""
    if not HF_DATASETS_AVAILABLE:
        raise ImportError("datasets library not installed. Run: pip install datasets")

    dataset = load_dataset(dataset_name, split=split)

    texts = []
    column_warned = False
    for i, item in enumerate(dataset):
        if max_samples > 0 and i >= max_samples:
            break
        if text_column in item:
            text_content = str(item[text_column])
            if text_content.strip():  # Only add non-empty text
                texts.append(text_content)
        elif not column_warned:
            available_cols = list(item.keys()) if hasattr(item, 'keys') else []
            raise ValueError(f"Column '{text_column}' not found in dataset. Available columns: {available_cols}")

    if not texts:
        raise ValueError(f"No text data found. Check that column '{text_column}' exists and contains non-empty strings.")

    return "\n".join(texts)


def stream_hf_dataset_text(dataset_name: str, split: str = "train", text_column: str = "text", max_samples: int = 0):
    """Stream a HuggingFace dataset yielding text chunks for memory-efficient processing."""
    if not HF_DATASETS_AVAILABLE:
        raise ImportError("datasets library not installed. Run: pip install datasets")

    # Use streaming mode to avoid loading entire dataset
    dataset = load_dataset(dataset_name, split=split, streaming=True)

    yielded_any = False
    for i, item in enumerate(dataset):
        if max_samples > 0 and i >= max_samples:
            break
        if text_column in item:
            text_content = str(item[text_column])
            if text_content.strip():
                yielded_any = True
                yield text_content
        elif i == 0:  # Check on first item only
            available_cols = list(item.keys()) if hasattr(item, 'keys') else []
            raise ValueError(f"Column '{text_column}' not found in dataset. Available columns: {available_cols}")

    if not yielded_any:
        raise ValueError(f"No text data found. Check that column '{text_column}' exists and contains non-empty strings.")


def stream_hf_dataset_sft(dataset_name: str, split: str = "train", max_samples: int = 0):
    """Stream a HuggingFace dataset yielding conversations in ChatML format."""
    if not HF_DATASETS_AVAILABLE:
        raise ImportError("datasets library not installed. Run: pip install datasets")

    dataset = load_dataset(dataset_name, split=split, streaming=True)

    for i, item in enumerate(dataset):
        if max_samples > 0 and i >= max_samples:
            break

        chatml = ""

        # Try common conversation formats
        if "messages" in item:
            for msg in item["messages"]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                chatml += f"<|{role}|>{content}<|end|>"
        elif "conversations" in item:
            for msg in item["conversations"]:
                role = msg.get("from", msg.get("role", "user"))
                if role == "human":
                    role = "user"
                elif role == "gpt":
                    role = "assistant"
                content = msg.get("value", msg.get("content", ""))
                chatml += f"<|{role}|>{content}<|end|>"
        elif "instruction" in item and "output" in item:
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            output = item.get("output", "")
            user_content = instruction
            if input_text:
                user_content += f"\n{input_text}"
            chatml = f"<|user|>{user_content}<|end|><|assistant|>{output}<|end|>"
        elif "prompt" in item and "completion" in item:
            chatml = f"<|user|>{item['prompt']}<|end|><|assistant|>{item['completion']}<|end|>"
        elif "question" in item and "answer" in item:
            chatml = f"<|user|>{item['question']}<|end|><|assistant|>{item['answer']}<|end|>"
        elif "text" in item:
            chatml = item["text"]

        if chatml:
            yield chatml
        elif i == 0:
            # First item didn't match any format - warn user
            available_keys = list(item.keys()) if hasattr(item, 'keys') else []
            raise ValueError(
                f"Dataset format not recognized. Available columns: {available_keys}. "
                f"Expected one of: messages, conversations, instruction+output, prompt+completion, question+answer, or text"
            )


def load_hf_dataset_sft(
    dataset_name: str,
    split: str = "train",
    max_samples: int = 0
) -> List[str]:
    """Load a HuggingFace dataset and return conversations in ChatML format."""
    if not HF_DATASETS_AVAILABLE:
        raise ImportError("datasets library not installed. Run: pip install datasets")

    dataset = load_dataset(dataset_name, split=split)

    conversations = []
    for i, item in enumerate(dataset):
        if max_samples > 0 and i >= max_samples:
            break

        chatml = ""

        # Try common conversation formats
        if "messages" in item:
            # OpenAI/ShareGPT format
            for msg in item["messages"]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                chatml += f"<|{role}|>{content}<|end|>"
        elif "conversations" in item:
            # Vicuna/ShareGPT format
            for msg in item["conversations"]:
                role = msg.get("from", msg.get("role", "user"))
                if role == "human":
                    role = "user"
                elif role == "gpt":
                    role = "assistant"
                content = msg.get("value", msg.get("content", ""))
                chatml += f"<|{role}|>{content}<|end|>"
        elif "instruction" in item and "output" in item:
            # Alpaca format
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            output = item.get("output", "")

            user_content = instruction
            if input_text:
                user_content += f"\n{input_text}"

            chatml = f"<|user|>{user_content}<|end|><|assistant|>{output}<|end|>"
        elif "prompt" in item and "completion" in item:
            # Simple prompt/completion format
            chatml = f"<|user|>{item['prompt']}<|end|><|assistant|>{item['completion']}<|end|>"
        elif "question" in item and "answer" in item:
            # Q&A format
            chatml = f"<|user|>{item['question']}<|end|><|assistant|>{item['answer']}<|end|>"
        elif "text" in item:
            # Plain text - assume it's already formatted or use as-is
            chatml = item["text"]

        if chatml:
            conversations.append(chatml)
        elif i == 0:
            # First item didn't match any format - warn user
            available_keys = list(item.keys()) if hasattr(item, 'keys') else []
            raise ValueError(
                f"Dataset format not recognized. Available columns: {available_keys}. "
                f"Expected one of: messages, conversations, instruction+output, prompt+completion, question+answer, or text"
            )

    if not conversations:
        raise ValueError("No conversations extracted from dataset. Check that the dataset format is supported.")

    return conversations


def get_learning_rate(step: int, warmup_steps: int, total_steps: int, max_lr: float) -> float:
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        return max_lr * (step / warmup_steps)

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max_lr * 0.5 * (1 + np.cos(np.pi * progress))


class AdamOptimizer:
    """Adam optimizer with momentum and adaptive learning rates."""

    def __init__(self, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # Timestep
        self.m = {}  # First moment (momentum)
        self.v = {}  # Second moment (velocity)

    def reset(self):
        """Reset optimizer state."""
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self):
        """Increment timestep (call once per training step)."""
        self.t += 1

    def update(self, param_id: str, param: np.ndarray, grad: np.ndarray, lr: float) -> None:
        """Update a parameter using Adam algorithm.

        Args:
            param_id: Unique identifier for this parameter
            param: The parameter array to update (modified in-place)
            grad: The gradient for this parameter
            lr: Current learning rate
        """
        # Initialize moment estimates if needed
        if param_id not in self.m:
            self.m[param_id] = np.zeros_like(param)
            self.v[param_id] = np.zeros_like(param)

        # Update biased first moment estimate
        self.m[param_id] = self.beta1 * self.m[param_id] + (1 - self.beta1) * grad

        # Update biased second raw moment estimate
        self.v[param_id] = self.beta2 * self.v[param_id] + (1 - self.beta2) * (grad ** 2)

        # Compute bias-corrected first moment estimate
        m_hat = self.m[param_id] / (1 - self.beta1 ** self.t)

        # Compute bias-corrected second raw moment estimate
        v_hat = self.v[param_id] / (1 - self.beta2 ** self.t)

        # Update parameters
        param -= lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


# =============================================================================
# TRAINING STATE
# =============================================================================

@dataclass
class TrainingState:
    """Complete training state for save/resume."""

    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)

    # Model and tokenizer (serialized)
    model_bytes: Optional[bytes] = None
    tokenizer_bytes: Optional[bytes] = None

    # Training progress
    current_epoch: int = 0
    current_batch: int = 0
    current_step: int = 0
    total_steps: int = 0

    # Training data
    training_data: str = ""

    # Metrics
    losses: List[float] = field(default_factory=list)


# =============================================================================
# TRAINER CLASS
# =============================================================================

class Trainer:
    """Training manager with pause/resume support."""

    def __init__(self):
        self.model: Optional[GabGPT] = None
        self.tokenizer: Optional[Tokenizer] = None
        self.batches: List[Tuple[np.ndarray, np.ndarray]] = []
        self.state = TrainingState()
        self.optimizer: Optional[AdamOptimizer] = None

        # Streaming mode
        self._use_streaming = False
        self._batch_generator: Optional[StreamingBatchGenerator] = None

        # Control flags
        self._running = False
        self._paused = False
        self._stop_requested = False

        # Manual import flags (set when user imports model/tokenizer via file upload)
        self._model_imported = False
        self._tokenizer_imported = False

        # Callbacks
        self.on_log: Optional[Callable[[str, str], None]] = None
        self.on_progress: Optional[Callable[[float], None]] = None
        self.on_stats: Optional[Callable[[Dict[str, Any]], None]] = None

    def log(self, message: str, level: str = 'info'):
        """Log a message."""
        if self.on_log:
            self.on_log(message, level)

    def _init_training(self, config: Dict[str, Any]) -> bool:
        """Initialize training from config."""
        try:
            self.state = TrainingState(config=config.copy())

            mode = config.get('mode', 'pretrain_hf')
            is_sft = mode.startswith('sft')
            self._use_streaming = config.get('use_streaming', False)

            self.log("=" * 60, 'info')
            self.log("STARTING NEW TRAINING SESSION", 'info')
            self.log("=" * 60, 'info')

            # Log config
            self.log(f"Mode: {mode}", 'info')
            self.log(f"Streaming: {'Enabled' if self._use_streaming else 'Disabled'}", 'info')
            self.log(f"Embedding Dim: {config['embed_dim']}", 'info')
            self.log(f"Attention Heads: {config['num_heads']}", 'info')
            self.log(f"Transformer Blocks: {config['num_blocks']}", 'info')
            self.log(f"Max Sequence Length: {config['max_seq_len']}", 'info')
            self.log("-" * 60, 'info')

            # Check if we're using an imported tokenizer
            if self._tokenizer_imported and self.tokenizer is not None:
                self.log("Using imported tokenizer", 'success')
                self.log(f"  Vocabulary size: {self.tokenizer.vocab_size}", 'info')
            else:
                # Create new tokenizer
                self.log("Creating tokenizer...", 'info')
                self.tokenizer = Tokenizer()

                # Reserve keywords (always for SFT, optional for pretrain)
                reserved = config.get('reserved_keywords', [])
                if is_sft:
                    # Ensure ChatML tokens are reserved for SFT
                    sft_tokens = ['<|user|>', '<|assistant|>', '<|system|>', '<|end|>', '<|pad|>']
                    for tok in sft_tokens:
                        if tok not in reserved:
                            reserved.append(tok)

                for kw in reserved:
                    token_id = self.tokenizer.reserve_token(kw)
                    self.log(f"  Reserved '{kw}' -> {token_id}", 'info')

            # Get pad token ID for SFT
            pad_token_id = self.tokenizer.encode('<|pad|>')[0] if is_sft else 0

            # Common config
            dataset_name = config.get('dataset_name', '')
            dataset_split = config.get('dataset_split', 'train')
            text_column = config.get('text_column', 'text')
            max_samples = config.get('max_samples', 0)
            batch_size = config.get('batch_size', 4)
            seq_length = config.get('seq_length', 128)
            num_merges = config.get('num_merges', 1000)

            # Load training data based on mode
            self.log("-" * 60, 'info')

            if self._use_streaming:
                # STREAMING MODE: Don't load all data into memory
                self.log(f"Setting up streaming from: {dataset_name}", 'info')
                self.log(f"  Split: {dataset_split}", 'info')
                if max_samples > 0:
                    self.log(f"  Max samples: {max_samples}", 'info')

                # Train tokenizer only if not imported
                if not self._tokenizer_imported:
                    # For tokenizer training, we sample a subset of the data
                    self.log("Sampling data for tokenizer training...", 'info')
                    tokenizer_sample_size = min(max_samples if max_samples > 0 else 1000, 1000)
                    sample_texts = []

                    if is_sft:
                        for i, text in enumerate(stream_hf_dataset_sft(dataset_name, dataset_split, tokenizer_sample_size)):
                            sample_texts.append(text)
                            if i >= tokenizer_sample_size - 1:
                                break
                    else:
                        for i, text in enumerate(stream_hf_dataset_text(dataset_name, dataset_split, text_column, tokenizer_sample_size)):
                            sample_texts.append(text)
                            if i >= tokenizer_sample_size - 1:
                                break

                    tokenizer_training_text = "\n".join(sample_texts)
                    self.log(f"Sampled {len(sample_texts)} items for tokenizer ({len(tokenizer_training_text):,} chars)", 'info')

                    # Train tokenizer on sample
                    self.log(f"Training tokenizer with {num_merges} merges...", 'info')

                    def tokenizer_callback(merge_num, total):
                        if self.on_progress:
                            self.on_progress(merge_num / total * 10)

                    self.tokenizer.train(tokenizer_training_text, num_merges, tokenizer_callback,
                                         log_callback=lambda msg: self.log(msg, 'info'))
                    self.log(f"Vocabulary size: {self.tokenizer.vocab_size}", 'success')

                # Create streaming batch generator
                self._batch_generator = StreamingBatchGenerator(
                    dataset_name=dataset_name,
                    tokenizer=self.tokenizer,
                    batch_size=batch_size,
                    seq_length=seq_length,
                    split=dataset_split,
                    text_column=text_column,
                    max_samples=max_samples,
                    is_sft=is_sft,
                    pad_token_id=pad_token_id,
                )

                # Estimate total batches for progress tracking
                self.log("Estimating dataset size...", 'info')
                estimated_batches = self._batch_generator.estimate_total_batches()
                self.log(f"Estimated ~{estimated_batches} batches per epoch", 'info')

                # Don't store training_data in streaming mode (memory savings!)
                self.state.training_data = ""
                self.batches = []  # Will use generator instead

            else:
                # NON-STREAMING MODE: Load all data into memory (original behavior)
                if mode == 'pretrain_hf':
                    self.log(f"Loading HF dataset: {dataset_name}", 'info')
                    self.log(f"  Split: {dataset_split}, Column: {text_column}", 'info')
                    if max_samples > 0:
                        self.log(f"  Max samples: {max_samples}", 'info')

                    training_text = load_hf_dataset_text(dataset_name, dataset_split, text_column, max_samples)
                    self.state.training_data = training_text
                    self.log(f"Loaded {len(training_text)} characters", 'success')

                elif mode == 'sft_hf':
                    self.log(f"Loading HF dataset for SFT: {dataset_name}", 'info')
                    self.log(f"  Split: {dataset_split}", 'info')
                    if max_samples > 0:
                        self.log(f"  Max samples: {max_samples}", 'info')

                    conversations = load_hf_dataset_sft(dataset_name, dataset_split, max_samples)
                    self.log(f"Loaded {len(conversations)} conversations", 'success')

                    training_text = "\n".join(conversations)
                    self.state.training_data = training_text
                    self.state.config['conversations'] = conversations

                else:
                    self.log(f"Unknown mode: {mode}", 'error')
                    return False

                # Train tokenizer only if not imported
                if not self._tokenizer_imported:
                    self.log(f"Training tokenizer with {num_merges} merges...", 'info')

                    def tokenizer_callback(merge_num, total):
                        if self.on_progress:
                            self.on_progress(merge_num / total * 10)

                    self.tokenizer.train(training_text, num_merges, tokenizer_callback,
                                         log_callback=lambda msg: self.log(msg, 'info'))
                    self.log(f"Vocabulary size: {self.tokenizer.vocab_size}", 'success')

            # Create model or reuse imported one
            self.log("-" * 60, 'info')
            if self._model_imported and self.model is not None:
                self.log("Using imported model (resuming training)", 'success')
                self.log(f"  Vocab size: {self.model.vocab_size}, Embed dim: {self.model.embed_dim}", 'info')
                self.log(f"  Heads: {self.model.num_heads}, Blocks: {self.model.num_blocks}", 'info')
            else:
                self.log("Creating model...", 'info')

                self.model = GabGPT(
                    vocab_size=self.tokenizer.vocab_size,
                    embed_dim=config['embed_dim'],
                    num_heads=config['num_heads'],
                    num_blocks=config['num_blocks'],
                    max_seq_len=config['max_seq_len'],
                )
                self.log("Model created", 'success')

            # Initialize optimizer
            use_adam = config.get('use_adam', True)
            if use_adam:
                self.optimizer = AdamOptimizer()
                self.log("Using Adam optimizer", 'info')
            else:
                self.optimizer = None
                self.log("Using SGD optimizer", 'info')

            # Estimate parameters
            embed_dim = config['embed_dim']
            hidden_dim = embed_dim * 4
            vocab_size = self.tokenizer.vocab_size
            num_blocks = config['num_blocks']
            max_seq_len = config['max_seq_len']

            param_count = (
                vocab_size * embed_dim +
                max_seq_len * embed_dim +
                num_blocks * (
                    2 * embed_dim +
                    4 * embed_dim * embed_dim +
                    2 * embed_dim +
                    embed_dim * hidden_dim + hidden_dim +
                    hidden_dim * embed_dim + embed_dim
                ) +
                2 * embed_dim +
                embed_dim * vocab_size + vocab_size
            )
            self.log(f"Estimated parameters: ~{param_count:,}", 'info')

            if not self._use_streaming:
                # Create batches (non-streaming mode only)
                self.log("-" * 60, 'info')
                self.log("Creating training batches...", 'info')

                if is_sft:
                    conversations = self.state.config.get('conversations', [training_text])
                    self.batches = create_sft_batches(
                        conversations, self.tokenizer, batch_size, seq_length, pad_token_id
                    )
                    self.log(f"Created {len(self.batches)} SFT batches (with padding)", 'info')
                else:
                    self.log("Encoding training text...", 'info')
                    tokens = self.tokenizer.encode(training_text,
                                                   log_callback=lambda msg: self.log(msg, 'info'))
                    self.log(f"Encoded {len(training_text):,} chars -> {len(tokens):,} tokens", 'info')
                    if len(tokens) == 0:
                        self.log("ERROR: Tokenization produced 0 tokens. Check your training data and tokenizer.", 'error')
                        return False
                    self.log(f"Compression ratio: {len(training_text) / len(tokens):.2f}x", 'info')
                    self.batches = create_batches(tokens, batch_size, seq_length,
                                                  log_callback=lambda msg: self.log(msg, 'info'))
                    self.log(f"Created {len(self.batches):,} batches", 'info')

                if not self.batches:
                    self.log("ERROR: No batches created! Data too short.", 'error')
                    return False

            # Calculate steps
            num_epochs = config.get('num_epochs', 10)

            if self._use_streaming:
                estimated_batches = self._batch_generator.estimate_total_batches()
                self.state.total_steps = num_epochs * estimated_batches
            else:
                self.state.total_steps = num_epochs * len(self.batches)

            warmup_ratio = config.get('warmup_ratio', 0.1)
            warmup_steps = int(self.state.total_steps * warmup_ratio)

            self.state.config['num_epochs'] = num_epochs
            self.state.config['warmup_steps'] = warmup_steps
            self.state.config['learning_rate'] = config.get('learning_rate', 0.001)
            self.state.config['use_streaming'] = self._use_streaming

            self.log(f"Total steps (estimated): {self.state.total_steps}", 'info')
            self.log(f"Warmup steps: {warmup_steps}", 'info')

            # Reset import flags after initialization (so future training sessions start fresh)
            self._model_imported = False
            self._tokenizer_imported = False

            return True

        except Exception as e:
            self.log(f"Initialization error: {str(e)}", 'error')
            import traceback
            self.log(traceback.format_exc(), 'error')
            return False

    def start(self, config: Dict[str, Any]) -> None:
        """Start training in a new thread."""
        if self._running:
            return

        self._stop_requested = False
        self._paused = False
        self._running = True  # Set running BEFORE init so UI shows status during tokenizer training

        # Check if we have a loaded session to continue from
        if self.model is not None and self.tokenizer is not None and self.state.current_step > 0:
            # Continue from loaded session
            self.log("=" * 60, 'info')
            self.log("CONTINUING FROM LOADED SESSION", 'info')
            self.log(f"Resuming at epoch {self.state.current_epoch + 1}, step {self.state.current_step}", 'info')
            self.log("=" * 60, 'info')

            # Recreate batches if needed
            if not self.batches and self.state.training_data:
                is_sft = self.state.config.get('mode', '').startswith('sft')
                batch_size = self.state.config.get('batch_size', 4)
                seq_length = self.state.config.get('seq_length', 128)

                if is_sft:
                    conversations = self.state.config.get('conversations', [])
                    pad_token_id = self.tokenizer.encode('<|pad|>')[0]
                    self.batches = create_sft_batches(
                        conversations, self.tokenizer, batch_size, seq_length, pad_token_id
                    )
                else:
                    tokens = self.tokenizer.encode(self.state.training_data)
                    self.batches = create_batches(tokens, batch_size, seq_length)

                self.log(f"Recreated {len(self.batches)} batches", 'info')

            self._train_loop()
        else:
            # Fresh start
            if not self._init_training(config):
                self._running = False  # Reset if init failed
                return

            self._train_loop()

    def _train_loop(self) -> None:
        """Main training loop."""
        try:
            self.log("=" * 60, 'info')
            self.log("STARTING TRAINING LOOP", 'info')
            self.log("=" * 60, 'info')

            num_epochs = self.state.config['num_epochs']
            lr_base = self.state.config['learning_rate']
            warmup_steps = self.state.config['warmup_steps']
            log_every = 100  # Fixed: log every 100 steps

            if self._use_streaming:
                self._train_loop_streaming(num_epochs, lr_base, warmup_steps, log_every)
            else:
                self._train_loop_batched(num_epochs, lr_base, warmup_steps, log_every)

            if not self._stop_requested:
                self.log("=" * 60, 'info')
                self.log("TRAINING COMPLETE!", 'success')
                self.log("=" * 60, 'info')

                # Auto-save to persistent storage if available
                self._auto_save_artifacts()

        except Exception as e:
            self.log(f"Training error: {str(e)}", 'error')
            import traceback
            self.log(traceback.format_exc(), 'error')
        finally:
            self._running = False

    def _train_loop_batched(self, num_epochs: int, lr_base: float, warmup_steps: int, log_every: int) -> None:
        """Training loop for non-streaming (batched) mode."""
        epoch_losses = []

        while self.state.current_epoch < num_epochs:
            if self._stop_requested:
                self.log("Training stopped by user", 'warning')
                break

            if self._paused:
                return  # Will resume later

            if self.state.current_batch == 0:
                self.log("-" * 60, 'info')
                self.log(f"EPOCH {self.state.current_epoch + 1}/{num_epochs}", 'info')
                epoch_losses = []

            # Process batch
            if self.state.current_batch < len(self.batches):
                inputs, targets = self.batches[self.state.current_batch]

                lr = get_learning_rate(
                    self.state.current_step,
                    warmup_steps,
                    self.state.total_steps,
                    lr_base
                )

                loss = self.model.train_step(inputs, targets, lr, self.optimizer)
                epoch_losses.append(loss)
                self.state.losses.append(loss)

                # Log progress
                if self.state.current_step % log_every == 0 or self.state.current_batch == 0:
                    self.log(
                        f"  Step {self.state.current_step + 1}/{self.state.total_steps} | "
                        f"Batch {self.state.current_batch + 1}/{len(self.batches)} | "
                        f"Loss: {loss:.6f} | LR: {lr:.6f}",
                        'info'
                    )

                # Update progress
                progress = (self.state.current_step + 1) / self.state.total_steps * 100
                if self.on_progress:
                    self.on_progress(progress)

                # Update stats
                avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
                if self.on_stats:
                    self.on_stats({
                        'epoch': self.state.current_epoch + 1,
                        'step': self.state.current_step + 1,
                        'loss': f"{loss:.4f}",
                        'avg_loss': f"{avg_loss:.4f}",
                    })

                self.state.current_step += 1
                self.state.current_batch += 1
            else:
                # End of epoch
                avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
                self.log(f"EPOCH {self.state.current_epoch + 1} COMPLETE | Avg Loss: {avg_loss:.6f}", 'success')

                self.state.current_epoch += 1
                self.state.current_batch = 0

                # Auto-save checkpoint after each epoch
                self._auto_save_artifacts()

    def _train_loop_streaming(self, num_epochs: int, lr_base: float, warmup_steps: int, log_every: int) -> None:
        """Training loop for streaming mode - memory efficient for large datasets."""
        epoch_losses = []

        while self.state.current_epoch < num_epochs:
            if self._stop_requested:
                self.log("Training stopped by user", 'warning')
                break

            if self._paused:
                return  # Will resume later

            self.log("-" * 60, 'info')
            self.log(f"EPOCH {self.state.current_epoch + 1}/{num_epochs} (streaming)", 'info')
            epoch_losses = []

            # Reset generator for new epoch
            self._batch_generator.reset()
            batch_count = 0

            for inputs, targets in self._batch_generator:
                if self._stop_requested:
                    self.log("Training stopped by user", 'warning')
                    return

                if self._paused:
                    return  # Will resume later

                lr = get_learning_rate(
                    self.state.current_step,
                    warmup_steps,
                    self.state.total_steps,
                    lr_base
                )

                loss = self.model.train_step(inputs, targets, lr, self.optimizer)
                epoch_losses.append(loss)
                self.state.losses.append(loss)

                batch_count += 1

                # Log progress
                if self.state.current_step % log_every == 0 or batch_count == 1:
                    self.log(
                        f"  Step {self.state.current_step + 1}/{self.state.total_steps} | "
                        f"Batch {batch_count} | "
                        f"Loss: {loss:.6f} | LR: {lr:.6f}",
                        'info'
                    )

                # Update progress (based on estimated total)
                progress = min((self.state.current_step + 1) / max(1, self.state.total_steps) * 100, 99.9)
                if self.on_progress:
                    self.on_progress(progress)

                # Update stats
                avg_loss = sum(epoch_losses[-100:]) / len(epoch_losses[-100:]) if epoch_losses else 0
                if self.on_stats:
                    self.on_stats({
                        'epoch': self.state.current_epoch + 1,
                        'step': self.state.current_step + 1,
                        'loss': f"{loss:.4f}",
                        'avg_loss': f"{avg_loss:.4f}",
                    })

                self.state.current_step += 1
                self.state.current_batch = batch_count

            # End of epoch
            avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
            self.log(f"EPOCH {self.state.current_epoch + 1} COMPLETE | "
                     f"Batches: {batch_count} | Avg Loss: {avg_loss:.6f}", 'success')

            # Update total_steps estimate based on actual batch count
            if self.state.current_epoch == 0:
                self.state.total_steps = num_epochs * batch_count
                self.log(f"Updated total steps estimate: {self.state.total_steps}", 'info')

            self.state.current_epoch += 1
            self.state.current_batch = 0

            # Auto-save checkpoint after each epoch
            self._auto_save_artifacts()

    def pause(self) -> None:
        """Pause training."""
        if self._running and not self._paused:
            self._paused = True
            self.log("Training paused", 'warning')

    def resume(self) -> None:
        """Resume paused training."""
        if self._paused:
            self._paused = False
            self.log("Training resumed", 'success')
            self._running = True
            self._train_loop()

    def stop(self) -> None:
        """Stop training."""
        self._stop_requested = True
        self._paused = False

    def _auto_save_artifacts(self) -> None:
        """Auto-save model and tokenizer to persistent storage after training."""
        persistent_dir = "/data"
        if not os.path.isdir(persistent_dir) or not os.access(persistent_dir, os.W_OK):
            self.log("Persistent storage not available, skipping auto-save", 'info')
            return

        try:
            model_path = os.path.join(persistent_dir, "cached_model.gab")
            tokenizer_path = os.path.join(persistent_dir, "cached_tokenizer.bpe")

            # Delete old artifacts
            if os.path.exists(model_path):
                os.remove(model_path)
            if os.path.exists(tokenizer_path):
                os.remove(tokenizer_path)

            # Save new
            if self.model:
                model_data = self.model.serialize()
                with open(model_path, 'wb') as f:
                    f.write(model_data)
                self.log(f"Model saved to cache ({len(model_data):,} bytes)", 'success')

            if self.tokenizer:
                tokenizer_data = self.tokenizer.serialize()
                with open(tokenizer_path, 'wb') as f:
                    f.write(tokenizer_data)
                self.log(f"Tokenizer saved to cache ({len(tokenizer_data):,} bytes)", 'success')

        except Exception as e:
            self.log(f"Auto-save failed: {str(e)}", 'warning')

    def export_model(self) -> Optional[bytes]:
        """Export just the model weights."""
        if self.model:
            return self.model.serialize()
        return None

    def export_tokenizer(self) -> Optional[bytes]:
        """Export just the tokenizer."""
        if self.tokenizer:
            return self.tokenizer.serialize()
        return None

    def import_model(self, data: bytes) -> Dict[str, Any]:
        """Import a model from bytes. Returns model hyperparameters."""
        try:
            self.model = GabGPT.deserialize(data)
            self._model_imported = True
            self.log(f"Model imported: vocab_size={self.model.vocab_size}, embed_dim={self.model.embed_dim}, "
                     f"num_heads={self.model.num_heads}, num_blocks={self.model.num_blocks}, "
                     f"max_seq_len={self.model.max_seq_len}", 'success')
            return {
                'vocab_size': self.model.vocab_size,
                'embed_dim': self.model.embed_dim,
                'num_heads': self.model.num_heads,
                'num_blocks': self.model.num_blocks,
                'max_seq_len': self.model.max_seq_len,
            }
        except Exception as e:
            self.log(f"Failed to import model: {str(e)}", 'error')
            raise

    def import_tokenizer(self, data: bytes) -> Dict[str, Any]:
        """Import a tokenizer from bytes. Returns tokenizer info."""
        try:
            self.tokenizer = Tokenizer.deserialize(data)
            self._tokenizer_imported = True
            num_merges = self.tokenizer.vocab_size - 256
            self.log(f"Tokenizer imported: vocab_size={self.tokenizer.vocab_size}, "
                     f"num_merges={num_merges}", 'success')
            return {
                'vocab_size': self.tokenizer.vocab_size,
                'num_merges': num_merges,
            }
        except Exception as e:
            self.log(f"Failed to import tokenizer: {str(e)}", 'error')
            raise

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_paused(self) -> bool:
        return self._paused


# =============================================================================
# GRADIO INTERFACE
# =============================================================================

# Global trainer instance
trainer = Trainer()
training_thread: Optional[threading.Thread] = None
log_buffer: List[str] = []
MAX_LOG_LINES = 500


def add_log(message: str, level: str = 'info'):
    """Add a log message to the buffer."""
    global log_buffer
    timestamp = time.strftime("%H:%M:%S")

    # Simple prefix for level
    prefix = {'info': '', 'success': '[OK] ', 'warning': '[WARN] ', 'error': '[ERR] '}
    log_buffer.append(f"[{timestamp}] {prefix.get(level, '')}{message}")

    # Limit buffer size to prevent memory issues
    if len(log_buffer) > MAX_LOG_LINES:
        log_buffer = log_buffer[-MAX_LOG_LINES:]


def get_logs() -> str:
    """Get formatted logs."""
    return "\n".join(log_buffer)


def clear_logs():
    """Clear log buffer."""
    global log_buffer
    log_buffer = []
    return ""


# Progress tracking
current_progress = 0.0
current_stats = {'epoch': 0, 'step': 0, 'loss': '-', 'avg_loss': '-'}


def update_progress(progress: float):
    global current_progress
    current_progress = progress


def update_stats(stats: dict):
    global current_stats
    current_stats = stats


# Set up trainer callbacks
trainer.on_log = add_log
trainer.on_progress = update_progress
trainer.on_stats = update_stats


def start_training(
    training_mode: str,
    dataset_name: str,
    dataset_split: str,
    text_column: str,
    max_samples: int,
    use_streaming: bool,
    training_data: str,
    embed_dim: int,
    num_heads: int,
    num_blocks: int,
    max_seq_len: int,
    num_merges: int,
    reserved_keywords: str,
    num_epochs: int,
    batch_size: int,
    seq_length: int,
    learning_rate: float,
    warmup_ratio: float,
    use_adam: bool,
):
    """Start training with given parameters."""
    global training_thread, current_progress, current_stats

    if trainer.is_running:
        return "Training already in progress", gr.update(), gr.update()

    # Reset state
    current_progress = 0.0
    current_stats = {'epoch': 0, 'step': 0, 'loss': '-', 'avg_loss': '-'}
    clear_logs()

    # Parse keywords
    keywords = [kw.strip() for kw in reserved_keywords.split(',') if kw.strip()]

    config = {
        'mode': training_mode,
        'dataset_name': dataset_name,
        'dataset_split': dataset_split,
        'text_column': text_column,
        'max_samples': int(max_samples),
        'use_streaming': bool(use_streaming),
        'training_data': training_data,
        'embed_dim': int(embed_dim),
        'num_heads': int(num_heads),
        'num_blocks': int(num_blocks),
        'max_seq_len': int(max_seq_len),
        'num_merges': int(num_merges),
        'reserved_keywords': keywords,
        'num_epochs': int(num_epochs),
        'batch_size': int(batch_size),
        'seq_length': int(seq_length),
        'learning_rate': float(learning_rate),
        'warmup_ratio': float(warmup_ratio),
        'use_adam': bool(use_adam),
    }

    # Start in thread
    training_thread = threading.Thread(target=trainer.start, args=(config,))
    training_thread.start()

    return "Training started", gr.update(interactive=False), gr.update(interactive=True)


def pause_training():
    """Pause training."""
    trainer.pause()
    return "Paused"


def resume_training():
    """Resume training."""
    global training_thread

    if trainer.is_paused:
        training_thread = threading.Thread(target=trainer.resume)
        training_thread.start()
        return "Resumed"
    return "Not paused"


def stop_training():
    """Stop training."""
    trainer.stop()
    return "Stopped"


def export_model_file():
    """Export model weights."""
    data = trainer.export_model()
    if data:
        return data, f"Model exported ({len(data):,} bytes)"
    return None, "No model to export"


def export_tokenizer_file():
    """Export tokenizer."""
    data = trainer.export_tokenizer()
    if data:
        return data, f"Tokenizer exported ({len(data):,} bytes)"
    return None, "No tokenizer to export"


def import_model_file(file):
    """Import model from file. Returns values and disables model config fields."""
    if file is None:
        # Returns: status, embed_dim, num_heads, num_blocks, max_seq_len, num_merges, reserved_keywords
        return ("No file selected",
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update())

    try:
        with open(file.name, 'rb') as f:
            data = f.read()

        params = trainer.import_model(data)
        # Calculate num_merges from vocab_size
        num_merges = params['vocab_size'] - 256
        return (
            f"Model imported ({len(data):,} bytes). Config fields locked.",
            gr.update(value=params['embed_dim'], interactive=False),
            gr.update(value=params['num_heads'], interactive=False),
            gr.update(value=params['num_blocks'], interactive=False),
            gr.update(value=params['max_seq_len'], interactive=False),
            gr.update(value=num_merges, interactive=False),
            gr.update(interactive=False),  # reserved_keywords also locked (tied to tokenizer in model)
        )
    except Exception as e:
        return (f"Failed to import model: {str(e)}",
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update())


def import_tokenizer_file(file):
    """Import tokenizer from file. Returns status and disables tokenizer config fields."""
    if file is None:
        return ("No file selected",
                gr.update(),  # num_merges
                gr.update())  # reserved_keywords

    try:
        with open(file.name, 'rb') as f:
            data = f.read()

        params = trainer.import_tokenizer(data)
        return (
            f"Tokenizer imported ({len(data):,} bytes). Vocab: {params['vocab_size']}. Config fields locked.",
            gr.update(value=params['num_merges'], interactive=False),
            gr.update(interactive=False),  # reserved_keywords disabled
        )
    except Exception as e:
        return (f"Failed to import tokenizer: {str(e)}",
                gr.update(),
                gr.update())


# =============================================================================
# PERSISTENT STORAGE (HuggingFace Spaces /data directory)
# =============================================================================

# HuggingFace Spaces persistent storage path
PERSISTENT_DIR = "/data"
ARTIFACT_MODEL_PATH = os.path.join(PERSISTENT_DIR, "cached_model.gab")
ARTIFACT_TOKENIZER_PATH = os.path.join(PERSISTENT_DIR, "cached_tokenizer.bpe")


def is_persistent_storage_available() -> bool:
    """Check if persistent storage is available (HF Spaces)."""
    return os.path.isdir(PERSISTENT_DIR) and os.access(PERSISTENT_DIR, os.W_OK)


def save_artifacts() -> str:
    """Save model and tokenizer to persistent storage."""
    if not is_persistent_storage_available():
        return "Persistent storage not available (not on HF Spaces or /data not mounted)"

    if not trainer.model or not trainer.tokenizer:
        return "No model/tokenizer to save"

    try:
        # Delete old artifacts first
        if os.path.exists(ARTIFACT_MODEL_PATH):
            os.remove(ARTIFACT_MODEL_PATH)
        if os.path.exists(ARTIFACT_TOKENIZER_PATH):
            os.remove(ARTIFACT_TOKENIZER_PATH)

        # Save new artifacts
        model_data = trainer.model.serialize()
        tokenizer_data = trainer.tokenizer.serialize()

        with open(ARTIFACT_MODEL_PATH, 'wb') as f:
            f.write(model_data)

        with open(ARTIFACT_TOKENIZER_PATH, 'wb') as f:
            f.write(tokenizer_data)

        return f"Saved to persistent storage (model: {len(model_data):,} bytes, tokenizer: {len(tokenizer_data):,} bytes)"

    except Exception as e:
        return f"Failed to save artifacts: {str(e)}"


def load_artifacts() -> str:
    """Load model and tokenizer from persistent storage."""
    if not is_persistent_storage_available():
        return "Persistent storage not available"

    if not os.path.exists(ARTIFACT_MODEL_PATH) or not os.path.exists(ARTIFACT_TOKENIZER_PATH):
        return "No cached artifacts found"

    try:
        with open(ARTIFACT_MODEL_PATH, 'rb') as f:
            model_data = f.read()

        with open(ARTIFACT_TOKENIZER_PATH, 'rb') as f:
            tokenizer_data = f.read()

        trainer.model = GabGPT.deserialize(model_data)
        trainer.tokenizer = Tokenizer.deserialize(tokenizer_data)

        return f"Loaded from cache (model: {len(model_data):,} bytes, tokenizer: {len(tokenizer_data):,} bytes)"

    except Exception as e:
        return f"Failed to load artifacts: {str(e)}"


def get_artifact_status() -> str:
    """Check if cached artifacts exist."""
    if not is_persistent_storage_available():
        return "Persistent storage: Not available"

    model_exists = os.path.exists(ARTIFACT_MODEL_PATH)
    tokenizer_exists = os.path.exists(ARTIFACT_TOKENIZER_PATH)

    if model_exists and tokenizer_exists:
        model_size = os.path.getsize(ARTIFACT_MODEL_PATH)
        tokenizer_size = os.path.getsize(ARTIFACT_TOKENIZER_PATH)
        model_mtime = os.path.getmtime(ARTIFACT_MODEL_PATH)
        model_date = time.strftime("%Y-%m-%d %H:%M", time.localtime(model_mtime))
        return f"Cached: model ({model_size:,} B), tokenizer ({tokenizer_size:,} B) - {model_date}"

    return "No cached artifacts"


def download_cached_model():
    """Get cached model file for download."""
    if os.path.exists(ARTIFACT_MODEL_PATH):
        return ARTIFACT_MODEL_PATH
    return None


def download_cached_tokenizer():
    """Get cached tokenizer file for download."""
    if os.path.exists(ARTIFACT_TOKENIZER_PATH):
        return ARTIFACT_TOKENIZER_PATH
    return None


def clear_artifacts() -> str:
    """Clear cached artifacts from persistent storage."""
    if not is_persistent_storage_available():
        return "Persistent storage not available"

    cleared = []
    try:
        if os.path.exists(ARTIFACT_MODEL_PATH):
            os.remove(ARTIFACT_MODEL_PATH)
            cleared.append("model")
        if os.path.exists(ARTIFACT_TOKENIZER_PATH):
            os.remove(ARTIFACT_TOKENIZER_PATH)
            cleared.append("tokenizer")

        if cleared:
            return f"Cleared: {', '.join(cleared)}"
        return "No cached artifacts to clear"

    except Exception as e:
        return f"Failed to clear artifacts: {str(e)}"


def refresh_ui():
    """Refresh UI elements."""
    status = "Idle"
    if trainer.is_running and not trainer.is_paused:
        status = "Training..."
    elif trainer.is_paused:
        status = "Paused"

    return (
        get_logs(),
        f"{current_progress:.1f}%",
        current_stats['epoch'],
        current_stats['step'],
        current_stats['loss'],
        current_stats['avg_loss'],
        status,
    )


def estimate_params(embed_dim, num_heads, num_blocks, max_seq_len, num_merges):
    """Estimate parameter count."""
    vocab_size = 256 + int(num_merges)
    hidden_dim = int(embed_dim) * 4
    embed_dim = int(embed_dim)
    num_blocks = int(num_blocks)
    max_seq_len = int(max_seq_len)

    param_count = (
        vocab_size * embed_dim +
        max_seq_len * embed_dim +
        num_blocks * (
            2 * embed_dim +
            4 * embed_dim * embed_dim +
            2 * embed_dim +
            embed_dim * hidden_dim + hidden_dim +
            hidden_dim * embed_dim + embed_dim
        ) +
        2 * embed_dim +
        embed_dim * vocab_size + vocab_size
    )

    if param_count >= 1e9:
        return f"~{param_count/1e9:.1f}B params (vocab ~{vocab_size:,})"
    elif param_count >= 1e6:
        return f"~{param_count/1e6:.1f}M params (vocab ~{vocab_size:,})"
    elif param_count >= 1e3:
        return f"~{param_count/1e3:.0f}K params (vocab ~{vocab_size:,})"
    return f"~{param_count} params (vocab ~{vocab_size:,})"


def apply_preset(preset_name):
    """Apply architecture preset and return params with estimate.

    If a model has been manually imported, returns gr.update() for all fields
    to prevent overwriting the imported model's configuration.
    """
    # Don't apply preset if model was manually imported
    if trainer._model_imported:
        # Return no-change updates for all 8 outputs
        return (gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update())

    presets = {
        'small': (64, 4, 4, 128, 500, 64, 0.001),
        'medium': (512, 8, 6, 512, 1500, 256, 0.0003),
        'large': (1536, 16, 24, 1024, 4000, 512, 0.0001),
    }

    if preset_name in presets:
        vals = presets[preset_name]
    else:
        vals = (512, 8, 6, 512, 1500, 256, 0.0003)

    # Calculate estimate for these values
    embed_dim, num_heads, num_blocks, max_seq_len, num_merges, seq_length, learning_rate = vals
    estimate = estimate_params(embed_dim, num_heads, num_blocks, max_seq_len, num_merges)

    return vals + (estimate,)


# Build Gradio interface
def create_ui():
    with gr.Blocks(title="GabGPT Trainer") as demo:
        with gr.Row():
            # Left column - Configuration
            with gr.Column(scale=1):
                gr.Markdown("## Training Mode")
                training_mode = gr.Radio(
                    choices=[
                        ("Pre-Train", "pretrain_hf"),
                        ("SFT", "sft_hf"),
                    ],
                    value="pretrain_hf",
                    label="",
                )

                gr.Markdown("### HuggingFace Dataset")
                dataset_name = gr.Textbox(
                    label="Dataset Name",
                    value="roneneldan/TinyStories",
                    placeholder="e.g., roneneldan/TinyStories, databricks/dolly-15k",
                )
                with gr.Row():
                    dataset_split = gr.Textbox(label="Split", value="train")
                    text_column = gr.Textbox(label="Text Column (Pre-Train only)", value="text")
                    max_samples = gr.Number(label="Max Samples (0=all)", value=1000, minimum=0)
                use_streaming = gr.Checkbox(
                    label="Enable Streaming (memory-efficient for large datasets)",
                    value=True,
                    info="Streams data on-the-fly instead of loading all into memory"
                )
                gr.Markdown("*SFT auto-detects: messages, conversations, instruction/output, prompt/completion, question/answer*", elem_classes=["info-text"])

                # Hidden - not used but needed for function signature
                training_data = gr.Textbox(visible=False, value="")

                gr.Markdown("## Model Architecture")
                with gr.Row():
                    embed_dim = gr.Number(label="Embedding Dim", value=512, minimum=8, step=8)
                    num_heads = gr.Number(label="Attention Heads", value=8, minimum=1)

                with gr.Row():
                    num_blocks = gr.Number(label="Transformer Blocks", value=6, minimum=1)
                    max_seq_len = gr.Number(label="Max Sequence Length", value=512, minimum=16, step=16)

                param_estimate = gr.Textbox(label="Estimated Parameters", interactive=False)

                with gr.Row():
                    preset_small = gr.Button("~300K", size="sm")
                    preset_medium = gr.Button("~20M", size="sm")
                    preset_large = gr.Button("~1B", size="sm")

                gr.Markdown("## Tokenizer")
                num_merges = gr.Number(label="BPE Merges", value=1500, minimum=0, step=50)
                reserved_keywords = gr.Textbox(
                    label="Reserved Keywords (comma-separated)",
                    value="<|user|>,<|assistant|>,<|system|>,<|end|>,<|pad|>,<|think|>",
                )

                gr.Markdown("## Training Config")
                with gr.Row():
                    num_epochs = gr.Number(label="Epochs", value=10, minimum=1)
                    batch_size = gr.Number(label="Batch Size", value=4, minimum=1)
                    seq_length = gr.Number(label="Sequence Length", value=256, minimum=8, step=8)

                with gr.Row():
                    learning_rate = gr.Number(label="Learning Rate", value=0.0003, minimum=0)
                    warmup_ratio = gr.Number(label="Warmup Ratio", value=0.1, minimum=0, maximum=1)
                    use_adam = gr.Checkbox(label="Use Adam Optimizer", value=True)

                gr.Markdown("## Artifacts (Persistent Storage)")
                artifact_status = gr.Textbox(label="Cache Status", value="", interactive=False, lines=1)
                with gr.Row():
                    save_artifact_btn = gr.Button("Save to Cache", size="sm")
                    load_artifact_btn = gr.Button("Load from Cache", size="sm")
                    clear_artifact_btn = gr.Button("Clear Cache", size="sm")
                with gr.Row():
                    download_cached_model_btn = gr.Button("Download Cached Model", size="sm")
                    download_cached_tokenizer_btn = gr.Button("Download Cached Tokenizer", size="sm")
                with gr.Row():
                    cached_model_file = gr.File(label="Model", height=50)
                    cached_tokenizer_file = gr.File(label="Tokenizer", height=50)

            # Right column - Training
            with gr.Column(scale=1):
                gr.Markdown("## Training Controls")
                with gr.Row():
                    status_text = gr.Textbox(label="Status", value="Idle", interactive=False)
                    progress_text = gr.Textbox(label="Progress", value="0%", interactive=False)

                with gr.Row():
                    start_btn = gr.Button("Start Training", variant="primary")
                    pause_btn = gr.Button("Pause")
                    resume_btn = gr.Button("Resume")
                    stop_btn = gr.Button("Stop", variant="stop")

                with gr.Row():
                    epoch_display = gr.Number(label="Epoch", value=0, interactive=False)
                    step_display = gr.Number(label="Step", value=0, interactive=False)
                    loss_display = gr.Textbox(label="Loss", value="-", interactive=False)
                    avg_loss_display = gr.Textbox(label="Avg Loss", value="-", interactive=False)

                gr.Markdown("## Training Log")
                log_output = gr.Textbox(
                    label="",
                    value="",
                    lines=20,
                    max_lines=20,
                    interactive=False,
                )

                with gr.Row():
                    clear_log_btn = gr.Button("Clear Log", size="sm")
                    refresh_btn = gr.Button("Refresh UI", size="sm")

                gr.Markdown("## Export & Import")
                save_status = gr.Textbox(label="Status", interactive=False, lines=1)

                with gr.Row():
                    export_model_btn = gr.Button("Export Model", size="sm")
                    export_tokenizer_btn = gr.Button("Export Tokenizer", size="sm")

                with gr.Row():
                    model_download = gr.File(label="Model (.gab)", height=60)
                    tokenizer_download = gr.File(label="Tokenizer (.bpe)", height=60)

                with gr.Row():
                    import_model_file_input = gr.File(label="Import Model (.gab)", file_types=[".gab"], height=60)
                    import_tokenizer_file_input = gr.File(label="Import Tokenizer (.bpe)", file_types=[".bpe"], height=60)

        # Event handlers
        def update_param_display(*args):
            return estimate_params(*args)

        # Parameter estimation
        for component in [embed_dim, num_heads, num_blocks, max_seq_len, num_merges]:
            component.change(
                fn=update_param_display,
                inputs=[embed_dim, num_heads, num_blocks, max_seq_len, num_merges],
                outputs=[param_estimate],
            )

        # Presets
        preset_small.click(
            fn=lambda: apply_preset('small'),
            outputs=[embed_dim, num_heads, num_blocks, max_seq_len, num_merges, seq_length, learning_rate, param_estimate],
        )
        preset_medium.click(
            fn=lambda: apply_preset('medium'),
            outputs=[embed_dim, num_heads, num_blocks, max_seq_len, num_merges, seq_length, learning_rate, param_estimate],
        )
        preset_large.click(
            fn=lambda: apply_preset('large'),
            outputs=[embed_dim, num_heads, num_blocks, max_seq_len, num_merges, seq_length, learning_rate, param_estimate],
        )

        # Training controls
        start_btn.click(
            fn=start_training,
            inputs=[
                training_mode, dataset_name, dataset_split, text_column, max_samples,
                use_streaming, training_data, embed_dim, num_heads, num_blocks, max_seq_len,
                num_merges, reserved_keywords, num_epochs, batch_size, seq_length,
                learning_rate, warmup_ratio, use_adam,
            ],
            outputs=[status_text, start_btn, pause_btn],
        )

        pause_btn.click(fn=pause_training, outputs=[status_text])
        resume_btn.click(fn=resume_training, outputs=[status_text])
        stop_btn.click(fn=stop_training, outputs=[status_text])

        # Refresh
        refresh_btn.click(
            fn=refresh_ui,
            outputs=[log_output, progress_text, epoch_display, step_display, loss_display, avg_loss_display, status_text],
        )

        clear_log_btn.click(fn=clear_logs, outputs=[log_output])

        # Initial load
        demo.load(
            fn=refresh_ui,
            outputs=[log_output, progress_text, epoch_display, step_display, loss_display, avg_loss_display, status_text],
        )

        # Export/Import
        def export_model_and_prepare():
            data, msg = export_model_file()
            if data:
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.gab') as f:
                    f.write(data)
                    return f.name, msg
            return None, msg

        def export_tokenizer_and_prepare():
            data, msg = export_tokenizer_file()
            if data:
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.bpe') as f:
                    f.write(data)
                    return f.name, msg
            return None, msg

        export_model_btn.click(
            fn=export_model_and_prepare,
            outputs=[model_download, save_status],
        )

        export_tokenizer_btn.click(
            fn=export_tokenizer_and_prepare,
            outputs=[tokenizer_download, save_status],
        )

        # Import handlers
        import_model_file_input.change(
            fn=import_model_file,
            inputs=[import_model_file_input],
            outputs=[save_status, embed_dim, num_heads, num_blocks, max_seq_len, num_merges, reserved_keywords],
        )

        import_tokenizer_file_input.change(
            fn=import_tokenizer_file,
            inputs=[import_tokenizer_file_input],
            outputs=[save_status, num_merges, reserved_keywords],
        )

        # Artifact handlers
        save_artifact_btn.click(
            fn=save_artifacts,
            outputs=[artifact_status],
        )

        load_artifact_btn.click(
            fn=load_artifacts,
            outputs=[artifact_status],
        )

        clear_artifact_btn.click(
            fn=clear_artifacts,
            outputs=[artifact_status],
        )

        download_cached_model_btn.click(
            fn=download_cached_model,
            outputs=[cached_model_file],
        )

        download_cached_tokenizer_btn.click(
            fn=download_cached_tokenizer,
            outputs=[cached_tokenizer_file],
        )

        # Load artifact status on startup
        demo.load(
            fn=get_artifact_status,
            outputs=[artifact_status],
        )

    return demo


# Main entry point
if __name__ == "__main__":
    demo = create_ui()
    demo.launch(ssr_mode=False)
