"""
gab.py - PyTorch GPT-2 style transformer for Supervised Fine-Tuning

A GPU-accelerated implementation with support for:
- Loss masking (train only on assistant responses)
- Attention masking (handle padding)
- .gab format for JS interop
- .pt checkpoints for training
"""

import math
import struct
import numpy as np
from typing import List, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# MODEL COMPONENTS
# ============================================================================

class MLPBlock(nn.Module):
    """Feed-forward network: Linear -> GELU -> Linear"""

    def __init__(self, embedding_dim: int, expansion_factor: int = 4):
        super().__init__()
        hidden_dim = embedding_dim * expansion_factor
        self.dense1 = nn.Linear(embedding_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dense2 = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dense2(x)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head causal self-attention with Flash Attention support"""

    def __init__(self, embedding_dim: int, num_heads: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.query_weights = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key_weights = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value_weights = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.output_weights = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_length, _ = x.shape

        # Project to Q, K, V
        queries = self.query_weights(x)
        keys = self.key_weights(x)
        values = self.value_weights(x)

        # Reshape for multi-head attention: (batch, seq, embed) -> (batch, heads, seq, head_dim)
        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Use Flash Attention via PyTorch 2.0+ scaled_dot_product_attention
        if attention_mask is None:
            # Pure causal attention - most efficient path
            attended = F.scaled_dot_product_attention(
                queries, keys, values,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True
            )
        else:
            # With attention mask - need to create combined causal + padding mask
            # Create causal mask: (seq, seq) where True = masked (cannot attend)
            causal_mask = torch.triu(
                torch.ones(seq_length, seq_length, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            # attention_mask: (batch, seq) where 1 = valid, 0 = pad
            # Expand: (batch, 1, 1, seq) - marks which keys to mask
            padding_mask_expanded = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
            # Combine: (batch, 1, seq, seq)
            combined_mask = causal_mask.unsqueeze(0).unsqueeze(0) | padding_mask_expanded
            # Convert bool mask to attention bias (-inf for masked positions)
            attn_mask = torch.zeros_like(combined_mask, dtype=queries.dtype)
            attn_mask.masked_fill_(combined_mask, float('-inf'))

            attended = F.scaled_dot_product_attention(
                queries, keys, values,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False  # We're providing our own causal mask
            )

        # Reshape back: (batch, heads, seq, head_dim) -> (batch, seq, embed)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embedding_dim)

        # Output projection
        output = self.output_weights(attended)
        return output


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LN -> Attention -> Residual -> LN -> MLP -> Residual"""

    def __init__(self, embedding_dim: int, num_heads: int):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.attention = MultiHeadAttention(embedding_dim, num_heads)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLPBlock(embedding_dim)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm attention with residual
        normed = self.layer_norm1(x)
        attended = self.attention(normed, attention_mask)
        x = x + attended

        # Pre-norm MLP with residual
        normed = self.layer_norm2(x)
        mlp_out = self.mlp(normed)
        x = x + mlp_out

        return x


class GabGPT(nn.Module):
    """GPT-2 style transformer language model"""

    def __init__(self, vocab_size: int, embedding_dim: int, num_heads: int,
                 num_blocks: int, max_seq_length: int):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.max_seq_length = max_seq_length

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_length, embedding_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads)
            for _ in range(num_blocks)
        ])

        # Output
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        scale = math.sqrt(1.0 / self.embedding_dim)
        nn.init.uniform_(self.token_embedding.weight, -scale, scale)
        nn.init.uniform_(self.position_embedding.weight, -scale, scale)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, input_tokens: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass

        Args:
            input_tokens: (batch_size, seq_length) token IDs
            attention_mask: (batch_size, seq_length) where 1=valid, 0=pad

        Returns:
            logits: (batch_size, seq_length, vocab_size)
        """
        batch_size, seq_length = input_tokens.shape

        # Create position indices
        positions = torch.arange(seq_length, device=input_tokens.device).unsqueeze(0).expand(batch_size, -1)

        # Embeddings
        token_embeds = self.token_embedding(input_tokens)
        position_embeds = self.position_embedding(positions)
        hidden = token_embeds + position_embeds

        # Transformer blocks
        for block in self.blocks:
            hidden = block(hidden, attention_mask)

        # Output
        hidden = self.final_norm(hidden)
        logits = self.output(hidden)

        return logits

    def generate(self, prompt_tokens: List[int], max_length: int,
                 temperature: float = 1.0, device: str = 'cpu') -> List[int]:
        """Generate tokens autoregressively"""
        self.eval()
        tokens = prompt_tokens.copy()

        with torch.no_grad():
            for _ in range(max_length):
                # Truncate if exceeds max sequence length
                input_tokens = tokens[-self.max_seq_length:]
                input_tensor = torch.tensor([input_tokens], dtype=torch.long, device=device)

                logits = self.forward(input_tensor)
                last_logits = logits[0, -1, :] / temperature

                probs = F.softmax(last_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                tokens.append(next_token)

        return tokens

    def train_step(self, input_tokens: torch.Tensor, target_tokens: torch.Tensor,
                   optimizer: torch.optim.Optimizer,
                   attention_mask: Optional[torch.Tensor] = None,
                   loss_mask: Optional[torch.Tensor] = None) -> float:
        """
        Single training step with optional loss masking for SFT

        Args:
            input_tokens: (batch, seq_length) input token IDs
            target_tokens: (batch, seq_length) target token IDs
            optimizer: optimizer to use
            attention_mask: (batch, seq_length) 1=valid, 0=pad (for attention)
            loss_mask: (batch, seq_length) 1=compute loss, 0=ignore (for SFT)

        Returns:
            loss value as float
        """
        self.train()
        optimizer.zero_grad()

        logits = self.forward(input_tokens, attention_mask)

        if loss_mask is None:
            # Standard loss on all tokens (pre-training style)
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                target_tokens.view(-1)
            )
        else:
            # Masked loss for SFT - only compute on response tokens
            loss_per_token = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                target_tokens.view(-1),
                reduction='none'
            )
            # Apply mask and compute mean over non-masked tokens
            loss_mask_flat = loss_mask.view(-1).float()
            masked_loss = loss_per_token * loss_mask_flat
            num_tokens = loss_mask_flat.sum()
            if num_tokens > 0:
                loss = masked_loss.sum() / num_tokens
            else:
                loss = masked_loss.sum()  # Edge case: no tokens to train on

        loss.backward()
        optimizer.step()

        return loss.item()

    # ========================================================================
    # .gab SERIALIZATION (JS interop)
    # ========================================================================

    def save_gab(self, path: str):
        """Save model weights in .gab format (compatible with gab.js)"""
        hidden_dim = self.embedding_dim * 4

        # Calculate buffer size
        buffer_size = 28  # header
        buffer_size += self.vocab_size * self.embedding_dim * 4  # token embeddings
        buffer_size += self.max_seq_length * self.embedding_dim * 4  # position embeddings

        for _ in range(self.num_blocks):
            buffer_size += 2 * self.embedding_dim * 4  # layerNorm1
            buffer_size += 4 * self.embedding_dim * self.embedding_dim * 4  # attention
            buffer_size += 2 * self.embedding_dim * 4  # layerNorm2
            buffer_size += hidden_dim * (self.embedding_dim + 1) * 4  # mlp.dense1
            buffer_size += self.embedding_dim * (hidden_dim + 1) * 4  # mlp.dense2

        buffer_size += 2 * self.embedding_dim * 4  # finalNorm
        buffer_size += self.embedding_dim * self.vocab_size * 4  # output weights
        buffer_size += self.vocab_size * 4  # output bias

        data = bytearray(buffer_size)
        offset = 0

        def write_float(val: float):
            nonlocal offset
            struct.pack_into('<f', data, offset, val)
            offset += 4

        def write_tensor_1d(tensor: torch.Tensor):
            for val in tensor.detach().cpu().numpy().flatten():
                write_float(float(val))

        def write_tensor_2d(tensor: torch.Tensor):
            for row in tensor.detach().cpu().numpy():
                for val in row:
                    write_float(float(val))

        # Header
        struct.pack_into('<I', data, offset, 0x47414231)  # 'GAB1'
        offset += 4
        struct.pack_into('<I', data, offset, 1)  # version
        offset += 4
        struct.pack_into('<I', data, offset, self.vocab_size)
        offset += 4
        struct.pack_into('<I', data, offset, self.embedding_dim)
        offset += 4
        struct.pack_into('<I', data, offset, self.num_heads)
        offset += 4
        struct.pack_into('<I', data, offset, self.num_blocks)
        offset += 4
        struct.pack_into('<I', data, offset, self.max_seq_length)
        offset += 4

        # Token embeddings
        write_tensor_2d(self.token_embedding.weight)

        # Position embeddings
        write_tensor_2d(self.position_embedding.weight)

        # Transformer blocks
        for block in self.blocks:
            # layerNorm1
            write_tensor_1d(block.layer_norm1.weight)
            write_tensor_1d(block.layer_norm1.bias)

            # attention weights (transpose for JS compatibility)
            write_tensor_2d(block.attention.query_weights.weight.T)
            write_tensor_2d(block.attention.key_weights.weight.T)
            write_tensor_2d(block.attention.value_weights.weight.T)
            write_tensor_2d(block.attention.output_weights.weight.T)

            # layerNorm2
            write_tensor_1d(block.layer_norm2.weight)
            write_tensor_1d(block.layer_norm2.bias)

            # mlp.dense1 - stored per-neuron (weights + bias)
            dense1_weights = block.mlp.dense1.weight.detach().cpu().numpy()
            dense1_bias = block.mlp.dense1.bias.detach().cpu().numpy()
            for n in range(hidden_dim):
                for w in dense1_weights[n]:
                    write_float(float(w))
                write_float(float(dense1_bias[n]))

            # mlp.dense2 - stored per-neuron (weights + bias)
            dense2_weights = block.mlp.dense2.weight.detach().cpu().numpy()
            dense2_bias = block.mlp.dense2.bias.detach().cpu().numpy()
            for n in range(self.embedding_dim):
                for w in dense2_weights[n]:
                    write_float(float(w))
                write_float(float(dense2_bias[n]))

        # finalNorm
        write_tensor_1d(self.final_norm.weight)
        write_tensor_1d(self.final_norm.bias)

        # output layer (transpose weights)
        write_tensor_2d(self.output.weight.T)
        write_tensor_1d(self.output.bias)

        with open(path, 'wb') as f:
            f.write(data)

    @staticmethod
    def load_gab(path: str, device: str = 'cpu') -> 'GabGPT':
        """Load model from .gab format (compatible with gab.js)"""
        with open(path, 'rb') as f:
            data = f.read()

        offset = 0

        def read_uint32() -> int:
            nonlocal offset
            val = struct.unpack_from('<I', data, offset)[0]
            offset += 4
            return val

        def read_float() -> float:
            nonlocal offset
            val = struct.unpack_from('<f', data, offset)[0]
            offset += 4
            return val

        def read_tensor_1d(length: int) -> torch.Tensor:
            vals = [read_float() for _ in range(length)]
            return torch.tensor(vals, dtype=torch.float32)

        def read_tensor_2d(rows: int, cols: int) -> torch.Tensor:
            vals = [[read_float() for _ in range(cols)] for _ in range(rows)]
            return torch.tensor(vals, dtype=torch.float32)

        # Header
        magic = read_uint32()
        if magic != 0x47414231:
            raise ValueError('Invalid GabGPT file format')

        version = read_uint32()
        if version != 1:
            raise ValueError(f'Unsupported GabGPT version: {version}')

        vocab_size = read_uint32()
        embedding_dim = read_uint32()
        num_heads = read_uint32()
        num_blocks = read_uint32()
        max_seq_length = read_uint32()

        hidden_dim = embedding_dim * 4

        # Create model
        model = GabGPT(vocab_size, embedding_dim, num_heads, num_blocks, max_seq_length)

        # Load token embeddings
        model.token_embedding.weight.data = read_tensor_2d(vocab_size, embedding_dim)

        # Load position embeddings
        model.position_embedding.weight.data = read_tensor_2d(max_seq_length, embedding_dim)

        # Load transformer blocks
        for block in model.blocks:
            # layerNorm1
            block.layer_norm1.weight.data = read_tensor_1d(embedding_dim)
            block.layer_norm1.bias.data = read_tensor_1d(embedding_dim)

            # attention weights (transpose back)
            block.attention.query_weights.weight.data = read_tensor_2d(embedding_dim, embedding_dim).T
            block.attention.key_weights.weight.data = read_tensor_2d(embedding_dim, embedding_dim).T
            block.attention.value_weights.weight.data = read_tensor_2d(embedding_dim, embedding_dim).T
            block.attention.output_weights.weight.data = read_tensor_2d(embedding_dim, embedding_dim).T

            # layerNorm2
            block.layer_norm2.weight.data = read_tensor_1d(embedding_dim)
            block.layer_norm2.bias.data = read_tensor_1d(embedding_dim)

            # mlp.dense1
            dense1_weights = []
            dense1_bias = []
            for n in range(hidden_dim):
                weights = [read_float() for _ in range(embedding_dim)]
                dense1_weights.append(weights)
                dense1_bias.append(read_float())
            block.mlp.dense1.weight.data = torch.tensor(dense1_weights, dtype=torch.float32)
            block.mlp.dense1.bias.data = torch.tensor(dense1_bias, dtype=torch.float32)

            # mlp.dense2
            dense2_weights = []
            dense2_bias = []
            for n in range(embedding_dim):
                weights = [read_float() for _ in range(hidden_dim)]
                dense2_weights.append(weights)
                dense2_bias.append(read_float())
            block.mlp.dense2.weight.data = torch.tensor(dense2_weights, dtype=torch.float32)
            block.mlp.dense2.bias.data = torch.tensor(dense2_bias, dtype=torch.float32)

        # finalNorm
        model.final_norm.weight.data = read_tensor_1d(embedding_dim)
        model.final_norm.bias.data = read_tensor_1d(embedding_dim)

        # output layer (transpose back)
        model.output.weight.data = read_tensor_2d(embedding_dim, vocab_size).T
        model.output.bias.data = read_tensor_1d(vocab_size)

        return model.to(device)

    # ========================================================================
    # .pt CHECKPOINT (full training state)
    # ========================================================================

    def save_checkpoint(self, path: str, optimizer: torch.optim.Optimizer,
                        step: int, loss: float = 0.0, **kwargs):
        """Save full training checkpoint"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step,
            'loss': loss,
            'config': {
                'vocab_size': self.vocab_size,
                'embedding_dim': self.embedding_dim,
                'num_heads': self.num_heads,
                'num_blocks': self.num_blocks,
                'max_seq_length': self.max_seq_length,
            },
            **kwargs
        }
        torch.save(checkpoint, path)

    @staticmethod
    def load_checkpoint(path: str, device: str = 'cpu',
                        learning_rate: float = 0.001) -> Tuple['GabGPT', torch.optim.Optimizer, int, float]:
        """
        Load full training checkpoint

        Returns:
            model, optimizer, step, loss
        """
        checkpoint = torch.load(path, map_location=device)

        config = checkpoint['config']
        model = GabGPT(
            vocab_size=config['vocab_size'],
            embedding_dim=config['embedding_dim'],
            num_heads=config['num_heads'],
            num_blocks=config['num_blocks'],
            max_seq_length=config['max_seq_length']
        ).to(device)

        model.load_state_dict(checkpoint['model_state_dict'])

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return model, optimizer, checkpoint['step'], checkpoint.get('loss', 0.0)


# ============================================================================
# SFT DATASET
# ============================================================================

class SFTDataset(Dataset):
    """
    Dataset for Supervised Fine-Tuning.

    Each item contains:
    - input_tokens: token IDs for input
    - target_tokens: token IDs for target (shifted by 1)
    - attention_mask: 1 for real tokens, 0 for padding
    - loss_mask: 1 for tokens to train on (assistant response), 0 for prompt/padding
    """

    def __init__(self, examples: List[dict]):
        """
        Args:
            examples: List of dicts with keys:
                - 'input_ids': List[int] - token IDs
                - 'loss_mask': List[int] - 1 for response tokens, 0 for prompt
        """
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        example = self.examples[idx]
        input_ids = example['input_ids']
        loss_mask = example['loss_mask']

        # Input is all tokens except last, target is all tokens except first
        input_tokens = torch.tensor(input_ids[:-1], dtype=torch.long)
        target_tokens = torch.tensor(input_ids[1:], dtype=torch.long)
        # Loss mask also shifts - we predict token i+1 from token i
        loss_mask_tensor = torch.tensor(loss_mask[1:], dtype=torch.long)

        return {
            'input_tokens': input_tokens,
            'target_tokens': target_tokens,
            'loss_mask': loss_mask_tensor,
        }


def sft_collate_fn(batch: List[dict], pad_token_id: int) -> dict:
    """
    Collate function that pads sequences to the same length.

    Args:
        batch: List of dicts from SFTDataset
        pad_token_id: Token ID to use for padding

    Returns:
        Dict with batched and padded tensors
    """
    # Find max length in batch
    max_len = max(item['input_tokens'].size(0) for item in batch)

    batch_size = len(batch)

    # Initialize tensors
    input_tokens = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    target_tokens = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    loss_mask = torch.zeros(batch_size, max_len, dtype=torch.long)

    # Fill in actual values
    for i, item in enumerate(batch):
        seq_len = item['input_tokens'].size(0)
        input_tokens[i, :seq_len] = item['input_tokens']
        target_tokens[i, :seq_len] = item['target_tokens']
        attention_mask[i, :seq_len] = 1
        loss_mask[i, :seq_len] = item['loss_mask']

    return {
        'input_tokens': input_tokens,
        'target_tokens': target_tokens,
        'attention_mask': attention_mask,
        'loss_mask': loss_mask,
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_learning_rate(step: int, warmup_steps: int, total_steps: int, max_lr: float) -> float:
    """Cosine learning rate schedule with warmup"""
    if step < warmup_steps:
        return max_lr * (step / warmup_steps)

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max_lr * 0.5 * (1 + math.cos(math.pi * progress))


if __name__ == '__main__':
    print("gab.py - GabGPT model for SFT")
    print("Used by app.py for fine-tuning. See app.py for usage.")
