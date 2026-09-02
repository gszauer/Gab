"""model_config.py — the one place the model shape is defined.

Used as the CLI defaults by pretrainer.py and as the trusted shape by
inference.py (a .weights file is a headerless flat float32 buffer — the
MiniGPT.js-compatible format — so it carries no config of its own).

The finetuner and the debug tools don't read this: they take a .checkpoint,
which stores its own config, so they always match the model they load.

IMPORTANT: these values must describe the model the .weights files were
trained with. If you train a differently-shaped model, change them here (or
pass explicit flags to the pretrainer).
"""

FEATURE_DIM = 768       # -f: embedding width
NUM_HEADS = 12          # -n: attention heads (head_dim = 768/12 = 64)
ROPE_BASE = 10000       # -r: RoPE theta base
NUM_BLOCKS = 13         # -b: transformer blocks
CONTEXT_LENGTH = 2048   # -c: training/attention window, in tokens
