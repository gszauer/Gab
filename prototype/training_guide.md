## Quick Reference

| Setting | What It Controls | Typical Range |
|---------|-----------------|---------------|
| **Tokenizer** | | |
| Number of Merges | Vocabulary size | 500 - 4000 |
| **Architecture** | | |
| Embedding Dimension | Richness of word meanings | 64 - 1536 |
| Attention Heads | Different "perspectives" on text | 4 - 16 |
| Transformer Blocks | Depth of understanding | 2 - 24 |
| Max Sequence Length | Memory window size | 128 - 1024 |
| **Training** | | |
| Epochs | Training passes through data | 5 - 100+ |
| Batch Size | Examples processed together | 1 - 16 |
| Learning Rate | Speed of learning | 0.0001 - 0.001 |
| Warmup Ratio | Gentle start to training | 0.05 - 0.2 |
| Use Adam Optimizer | Smart adaptive learning | On/Off. Ignores Learning Rate & warmup if on |

---

## Part 1: Foundations

### What is a Language Model?

A language model is a **next-word predictor**. When you type "The cat sat on the", the model predicts that "mat" is a likely next word. It does this repeatedly, one word at a time, building complete responses.

```
You type:        "What is the capital of France?"
Model predicts:  "What is the capital of France? The"
Model predicts:  "What is the capital of France? The answer"
Model predicts:  "What is the capital of France? The answer is"
Model predicts:  "What is the capital of France? The answer is Paris"
Model predicts:  "What is the capital of France? The answer is Paris."
```

The model doesn't "know" things the way humans do. It has learned patterns from text—it has seen "capital of France" followed by "Paris" so many times that this pattern is burned into its memory.

### How Does Training Work?

Training is how the model learns patterns. It "learns" by scanning the text from start to end one word at a time, and tries to guess what the next word is. During training it might be given a sentance like "The cat sat on the mat", training would:

1. The model sees text like "The cat sat on the"
2. It guesses the next word (maybe "floor")
3. It checks the real answer ("mat")
4. It adjusts its internal numbers to make "mat" more likely next time

Repeat this billions of times, and you get a model that can write convincingly.

### Understanding Loss

**Loss** is your primary feedback signal during training. It measures how wrong the model's predictions are, in "nats" (natural log units).

```
Model predicts next word probabilities:
  "cat": 10%
  "floor": 60%  ← model's best guess
  "mat": 5%
  "hat": 25%

Actual next word: "mat"

Loss is HIGH because the model was confident about "floor" but the answer was "mat"
```

| Loss | What It Means |
|------|---------------|
| 8-10 | Random guessing. Model hasn't learned anything yet |
| 5-7 | Basic patterns emerging. Common words recognized |
| 3-4 | Decent understanding. Grammar mostly correct |
| 2-3 | Good model. Coherent sentences, understands context |
| 1-2 | Strong model. Nuanced understanding |
| <1 | Exceptional (or possibly memorized!) |

**What to watch for:**
- Loss decreasing steadily = good
- Loss stuck high = learning rate too low or model too small
- Loss jumping wildly = learning rate too high
- Loss going back up = memorization (try stopping earlier)

---

## Part 2: The Tokenizer

Before the model can read text, it needs to convert words into numbers. The **tokenizer** handles this conversion. A token is a chunk of text that the model treats as a single unit. Tokens aren't always complete words—they can be sub-word pieces:

```
"Hello world" → ["Hello", " world"]    (notice the space is included!)

"unhappiness" → ["un", "happiness"]

"ChatGPT" → ["Chat", "G", "PT"]
```

Why break words apart? Because language has patterns at different levels. The prefix "un-" means "not" in many words (unhappy, unclear, unfair). By learning "un" as a separate token, the model can apply this knowledge across many words.

ChatGPT and The Gab tokenizer uses **Byte Pair Encoding** (BPE). Here's how it works:

1. **Start with individual characters.** Every letter and symbol is its own token. Reserve 0-255 for ASCII/UTF8.
2. **Count pairs.** Scan the text and count how often each pair of adjacent tokens appears.
3. **Merge the most common pair.** Combine the most frequent pair into a new token.
4. **Repeat.** Keep merging until you've done it a set number of times.

Example:

```
Training text: "the cat and the dog and the bird"

Start: Every character is its own token
  |t|h|e| |c|a|t| |a|n|d| |t|h|e| |d|o|g| |a|n|d| |t|h|e| |b|i|r|d|

Merge 1: "t"+"h" appears 3 times → create token "th"
  |th|e| |c|a|t| |a|n|d| |th|e| |d|o|g| |a|n|d| |th|e| |b|i|r|d|

Merge 2: "th"+"e" appears 3 times → create token "the"
  |the| |c|a|t| |a|n|d| |the| |d|o|g| |a|n|d| |the| |b|i|r|d|

Merge 3: "the"+" " appears 3 times → create token "the "
  |the |c|a|t| |a|n|d| |the |d|o|g| |a|n|d| |the |b|i|r|d|

...after more merges, common words become single tokens!
```

Notice how "the" quickly became one token because it appeared so often. Rare words like "bird" stay as individual characters longer.

### [CONFIG OPTION] Number of Merges

This controls how many times the merging process runs.

- **More merges = Larger vocabulary.** Common words become single tokens.
- **Fewer merges = Smaller vocabulary.** More words get split into pieces.

| Merges | Best For |
|--------|----------|
| 500 | Small experiments, limited text |
| 1,500 | Medium projects |
| 5,000 | Larger projects with diverse text |
| 50,000 | GPT-2/3 scale |
| 100,000 | GPT-4 scale |

### [CONFIG OPTION] Reserved Tokens

You may want to reserve special tokens for conversation structure. These are common tokens that are used by ChatGPT to format a conversation. We will use them later:

| Token | Purpose |
|-------|---------|
| `<\|user\|>` | Marks the start of a user message |
| `<\|assistant\|>` | Marks the start of the AI's response |
| `<\|end\|>` | Marks the end of a message |
| `<\|pad\|>` | Padding for batching (internal use) |
| `<\|system\|>` | System instructions (optional) |
| `<\|think\|>` | Start of "thinking" section (optional) |

You can also reserve common suffixes like `ing ` (with trailing space) since words often end this way. You could also reserve ` un`. Notice the space is in front this time.

---

## Part 3: Model Architecture

These settings define the "shape" of your model, how big it is and how it processes information.

### [CONFIG OPTION] Embedding Dimension

Every token is represented as a list of numbers. The **embedding dimension** is how long that list is.

```
If embedding dimension is 64:
  "cat" = [0.12, -0.34, 0.56, ... (64 numbers total)]
  "dog" = [0.15, -0.29, 0.61, ... (64 numbers total)]
```

These numbers capture meaning. Similar words have similar numbers. The model learns during training that "cat" and "kitten" should have similar number patterns. Think of it like every number in the embedding dimension is one fact about the word that the AI can learn. The larger the dimension, the more neuance words have.


| Size | Description |
|------|-------------|
| Small (64) | Basic meaning. "Cat" and "dog" are both "animals" |
| Medium (512) | Richer meaning. Can distinguish "cat" from "kitten" |
| Large (1536+) | Nuanced meaning. Understands "tabby cat" vs "Siamese cat" |


### [CONFIG OPTION] Attention Heads

**Attention** is how the model connects words to each other. When you read "The cat sat on the mat because **it** was tired", how do you know "it" refers to "the cat"? You pay attention to the context. The model does the same thing mathematically.

Each **attention head** looks at the text from a different angle:
- Head 1 might focus on subjects and verbs ("cat" → "sat")
- Head 2 might focus on pronouns and their references ("it" → "cat")
- Head 3 might focus on describing words ("tired" → "cat")

| Heads | Use Case |
|-------|----------|
| 4 | Simple text, basic relationships |
| 8 | Standard choice for most projects |
| 16+ | Complex text with many relationships |

**Important:** The embedding dimension must be divisible by the number of heads. If embedding is 64 and you have 4 heads, each head works with 16 dimensions (64 ÷ 4 = 16).

**Rule of thumb:** embedding dimension ÷ attention heads = 64 (or at least 32-128).

### [CONFIG OPTION] Transformer Blocks

A **transformer block** is one complete round of processing (thinking), ie how many times does the AI look at a sentance as it's learning. Stacking multiple blocks is like re-reading a sentence multiple times:

- Block 1: Basic understanding ("there's a cat, there's a mat")
- Block 2: Relationships ("the cat is ON the mat")
- Block 3: Deeper meaning ("this is describing a scene")
- Block 4+: Nuance and context

| Blocks | Capability |
|--------|------------|
| 2-4 | Simple patterns, short responses |
| 6 | Good for most conversational tasks |
| 12+ | Complex reasoning, longer context |

Deeper networks are smarter than wider ones for the same number of parameters. Each transformer block is a "round of thinking", more blocks means more reasoning steps.

```
Same ~20M parameters, different architectures:

Wide (less capable):        Deep (more capable):
  Embedding: 1024             Embedding: 512
  Blocks: 3                   Blocks: 6
  Heads: 8                    Heads: 8
```

### [CONFIG OPTION] Max Sequence Length

This is the maximum number of tokens the model can process. The model's "memory window."

A rough rule: 1 token ≈ 0.75 words ≈ 4 characters. So:
- 128 tokens ≈ 96 words ≈ 512 characters
- 512 tokens ≈ 380 words ≈ 2,000 characters
- 1024 tokens ≈ 750 words ≈ 4,000 characters

Attention compares every token to every other token. With 128 tokens, that's 16,384 comparisons. With 512 tokens, it's 262,144 comparisons. You have to balance sequence length with the available memory of your computer.

| Length | Typical Use |
|--------|-------------|
| 512 | Short responses, simple Q&A |
| 2048 | Full conversations, longer context |
| 4096 | Extended discussions, complex tasks |

---

## Part 4: Training Parameters

These settings control the learning process, how fast the model learns and how it processes training data.

### [CONFIG OPTION] Epochs

An **epoch** is one complete pass through your training data. If you have 1000 examples and run 10 epochs, the model sees each example 10 times.

> When training on text, every word is an example. "The cat sat" has two examples: Given "the" predict "cat"; given "The cat" predict "sat".

```
Epoch 1: Model sees all data (makes lots of mistakes)
Epoch 2: Model sees all data again (fewer mistakes)
Epoch 3: Model sees all data again (getting better)
...and so on
```

| Epochs | Situation |
|--------|-----------|
| 1-10 | Large diverse dataset |
| 10-100 | Medium dataset |
| 100+ | Small dataset |

The model needs to see lots of tokens. If your training data doesn't have enough, you need more epochs. For example, the Bee Movie script might need ~150 epochs.

### [CONFIG OPTION] Batch Size

Instead of learning from one example at a time, the model learns from a **batch** of examples together. This isn't predicting more words at once—it's predicting multiple sentences in parallel. Example with batch size 4:
- sentance one is "Dog is an animal"
- sentance two is "Cat is an animal"
- sentance three is "cat is a pet"
- sentance four is "animals are all pets"

The model learns to predict these four at the same time. If we look at how predicting a batch of four works, we can get some intuition as to how AI can learn.

```
Dog -> is
Cat -> is
Cat -> is
Animals -> are
```


```
Dog is -> an
Cat is -> an
Cat is -> a
Animals are -> all
```

```
Dog is an -> animal
Cat is an -> animal
Cat is a -> pet
Animals are all -> pets
```

You can see how there are patterns to be found by considering all four training examples at once.  With batching we learn ritcher information. The learning is more stable (one weird example doesn't throw everything off). And because work is batched, training is faster.

| Batch Size | Tradeoff |
|------------|----------|
| 1-2 | Noisy updates, but works with limited memory |
| 4-8 | Good balance for most cases |
| 16+ | Smoother learning, needs more memory |

Memory is the limiting factor. Start with a small batch, check memory usage, and adjust.

### [CONFIG OPTION] Sequence Length (Training)

This is how many tokens you train on at once during each training step—different from Max Sequence Length in architecture.

```
Book: "Once upon a time there was a princess who lived in a castle..."

With sequence length 8:
  Chunk 1: "Once upon a time there was a princess"
  Chunk 2: "who lived in a castle the castle was"
  ...
```

**Should match or be smaller than Max Sequence Length.** You can't train on 512-token chunks if your model only supports 256 tokens.

> In Gab, text is converted to tokens, then grouped into chunks (based on sequence length). Chunks are automatically grouped into batches.

### [CONFIG OPTION] Learning Rate

The most important training parameter. The **learning rate** controls how much the model adjusts after each batch.

Imagine learning to throw darts:
- **High learning rate:** Dramatic adjustments after each throw. You might overshoot and never settle.
- **Low learning rate:** Tiny adjustments. You'll eventually hit the target, but it takes forever.
- **Just right:** Sensible adjustments, steady improvement.

```
Learning rate too high:
  Loss: 5.2 → 3.1 → 7.8 → 4.2 → 9.1  (jumping around)

Learning rate too low:
  Loss: 5.2 → 5.1 → 5.0 → 4.99 → 4.98  (painfully slow)

Learning rate just right:
  Loss: 5.2 → 4.1 → 3.2 → 2.5 → 2.0  (steady improvement)
```

| Model Size | Learning Rate |
|------------|---------------|
| Small (~300K params) | 0.001 (1e-3) |
| Medium (~20M params) | 0.0003 (3e-4) |
| Large (~1B params) | 0.0001 (1e-4) |

**Rule of thumb:** Bigger models need smaller learning rates.

###  [CONFIG OPTION] Warmup Ratio

Training starts with a **warmup period** where the learning rate gradually increases from zero to the target value.

If warmup ratio is 0.1 (10%) and you train for 1000 steps:
- Steps 1-100: Learning rate slowly rises from 0 to target
- Steps 101-1000: Learning rate at full strength

**Why warmup?** At the start, the model's weights are random. Large updates could send them in wild directions. Warming up lets the model find its footing first. **Recommended:** 0.1 (10%) is a good default. More warmup for larger models.

###  [CONFIG OPTION]  Use Adam Optimizer

**SGD (Stochastic Gradient Descent)** is the basic learning algorithm—it uses Learning Rate and Warmup Ratio and updates all parts of the model at the same rate.

**Adam** (Adaptive Moment Estimation) is smarter. It adapts the learning rate for each parameter individually. If one part of the model needs bigger updates, Adam gives it bigger updates.

When Adam is enabled, Learning Rate and Warmup Ratio are ignored—Adam figures out the right values after just a few hundred examples.

**Recommended:** Use Adam. Disable only for a more artisanal experience.

---

## Part 5: Training Modes

### Pre-Training

**What it is:** Teaching the model language from raw text.

Feed the model books, articles, websites—any text. The model learns to predict the next word. Through this, it learns grammar, facts, writing styles, and reasoning patterns. The data is unformatted, just big blobs of text. We're teaching the model only to predict the next word.

After pre-training the model is an auto-compleate or text regressor. You can give it a partial text, and the model will do its best to finish it for you. But it does not understand conversations.

**When to use:** Starting from scratch, building a foundation model.

**Data format:** Plain text. The model learns by predicting the next word.

### Supervised Fine-Tuning (SFT)

**What it is:** Teaching the model to have conversations.

A pre-trained model knows language but doesn't know how to chat. SFT shows it examples of good conversations. It uses special tokens like <|user|> and <|assistant|> to delimit parts of text. If shown enough of these, the model will figure out that <|user|> text is provided, and how the <|assistant|> should sound when responding.

**When to use:** After pre-training, to create a chatbot.

**Data format:** Conversations in ChatML format:

```
<|user|>What is the capital of France?<|end|>
<|assistant|>The capital of France is Paris.<|end|>

<|user|>Tell me a joke<|end|>
<|assistant|>Why don't scientists trust atoms? Because they make up everything!<|end|>
```

The model learns that after `<|user|>question<|end|><|assistant|>`, it should generate a helpful answer.

### Other Fine-Tuning Tasks

Fine-tuning isn't just for chatbots. You can use fine tuning to teach the AI anything. What special tokens are, and how they are interpreted is up to you:

**Summarization:**
```
<|original|>Summarize this article: [long article text]<|end|>
<|summary|>The article discusses three main points...<|end|>
```

**Translation:**
```
<|english|>Translate to French: Hello, how are you?<|end|>
<|french|>Bonjour, comment allez-vous?<|end|>
```

**Code Generation:**
```
<|user|>Write a Python function to calculate factorial<|end|>
<|coder|>def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)<|end|>
```

**Reasoning with Thinking:**

When you see a "thinking" model, it just generates a special <|think|> tag with extra text that is trained to be factual and step by step. Because the AI always predicts the next word, if it generates a think tag, the assistant tag will be influenced by the words of the think tag.

```
<|user|>If a train travels 60 mph for 2.5 hours, how far does it go?<|end|>
<|think|>Distance = speed × time. Speed is 60 mph, time is 2.5 hours. 60 × 2.5 = 150 miles.<|/think|>
<|assistant|>The train travels 150 miles.<|end|>
```
