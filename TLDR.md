# Building GPT From Scratch: A Complete Guide

This tutorial ia a companion guide for [The gift of gab](https://giftofgab.chat/), it teaches you to build a GPT-style language model from scratch. You'll build a working text generator first, then learn how to train it. 

## Table of Contents

### Part I: What Are We Building?
- Chapter 1: The Big Picture
- Chapter 2: Core Concepts 

### Part II: Building the Inference Engine
- Chapter 3: Tokenization
- Chapter 4: Embeddings
- Chapter 5: Attention (The Core Innovation)
- Chapter 6: The Feed-Forward Network
- Chapter 7: Layer Normalization
- Chapter 8: The Transformer Block
- Chapter 9: Assembling the Complete Model
- Chapter 10: Text Generation

### Part III: How the Model Learns
- Chapter 11: What Is Training?
- Chapter 12: Loss Functions
- Chapter 13: Gradients and Backpropagation
- Chapter 14: Implementing the Backward Pass
- Chapter 15: The Training Loop
- Chapter 16: Hyperparameters

# Part I: What Are We Building?

## Chapter 1: The Big Picture

A language model predicts the next word in a sequence.

```
Input:  "The cat sat on the"
Output: Probability distribution over all possible next words
        "mat"      → 15%
        "floor"    → 12%
        "couch"    → 8%
        "table"    → 5%
        ...
        "elephant" → 0.001%
```

When you chat with an AI, the next token depends on all previous tokens:

```
You type:        "The cat sat on the"
Model predicts:  "The cat sat on the mat"
Model predicts:  "The cat sat on the mat, "
Model predicts:  "The cat sat on the mat, and"
Model predicts:  "The cat sat on the mat, and ate"
Model predicts:  "The cat sat on the mat, and ate dinner"
```

One word at a time, always asking: "Given everything so far, what comes next?"

### The Architecture

Here's what we're building:

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT TEXT                              │
│                    "The cat sat on the"                         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         TOKENIZER                               │
│                                                                 │
│  Converts text to numbers.                                      │
│  "The" → 464, "cat" → 9246, "sat" → 3332, ...                   │
│                                                                 │
│  Output: [464, 9246, 3332, 319, 262]                            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       EMBEDDINGS                                │
│                                                                 │
│  Converts each number to a vector of 256 numbers.               │
│  Also adds position information (where in the sequence).        │
│                                                                 │
│  Output: 5 vectors, each with 256 numbers                       │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                  TRANSFORMER BLOCK #1                           │
│                                                                 │
│    ┌─────────────────────────────────────────────────────┐      │
│    │  1. Layer Normalization (stabilize values)          │      │
│    │  2. Attention (let positions look at each other)    │      │
│    │  3. Layer Normalization                             │      │
│    │  4. Feed-Forward Network (process each position)    │      │
│    └─────────────────────────────────────────────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                               ...
                (repeat for N transformer blocks)
                               ...
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FINAL LAYER NORM                              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   OUTPUT PROJECTION                             │
│                                                                 │
│  Converts 256-dim vector to vocabulary size (e.g., 5000).       │
│  These are "logits" - raw scores for each possible word.        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        SOFTMAX                                  │
│                                                                 │
│  Converts raw scores to probabilities (0 to 1, sum to 1).       │
│                                                                 │
│  [2.1, 0.5, -1.2, ...] → [0.15, 0.03, 0.006, ...]               │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              SAMPLE NEXT TOKEN                                  │
│                                                                 │
│  Pick one word based on probabilities.                          │
│  → "mat" (sampled with 15% probability)                         │
└─────────────────────────────────────────────────────────────────┘
```

We'll build each component, then connect them. By the end of Part II, you'll have a working text generator. Part III will teach you how to train it.

## Chapter 2: Core Concepts

Before we build, let's establish just enough vocabulary to understand the code. 

### Vectors and Matrices

https://gabormakesgames.com/blog_vectors.html

A **vector** is a list of numbers:

```
[0.5, -0.3, 0.8, 0.1]
```

In our model, each word is represented as a vector. A 256-dimensional vector means 256 numbers that together capture "what this word means."

https://gabormakesgames.com/blog_matrices.html

A **matrix** is a 2D grid of numbers:

```
[[0.1, 0.2, 0.3],
 [0.4, 0.5, 0.6],
 [0.7, 0.8, 0.9]]
```

Matrices store the model's learned knowledge. A matrix with 256 rows and 256 columns contains 65,536 learned numbers (parameters).

### The Dot Product

The dot product of two vectors multiplies corresponding elements and sums them:

```
[1, 2, 3] · [4, 5, 6] = 1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32
```

Why does this matter? The dot product measures **similarity**:
- Large positive: vectors point the same direction (similar)
- Near zero: vectors are perpendicular (unrelated)
- Large negative: vectors point opposite directions (dissimilar)

We convert all words into tokens. Map tokens to vectors, then compare those vectors with the dot product. The result of the "dot" product tells us how related two words are to each other. For example ```dot(cat, fur)``` could be a large positive, but ```dot(cat, soccer)``` could be a large negative. It depends on how "aligned" the words are.

### Matrix Multiplication

Matrix multiplication transforms vectors. It's how the model applies learned transformations to data. Vectors and matrices can be multiplied, if a vector is treated as a one dimensional matrix.

To multiply a vector by a matrix, we compute dot products. Each output element is the dot product of the input vector with one column of the matrix:

```
input vector: [2, 3]

matrix:  [[0.5, 0.1, 0.8],
          [0.2, 0.4, 0.3]]

Output element 0 = input · column 0 = 2×0.5 + 3×0.2 = 1.0 + 0.6 = 1.6
Output element 1 = input · column 1 = 2×0.1 + 3×0.4 = 0.2 + 1.2 = 1.4
Output element 2 = input · column 2 = 2×0.8 + 3×0.3 = 1.6 + 0.9 = 2.5

output vector: [1.6, 1.4, 2.5]
```

The input vector is treated as a matrix ```(1x2)``` and the matrix is bigger ```(2x3)```. Notice how the inner dimensions match, this is a must for matrix multiplication. Also, notice that the resulting matrixes shape is the outer dimensions ```(1x3)```

```javascript
function matrixMultiply(vector, matrix) {
    // vector: [inputDim]
    // matrix: [inputDim][outputDim]  
    // returns: [outputDim]
    
    const inputDim = vector.length;
    const outputDim = matrix[0].length;
    const result = [];
    
    for (let j = 0; j < outputDim; j++) {
        // Dot product of input vector with column j
        let sum = 0;
        for (let i = 0; i < inputDim; i++) {
            sum += vector[i] * matrix[i][j];
        }
        result[j] = sum;
    }
    
    return result;
}
```

Why is this useful? Different weight values create different transformations. A matrix can rotate, scale, project, or combine inputs in complex ways. The model learns weights that transform inputs into useful representations.

Projecting a vector from one space or another is done with matrix multiplication. When you multiply a vector by a matrix, the output dimension is determined by the matrix shape: vector [N] × matrix [N][M] = result [M]
 
* If M < N: you're projecting down (compressing)
* If M > N: you're projecting up (expanding)
* If M = N: same dimension (transforming)

### Softmax: Turning Scores into Probabilities

The model outputs raw scores called "logits" — they can be any number from negative infinity to positive infinity. We need probabilities: numbers between 0 and 1 that sum to 1. Softmax does this conversion.

Given a set of numbers, softmax exponentiates them, sums them, then divides each number by the sum. The resulting list is a list of probabilities.

```
scores:        [2.0,  1.0,  0.5]

Step 1 - Exponentiate each score:
    e^2.0 = 7.39
    e^1.0 = 2.72
    e^0.5 = 1.65

Step 2 - Sum them:
    7.39 + 2.72 + 1.65 = 11.76

Step 3 - Divide each by the sum:
    7.39 / 11.76 = 0.63 (63%)
    2.72 / 11.76 = 0.23 (23%)
    1.65 / 11.76 = 0.14 (14%)

probabilities: [0.63, 0.23, 0.14]  ← sums to 1.0
```

Notice how the highest score (2.0) gets 63% of the probability mass, even though it's only 2× the lowest score (0.5 vs 1.0). Exponentiation makes the winner "win bigger."

**Why exponentiate?**

We need a function that:
1. Makes all values positive (probabilities can't be negative)
2. Preserves ranking (higher score = higher probability)
3. Makes outputs sum to 1

Exponentiation (e^x) solves requirements 1 and 2:
- e^x is always positive, even for negative x
- e^x is monotonically increasing: if a > b, then e^a > e^b
- e^x amplifies differences: large scores become MUCH larger than small scores

Then we divide by the sum to satisfy requirement 3.

**The numerical stability trick:**

With large scores, e^x can overflow (become infinity). We subtract the maximum score first:

```
scores: [1000, 999, 998]

Naive: e^1000 = overflow!

Better: subtract max first
    [1000-1000, 999-1000, 998-1000] = [0, -1, -2]
    e^0 = 1.0
    e^-1 = 0.37
    e^-2 = 0.14
    
    Normalize: [0.66, 0.24, 0.09]
```

This gives the same answer (the subtraction cancels out in the division) but avoids overflow.

```javascript
function softmax(scores) {
    // Find max for numerical stability
    let max = scores[0];
    for (let i = 1; i < scores.length; i++) {
        if (scores[i] > max) max = scores[i];
    }
    
    // Exponentiate (subtract max to prevent overflow)
    const exps = [];
    let sum = 0;
    for (let i = 0; i < scores.length; i++) {
        const e = Math.exp(scores[i] - max);
        exps.push(e);
        sum += e;
    }
    
    // Normalize (divide by sum so they sum to 1)
    const probs = [];
    for (let i = 0; i < exps.length; i++) {
        probs.push(exps[i] / sum);
    }
    
    return probs;
}
```

### Activation Functions

After matrix multiplication, we apply a non-linear function. Without this, stacking layers would be pointless (two linear transforms = one linear transform). Adding non-liniarity to the mix is how the LLM learns.

GPT uses **GELU** (Gaussian Error Linear Unit). Key properties:

- For large positive inputs: output ≈ input (passes through unchanged)
- For large negative inputs: output ≈ 0 (blocked)
- Smooth transition around 0
- Slight dip below 0 for small negative inputs (minimum ≈ -0.17 at x ≈ -0.75)

This smooth curve is important for training. Unlike ReLU (which has a sharp corner at 0), GELU is differentiable everywhere, which helps gradient flow.

```javascript
function gelu(x) {
    const c = Math.sqrt(2.0 / Math.PI);
    return 0.5 * x * (1.0 + Math.tanh(c * (x + 0.044715 * x * x * x)));
}
```

Why non-linearity matters: A linear function can only draw straight decision boundaries. Non-linear activations let the model learn curved, complex patterns. Stacking linear + non-linear layers builds up increasingly sophisticated representations.

That's all the math you need. Let's build!


# Part II: Building the Inference Engine

## Chapter 3: Tokenization

The tokenizer converts text to numbers and back. This is the first and last step of the pipeline—text goes in, tokens go to the model, tokens come out, text goes to the user.

Unlike the neural network components that follow, the tokenizer is a complete, self-contained system. We'll cover everything here: why we tokenize, how to train a tokenizer, and how to use it.


### Why Tokenize?

Neural networks work with numbers, not characters. We need a mapping.

The simplest approach maps each character to a number:

```
"hello" → [104, 101, 108, 108, 111]  (ASCII codes)
```

This works but is inefficient. The word "the" becomes 3 tokens. A 1000-word essay becomes ~5000 tokens. Since attention computes pairwise interactions (O(n²) complexity), this gets expensive fast.

Better: group common character sequences into single tokens.

```
"hello" → [15496]  (one token!)
"the"   → [1169]   (one token!)
```

This is what Byte Pair Encoding (BPE) does. GPT-2 uses ~50,000 tokens.


### How BPE Works

BPE learns which character sequences to merge by counting frequencies in training text:

1. Start: vocabulary = all 256 possible bytes
2. Count all adjacent pairs in training text
3. Find the most frequent pair (e.g., "th" appears 50,000 times)
4. Create a new token for that pair: "th" → token 256
5. Replace all occurrences in training text
6. Repeat steps 2-5 for N iterations

After training, common words are single tokens, rare words split into pieces Let's trace through a tiny example:

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


### The Tokenizer Structure

The tokenizer needs two data structures:
- **vocabulary**: Maps token ID → bytes (for decoding)
- **merges**: Maps (token1, token2) → merged token (for encoding)

The order of merges matters! Earlier merges represent more frequent patterns and must be applied first during encoding. Let's start the ```Tokenizer``` implementation by reserving the first 256 tokens.

```javascript
class Tokenizer {
    constructor() {
        this.merges = new Map();      // "tok1,tok2" → newTokenId
        this.vocabulary = new Map();   // tokenId → [bytes]
        
        // Initialize with single-byte tokens (0-255)
        for (let i = 0; i < 256; i++) {
            this.vocabulary.set(i, [i]);
        }
    }
}
```

**Why start with 256 tokens?**

A byte can have values 0-255. By giving each byte value its own token, we guarantee ANY text can be tokenized—even unknown characters fall back to their raw bytes. Token 65 represents byte 65, which is ASCII 'A'. Token 32 represents byte 32, which is space. And so on.


### Why Bytes? Why Not Characters?

JavaScript strings are UTF-16 internally, but we convert to UTF-8 bytes. Why?

1. **UTF-8 is compact for ASCII**: English text uses 1 byte per character
2. **Universal coverage**: Any Unicode character can be represented (1-4 bytes each)
3. **Fixed base vocabulary**: Always exactly 256 possible byte values

```javascript
// Convert string to UTF-8 bytes
stringToBytes(text) {
    // TextEncoder converts JavaScript's UTF-16 string to UTF-8 bytes
    const encoder = new TextEncoder();
    const uint8Array = encoder.encode(text);
    
    // Convert typed array to regular array for easier manipulation
    const bytes = [];
    for (let i = 0; i < uint8Array.length; i++) {
        bytes.push(uint8Array[i]);
    }
    return bytes;
}

// Convert UTF-8 bytes back to string
bytesToString(bytes) {
    // TextDecoder reverses the process
    return new TextDecoder().decode(new Uint8Array(bytes));
}
```

Example:
```
"Hello" → [72, 101, 108, 108, 111]     (5 bytes, ASCII)
"你好"  → [228, 189, 160, 229, 165, 189] (6 bytes, UTF-8 Chinese)
```


### Training the Tokenizer

Training finds the most frequent pairs and creates merge rules:

```javascript
train(text, numMerges) {
    // Convert text to byte tokens
    let tokens = this.stringToBytes(text);
    
    console.log("Training BPE on " + tokens.length + " bytes");
    console.log("Target vocabulary size: " + (256 + numMerges));
    
    for (let i = 0; i < numMerges; i++) {
        // Find the most frequent adjacent pair
        const pair = this.findMostFrequentPair(tokens);
        
        if (pair === null) {
            console.log("No more pairs to merge at iteration " + i);
            break;
        }
        
        // Create a new token for this pair
        const newToken = this.createMerge(pair[0], pair[1]);
        
        // Replace all occurrences of the pair with the new token
        tokens = this.applyMerge(tokens, pair[0], pair[1], newToken);
        
        if ((i + 1) % 100 === 0) {
            console.log("Merge " + (i + 1) + "/" + numMerges + 
                " | Vocab size: " + this.vocabulary.size +
                " | Tokens remaining: " + tokens.length);
        }
    }
    
    console.log("Training complete. Final vocabulary size: " + this.vocabulary.size);
}
```

**Finding the most frequent pair:**

We scan through all adjacent token pairs and count how often each one appears. We use a string key like `"97,98"` to represent the pair (token 97, token 98).

> Why a comma-separated string? JavaScript Map keys need to be comparable. Two arrays `[97, 98]` and `[97, 98]` are different objects, so they wouldn't match as Map keys. But the strings `"97,98"` and `"97,98"` are equal.

```javascript
findMostFrequentPair(tokens) {
    const pairCounts = new Map();
    
    // Count all adjacent pairs
    // tokens = [72, 101, 108, 108, 111] for "Hello"
    // pairs: (72,101), (101,108), (108,108), (108,111)
    for (let i = 0; i < tokens.length - 1; i++) {
        const key = tokens[i] + "," + tokens[i + 1];  // e.g., "108,108"
        const count = pairCounts.get(key) || 0;
        pairCounts.set(key, count + 1);
    }
    
    // Find the most frequent pair
    let bestPair = null;
    let bestCount = 1;  // Must appear more than once to be worth merging
    
    for (const [key, count] of pairCounts) {
        if (count > bestCount) {
            bestCount = count;
            bestPair = key;
        }
    }
    
    if (bestPair === null) {
        return null;  // No pair appears more than once
    }
    
    // Parse "tok1,tok2" back into [tok1, tok2]
    const parts = bestPair.split(",");
    return [parseInt(parts[0]), parseInt(parts[1])];
}
```

**Creating a merge rule:**

When we find a frequent pair, we create a new token for it. We record:
1. The merge rule: (tok1, tok2) → newToken
2. The vocabulary entry: newToken → concatenated bytes

```javascript
createMerge(tok1, tok2) {
    const key = tok1 + "," + tok2;
    
    // Check if merge already exists
    if (this.merges.has(key)) {
        return this.merges.get(key);
    }
    
    // Assign the next available token ID
    const newToken = this.vocabulary.size;  // e.g., 256 for first merge
    
    // Record the merge rule: "97,98" → 256
    this.merges.set(key, newToken);
    
    // Record what bytes this token represents
    // If tok1=97 represents [97] and tok2=98 represents [98]
    // Then newToken represents [97, 98]
    const bytes1 = this.vocabulary.get(tok1);
    const bytes2 = this.vocabulary.get(tok2);
    this.vocabulary.set(newToken, [...bytes1, ...bytes2]);
    
    return newToken;
}
```

**Applying a merge to the token sequence:**

Once we have a merge rule, we scan through the token sequence and replace every occurrence of the pair with the new merged token.

```javascript
applyMerge(tokens, tok1, tok2, newToken) {
    const result = [];
    let i = 0;
    
    // Scan through tokens
    while (i < tokens.length) {
        // Check if current position starts the pair we're looking for
        if (i < tokens.length - 1 && 
            tokens[i] === tok1 && 
            tokens[i + 1] === tok2) {
            // Found the pair! Replace both tokens with the merged token
            result.push(newToken);
            i += 2;  // Skip past BOTH tokens of the pair
        } else {
            // Not a match, keep the token as-is
            result.push(tokens[i]);
            i += 1;
        }
    }
    
    return result;
}
```

### Using the Trained Tokenizer: Encoding

Encoding converts text to tokens by converting the string to bytes, and applying each learned merge rule in order. The order matters, we must apply merge rules in the order they where learned.

```javascript
encode(text) {
    // Step 1: Convert string to byte tokens
    // "the" → [116, 104, 101]  (ASCII: t=116, h=104, e=101)
    let tokens = this.stringToBytes(text);
    
    // Step 2: Apply each merge rule in the order they were learned
    // Map preserves insertion order in JavaScript, which is critical!
    for (const [mergeKey, newToken] of this.merges) {
        // Parse "116,104" back to [116, 104]
        const [tok1, tok2] = mergeKey.split(",").map(Number);
        // Replace all occurrences of this pair
        tokens = this.applyMerge(tokens, tok1, tok2, newToken);
    }
    
    return tokens;
}
```

**Why must merges apply in order?**

Merges build on each other. If we learned:
1. "t" + "h" → 256
2. "256" + "e" → 257 (i.e., "th" + "e" = "the")

We must apply merge 1 first to create token 256, otherwise merge 2 would never find its input pair (256, 101).

### Using the Trained Tokenizer: Decoding

Decoding is simpler. We just look up what bytes each token represents and concatenate them:

```javascript
decode(tokens) {
    const bytes = [];
    
    // For each token, look up its bytes and append them
    for (const token of tokens) {
        const tokenBytes = this.vocabulary.get(token);
        if (tokenBytes) {
            for (const b of tokenBytes) {
                bytes.push(b);
            }
        }
    }
    
    // Convert bytes back to string
    return this.bytesToString(bytes);
}
```
Notice that decoding doesn't need merge rules at all—each token directly maps to its bytes. This is why we store the byte sequence for each token when we create it.

### Special Tokens

Chat models need markers for conversation structure:

- `<|user|>` — Start of user message
- `<|assistant|>` — Start of assistant response  
- `<|end|>` — End of a turn
- `<|pad|>` — Padding for batching

These should be reserved BEFORE training the main vocabulary, guaranteeing they become single tokens that won't be split:

```javascript
reserveToken(tokenString) {
    // First, encode using existing merges to reuse any existing tokens
    // This is important! If we already have "<|im_" as a token from
    // reserving "<|im_start|>", we should reuse it for "<|im_end|>"
    let tokens = this.encode(tokenString);
    
    // If already a single token, we're done
    if (tokens.length === 1) {
        return tokens[0];
    }
    
    // Merge remaining tokens left-to-right until we have a single token
    while (tokens.length > 1) {
        const newToken = this.createMerge(tokens[0], tokens[1]);
        tokens = [newToken, ...tokens.slice(2)];
    }
    
    return tokens[0];
}
```


### Complete Tokenizer Usage

```javascript
// Create tokenizer
const tokenizer = new Tokenizer();

// Reserve special tokens FIRST
const USER = tokenizer.reserveToken("<|user|>");
const ASSISTANT = tokenizer.reserveToken("<|assistant|>");
const END = tokenizer.reserveToken("<|end|>");
const PAD = tokenizer.reserveToken("<|pad|>");

// Train on your text corpus
const trainingText = loadTextFile("training_data.txt");
tokenizer.train(trainingText, 5000);  // Learn 5000 merges

// Now use it
const tokens = tokenizer.encode("Hello, world!");
console.log(tokens);  // [15496, 11, 995, 0]

const text = tokenizer.decode(tokens);
console.log(text);  // "Hello, world!"

// Vocabulary size
console.log(tokenizer.vocabulary.size);  // 256 + 4 special + 5000 merges = 5260
```

## Chapter 4: Embeddings

The tokenizer gives us integers like [464, 9246, 3332]. But integers are arbitrary—9246 doesn't "mean" anything more than 9245. We need richer representations.

Embeddings convert token IDs to vectors, and those vectors can capture meaning. Embedding dimension refers to how big these vectors are. The larger the embedding dimension the more meaning each token has.

### Token Embeddings

An embedding is a lookup table: each token ID maps to a vector.

```
Token 0    → [0.12, -0.34, 0.56, ...]    (256 numbers)
Token 1    → [0.45, 0.23, -0.12, ...]
Token 2    → [-0.33, 0.67, 0.89, ...]
...
Token 9246 → [0.78, -0.45, 0.11, ...]    ("cat")
...
```

In code, it's a 2D array with shape [vocabSize][embeddingDim]:

```javascript
class TokenEmbedding {
    constructor(vocabSize, embeddingDim) {
        this.weights = [];
        
        // Initialize with small random values
        for (let v = 0; v < vocabSize; v++) {
            this.weights[v] = [];
            for (let d = 0; d < embeddingDim; d++) {
                this.weights[v][d] = (Math.random() - 0.5) * 0.02;
            }
        }
    }
    
    forward(tokenIds) {
        // tokenIds: [seqLen] array of integers
        // returns: [seqLen][embeddingDim] array of vectors
        
        const output = [];
        for (let t = 0; t < tokenIds.length; t++) {
            const id = tokenIds[t];
            output[t] = this.weights[id].slice();  // Copy the row
        }
        return output;
    }
}
```

After training, similar words will have similar vectors:

```
"cat" → [0.8, -0.2, 0.5, ...]
"dog" → [0.7, -0.1, 0.6, ...]   (nearby!)
"car" → [-0.3, 0.9, 0.1, ...]   (far away)
```

The model learns these representations from data. Words that appear in similar contexts end up with similar vectors.

### Positional Embeddings

There's a problem: attention (coming next) doesn't inherently know about order. It sees tokens as a SET, not a SEQUENCE. To attention, "dog bites man" and "man bites dog" would look identical!

Positional embeddings add order information. Each position (0, 1, 2, ...) gets its own vector, added to the token embedding. This makes each token unique to it's position:

```
Position 0 → [0.1, 0.2, -0.1, ...]
Position 1 → [0.3, -0.1, 0.2, ...]
Position 2 → [0.2, 0.3, 0.0, ...]
```

```javascript
class PositionalEmbedding {
    constructor(maxSeqLen, embeddingDim) {
        this.weights = [];
        
        for (let pos = 0; pos < maxSeqLen; pos++) {
            this.weights[pos] = [];
            for (let d = 0; d < embeddingDim; d++) {
                this.weights[pos][d] = (Math.random() - 0.5) * 0.02;
            }
        }
    }
    
    forward(seqLen) {
        // returns: [seqLen][embeddingDim]
        const output = [];
        for (let pos = 0; pos < seqLen; pos++) {
            output[pos] = this.weights[pos].slice();
        }
        return output;
    }
}
```

### Combining Embeddings

The input to the transformer is token embedding + position embedding:

```javascript
function combineEmbeddings(tokenEmb, posEmb) {
    // Both: [seqLen][embeddingDim]
    const seqLen = tokenEmb.length;
    const dim = tokenEmb[0].length;
    
    const combined = [];
    for (let t = 0; t < seqLen; t++) {
        combined[t] = [];
        for (let d = 0; d < dim; d++) {
            combined[t][d] = tokenEmb[t][d] + posEmb[t][d];
        }
    }
    return combined;
}
```


## Chapter 5: Attention (The Core Innovation)

Attention is the breakthrough that makes transformers work. It lets each position in a sequence "look at" all other positions and decide which ones are relevant. This chapter explains attention in depth. Take your time here—it's the heart of the model.

### The Problem Attention Solves

Consider: "The cat sat on the ___"

To predict the blank, we need to know:
- There's a cat (might sit on cat-related things)
- The action is "sat" (needs a surface)
- Phrase structure suggests a location noun

Without attention, each position processes in isolation. Position 5 ("the") would have no direct information about positions 0-4. Each position would only know about itself—its own embedding vector. To share information between positions, you'd need some other mechanism, like passing a single summary vector between layers. That's a severe bottleneck: how do you compress "there's a cat and it sat" into one vector?

Attention solves this by letting position 5 directly access ALL previous positions. Each position can ask: "Which other positions have information I need?" and then pull information from them.

### What You'll Implement vs What Gets Learned

Before diving in, let's be clear about what YOU write as an engineer versus what the model LEARNS during training:

**You implement (fixed algorithm):**
- The matrix multiplications that create Q, K, V
- The dot product that computes attention scores
- The softmax that converts scores to weights
- The weighted sum that produces output
- The causal mask that prevents looking at future tokens

**The model learns (trained parameters):**
- Wq, Wk, Wv: the weight matrices that project inputs to queries, keys, and values
- Wo: the output projection matrix
- These matrices start as random numbers and get adjusted during training

### The Query-Key-Value Framework

Attention uses three projections of the input. Each token's embedding vector gets transformed three different ways:

- **Query (Q)**: A vector representing "what information this position is looking for"
- **Key (K)**: A vector representing "what information this position contains"
- **Value (V)**: A vector containing "the actual information to pass along if selected"

The dot product of a query and key measures how relevant that key is to that query. When Q[i] · K[j] is large and positive, it means position i's query "matches" position j's key—position j has information that position i is looking for. When the dot product is small or negative, position j isn't relevant to position i's current needs.

### The Weight Matrices: What Do They Actually Do?

The input to attention is X, a matrix of shape [seqLen, embeddingDim]. Each row is one token's embedding vector. We multiply X by three learned matrices:

```
Q = X × Wq    where Wq has shape [embeddingDim, embeddingDim]
K = X × Wk    where Wk has shape [embeddingDim, embeddingDim]
V = X × Wv    where Wv has shape [embeddingDim, embeddingDim]
```

> What does multiplying by Wq actually DO? It's a linear transformation—a rotation, scaling, and projection of the input vector into a new space. Different weight values create different transformations.

**Why three separate matrices?** Because the same word needs to behave differently in each role:

Consider the word "cat" in "The cat sat on the mat":
- As a **query** (when "cat" is looking at other words): "cat" might be looking for verbs (what did I do?) or adjectives (what kind of cat?). Wq learns to project "cat" into a query vector that will match keys of action words.
- As a **key** (when other words are looking at "cat"): "cat" advertises that it's an animal, a noun, a subject. Wk learns to project "cat" into a key vector that action verbs will match against.
- As a **value** (the information "cat" provides): When another position decides "cat" is relevant, what information should flow? The semantic content—"small furry animal." Wv learns to project "cat" into useful information to pass along.

The model learns what to look for, what to advertise, and what to transmit—all by adjusting Wq, Wk, and Wv during training.

### Step-by-Step: Computing Attention

Let's trace through the algorithm with concrete numbers. Suppose:
- Sequence length: 4 tokens
- Embedding dimension: 4 (small for illustration; real models use 256, 768, etc.)

**Input X** (each row is a token's embedding):
```
X = [
  [0.1, 0.2, 0.3, 0.4],   // token 0: "The"
  [0.5, 0.6, 0.1, 0.2],   // token 1: "cat"
  [0.3, 0.1, 0.4, 0.5],   // token 2: "sat"
  [0.2, 0.4, 0.5, 0.3],   // token 3: "on"
]
```

**Step 1: Project to Q, K, V**

Multiply each token's embedding by the weight matrices. This is just matrix multiplication, applied row by row.

```javascript
function project(X, W) {
    // X: [seqLen][embeddingDim]
    // W: [embeddingDim][embeddingDim]
    // returns: [seqLen][embeddingDim]
    
    const seqLen = X.length;
    const dim = W[0].length;
    const result = [];
    
    for (let t = 0; t < seqLen; t++) {
        result[t] = [];
        for (let d = 0; d < dim; d++) {
            // Dot product of X[t] with column d of W
            let sum = 0;
            for (let i = 0; i < dim; i++) {
                sum += X[t][i] * W[i][d];
            }
            result[t][d] = sum;
        }
    }
    return result;
}

// In your attention function:
const Q = project(X, this.Wq);  // [seqLen][embeddingDim]
const K = project(X, this.Wk);  // [seqLen][embeddingDim]
const V = project(X, this.Wv);  // [seqLen][embeddingDim]
```

After projection (with some example learned weights), suppose we have:
```
Q = [                       K = [                       V = [
  [0.5, 0.3, 0.2, 0.1],       [0.4, 0.2, 0.3, 0.1],       [0.6, 0.1, 0.2, 0.3],
  [0.4, 0.6, 0.1, 0.2],       [0.3, 0.5, 0.2, 0.4],       [0.5, 0.3, 0.4, 0.2],
  [0.3, 0.2, 0.5, 0.4],       [0.2, 0.3, 0.6, 0.3],       [0.4, 0.5, 0.3, 0.1],
  [0.2, 0.4, 0.3, 0.5],       [0.1, 0.4, 0.4, 0.5],       [0.3, 0.4, 0.5, 0.2],
]                           ]                           ]
```

**Step 2: Compute Attention Scores**

For each pair of positions (i, j), compute how much position i should attend to position j. This is the dot product of Q[i] and K[j], scaled by √embeddingDim.

```
score[i][j] = (Q[i] · K[j]) / √embeddingDim
```

Remember from Chapter 2: dot products measure alignment. If Q[i] and K[j] point in the same direction, their dot product is large—position j has what position i is looking for. Why divide by √embeddingDim? Without scaling, dot products grow with dimension (more terms to sum). Large values would push softmax toward one-hot outputs, making gradients vanish during training. Scaling keeps values in a reasonable range.

```javascript
function computeAttentionScores(Q, K) {
    // Q: [seqLen][embeddingDim]
    // K: [seqLen][embeddingDim]
    // returns: [seqLen][seqLen] matrix of scores
    
    const seqLen = Q.length;
    const embeddingDim = Q[0].length;
    const scale = Math.sqrt(embeddingDim);
    const scores = [];
    
    for (let i = 0; i < seqLen; i++) {
        scores[i] = [];
        for (let j = 0; j < seqLen; j++) {
            // Dot product of Q[i] and K[j]
            let dot = 0;
            for (let d = 0; d < embeddingDim; d++) {
                dot += Q[i][d] * K[j][d];
            }
            scores[i][j] = dot / scale;
        }
    }
    return scores;
}
```

Example computation for score[1][0] (how much should "cat" attend to "The"?):

```
Q[1] = [0.4, 0.6, 0.1, 0.2]
K[0] = [0.4, 0.2, 0.3, 0.1]

dot = 0.4×0.4 + 0.6×0.2 + 0.1×0.3 + 0.2×0.1
    = 0.16 + 0.12 + 0.03 + 0.02
    = 0.33

score[1][0] = 0.33 / √4 = 0.33 / 2 = 0.165
```

After computing all pairs:
```
scores = [
  [0.25,  ...,  ...,  ...],    // token 0's scores for all positions
  [0.165, 0.38, ...,  ...],    // token 1's scores
  [0.12,  0.22, 0.31, ...],    // token 2's scores
  [0.08,  0.19, 0.27, 0.35],   // token 3's scores
]
```

What do these numbers tell us? 
- Score[1][0] = 0.165 means "cat" has some relevance to "The", the query vector for "cat" partially aligns with the key vector for "The". 
- Score[1][1] = 0.38 is higher, meaning "cat" finds itself more relevant. 

These scores are raw relevance measures, not probabilities yet. Notice they don't sum to 1, and they could even be negative if vectors point in opposite directions. 

**Step 3: Apply Causal Mask**

In language modeling, position i cannot look ahead at positions j > i. That would be cheating, ie using future tokens to predict the present. We enforce this by setting future scores to -∞. After softmax, e^(-∞) = 0, so future positions contribute nothing.

```javascript
function applyCausalMask(scores) {
    const seqLen = scores.length;
    
    for (let i = 0; i < seqLen; i++) {
        for (let j = 0; j < seqLen; j++) {
            if (j > i) {
                // Position i cannot look at position j (j is in the future)
                scores[i][j] = -Infinity;
            }
        }
    }
    return scores;
}
```

After masking:
```
scores = [
  [0.25,  -∞,   -∞,   -∞  ],   // token 0 can only see itself
  [0.165, 0.38, -∞,   -∞  ],   // token 1 can see 0 and itself
  [0.12,  0.22, 0.31, -∞  ],   // token 2 can see 0, 1, and itself
  [0.08,  0.19, 0.27, 0.35],   // token 3 can see everyone
]
```

**Step 4: Softmax to Get Attention Weights**

Convert scores to probabilities using the softmax function from Chapter 2. Each row gets softmaxed independently, so weights in each row sum to 1.

```javascript
function attentionWeights(scores) {
    const seqLen = scores.length;
    const weights = [];
    
    for (let i = 0; i < seqLen; i++) {
        // Softmax across row i (reuse function from Chapter 2)
        weights[i] = softmax(scores[i]);
    }
    return weights;
}
```

After softmax:
```
weights = [
  [1.0,   0,     0,     0    ],   // token 0: 100% attention on itself
  [0.35,  0.65,  0,     0    ],   // token 1: 35% on "The", 65% on itself
  [0.24,  0.32,  0.44,  0    ],   // token 2: distributed across 0,1,2
  [0.15,  0.22,  0.28,  0.35 ],   // token 3: most on itself, least on "The"
]
```

Notice how the masked positions become exactly 0 (since e^(-∞) = 0). Also notice that later tokens spread attention across more positions—they have more context available.

**Step 5: Weighted Sum of Values**

Each position's output is a weighted combination of ALL value vectors, using the attention weights.

```
output[i] = Σⱼ weights[i][j] × V[j]
```

This is where information actually flows. Position i pulls information from all positions j, weighted by how relevant each j was (determined by Q-K matching).

```javascript
function computeAttentionOutput(weights, V) {
    // weights: [seqLen][seqLen]
    // V: [seqLen][embeddingDim]
    // returns: [seqLen][embeddingDim]
    
    const seqLen = weights.length;
    const embeddingDim = V[0].length;
    const output = [];
    
    for (let i = 0; i < seqLen; i++) {
        output[i] = [];
        for (let d = 0; d < embeddingDim; d++) {
            // Weighted sum of V[j][d] for all j
            let sum = 0;
            for (let j = 0; j < seqLen; j++) {
                sum += weights[i][j] * V[j][d];
            }
            output[i][d] = sum;
        }
    }
    return output;
}
```

Example for output[1] (what information does "cat" receive?):
```
weights[1] = [0.35, 0.65, 0, 0]
V[0] = [0.6, 0.1, 0.2, 0.3]
V[1] = [0.5, 0.3, 0.4, 0.2]

output[1][0] = 0.35×0.6 + 0.65×0.5 = 0.21 + 0.325 = 0.535
output[1][1] = 0.35×0.1 + 0.65×0.3 = 0.035 + 0.195 = 0.23
output[1][2] = 0.35×0.2 + 0.65×0.4 = 0.07 + 0.26 = 0.33
output[1][3] = 0.35×0.3 + 0.65×0.2 = 0.105 + 0.13 = 0.235

output[1] = [0.535, 0.23, 0.33, 0.235]
```

The output for "cat" is a blend of the value vectors from "The" (35%) and "cat" (65%).

**Step 6: Output Projection**

Finally, we apply one more learned transformation to the attention output:

```javascript
const finalOutput = project(output, this.Wo);  // Wo: [embeddingDim][embeddingDim]
```

Why do we need this? The weighted sum of values gives us a blend of information from other positions. But the model might need to transform this blend before passing it to the next layer. ```Wo``` lets the model learn:
- Which aspects of the attended information are most useful
- How to combine features from the attention output
- How to format the output for downstream layers

Without ```Wo```, the attention output would be constrained to live in the same "space" as the value vectors. Wo gives the model flexibility to reshape this information.

### Putting It All Together

Here's the complete single-head attention implementation:

```javascript
class Attention {
    constructor(embeddingDim) {
        this.embeddingDim = embeddingDim;
        
        // Learned weight matrices (initialized randomly, trained later)
        this.Wq = randomMatrix(embeddingDim, embeddingDim);
        this.Wk = randomMatrix(embeddingDim, embeddingDim);
        this.Wv = randomMatrix(embeddingDim, embeddingDim);
        this.Wo = randomMatrix(embeddingDim, embeddingDim);
    }
    
    forward(X) {
        // X: [seqLen][embeddingDim]
        const seqLen = X.length;
        
        // Step 1: Project to Q, K, V
        const Q = this.project(X, this.Wq);
        const K = this.project(X, this.Wk);
        const V = this.project(X, this.Wv);
        
        // Step 2: Compute attention scores
        const scores = this.computeScores(Q, K);
        
        // Step 3: Apply causal mask
        this.applyCausalMask(scores);
        
        // Step 4: Softmax to get weights
        const weights = this.computeWeights(scores);
        
        // Step 5: Weighted sum of values
        const attended = this.applyWeights(weights, V);
        
        // Step 6: Output projection
        const output = this.project(attended, this.Wo);
        
        return output;
    }
    
    project(X, W) {
        const seqLen = X.length;
        const dim = this.embeddingDim;
        const result = [];
        
        for (let t = 0; t < seqLen; t++) {
            result[t] = [];
            for (let d = 0; d < dim; d++) {
                let sum = 0;
                for (let i = 0; i < dim; i++) {
                    sum += X[t][i] * W[i][d];
                }
                result[t][d] = sum;
            }
        }
        return result;
    }
    
    computeScores(Q, K) {
        const seqLen = Q.length;
        const scale = Math.sqrt(this.embeddingDim);
        const scores = [];
        
        for (let i = 0; i < seqLen; i++) {
            scores[i] = [];
            for (let j = 0; j < seqLen; j++) {
                let dot = 0;
                for (let d = 0; d < this.embeddingDim; d++) {
                    dot += Q[i][d] * K[j][d];
                }
                scores[i][j] = dot / scale;
            }
        }
        return scores;
    }
    
    applyCausalMask(scores) {
        const seqLen = scores.length;
        for (let i = 0; i < seqLen; i++) {
            for (let j = i + 1; j < seqLen; j++) {
                scores[i][j] = -Infinity;
            }
        }
    }
    
    computeWeights(scores) {
        const weights = [];
        for (let i = 0; i < scores.length; i++) {
            weights[i] = softmax(scores[i]);
        }
        return weights;
    }
    
    applyWeights(weights, V) {
        const seqLen = weights.length;
        const dim = this.embeddingDim;
        const output = [];
        
        for (let i = 0; i < seqLen; i++) {
            output[i] = [];
            for (let d = 0; d < dim; d++) {
                let sum = 0;
                for (let j = 0; j < seqLen; j++) {
                    sum += weights[i][j] * V[j][d];
                }
                output[i][d] = sum;
            }
        }
        return output;
    }
}

// Helper: create a random matrix
function randomMatrix(rows, cols) {
    const m = [];
    for (let r = 0; r < rows; r++) {
        m[r] = [];
        for (let c = 0; c < cols; c++) {
            m[r][c] = (Math.random() - 0.5) * 0.02;
        }
    }
    return m;
}
```

### What Does Attention Learn?

After training, the weight matrices encode useful patterns:

**Wq and Wk together determine WHAT positions attend to:**
- Some attention patterns are syntactic: verbs attend to their subjects
- Some are positional: attend to the immediately previous token
- Some are semantic: pronouns attend to their referents

**Wv determines WHAT information flows:**
- Might extract "this is a noun" or "this is past tense"
- Might pass along semantic features relevant to prediction

**Wo determines HOW to use the attended information:**
- Transforms the blended values into a useful representation
- Learns which combinations of attended features matter

You don't program these patterns. The model discovers them by seeing millions of examples and adjusting weights to predict the next token better.

### Multi-Head Attention

One attention pattern isn't enough. Position 4 might need to:
- Attend to "cat" (what kind of thing?)
- Attend to "sat" (what action?)
- Attend to "on" (preposition, location coming?)
- Attend to previous "the" (grammar pattern?)

Multi-head attention runs MULTIPLE attention operations in parallel:

```
┌────────────────────────────────────────────────────────────┐
│                    Input (256 dims)                        │
└────────────────────────────────────────────────────────────┘
        │               │               │               │
        ▼               ▼               ▼               ▼
┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐
│   Head 1   │  │   Head 2   │  │   Head 3   │  │   Head 4   │
│  (64 dim)  │  │  (64 dim)  │  │  (64 dim)  │  │  (64 dim)  │
│            │  │            │  │            │  │            │
│ Q₁,K₁,V₁   │  │ Q₂,K₂,V₂   │  │ Q₃,K₃,V₃   │  │ Q₄,K₄,V₄   │
│            │  │            │  │            │  │            │
│ Attention  │  │ Attention  │  │ Attention  │  │ Attention  │
│            │  │            │  │            │  │            │
│ Out (64)   │  │ Out (64)   │  │ Out (64)   │  │ Out (64)   │
└────────────┘  └────────────┘  └────────────┘  └────────────┘
        │               │               │               │
        └───────────────┴───────────────┴───────────────┘
                                │
                                ▼
                ┌──────────────────────────────┐
                │   Concatenate (256 dims)     │
                └──────────────────────────────┘
                                │
                                ▼
                ┌──────────────────────────────┐
                │  Output Projection (256→256) │
                └──────────────────────────────┘
```

With 4 heads and 256 dimensions:
- Split 256 dims into 4 heads of 64 dims each
- Each head has its own Wq, Wk, Wv (64×64 each)
- Each head computes attention independently  
- Concatenate outputs: 4 × 64 = 256
- Final projection mixes information across heads

Different heads learn different patterns. After training:
- Head 1 might track the subject of the sentence
- Head 2 might track the most recent noun
- Head 3 might track verbs
- Head 4 might track punctuation/structure

### Implementing Multi-Head Attention

```javascript
class MultiHeadAttention {
    constructor(embeddingDim, numHeads) {
        this.embeddingDim = embeddingDim;
        this.numHeads = numHeads;
        this.headDim = embeddingDim / numHeads;
        
        if (embeddingDim % numHeads !== 0) {
            throw new Error("embeddingDim must be divisible by numHeads");
        }
        
        // Initialize weight matrices with small random values
        const scale = Math.sqrt(2.0 / embeddingDim);
        this.Wq = this.randomMatrix(embeddingDim, embeddingDim, scale);
        this.Wk = this.randomMatrix(embeddingDim, embeddingDim, scale);
        this.Wv = this.randomMatrix(embeddingDim, embeddingDim, scale);
        this.Wo = this.randomMatrix(embeddingDim, embeddingDim, scale);
    }
    
    randomMatrix(rows, cols, scale) {
        const m = [];
        for (let i = 0; i < rows; i++) {
            m[i] = [];
            for (let j = 0; j < cols; j++) {
                m[i][j] = (Math.random() - 0.5) * scale;
            }
        }
        return m;
    }
```

The forward pass:

```javascript
    forward(input) {
        // input: [seqLen][embeddingDim]
        const seqLen = input.length;
        
        // Step 1: Project input to Q, K, V
        const Q = this.matmul(input, this.Wq);
        const K = this.matmul(input, this.Wk);
        const V = this.matmul(input, this.Wv);
        
        // Step 2-5: Compute attention for each head
        const scale = 1.0 / Math.sqrt(this.headDim);
        
        // Output accumulator
        const attnOutput = [];
        for (let t = 0; t < seqLen; t++) {
            attnOutput[t] = new Array(this.embeddingDim).fill(0);
        }
        
        // Process each head
        for (let h = 0; h < this.numHeads; h++) {
            const headStart = h * this.headDim;
            
            // For each query position
            for (let i = 0; i < seqLen; i++) {
                
                // Compute attention scores: Q[i] · K[j] for all j ≤ i
                const scores = [];
                for (let j = 0; j < seqLen; j++) {
                    if (j <= i) {
                        // Dot product for this head's dimensions
                        let dot = 0;
                        for (let d = 0; d < this.headDim; d++) {
                            dot += Q[i][headStart + d] * K[j][headStart + d];
                        }
                        scores[j] = dot * scale;
                    } else {
                        // Causal mask: cannot attend to future
                        scores[j] = -Infinity;
                    }
                }
                
                // Softmax to get attention weights
                const weights = this.softmax(scores);
                
                // Weighted sum of values
                for (let j = 0; j <= i; j++) {
                    for (let d = 0; d < this.headDim; d++) {
                        attnOutput[i][headStart + d] += 
                            weights[j] * V[j][headStart + d];
                    }
                }
            }
        }
        
        // Step 6: Output projection
        const output = this.matmul(attnOutput, this.Wo);
        
        return output;
    }
```

Helper methods:

```javascript
    matmul(input, weights) {
        // input: [seqLen][inDim]
        // weights: [inDim][outDim]
        // returns: [seqLen][outDim]
        
        const seqLen = input.length;
        const inDim = weights.length;
        const outDim = weights[0].length;
        
        const output = [];
        for (let t = 0; t < seqLen; t++) {
            output[t] = [];
            for (let j = 0; j < outDim; j++) {
                let sum = 0;
                for (let i = 0; i < inDim; i++) {
                    sum += input[t][i] * weights[i][j];
                }
                output[t][j] = sum;
            }
        }
        return output;
    }
    
    softmax(scores) {
        let max = -Infinity;
        for (let i = 0; i < scores.length; i++) {
            if (scores[i] > max) max = scores[i];
        }
        if (max === -Infinity) max = 0;
        
        const exps = [];
        let sum = 0;
        for (let i = 0; i < scores.length; i++) {
            const e = scores[i] === -Infinity ? 0 : Math.exp(scores[i] - max);
            exps.push(e);
            sum += e;
        }
        
        const result = [];
        for (let i = 0; i < exps.length; i++) {
            result.push(exps[i] / (sum + 1e-10));
        }
        return result;
    }
}
```

## Chapter 6: The Feed-Forward Network (MLP)

After attention gathers information from other positions, the MLP (multi layer preceptron) processes each position independently. It's a simple two-layer network:

```
Input (256) → Linear (256→1024) → GELU → Linear (1024→256) → Output
```

### Why the MLP?

Attention is great at ROUTING information between positions, but it's essentially just computing weighted averages. The MLP adds:

1. **Non-linear transformations**: Can implement complex "if-then" logic
2. **Position-wise processing**: Refines information at each position
3. **Capacity**: Stores learned "knowledge" in its weight matrices

> Research suggests MLPs store "factual knowledge" while attention handles "routing." For example, the MLP might encode "Paris is the capital of France" while attention retrieves that fact when relevant.

### The 4× Expansion

The MLP expands to 4× the embedding dimension, then contracts back:

```
256 → 1024 → 256
```

Why expand? More dimensions in the hidden layer = more capacity to learn complex transformations. The 4× factor is standard across transformers.

## Implementing the MLP

```javascript
class MLP {
    constructor(embeddingDim, expansionFactor = 4) {
        this.embeddingDim = embeddingDim;
        this.hiddenDim = embeddingDim * expansionFactor;
        
        const scale1 = Math.sqrt(2.0 / embeddingDim);
        this.W1 = this.randomMatrix(embeddingDim, this.hiddenDim, scale1);
        this.b1 = new Array(this.hiddenDim).fill(0);
        
        const scale2 = Math.sqrt(2.0 / this.hiddenDim);
        this.W2 = this.randomMatrix(this.hiddenDim, embeddingDim, scale2);
        this.b2 = new Array(embeddingDim).fill(0);
    }
```

The constructor initializes two layers of weights and biases. The first layer expands from `embeddingDim` to `hiddenDim` (4× larger), and the second contracts back. The scale factors implement "He initialization"—without this scaling, values would explode or vanish as they pass through layers, making training unstable.

```javascript
    randomMatrix(rows, cols, scale) {
        const m = [];
        for (let i = 0; i < rows; i++) {
            m[i] = [];
            for (let j = 0; j < cols; j++) {
                m[i][j] = (Math.random() - 0.5) * scale;
            }
        }
        return m;
    }
```

This creates a 2D array filled with small random values. Random initialization "breaks symmetry"—if all weights started identical, every neuron would compute the same thing and the network couldn't learn diverse features.

```javascript
    gelu(x) {
        const c = Math.sqrt(2.0 / Math.PI);
        return 0.5 * x * (1.0 + Math.tanh(c * (x + 0.044715 * x * x * x)));
    }
```

GELU (Gaussian Error Linear Unit) is a smooth activation function. For large positive inputs, it passes them through nearly unchanged. For large negative inputs, it outputs nearly zero. Unlike ReLU which has a sharp corner at zero, GELU is smooth everywhere, which helps gradients flow during training.

```javascript
    forward(input) {
        const seqLen = input.length;
        const output = [];
        
        for (let t = 0; t < seqLen; t++) {
            // First linear layer: expand to hiddenDim
            const hidden = [];
            for (let j = 0; j < this.hiddenDim; j++) {
                let sum = this.b1[j];
                for (let i = 0; i < this.embeddingDim; i++) {
                    sum += input[t][i] * this.W1[i][j];
                }
                hidden[j] = this.gelu(sum);
            }
            
            // Second linear layer: contract back to embeddingDim
            output[t] = [];
            for (let j = 0; j < this.embeddingDim; j++) {
                let sum = this.b2[j];
                for (let i = 0; i < this.hiddenDim; i++) {
                    sum += hidden[i] * this.W2[i][j];
                }
                output[t][j] = sum;
            }
        }
        
        return output;
    }
}
```

The forward pass processes each token position independently—this is different from attention, which explicitly mixes positions. For each token, we first compute a weighted sum across all input dimensions for each hidden neuron, apply GELU, then compute another weighted sum to produce the output. The result has the same shape as the input: `[seqLen][embeddingDim]`


## Chapter 7: Layer Normalization

Deep networks have a problem: as values pass through many layers, they can explode (become huge) or vanish (become tiny). Layer normalization keeps values in a reasonable range. Imagine multiplying by 1.1 at each layer:

```
Layer 1:  value = 1.0 × 1.1 = 1.1
Layer 2:  value = 1.1 × 1.1 = 1.21
Layer 10: value = 1.1^10 ≈ 2.6
Layer 50: value = 1.1^50 ≈ 117
Layer 100: value = 1.1^100 ≈ 13,781
```

Values explode. Similarly, multiplying by 0.9 makes values vanish. Normalization resets the scale at each layer.

### How Layer Norm Works

For each position's vector:

1. Compute mean: μ = average of all elements
2. Compute variance: σ² = average of squared deviations from mean
3. Normalize: x̂ = (x - μ) / √(σ² + ε)
4. Scale and shift: y = γ × x̂ + β

The ε (epsilon, ~1e-5) prevents division by zero.
The γ (gamma) and β (beta) are learned parameters that let the model undo normalization if needed.

**Example:**

```
Input: [2.0, 4.0, 6.0, 8.0]

Mean: (2 + 4 + 6 + 8) / 4 = 5.0

Variance: ((2-5)² + (4-5)² + (6-5)² + (8-5)²) / 4 = 5.0

Std dev: √5 ≈ 2.24

Normalized: [(2-5)/2.24, (4-5)/2.24, (6-5)/2.24, (8-5)/2.24]
          = [-1.34, -0.45, 0.45, 1.34]

After scale (γ=1) and shift (β=0): [-1.34, -0.45, 0.45, 1.34]
```

The output has mean ≈ 0 and variance ≈ 1.

### Implementing Layer Normalization

```javascript
class LayerNorm {
    constructor(dim) {
        this.dim = dim;
        this.eps = 1e-5;
        
        // Learned parameters (initialize to identity: scale=1, shift=0)
        this.gamma = new Array(dim).fill(1.0);
        this.beta = new Array(dim).fill(0.0);
    }
    
    forward(input) {
        // input: [seqLen][dim]
        const seqLen = input.length;
        const output = [];
        
        for (let t = 0; t < seqLen; t++) {
            // Compute mean
            let mean = 0;
            for (let d = 0; d < this.dim; d++) {
                mean += input[t][d];
            }
            mean /= this.dim;
            
            // Compute variance
            let variance = 0;
            for (let d = 0; d < this.dim; d++) {
                const diff = input[t][d] - mean;
                variance += diff * diff;
            }
            variance /= this.dim;
            
            // Normalize, scale, and shift
            const stdInv = 1.0 / Math.sqrt(variance + this.eps);
            output[t] = [];
            
            for (let d = 0; d < this.dim; d++) {
                const normalized = (input[t][d] - mean) * stdInv;
                output[t][d] = this.gamma[d] * normalized + this.beta[d];
            }
        }
        
        return output;
    }
}
```

### Pre-Norm vs Post-Norm

- **Original Transformer**: normalize AFTER attention/MLP (post-norm)
- **GPT-2 and modern models**: normalize BEFORE attention/MLP (pre-norm)

Pre-norm is more stable for deep networks. We use pre-norm.


## Chapter 8: The Transformer Block

A transformer block combines attention and MLP with layer norms and residual connections. This is the fundamental repeating unit, the T in GPT.

### Residual Connections

A residual connection adds the input to the output:

```
output = input + sublayer(input)
```

Visually:

```
input ─────────────────────────────┐
           │                       │
           ▼                       │
     ┌───────────┐                 │
     │  Sublayer │                 │
     └───────────┘                 │
           │                       │
           ▼                       │
         [ + ] ◄───────────────────┘
           │
           ▼
        output
```

Why residual connections?

1. **Gradient flow**: Creates a direct path for gradients during training. Without residuals, gradients must flow through every layer and often vanish. With residuals, gradients have a "shortcut."

2. **Easy identity**: At initialization, sublayers output near-zero values. The residual ensures output ≈ input, so the model starts as roughly identity. It learns to ADD useful transformations.

3. **Deep networks**: Makes 100+ layer networks trainable.

### The Pre-Norm Transformer Block

```
input
  │
  ├───────────────────────────────────┐
  ▼                                   │
LayerNorm                             │
  │                                   │
  ▼                                   │
Attention                             │
  │                                   │
  ▼                                   │
[ + ] ◄───────────────────────────────┘  (residual)
  │
  ├───────────────────────────────────┐
  ▼                                   │
LayerNorm                             │
  │                                   │
  ▼                                   │
 MLP                                  │
  │                                   │
  ▼                                   │
[ + ] ◄───────────────────────────────┘  (residual)
  │
  ▼
output
```

### Implementing the Transformer Block

```javascript
class TransformerBlock {
    constructor(embeddingDim, numHeads) {
        this.embeddingDim = embeddingDim;
        
        this.norm1 = new LayerNorm(embeddingDim);
        this.attention = new MultiHeadAttention(embeddingDim, numHeads);
        this.norm2 = new LayerNorm(embeddingDim);
        this.mlp = new MLP(embeddingDim);
    }
    
    forward(input) {
        // input: [seqLen][embeddingDim]
        const seqLen = input.length;
        
        // Sub-block 1: LayerNorm → Attention → Residual
        const normed1 = this.norm1.forward(input);
        const attnOut = this.attention.forward(normed1);
        
        const afterAttn = [];
        for (let t = 0; t < seqLen; t++) {
            afterAttn[t] = [];
            for (let d = 0; d < this.embeddingDim; d++) {
                afterAttn[t][d] = input[t][d] + attnOut[t][d];  // Residual
            }
        }
        
        // Sub-block 2: LayerNorm → MLP → Residual
        const normed2 = this.norm2.forward(afterAttn);
        const mlpOut = this.mlp.forward(normed2);
        
        const output = [];
        for (let t = 0; t < seqLen; t++) {
            output[t] = [];
            for (let d = 0; d < this.embeddingDim; d++) {
                output[t][d] = afterAttn[t][d] + mlpOut[t][d];  // Residual
            }
        }
        
        return output;
    }
}
```


## Chapter 9: Assembling the Complete Model

Now we connect all the pieces into a complete GPT.

### The GPT Class

```javascript
class GPT {
    constructor(config) {
        this.vocabSize = config.vocabSize;
        this.embeddingDim = config.embeddingDim;
        this.numHeads = config.numHeads;
        this.numLayers = config.numLayers;
        this.maxSeqLen = config.maxSeqLen;
        
        // Embeddings
        this.tokenEmbed = new TokenEmbedding(
            config.vocabSize, 
            config.embeddingDim
        );
        this.posEmbed = new PositionalEmbedding(
            config.maxSeqLen, 
            config.embeddingDim
        );
        
        // Transformer blocks
        this.blocks = [];
        for (let i = 0; i < config.numLayers; i++) {
            this.blocks.push(new TransformerBlock(
                config.embeddingDim, 
                config.numHeads
            ));
        }
        
        // Final layer norm
        this.finalNorm = new LayerNorm(config.embeddingDim);
        
        // Output projection: embeddingDim → vocabSize
        const scale = Math.sqrt(2.0 / config.embeddingDim);
        this.outputWeights = [];
        for (let i = 0; i < config.embeddingDim; i++) {
            this.outputWeights[i] = [];
            for (let j = 0; j < config.vocabSize; j++) {
                this.outputWeights[i][j] = (Math.random() - 0.5) * scale;
            }
        }
    }
```

### The Forward Pass

```javascript
    forward(tokenIds) {
        // tokenIds: [seqLen] array of integers
        const seqLen = tokenIds.length;
        
        if (seqLen > this.maxSeqLen) {
            throw new Error("Sequence too long: " + seqLen + 
                " > " + this.maxSeqLen);
        }
        
        // Step 1: Embeddings
        const tokenVecs = this.tokenEmbed.forward(tokenIds);
        const posVecs = this.posEmbed.forward(seqLen);
        
        // Combine: add token and position embeddings
        let hidden = [];
        for (let t = 0; t < seqLen; t++) {
            hidden[t] = [];
            for (let d = 0; d < this.embeddingDim; d++) {
                hidden[t][d] = tokenVecs[t][d] + posVecs[t][d];
            }
        }
        
        // Step 2: Transformer blocks
        for (const block of this.blocks) {
            hidden = block.forward(hidden);
        }
        
        // Step 3: Final layer norm
        hidden = this.finalNorm.forward(hidden);
        
        // Step 4: Project to vocabulary (get logits)
        const logits = [];
        for (let t = 0; t < seqLen; t++) {
            logits[t] = [];
            for (let v = 0; v < this.vocabSize; v++) {
                let sum = 0;
                for (let d = 0; d < this.embeddingDim; d++) {
                    sum += hidden[t][d] * this.outputWeights[d][v];
                }
                logits[t][v] = sum;
            }
        }
        
        // Step 5: Softmax to get probabilities
        const probs = [];
        for (let t = 0; t < seqLen; t++) {
            probs[t] = this.softmax(logits[t]);
        }
        
        return probs;
    }
    
    softmax(logits) {
        let max = logits[0];
        for (let i = 1; i < logits.length; i++) {
            if (logits[i] > max) max = logits[i];
        }
        
        const exps = [];
        let sum = 0;
        for (let i = 0; i < logits.length; i++) {
            const e = Math.exp(logits[i] - max);
            exps.push(e);
            sum += e;
        }
        
        const probs = [];
        for (let i = 0; i < logits.length; i++) {
            probs.push(exps[i] / sum);
        }
        return probs;
    }
}
```

## Chapter 10: Text Generation

With the forward pass working, we can generate text. The process is simple:

1. Start with a prompt
2. Run forward pass to get probabilities for next token
3. Sample a token from that distribution
4. Append the token to the sequence
5. Repeat until done

### Basic Generation

```javascript
generate(startTokens, maxNewTokens) {
    let tokens = startTokens.slice();  // Copy
    
    for (let i = 0; i < maxNewTokens; i++) {
        // Truncate if exceeding max length
        const input = tokens.slice(-this.maxSeqLen);
        
        // Forward pass
        const probs = this.forward(input);
        
        // Get probabilities for position AFTER the last token
        const lastProbs = probs[probs.length - 1];
        
        // Sample next token
        const nextToken = this.sample(lastProbs);
        
        // Append
        tokens.push(nextToken);
    }
    
    return tokens;
}
```

### Greedy Sampling

The simplest approach: always pick the highest-probability token.

```javascript
sampleGreedy(probs) {
    let bestIdx = 0;
    let bestProb = probs[0];
    
    for (let i = 1; i < probs.length; i++) {
        if (probs[i] > bestProb) {
            bestProb = probs[i];
            bestIdx = i;
        }
    }
    
    return bestIdx;
}
```

Greedy is deterministic but often boring and repetitive.

### Temperature Sampling

Temperature controls randomness:
- **T = 1.0**: Use probabilities as-is
- **T < 1.0**: Sharper (more confident, less random)
- **T > 1.0**: Flatter (more random, more "creative")

Temperature scales logits before softmax:

```
scaledLogits[i] = logits[i] / temperature
```

```javascript
sampleWithTemperature(probs, temperature) {
    if (temperature <= 0) {
        return this.sampleGreedy(probs);
    }
    
    // Convert back to logits, scale, re-softmax
    const logits = [];
    for (let i = 0; i < probs.length; i++) {
        logits[i] = Math.log(probs[i] + 1e-10) / temperature;
    }
    
    const scaledProbs = this.softmax(logits);
    
    // Sample from distribution
    const rand = Math.random();
    let cumulative = 0;
    
    for (let i = 0; i < scaledProbs.length; i++) {
        cumulative += scaledProbs[i];
        if (rand < cumulative) {
            return i;
        }
    }
    
    return scaledProbs.length - 1;
}
```

Visualization:

```
Original probs:     [0.50, 0.30, 0.15, 0.05]

Temperature 0.5:    [0.70, 0.22, 0.07, 0.01]  (sharper)
                     ↑↑     ↓     ↓     ↓

Temperature 2.0:    [0.35, 0.28, 0.22, 0.15]  (flatter)
                     ↓     ↑     ↑↑    ↑↑↑
```

### Top-K Sampling

Only consider the top K most likely tokens:

```javascript
sampleTopK(probs, k, temperature) {
    // Find top k tokens
    const indexed = probs.map((p, i) => ({ prob: p, idx: i }));
    indexed.sort((a, b) => b.prob - a.prob);
    const topK = indexed.slice(0, k);
    
    // Re-normalize
    let sum = 0;
    for (const item of topK) sum += item.prob;
    
    // Apply temperature and sample
    const rand = Math.random();
    let cumulative = 0;
    
    for (const item of topK) {
        cumulative += item.prob / sum;
        if (rand < cumulative) {
            return item.idx;
        }
    }
    
    return topK[k - 1].idx;
}
```

### Top-P (Nucleus) Sampling

Select the smallest set of tokens whose cumulative probability exceeds P:

```javascript
sampleTopP(probs, p, temperature) {
    const indexed = probs.map((prob, idx) => ({ prob, idx }));
    indexed.sort((a, b) => b.prob - a.prob);
    
    // Find smallest set summing to >= p
    let cumulative = 0;
    const nucleus = [];
    for (const item of indexed) {
        nucleus.push(item);
        cumulative += item.prob;
        if (cumulative >= p) break;
    }
    
    // Re-normalize and sample from nucleus
    let sum = 0;
    for (const item of nucleus) sum += item.prob;
    
    const rand = Math.random();
    cumulative = 0;
    for (const item of nucleus) {
        cumulative += item.prob / sum;
        if (rand < cumulative) {
            return item.idx;
        }
    }
    
    return nucleus[nucleus.length - 1].idx;
}
```

### Putting It Together: Generate Text

```javascript
generate(startTokens, maxNewTokens, options = {}) {
    const temperature = options.temperature || 1.0;
    const topK = options.topK || 0;
    const topP = options.topP || 1.0;
    const stopToken = options.stopToken;
    
    let tokens = startTokens.slice();
    
    for (let i = 0; i < maxNewTokens; i++) {
        const input = tokens.slice(-this.maxSeqLen);
        const probs = this.forward(input);
        const lastProbs = probs[probs.length - 1];
        
        let nextToken;
        if (topK > 0) {
            nextToken = this.sampleTopK(lastProbs, topK, temperature);
        } else if (topP < 1.0) {
            nextToken = this.sampleTopP(lastProbs, topP, temperature);
        } else {
            nextToken = this.sampleWithTemperature(lastProbs, temperature);
        }
        
        if (stopToken !== undefined && nextToken === stopToken) {
            break;
        }
        
        tokens.push(nextToken);
    }
    
    return tokens;
}
```

### Complete Generation Example

```javascript
// TODO: Fill in the tokenizer and model setup

const prompt = "The cat sat on the";
const promptTokens = tokenizer.encode(prompt);

const generatedTokens = model.generate(promptTokens, 50, {
    temperature: 0.8,
    topP: 0.9
});

const generatedText = tokenizer.decode(generatedTokens);
console.log(generatedText);
```

With random weights, this produces gibberish. But structurally, this is a working language model! Part III teaches how to train it on data so it generates coherent text.


# Part III: How the Model Learns

You now have a working inference engine. You can feed it tokens, get probabilities, sample text. But with random weights, it outputs gibberish. Training adjusts the weights so the model produces useful output. This part explains how.


## Chapter 11: What Is Training?

Training is simple in concept:

1. Show the model some text
2. Ask it to predict each next token
3. Measure how wrong it was (the "loss")
4. Adjust weights to be less wrong
5. Repeat millions of times

After enough repetitions, the model learns patterns in language. This is our Stoatic Gradient Descenet trainer.

### The Training Data

Language models learn from (input, target) pairs where the target is just the input shifted by one position:

```
Text: "The cat sat on the mat"

Tokens: [The] [cat] [sat] [on] [the] [mat]

Training pairs:
    Input:  [The]                    Target: [cat]
    Input:  [The, cat]               Target: [sat]
    Input:  [The, cat, sat]          Target: [on]
    Input:  [The, cat, sat, on]      Target: [the]
    Input:  [The, cat, sat, on, the] Target: [mat]
```

The model learns: "Given these tokens, what comes next?" In practice, we process all positions at once. The forward pass produces predictions for ALL positions simultaneously. We compare each prediction to its target.

### The Training Loop (Pseudocode)

```
for each epoch (pass through the data):
    for each batch of text:
        
        # Forward pass
        predictions = model.forward(input_tokens)
        
        # Measure error
        loss = compute_loss(predictions, target_tokens)
        
        # Compute gradients (how to adjust each weight)
        gradients = backpropagate(loss)
        
        # Update weights
        for each weight in model:
            weight = weight - learning_rate * gradient
```

### What Changes During Training?

Everything with learned parameters. For a 25M parameter model, that's 25 million numbers being adjusted. Paramaters include:

- Token embeddings (vocabSize × embeddingDim numbers)
- Position embeddings (maxSeqLen × embeddingDim numbers)
- Attention weights: Wq, Wk, Wv, Wo for each layer
- MLP weights: W1, b1, W2, b2 for each layer
- LayerNorm parameters: gamma, beta for each norm
- Output projection weights


## Chapter 12: Loss Functions

The loss function measures how wrong the model is. The goal of training is to minimize this number.

### Cross-Entropy Loss

For language modeling, the loss function of choice is cross-entropy loss:

```
loss = -log(probability of correct token)
```

The negative log has useful properties:
- Maps probability (0,1) to loss (0,∞)
- Probability 1.0 → loss 0 (perfect)
- Probability 0.0 → loss ∞ (terrible)
- Steeper penalty as probability decreases

If the model predicts the correct token with:

- **90% probability** → loss = -log(0.9) = 0.105 (low - good!)
- **50% probability** → loss = -log(0.5) = 0.693 (medium)
- **10% probability** → loss = -log(0.1) = 2.303 (high - bad!)
- **1% probability** → loss = -log(0.01) = 4.605 (very high - very bad!)

For a sequence, we average the loss across all positions:

```javascript
function computeLoss(probs, targets) {
    // probs: [seqLen][vocabSize] - model's predictions
    // targets: [seqLen] - correct token IDs
    
    let totalLoss = 0;
    
    for (let t = 0; t < targets.length; t++) {
        const correctToken = targets[t];
        const probOfCorrect = probs[t][correctToken];
        totalLoss += -Math.log(probOfCorrect + 1e-10);  // +1e-10 prevents log(0)
    }
    
    return totalLoss / targets.length;  // Average
}
```

### Perplexity: A More Intuitive Metric

Perplexity is another way to look at loss. Itanswers the question: "On average, how many tokens is the model effectively choosing between?" 

We can think of perplexity as measuring the model's confusion. A perplexity of 10 means the model is, on average, as uncertain as if it were randomly picking from 10 equally likely choices. **Perplexity = e^(average loss)**

```
function computePerplexity(probs, targets) {
    const avgLoss = computeLoss(probs, targets);
    return Math.exp(avgLoss);
}
```

## Chapter 13: Gradients and Backpropagation

We can make a prediciton (forward pass) and measure loss. Now we need to figure out how to adjust each weight to reduce the loss. This is what gradients tell us. Gradient and error are interchangable in the context of this tutorail.

### What Is a Gradient?

The gradient of a weight tells us: "If I increase this weight slightly, does the loss go up or down?". Gradient is interchangable with error.

- **Positive gradient**: Increasing weight increases loss → DECREASE the weight
- **Negative gradient**: Increasing weight decreases loss → INCREASE the weight
- **Zero gradient**: Weight doesn't affect loss right now → No change needed

The **update rule**:

```
new_weight = old_weight - learning_rate * gradient
```

We subtract because we want to move OPPOSITE to the gradient, toward lower loss.

### The Chain Rule

Neural networks are layers stacked on layers. The final loss depends on the output layer, which depends on the layer before it, which depends on the layer before that, all the way back to the weights we want to update. The chain rule lets us trace this dependency backward, and tells us how much each paramater shoudl be updated.

Let's build a small network that still has depth: two inputs, a hidden layer with one neuron, and an output layer with one neuron. It's too small to do anything useful, but it can show how gradients flow backward through multiple layers.

```
inputs: x0 = 2, x1 = 3
    ↓
[Layer 1: h = x0*w0 + x1*w1]  →  h = 2*0.5 + 3*0.5 = 2.5
    ↓
[Layer 2: y = h * w2]         →  y = 2.5 * 4 = 10
    ↓
[Loss: squared error]         →  L = (y - target)² = (10 - 20)² = 100
```

We want to update `w0` and `w1` to reduce the loss. But they don't touch the loss directly. They affect `h`, which affects `y`, which affects the `loss`. The chain rule says: multiply the rates of change along this path.

```javascript
// Forward pass - compute all intermediate values
const x0 = 2, x1 = 3;
const w0 = 0.5, w1 = 0.5, w2 = 4;
const target = 20;

const h = x0 * w0 + x1 * w1;     // h = 2.5
const y = h * w2;                // y = 10
const loss = (y - target) ** 2;  // loss = 100

// Backward pass - compute gradients by working backward
const dL_dy = 2 * (y - target);  // How does loss change with y?  = -20
const dy_dh = w2;                // How does y change with h?     = 4
const dh_dw0 = x0;               // How does h change with w0?    = 2
const dh_dw1 = x1;               // How does h change with w1?    = 3

// Chain rule: multiply along the path from loss to each weight
const dL_dh = dL_dy * dy_dh;           // = -20 × 4 = -80
const dL_dw0 = dL_dh * dh_dw0;         // = -80 × 2 = -160
const dL_dw1 = dL_dh * dh_dw1;         // = -80 × 3 = -240
```

The gradients are negative, meaning: nudging these weights up would decrease the loss. So we nudge them up.

```javascript
const learningRate = 0.001;
const w0_new = w0 - learningRate * dL_dw0;  // 0.5 - 0.001 × (-160) = 0.66
const w1_new = w1 - learningRate * dL_dw1;  // 0.5 - 0.001 × (-240) = 0.74
```

Let's verify that this is driving loss down:

```javascript
const learningRate = 0.001;
const w0_new = w0 - learningRate * dL_dw0;  // 0.5 - 0.001 × (-160) = 0.66
const w1_new = w1 - learningRate * dL_dw1;  // 0.5 - 0.001 × (-240) = 0.74

const h_new = x0 * w0_new + x1 * w1_new;  // 2 × 0.66 + 3 × 0.74 = 3.54
const y_new = h_new * w2;                  // 3.54 × 4 = 14.16
const loss_new = (y_new - target) ** 2;    // (14.16 - 20)² = 34.1

// Loss dropped from 100 to 34. Much better.
```

**The Pattern**

Each step in the backward pass answers a simple question:

```
dL_dy:  "If y increases by 1, how much does loss change?"
dy_dh:  "If h increases by 1, how much does y change?"
dh_dw0: "If w0 increases by 1, how much does h change?"
```

Multiply them together and you get: "If w0 increases by 1, how much does loss change?" That's exactly what we need to know which direction to update w0. 

Think of it as a chain of cause and effect. The effects multiply as they propagate through the network:

- w0 changes by 1 → h changes by 2 (because dh_dw0 = x0 = 2)
- h changes by 2 → y changes by 8 (because dy_dh = w2 = 4, and 2 × 4 = 8)
- y changes by 8 → loss changes by -160 (because dL_dy = -20, and 8 × -20 = -160)

**Longer Chains**

More layers just means more terms to multiply:

```
dL/dw = dL/d(layer_n) × d(layer_n)/d(layer_n-1) × ... × d(layer_2)/dw
```

Each layer contributes one term. Backpropagation computes these terms efficiently by working backward: compute each derivative once, pass it to the previous layer, repeat.

**The Backward Pass Pattern**

Every layer in backpropagation does the same thing:

1. Receive the gradient from the layer above (how much the loss changes with respect to this layer's output)
2. Compute gradient with respect to weights (to update them)
3. Compute gradient with respect to input (to pass to the layer below)

Steps 2 and 3 both use the chain rule: multiply the received gradient by the local derivative.

```javascript
// Inside any layer's backward pass:
inputGradient = receivedGradient * (derivative of output with respect to input)
weightGradient = receivedGradient * (derivative of output with respect to weight)
```

### Backpropagation

Backpropagation computes all gradients efficiently by following these steps:

1. Forward pass: Compute all intermediate values, save them
2. Start at the loss, compute gradient w.r.t. final output
3. Work backward through each layer:
   - Compute gradient w.r.t. layer's input (to pass to previous layer)
   - Compute gradient w.r.t. layer's weights (to update them)
4. Update all weights

We start from the loss and work BACKWARD through the network. At each step, we use the gradient from the layer above to compute gradients for the current layer.

## Chapter 14: Implementing the Backward Pass

Now we implement backpropagation for each component. This is the most complex part of the tutorial, but each piece follows the same pattern.

It's not shown in most code examples below, but the forward pass of each layer must SAVE intermediate values (this.cache) for the backward pass to use. Every forward pass must save values needed by the backward pass. Add caching to each forward method:

```javascript
forward(input) {
    this.cache = {
        input: input,
        // ... other intermediate values
    };
    
    // ... computation
    
    return output;
}
```

### The Gradient Starting Point

For cross-entropy loss with softmax, the gradient of loss with respect to logits is:

```
gradLogits[i] = probs[i] - (1 if i == correct else 0)
```

In other words:
- For the correct token: gradient = probability - 1
- For wrong tokens: gradient = probability

No complex derivative needed! The math works out.

```javascript
function computeLossGradient(probs, target) {
    // probs: [vocabSize] - softmax output
    // target: integer - correct token ID
    // returns: [vocabSize] - gradient w.r.t. logits
    
    const grad = [];
    for (let v = 0; v < probs.length; v++) {
        if (v === target) {
            grad[v] = probs[v] - 1;  // Push this probability UP
        } else {
            grad[v] = probs[v];       // Push these probabilities DOWN
        }
    }
    return grad;
}
```

### Backward Through a Linear Layer

A linear layer computes: `output = input × W + b`

Gradients:
- `d(loss)/d(input) = gradOutput × Wᵀ` (to pass to previous layer)
- `d(loss)/d(W) = inputᵀ × gradOutput` (to update W)
- `d(loss)/d(b) = gradOutput` (to update b)

```javascript
function linearBackward(input, weights, biases, gradOutput, learningRate) {
    // input: [inDim]
    // weights: [inDim][outDim]
    // gradOutput: [outDim]
    
    const inDim = input.length;
    const outDim = gradOutput.length;
    
    // Gradient w.r.t. input (to pass backward)
    const gradInput = new Array(inDim).fill(0);
    for (let i = 0; i < inDim; i++) {
        for (let j = 0; j < outDim; j++) {
            gradInput[i] += gradOutput[j] * weights[i][j];
        }
    }
    
    // Update weights
    for (let i = 0; i < inDim; i++) {
        for (let j = 0; j < outDim; j++) {
            weights[i][j] -= learningRate * input[i] * gradOutput[j];
        }
    }
    
    // Update biases
    for (let j = 0; j < outDim; j++) {
        biases[j] -= learningRate * gradOutput[j];
    }
    
    return gradInput;
}
```

### Backward Through GELU

For GELU activation, we need its derivative:

```javascript
class MLP {
    // ... (forward code from before)
    
    geluDerivative(x) {
        const c = Math.sqrt(2.0 / Math.PI);
        const x3 = x * x * x;
        const inner = c * (x + 0.044715 * x3);
        const tanhInner = Math.tanh(inner);
        const sech2 = 1.0 - tanhInner * tanhInner;
        const innerDeriv = c * (1.0 + 3.0 * 0.044715 * x * x);
        return 0.5 * (1.0 + tanhInner) + 0.5 * x * sech2 * innerDeriv;
    }
    
    backward(gradOutput, learningRate) {
        const seqLen = this.cache.seqLen;
        const gradInput = [];
        
        for (let t = 0; t < seqLen; t++) {
            // Backward through second linear layer
            const gradActivated = new Array(this.hiddenDim).fill(0);
            for (let i = 0; i < this.hiddenDim; i++) {
                for (let j = 0; j < this.embeddingDim; j++) {
                    gradActivated[i] += gradOutput[t][j] * this.W2[i][j];
                }
            }
            
            // Update W2, b2
            for (let i = 0; i < this.hiddenDim; i++) {
                for (let j = 0; j < this.embeddingDim; j++) {
                    this.W2[i][j] -= learningRate * 
                        this.cache.activated[t][i] * gradOutput[t][j];
                }
            }
            for (let j = 0; j < this.embeddingDim; j++) {
                this.b2[j] -= learningRate * gradOutput[t][j];
            }
            
            // Backward through GELU
            const gradHidden = [];
            for (let i = 0; i < this.hiddenDim; i++) {
                gradHidden[i] = gradActivated[i] * 
                    this.geluDerivative(this.cache.hidden[t][i]);
            }
            
            // Backward through first linear layer
            gradInput[t] = new Array(this.embeddingDim).fill(0);
            for (let i = 0; i < this.embeddingDim; i++) {
                for (let j = 0; j < this.hiddenDim; j++) {
                    gradInput[t][i] += gradHidden[j] * this.W1[i][j];
                }
            }
            
            // Update W1, b1
            for (let i = 0; i < this.embeddingDim; i++) {
                for (let j = 0; j < this.hiddenDim; j++) {
                    this.W1[i][j] -= learningRate * 
                        this.cache.input[t][i] * gradHidden[j];
                }
            }
            for (let j = 0; j < this.hiddenDim; j++) {
                this.b1[j] -= learningRate * gradHidden[j];
            }
        }
        
        return gradInput;
    }
}
```


### Backward Through Layer Norm

Layer norm's backward pass is tricky because the output depends on the mean and variance, which depend on ALL inputs.

```javascript
class LayerNorm {
    // ... (forward code from before, must cache input, mean, variance, normalized)
    
    backward(gradOutput, learningRate) {
        const seqLen = this.cache.seqLen;
        const gradInput = [];
        
        // Accumulate gradients for gamma and beta
        const gradGamma = new Array(this.dim).fill(0);
        const gradBeta = new Array(this.dim).fill(0);
        
        for (let t = 0; t < seqLen; t++) {
            const mean = this.cache.mean[t];
            const variance = this.cache.variance[t];
            const stdInv = 1.0 / Math.sqrt(variance + this.eps);
            
            // Gradients for gamma and beta
            for (let d = 0; d < this.dim; d++) {
                gradGamma[d] += gradOutput[t][d] * this.cache.normalized[t][d];
                gradBeta[d] += gradOutput[t][d];
            }
            
            // Gradient for input (complex due to mean/variance dependencies)
            let sumGradNorm = 0;
            let sumGradVar = 0;
            
            for (let d = 0; d < this.dim; d++) {
                const gradNorm = gradOutput[t][d] * this.gamma[d];
                sumGradNorm += gradNorm;
                sumGradVar += gradNorm * (this.cache.input[t][d] - mean);
            }
            
            const gradVar = sumGradVar * -0.5 * Math.pow(variance + this.eps, -1.5);
            const gradMean = -stdInv * sumGradNorm;
            
            gradInput[t] = [];
            for (let d = 0; d < this.dim; d++) {
                const gradNorm = gradOutput[t][d] * this.gamma[d];
                gradInput[t][d] = gradNorm * stdInv 
                    + gradVar * 2.0 * (this.cache.input[t][d] - mean) / this.dim
                    + gradMean / this.dim;
            }
        }
        
        // Update gamma and beta
        for (let d = 0; d < this.dim; d++) {
            this.gamma[d] -= learningRate * gradGamma[d];
            this.beta[d] -= learningRate * gradBeta[d];
        }
        
        return gradInput;
    }
}
```

### Backward Through Attention

Attention's backward pass is the most complex. Gradients flow through:
1. Output projection
2. Weighted sum of values
3. Softmax
4. Score computation (Q·K)
5. Q, K, V projections

```javascript
class MultiHeadAttention {
    // Forward must cache: input, Q, K, V, attnWeights, attnOutput
    
    backward(gradOutput, learningRate) {
        const seqLen = this.cache.seqLen;
        
        // Step 1: Backward through output projection
        const gradAttnOutput = [];
        for (let t = 0; t < seqLen; t++) {
            gradAttnOutput[t] = new Array(this.embeddingDim).fill(0);
            for (let i = 0; i < this.embeddingDim; i++) {
                for (let j = 0; j < this.embeddingDim; j++) {
                    gradAttnOutput[t][i] += gradOutput[t][j] * this.Wo[i][j];
                }
            }
        }
        
        // Update Wo
        for (let i = 0; i < this.embeddingDim; i++) {
            for (let j = 0; j < this.embeddingDim; j++) {
                let grad = 0;
                for (let t = 0; t < seqLen; t++) {
                    grad += this.cache.attnOutput[t][i] * gradOutput[t][j];
                }
                this.Wo[i][j] -= learningRate * grad;
            }
        }
        
        // Step 2-4: Backward through attention mechanism
        const gradQ = [];
        const gradK = [];
        const gradV = [];
        for (let t = 0; t < seqLen; t++) {
            gradQ[t] = new Array(this.embeddingDim).fill(0);
            gradK[t] = new Array(this.embeddingDim).fill(0);
            gradV[t] = new Array(this.embeddingDim).fill(0);
        }
        
        const scale = 1.0 / Math.sqrt(this.headDim);
        
        for (let h = 0; h < this.numHeads; h++) {
            const headStart = h * this.headDim;
            
            for (let i = 0; i < seqLen; i++) {
                const attnWeights = this.cache.attnWeights[h][i];
                
                // Gradient w.r.t. V
                for (let j = 0; j <= i; j++) {
                    for (let d = 0; d < this.headDim; d++) {
                        gradV[j][headStart + d] += 
                            attnWeights[j] * gradAttnOutput[i][headStart + d];
                    }
                }
                
                // Gradient w.r.t. attention weights
                const gradWeights = new Array(seqLen).fill(0);
                for (let j = 0; j <= i; j++) {
                    for (let d = 0; d < this.headDim; d++) {
                        gradWeights[j] += 
                            gradAttnOutput[i][headStart + d] * 
                            this.cache.V[j][headStart + d];
                    }
                }
                
                // Backward through softmax
                let dot = 0;
                for (let j = 0; j <= i; j++) {
                    dot += attnWeights[j] * gradWeights[j];
                }
                
                const gradScores = [];
                for (let j = 0; j <= i; j++) {
                    gradScores[j] = attnWeights[j] * (gradWeights[j] - dot) * scale;
                }
                
                // Gradient w.r.t. Q and K
                for (let j = 0; j <= i; j++) {
                    for (let d = 0; d < this.headDim; d++) {
                        gradQ[i][headStart + d] += 
                            gradScores[j] * this.cache.K[j][headStart + d];
                        gradK[j][headStart + d] += 
                            gradScores[j] * this.cache.Q[i][headStart + d];
                    }
                }
            }
        }
        
        // Step 5: Backward through Q, K, V projections
        const gradInput = [];
        for (let t = 0; t < seqLen; t++) {
            gradInput[t] = new Array(this.embeddingDim).fill(0);
            
            for (let i = 0; i < this.embeddingDim; i++) {
                for (let j = 0; j < this.embeddingDim; j++) {
                    gradInput[t][i] += gradQ[t][j] * this.Wq[i][j];
                    gradInput[t][i] += gradK[t][j] * this.Wk[i][j];
                    gradInput[t][i] += gradV[t][j] * this.Wv[i][j];
                }
            }
        }
        
        // Update Wq, Wk, Wv
        for (let i = 0; i < this.embeddingDim; i++) {
            for (let j = 0; j < this.embeddingDim; j++) {
                let gq = 0, gk = 0, gv = 0;
                for (let t = 0; t < seqLen; t++) {
                    gq += this.cache.input[t][i] * gradQ[t][j];
                    gk += this.cache.input[t][i] * gradK[t][j];
                    gv += this.cache.input[t][i] * gradV[t][j];
                }
                this.Wq[i][j] -= learningRate * gq;
                this.Wk[i][j] -= learningRate * gk;
                this.Wv[i][j] -= learningRate * gv;
            }
        }
        
        return gradInput;
    }
}
```

### Backward Through Transformer Block

The transformer block uses residual connections. For residuals, gradients ADD from both paths:

```javascript
class TransformerBlock {
    backward(gradOutput, learningRate) {
        const seqLen = gradOutput.length;
        
        // Backward through second sub-block (MLP)
        const gradNormed2 = this.mlp.backward(gradOutput, learningRate);
        const gradAfterAttn = this.norm2.backward(gradNormed2, learningRate);
        
        // Add gradient from residual connection
        for (let t = 0; t < seqLen; t++) {
            for (let d = 0; d < this.embeddingDim; d++) {
                gradAfterAttn[t][d] += gradOutput[t][d];
            }
        }
        
        // Backward through first sub-block (Attention)
        const gradNormed1 = this.attention.backward(gradAfterAttn, learningRate);
        const gradInput = this.norm1.backward(gradNormed1, learningRate);
        
        // Add gradient from residual connection
        for (let t = 0; t < seqLen; t++) {
            for (let d = 0; d < this.embeddingDim; d++) {
                gradInput[t][d] += gradAfterAttn[t][d];
            }
        }
        
        return gradInput;
    }
}
```

### Backward Through the Full GPT

```javascript
class GPT {
    backward(targets, learningRate) {
        const seqLen = targets.length;
        
        // Step 1: Gradient of loss w.r.t. logits
        const gradLogits = [];
        for (let t = 0; t < seqLen; t++) {
            const probs = this.softmax(this.cache.logits[t]);
            gradLogits[t] = [];
            for (let v = 0; v < this.vocabSize; v++) {
                gradLogits[t][v] = probs[v] - (v === targets[t] ? 1 : 0);
            }
        }
        
        // Step 2: Backward through output projection
        const gradFinalHidden = [];
        for (let t = 0; t < seqLen; t++) {
            gradFinalHidden[t] = new Array(this.embeddingDim).fill(0);
            for (let d = 0; d < this.embeddingDim; d++) {
                for (let v = 0; v < this.vocabSize; v++) {
                    gradFinalHidden[t][d] += 
                        gradLogits[t][v] * this.outputWeights[d][v];
                }
            }
        }
        
        // Update output weights
        for (let d = 0; d < this.embeddingDim; d++) {
            for (let v = 0; v < this.vocabSize; v++) {
                let grad = 0;
                for (let t = 0; t < seqLen; t++) {
                    grad += this.cache.finalHidden[t][d] * gradLogits[t][v];
                }
                this.outputWeights[d][v] -= learningRate * grad;
            }
        }
        
        // Step 3: Backward through final layer norm
        let gradHidden = this.finalNorm.backward(gradFinalHidden, learningRate);
        
        // Step 4: Backward through transformer blocks (reverse order!)
        for (let i = this.numLayers - 1; i >= 0; i--) {
            gradHidden = this.blocks[i].backward(gradHidden, learningRate);
        }
        
        // Step 5: Update embeddings
        for (let t = 0; t < seqLen; t++) {
            const tokenId = this.cache.tokenIds[t];
            for (let d = 0; d < this.embeddingDim; d++) {
                this.tokenEmbed.weights[tokenId][d] -= 
                    learningRate * gradHidden[t][d];
                this.posEmbed.weights[t][d] -= 
                    learningRate * gradHidden[t][d];
            }
        }
    }
}
```

## Chapter 15: The Training Loop

Now we put it all together into a complete training system. First we must prepare our training text by tokenizing it, and splitting tokens on sequence length.

```javascript
function prepareTrainingData(text, tokenizer, seqLength) {
    // Tokenize entire text
    const tokens = tokenizer.encode(text);
    
    // Create training examples
    const examples = [];
    
    for (let i = 0; i < tokens.length - seqLength; i += seqLength) {
        const input = tokens.slice(i, i + seqLength);
        const target = tokens.slice(i + 1, i + seqLength + 1);
        examples.push({ input, target });
    }
    
    return examples;
}
```

In our examples so far, learning rate has been a small, static number. We can adjust our learning rate to be dynamic. Learn a little at the start of training, a lot in the middle, and then decay. This makes our weights stay more consistent. Start low (warmup), peak, then decay:

```javascript
function getLearningRate(step, warmupSteps, totalSteps, maxLR) {
    if (step < warmupSteps) {
        // Linear warmup
        return maxLR * (step / warmupSteps);
    }
    
    // Cosine decay
    const progress = (step - warmupSteps) / (totalSteps - warmupSteps);
    return maxLR * 0.5 * (1 + Math.cos(Math.PI * progress));
}
```

And we need a helper to shuffle arrays


```javascript
function shuffle(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
}
```

Now for the training loop.

```javascript
function train(model, tokenizer, text, config) {
    console.log("Preparing training data...");
    const examples = prepareTrainingData(text, tokenizer, config.seqLength);
    console.log("Training examples: " + examples.length);
    
    const totalSteps = config.numEpochs * examples.length;
    let globalStep = 0;
    
    for (let epoch = 0; epoch < config.numEpochs; epoch++) {
        let epochLoss = 0;
        const startTime = Date.now();
        
        // Shuffle examples each epoch
        shuffle(examples);
        
        for (let i = 0; i < examples.length; i++) {
            const { input, target } = examples[i];
            
            // Get learning rate (with warmup and decay)
            const lr = getLearningRate(
                globalStep,
                config.warmupSteps,
                totalSteps,
                config.maxLearningRate
            );
            
            // Forward pass
            const probs = model.forward(input);
            
            // Compute loss
            let loss = 0;
            for (let t = 0; t < target.length; t++) {
                loss += -Math.log(probs[t][target[t]] + 1e-10);
            }
            loss /= target.length;
            epochLoss += loss;
            
            // Backward pass (updates weights)
            model.backward(target, lr);
            
            globalStep++;
            
            // Logging
            if (globalStep % config.logEvery === 0) {
                console.log(
                    "Step " + globalStep + "/" + totalSteps +
                    " | Loss: " + loss.toFixed(4) +
                    " | LR: " + lr.toExponential(2)
                );
            }
        }
        
        // Epoch summary
        const avgLoss = epochLoss / examples.length;
        const perplexity = Math.exp(avgLoss);
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        
        console.log("═".repeat(50));
        console.log("Epoch " + (epoch + 1) + "/" + config.numEpochs);
        console.log("Average Loss: " + avgLoss.toFixed(4));
        console.log("Perplexity: " + perplexity.toFixed(2));
        console.log("Time: " + elapsed + "s");
        console.log("═".repeat(50));
        
        // Generate sample
        if (config.generateSamples) {
            const prompt = tokenizer.encode("The");
            const generated = model.generate(prompt, 50, { temperature: 0.8 });
            console.log("Sample: " + tokenizer.decode(generated));
        }
    }
    
    console.log("Training complete!");
}
```

## Chapter 16: Hyperparameters

Hyperparameters are settings you choose BEFORE training. They are paramaters that affect how the models paramaters are determined. This chapter explains each one and provides recommended values.

### Architecture Hyperparameters

**VOCAB SIZE (vocabSize)**: Number of unique tokens. Larger vocabulary = better sub-word compression, more memory usage
**EMBEDDING DIMENSION (embeddingDim)**: Size of token/position vectors. Larger = more expressive, more parameters
**NUMBER OF HEADS (numHeads)**:  Parallel attention patterns per layer. More heads = more patterns, smaller per-head dimension
**NUMBER OF LAYERS (numLayers)**: Depth of the model (transformer blocks). More = more complex patterns, harder to train
**MAX SEQUENCE LENGTH (maxSeqLen)**: Longest sequence the model handles. Longer = more context, O(n²) attention cost

### Training Hyperparameters

**LEARNING RATE (maxLearningRate)**: How big of a step when updating weights. Too high = unstable, too low = slow
**WARMUP STEPS (warmupSteps)**: Steps to linearly increase learning rate from 0. Prevents instability at start when gradients are noisy
**SEQUENCE LENGTH (seqLength)**: Length of training sequences. Longer = more context per example, more memory
**NUMBER OF EPOCHS (numEpochs)**: Passes through the training data. More = better learning, risk of overfitting
