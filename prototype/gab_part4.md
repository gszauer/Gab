# Part 4: Attention

In this part, we'll transform our RNN-based model into a GPT-2 style transformer. We'll keep our tokenizer and embedding layer, but replace the recurrent core with attention mechanisms. Here's what we're building:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Token IDs  │───►│  Token +    │───►│ Transformer │───►│   Output    │───►│Probabilities│
└─────────────┘    │ Positional  │    │   Blocks    │    │    Layer    │    └─────────────┘
                   │ Embeddings  │    └─────────────┘    └─────────────┘
                   └─────────────┘           │
                                             ▼
                                    ┌─────────────────┐
                                    │  × N Blocks:    │
                                    │  - LayerNorm    │
                                    │  - Attention    │
                                    │  - LayerNorm    │
                                    │  - MLP          │
                                    │  - Residuals    │
                                    └─────────────────┘
```

## What We Have vs. What We Need

| Component | ChatRNN (Part 3) | GPT-2 Style Transformer |
|-----------|------------------|-------------------------|
| Tokenization | BPE ✓ | BPE ✓ |
| Embeddings | Token embeddings ✓ | Token + **Positional** embeddings |
| Sequence processing | RNN (sequential) | **Self-Attention** (parallel) |
| Architecture | RNN layers with residuals | **Transformer decoder blocks** |
| Normalization | None | **Layer Normalization** |
| Feed-forward | None (RNN hidden state only) | **MLP blocks** after attention |

We already have the tokenizer and embedding layer from Part 3. Now we need to add positional embeddings, build the attention mechanism, implement layer normalization, and create MLP blocks. Let's start.

# Positional Embeddings

RNNs know token order implicitly, they process tokens one at a time, so position is baked into the computation. But attention processes all tokens simultaneously. Without positional information, "the cat sat on the mat" and "mat the on sat cat the" would look identical to the network.

Positional embeddings solve this by adding position-specific vectors to each token embedding. Token 0 gets one vector added, token 1 gets a different vector, and so on. The network learns these vectors during training, just like token embeddings.

```
Token:     "The"      "cat"      "sat"
            │          │          │
            ▼          ▼          ▼
Token    [0.12,     [0.45,     [0.23,
Embed:   -0.34,      0.67,     -0.89,
          ...]        ...]       ...]
            +          +          +
Position [0.01,     [0.08,     [0.15,    ← Different for each position
Embed:    0.02,     -0.03,      0.11,
          ...]        ...]       ...]
            =          =          =
Final:   [0.13,     [0.53,     [0.38,
         -0.32,      0.64,     -0.78,
          ...]        ...]       ...]
```

We'll implement positional embeddings using composition, wrapping the existing `EmbeddingLayer` rather than modifying it:

```javascript
class PositionalEmbeddingLayer {
    tokenEmbedding = null;
    positionWeights = null;
    maxSequenceLength = 0;
    embeddingDim = 0;
    cachedLength = 0;

    constructor(vocabSize, embeddingDim, maxSequenceLength) {
        this.embeddingDim = embeddingDim;
        this.maxSequenceLength = maxSequenceLength;
        
        // Reuse existing token embedding layer
        this.tokenEmbedding = new EmbeddingLayer(vocabSize, embeddingDim);
        
        // Initialize learnable position embeddings
        const scale = Math.sqrt(1.0 / embeddingDim);
        this.positionWeights = new Array(maxSequenceLength);
        for (let pos = 0; pos < maxSequenceLength; pos++) {
            this.positionWeights[pos] = new Array(embeddingDim);
            for (let d = 0; d < embeddingDim; d++) {
                this.positionWeights[pos][d] = (Math.random() * 2 - 1) * scale;
            }
        }
    }
}
```

The `maxSequenceLength` parameter sets the longest sequence we can handle. GPT-2 used 1024 positions; we'll use smaller values for our examples. The position weights are initialized the same way as token embeddings—small random values scaled by embedding dimension.

### Positional Embedding Forward Pass

The forward pass looks up token embeddings, then adds the corresponding position embedding to each:

```javascript
// ... class PositionalEmbeddingLayer
    forward(inputTokens) {
        // Get token embeddings from the wrapped layer
        const tokenEmbeddings = this.tokenEmbedding.forward(inputTokens);
        this.cachedLength = inputTokens.length;
        
        // Add position embeddings
        const output = new Array(inputTokens.length);
        for (let pos = 0; pos < inputTokens.length; pos++) {
            output[pos] = new Array(this.embeddingDim);
            for (let d = 0; d < this.embeddingDim; d++) {
                output[pos][d] = tokenEmbeddings[pos][d] + this.positionWeights[pos][d];
            }
        }
        
        return output;
    }
```

If the input is `[42, 17, 8]`, we look up embeddings for tokens 42, 17, and 8, then add position embeddings 0, 1, and 2 respectively. The result is a sequence of vectors where each vector encodes both "what token is this?" and "where in the sequence is it?"

### Positional Embedding Backward Pass

During backpropagation, gradients flow to both the token embeddings and position embeddings. The addition operation during forward pass means the same gradient goes to both:

```javascript
// ... class PositionalEmbeddingLayer
    backward(outputGradients, learningRate) {
        // Update position embeddings
        for (let pos = 0; pos < this.cachedLength; pos++) {
            for (let d = 0; d < this.embeddingDim; d++) {
                this.positionWeights[pos][d] -= learningRate * outputGradients[pos][d];
            }
        }
        
        // Pass gradients to token embeddings
        this.tokenEmbedding.backward(outputGradients, learningRate);
        
        return null; // First layer, no gradients to pass back
    }
```

> Different models handle positions differently. GPT-1 used learned positional embeddings like we're implementing here. The original Transformer paper used fixed sinusoidal patterns. Modern models like RoPE (Rotary Position Embedding) encode positions through rotation matrices. Learned embeddings are simplest to implement and work well in practice.

# Layer Normalization

Before diving into attention, we need layer normalization. Deep networks suffer from internal covariate shift: as earlier layers update during training, the distribution of inputs to later layers changes, making learning unstable. Layer normalization fixes this by normalizing activations at each layer.

For each position in the sequence, we compute the mean and variance across the feature dimension, then normalize so the values have mean 0 and variance 1. We then apply learnable scale (gamma) and shift (beta) parameters that let the network undo the normalization if that's beneficial.

```
Input:  [2.0, -1.0, 4.0, 1.0]
                │
                ▼
Mean = (2 + -1 + 4 + 1) / 4 = 1.5
Var  = average of squared deviations = 3.25
                │
                ▼
Normalized: [(2-1.5)/√3.25, (-1-1.5)/√3.25, (4-1.5)/√3.25, (1-1.5)/√3.25]
          = [0.28, -1.39, 1.39, -0.28]
                │
                ▼
Output: gamma * normalized + beta  (elementwise)
```

The gamma and beta parameters are learnable, one value per feature dimension. They start at 1 and 0 respectively, meaning initially the layer just normalizes. During training, the network can learn to scale and shift specific features.

```javascript
class LayerNormalization {
    gamma = null;       // Scale parameters (learnable)
    beta = null;        // Shift parameters (learnable)
    featureSize = 0;
    epsilon = 1e-5;     // Prevents division by zero
    
    // Cached values for backpropagation
    cachedInputs = null;
    cachedMean = null;
    cachedVariance = null;
    cachedNormalized = null;

    constructor(featureSize) {
        this.featureSize = featureSize;
        
        // Initialize gamma to 1 (no scaling initially)
        this.gamma = new Array(featureSize);
        for (let i = 0; i < featureSize; i++) {
            this.gamma[i] = 1.0;
        }
        
        // Initialize beta to 0 (no shifting initially)
        this.beta = new Array(featureSize);
        for (let i = 0; i < featureSize; i++) {
            this.beta[i] = 0.0;
        }
    }
}
```

### Layer Normalization Forward Pass

The forward pass normalizes each position in the sequence independently:

```javascript
// ... class LayerNormalization
    forward(inputs) {
        const seqLength = inputs.length;
        
        // Initialize caches
        this.cachedInputs = new Array(seqLength);
        this.cachedMean = new Array(seqLength);
        this.cachedVariance = new Array(seqLength);
        this.cachedNormalized = new Array(seqLength);
        
        const outputs = new Array(seqLength);
        
        for (let t = 0; t < seqLength; t++) {
            // Cache input
            this.cachedInputs[t] = new Array(this.featureSize);
            for (let i = 0; i < this.featureSize; i++) {
                this.cachedInputs[t][i] = inputs[t][i];
            }
            
            // Compute mean
            let mean = 0;
            for (let i = 0; i < this.featureSize; i++) {
                mean += inputs[t][i];
            }
            mean /= this.featureSize;
            this.cachedMean[t] = mean;
            
            // Compute variance
            let variance = 0;
            for (let i = 0; i < this.featureSize; i++) {
                const diff = inputs[t][i] - mean;
                variance += diff * diff;
            }
            variance /= this.featureSize;
            this.cachedVariance[t] = variance;
            
            // Normalize and apply gamma/beta
            const stdDev = Math.sqrt(variance + this.epsilon);
            this.cachedNormalized[t] = new Array(this.featureSize);
            outputs[t] = new Array(this.featureSize);
            
            for (let i = 0; i < this.featureSize; i++) {
                const normalized = (inputs[t][i] - mean) / stdDev;
                this.cachedNormalized[t][i] = normalized;
                outputs[t][i] = this.gamma[i] * normalized + this.beta[i];
            }
        }
        
        return outputs;
    }
```

Each position gets normalized independently. We cache everything needed for backpropagation: the original inputs, mean, variance, and normalized values.

### Layer Normalization Backward Pass

The backward pass is more complex because each output depends on all inputs at that position (through the mean and variance computation):

```javascript
// ... class LayerNormalization
    backward(outputGradients, learningRate) {
        const seqLength = outputGradients.length;
        const inputGradients = new Array(seqLength);
        
        // Accumulate gradients for gamma and beta across all positions
        const gammaGrad = new Array(this.featureSize);
        const betaGrad = new Array(this.featureSize);
        for (let i = 0; i < this.featureSize; i++) {
            gammaGrad[i] = 0;
            betaGrad[i] = 0;
        }
        
        for (let t = 0; t < seqLength; t++) {
            const mean = this.cachedMean[t];
            const variance = this.cachedVariance[t];
            const stdDev = Math.sqrt(variance + this.epsilon);
            
            // Accumulate gamma and beta gradients
            for (let i = 0; i < this.featureSize; i++) {
                gammaGrad[i] += outputGradients[t][i] * this.cachedNormalized[t][i];
                betaGrad[i] += outputGradients[t][i];
            }
            
            // Compute gradient with respect to normalized values
            const dNormalized = new Array(this.featureSize);
            for (let i = 0; i < this.featureSize; i++) {
                dNormalized[i] = outputGradients[t][i] * this.gamma[i];
            }
            
            // Gradient through normalization 
            // d/dx of (x - mean) / std involves terms for both the direct path
            // and the paths through mean and variance
            
            let dVariance = 0;
            for (let i = 0; i < this.featureSize; i++) {
                dVariance += dNormalized[i] * (this.cachedInputs[t][i] - mean);
            }
            dVariance *= -0.5 * Math.pow(variance + this.epsilon, -1.5);
            
            let dMean = 0;
            for (let i = 0; i < this.featureSize; i++) {
                dMean += dNormalized[i] * (-1.0 / stdDev);
            }
            dMean += dVariance * (-2.0 / this.featureSize) * 
                     this.#sum(this.cachedInputs[t], mean);
            
            // Final input gradients
            inputGradients[t] = new Array(this.featureSize);
            for (let i = 0; i < this.featureSize; i++) {
                inputGradients[t][i] = 
                    dNormalized[i] / stdDev +
                    dVariance * 2.0 * (this.cachedInputs[t][i] - mean) / this.featureSize +
                    dMean / this.featureSize;
            }
        }
        
        // Update gamma and beta
        for (let i = 0; i < this.featureSize; i++) {
            this.gamma[i] -= learningRate * gammaGrad[i];
            this.beta[i] -= learningRate * betaGrad[i];
        }
        
        return inputGradients;
    }
    
    #sum(arr, subtract) {
        let result = 0;
        for (let i = 0; i < arr.length; i++) {
            result += arr[i] - subtract;
        }
        return result;
    }
```

The gradient computation looks intimidating, but it follows the chain rule through three paths: the direct path from normalized output to input, the path through the mean, and the path through the variance. Each input value affects the mean, which affects all normalized values. Same for variance. These dependencies create cross-terms in the gradient.

# Self-Attention

RNNs process tokens one at a time, carrying forward a hidden state. This creates two problems:

1. **Information compression**: By the time an RNN reaches token 50, information about token 1 has been compressed and re-compressed through 49 state updates. Details get lost.

2. **No direct connections**: Token 50 can't directly "look at" token 1. It only sees what the hidden state remembers.

Consider the sentence: "The knight dropped his sword because it was heavy."

What does "it" refer to? The knight or the sword? As humans, we instantly connect "it" to "sword" because swords are heavy. An RNN would have to hope this connection survived the hidden state compression. Self-attention solves this by letting every token directly look at every other token and decide what's relevant.

hink of self-attention as every word performing a database lookup against all other words. For each word:                  
                         
1. Create a QUERY   - "What am I looking for?"        
2. Every word has a KEY   - "What do I contain?"     
3. Every word has a VALUE - "Here's my actual info"   
4. Compare my QUERY against all KEYs (similarity scores) 
5. Convert scores to percentages (softmax)      
6. Grab a weighted mix of all VALUEs             

# Query, Key, Value - What Are They?

These three vectors are computed from each token's embedding using learned weight matrices:

```javascript
// Each token embedding gets transformed three ways:
query = embedding × weightsQ   // "What am I looking for?"
key   = embedding × weightsK   // "What do I contain?"  
value = embedding × weightsV   // "My actual information"
```

**Why three separate transformations?**

Query and Key exist for *matching*. The network learns to make pronouns' queries match nouns' keys, verbs' queries match subjects' keys, etc.

Value is for *content*. Once you've decided "sword" is relevant to "it", you don't want the matching score - you want the actual semantic content of "sword".

- **Key** = The book's catalog entry (what you search by)
- **Value** = The book's contents (what you actually read)
- **Query** = Your search terms

The Q/K/V pattern appears everywhere in databases. Self-attention is essentially a differentiable, learned lookup table where the matching function is trained end-to-end. Let's trace through a sentence: `["The", "cat", "sat"]`

Each word starts as an embedding vector (say, 4 dimensions for simplicity):

```
"The" embedding: [0.2, -0.1, 0.5, 0.3]
"cat" embedding: [0.8, 0.4, -0.2, 0.1]
"sat" embedding: [-0.3, 0.6, 0.1, 0.7]
```

**Step 1: Compute Q, K, V for each token**

```
"The": query=[...], key=[...], value=[...]
"cat": query=[...], key=[...], value=[...]
"sat": query=[...], key=[...], value=[...]
```

**Step 2: Each token scores against all keys**

Let's focus on "sat" computing its scores:

```
score("sat" → "The") = dot(query_sat, key_The) = 0.3
score("sat" → "cat") = dot(query_sat, key_cat) = 1.8  // High! Verbs look for subjects
score("sat" → "sat") = dot(query_sat, key_sat) = 0.5

scores = [0.3, 1.8, 0.5]
```

**Step 3: Convert to percentages (softmax)**

```
weights = softmax([0.3, 1.8, 0.5]) = [0.15, 0.62, 0.23]
                                      │     │     │
                                     The   cat   sat
```

"sat" puts 62% of its attention on "cat", 23% on itself, 15% on "The".

**Step 4: Weighted sum of values**

```
output_sat = 0.15 × value_The + 0.62 × value_cat + 0.23 × value_sat
```

The output for "sat" is now enriched with information from "cat" - the subject it was looking for.

# The Causal Mask

For language models, there's a constraint: when predicting the next word, you can't look at future words (they don't exist yet during generation). Token at position `i` can only see positions `0` through `i`:

```
                    Positions being looked at:
                    0       1       2       3
                  "The"   "cat"   "sat"   "on"
                ┌───────────────────────────────┐
Position 0 "The"│  ✓       ✗       ✗       ✗   │  Can only see itself
Position 1 "cat"│  ✓       ✓       ✗       ✗   │  Can see "The" and itself
Position 2 "sat"│  ✓       ✓       ✓       ✗   │  Can see "The", "cat", itself
Position 3 "on" │  ✓       ✓       ✓       ✓   │  Can see everything before
                └───────────────────────────────┘
```

We implement this by setting "future" scores to negative infinity before softmax. After softmax, negative infinity becomes 0%, so future tokens contribute nothing.

```javascript
// Before masking: scores = [0.3, 1.8, 0.5, 0.9]
// After masking:  scores = [0.3, 1.8, -Infinity, -Infinity]
// After softmax:  weights = [0.18, 0.82, 0.0, 0.0]
```

# The Scaling Factor

Before softmax, we divide scores by `sqrt(dimension)`. Why? Dot products grow larger as dimensions increase. Large values going into softmax produce extremely peaked distributions (nearly one-hot), which causes gradient problems during training. Scaling keeps the values in a reasonable range.

```javascript
const scale = Math.sqrt(headDim);
for (let j = 0; j < scores.length; j++) {
    scores[j] /= scale;
}
```

> This is why it's called "Scaled Dot-Product Attention" in the literature. The scaling is a practical necessity, not a theoretical insight.

# Multi-Head Attention

Each attention head has its own Q, K, V weight matrices that project from the **full embedding dimension** down to a smaller **head dimension**. If your embedding is 64-dimensional and you have 4 heads, each head projects the full 64-dimensional input down to 16 dimensions.

```
Input embedding: [64 dimensions]
         │
         ├──► Head 0: weightsQ[64][16] ──► query[16]
         ├──► Head 1: weightsQ[64][16] ──► query[16]
         ├──► Head 2: weightsQ[64][16] ──► query[16]
         └──► Head 3: weightsQ[64][16] ──► query[16]
```

Each head sees the **entire** input vector, but projects it into its own 16-dimensional subspace. Think of it like four different cameras photographing the same scene from different angles—each captures the whole scene, but emphasizes different aspects.

The weight matrices learn these projections during training. Head 0 might learn to project inputs in a way that captures syntactic relationships. Head 1 might learn projections that capture semantic similarity. Each head develops its own "perspective" on the data.

After attention, each head produces a 16-dimensional output for each position. We concatenate these outputs:

```
Head 0 output: [16] ─┐
Head 1 output: [16] ─┼──► Concatenated: [64]
Head 2 output: [16] ─┤
Head 3 output: [16] ─┘
```

The concatenated output is back to 64 dimensions (16 × 4 = 64). We then apply an output projection—a learned linear transformation that mixes information across the head dimensions:

```
Concatenated [64] ──► Output Projection [64][64] ──► Final Output [64]
```

This output projection is crucial. Without it, the heads would operate completely independently. The projection lets the model combine insights from different heads—"Head 1 found the subject, Head 3 found the verb, now let's combine them."


>  Projecting a vector or matrix from one space or another is done with matrix multiplication. When you multiply a vector by a matrix, the output dimension is determined by the matrix shape: ```vector [N] × matrix [N][M] = result [M]``` 1. If M < N: you're projecting down (compressing) 2. If M > N: you're projecting up (expanding) 3.If M = N: same dimension (transforming)

We'll build two classes:
1. **AttentionHead** - A single attention head with its own Q/K/V weights. This is the core computation unit.
2. **MultiHeadAttentionLayer** - Contains N AttentionHead instances, handles concatenation and output projection.

# Helper Functions

Before building the classes, we need some utilities that both will use.

## Matrix Creation

```javascript
#createMatrix(rows, cols, fillValue = 0) {
    const matrix = new Array(rows);
    for (let i = 0; i < rows; i++) {
        matrix[i] = new Array(cols);
        for (let j = 0; j < cols; j++) {
            matrix[i][j] = fillValue;
        }
    }
    return matrix;
}
```

Creates a 2D array initialized to a fill value. We use regular nested arrays rather than typed arrays because we need flexibility with dimensions.

## Random Matrix Initialization

```javascript
#randomMatrix(rows, cols, scale) {
    const matrix = new Array(rows);
    for (let i = 0; i < rows; i++) {
        matrix[i] = new Array(cols);
        for (let j = 0; j < cols; j++) {
            matrix[i][j] = (Math.random() * 2 - 1) * scale;
        }
    }
    return matrix;
}
```

Weight matrices need random initialization. The `scale` parameter controls the initial magnitude - we use Xavier initialization to keep activations stable.

## Vector-Matrix Multiplication

```javascript
#vectorMatrixMultiply(vector, matrix) {
    // vector: [inputDim], matrix: [inputDim][outputDim]
    // result: [outputDim]
    const outputDim = matrix[0].length;
    const result = new Array(outputDim);
    
    for (let j = 0; j < outputDim; j++) {
        let sum = 0;
        for (let i = 0; i < vector.length; i++) {
            sum += vector[i] * matrix[i][j];
        }
        result[j] = sum;
    }
    
    return result;
}
```

The Q, K, V projections require multiplying each embedding vector by a weight matrix. This computes `result[j] = Σ vector[i] * matrix[i][j]` for each output dimension.

## Dot Product

```javascript
#dot(a, b) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}
```

Comparing queries to keys requires dot products. The dot product measures similarity - high values mean the vectors point in similar directions.

## Softmax

```javascript
#softmax(values) {
    // Find max for numerical stability
    let max = values[0];
    for (let i = 1; i < values.length; i++) {
        if (values[i] > max) max = values[i];
    }
    
    // Compute exp(value - max) and sum
    const exp = new Array(values.length);
    let sum = 0;
    for (let i = 0; i < values.length; i++) {
        exp[i] = Math.exp(values[i] - max);
        sum += exp[i];
    }
    
    // Normalize
    const result = new Array(values.length);
    for (let i = 0; i < values.length; i++) {
        result[i] = exp[i] / sum;
    }
    
    return result;
}
```

Converts scores to probabilities:
1. **Subtract max first** - Prevents numerical overflow. `exp(1000)` is infinity, but `exp(0) = 1`.
2. **Exponentiate** - Converts scores to positive values, amplifies differences.
3. **Normalize** - Divides by sum so outputs sum to 1.0.

# The AttentionHead Class

An attention head is the core computation unit. It has its own Q, K, V weight matrices and performs the full attention computation on a reduced dimension.

```javascript
class AttentionHead {
    weightsQ = null;  // [embeddingDim][headDim]
    weightsK = null;  // [embeddingDim][headDim]
    weightsV = null;  // [embeddingDim][headDim]
    
    embeddingDim = 0;
    headDim = 0;
    scale = 0;
    
    // Caches for backpropagation
    cachedInputs = null;
    cachedQ = null;
    cachedK = null;
    cachedV = null;
    cachedScores = null;
    cachedWeights = null;

    constructor(embeddingDim, headDim) {
        this.embeddingDim = embeddingDim;
        this.headDim = headDim;
        this.scale = Math.sqrt(headDim);
        
        const initScale = Math.sqrt(2.0 / (embeddingDim + headDim));
        
        this.weightsQ = this.#randomMatrix(embeddingDim, headDim, initScale);
        this.weightsK = this.#randomMatrix(embeddingDim, headDim, initScale);
        this.weightsV = this.#randomMatrix(embeddingDim, headDim, initScale);
    }
}
```

**What's stored:**
- Three weight matrices projecting from `embeddingDim` down to `headDim`
- The scale factor (precomputed `sqrt(headDim)`)
- Caches for backpropagation

Note that `weightsQ`, `weightsK`, and `weightsV` are `[embeddingDim][headDim]` - they project the full embedding down to the head's smaller dimension.

## AttentionHead Forward Pass

```javascript
forward(inputs) {
    // inputs: [seqLen][embeddingDim]
    const seqLen = inputs.length;
    this.cachedInputs = inputs;
    
    // Step 1: Project inputs to Q, K, V in head dimension
    this.cachedQ = new Array(seqLen);
    this.cachedK = new Array(seqLen);
    this.cachedV = new Array(seqLen);
    
    for (let i = 0; i < seqLen; i++) {
        this.cachedQ[i] = this.#vectorMatrixMultiply(inputs[i], this.weightsQ);
        this.cachedK[i] = this.#vectorMatrixMultiply(inputs[i], this.weightsK);
        this.cachedV[i] = this.#vectorMatrixMultiply(inputs[i], this.weightsV);
    }
    
    // Step 2: Compute attention scores
    this.cachedScores = this.#createMatrix(seqLen, seqLen);
    
    for (let i = 0; i < seqLen; i++) {
        for (let j = 0; j < seqLen; j++) {
            if (j > i) {
                // Causal mask: can't attend to future positions
                this.cachedScores[i][j] = -Infinity;
            } else {
                this.cachedScores[i][j] = this.#dot(this.cachedQ[i], this.cachedK[j]) / this.scale;
            }
        }
    }
    
    // Step 3: Softmax to get attention weights
    this.cachedWeights = new Array(seqLen);
    for (let i = 0; i < seqLen; i++) {
        this.cachedWeights[i] = this.#softmax(this.cachedScores[i]);
    }
    
    // Step 4: Weighted sum of values
    const outputs = new Array(seqLen);
    for (let i = 0; i < seqLen; i++) {
        outputs[i] = new Array(this.headDim);
        for (let d = 0; d < this.headDim; d++) {
            let sum = 0;
            for (let j = 0; j <= i; j++) {
                sum += this.cachedWeights[i][j] * this.cachedV[j][d];
            }
            outputs[i][d] = sum;
        }
    }
    
    return outputs;  // [seqLen][headDim]
}
```

The forward pass implements exactly what we described earlier:
1. Project each input to Q, K, V vectors (now in `headDim` space)
2. Compute scores between all query-key pairs, with causal masking
3. Softmax each row to get attention weights
4. Weighted sum of values

The output is `[seqLen][headDim]` - each position now contains information gathered from other positions.

## AttentionHead Backward Pass

The backward pass traces the forward pass in reverse, computing gradients for the weights and passing gradients to the input:

```javascript
backward(outputGradients, learningRate) {
    // outputGradients: [seqLen][headDim]
    const seqLen = this.cachedInputs.length;
    
    // Accumulators for weight gradients
    const weightsQGrad = this.#createMatrix(this.embeddingDim, this.headDim);
    const weightsKGrad = this.#createMatrix(this.embeddingDim, this.headDim);
    const weightsVGrad = this.#createMatrix(this.embeddingDim, this.headDim);
    
    // Gradients to pass to previous layer
    const inputGradients = new Array(seqLen);
    for (let i = 0; i < seqLen; i++) {
        inputGradients[i] = new Array(this.embeddingDim);
        for (let d = 0; d < this.embeddingDim; d++) {
            inputGradients[i][d] = 0;
        }
    }
    
    // ====== Step 4 backward: Weighted sum of values ======
    // output[i] = Σ weights[i][j] * V[j]
    // Need gradients for weights and V
    
    const weightsGrad = this.#createMatrix(seqLen, seqLen);
    const vGrad = new Array(seqLen);
    for (let i = 0; i < seqLen; i++) {
        vGrad[i] = new Array(this.headDim);
        for (let d = 0; d < this.headDim; d++) {
            vGrad[i][d] = 0;
        }
    }
    
    for (let i = 0; i < seqLen; i++) {
        for (let j = 0; j <= i; j++) {
            for (let d = 0; d < this.headDim; d++) {
                weightsGrad[i][j] += outputGradients[i][d] * this.cachedV[j][d];
                vGrad[j][d] += outputGradients[i][d] * this.cachedWeights[i][j];
            }
        }
    }
    
    // ====== Step 3 backward: Softmax ======
    // weights = softmax(scores)
    const scoresGrad = this.#createMatrix(seqLen, seqLen);
    
    for (let i = 0; i < seqLen; i++) {
        for (let j = 0; j <= i; j++) {
            for (let k = 0; k <= i; k++) {
                if (j === k) {
                    scoresGrad[i][j] += weightsGrad[i][k] * this.cachedWeights[i][j] * (1 - this.cachedWeights[i][j]);
                } else {
                    scoresGrad[i][j] += weightsGrad[i][k] * (-this.cachedWeights[i][j] * this.cachedWeights[i][k]);
                }
            }
        }
    }
    
    // ====== Step 2 backward: Score computation ======
    // scores[i][j] = dot(Q[i], K[j]) / scale
    const qGrad = new Array(seqLen);
    const kGrad = new Array(seqLen);
    for (let i = 0; i < seqLen; i++) {
        qGrad[i] = new Array(this.headDim);
        kGrad[i] = new Array(this.headDim);
        for (let d = 0; d < this.headDim; d++) {
            qGrad[i][d] = 0;
            kGrad[i][d] = 0;
        }
    }
    
    for (let i = 0; i < seqLen; i++) {
        for (let j = 0; j <= i; j++) {
            const scaledGrad = scoresGrad[i][j] / this.scale;
            
            for (let d = 0; d < this.headDim; d++) {
                qGrad[i][d] += scaledGrad * this.cachedK[j][d];
                kGrad[j][d] += scaledGrad * this.cachedQ[i][d];
            }
        }
    }
    
    // ====== Step 1 backward: Q, K, V projections ======
    // Q = input × weightsQ, etc.
    // Gradients flow through all three paths to input
    
    for (let i = 0; i < seqLen; i++) {
        // Q path
        for (let d = 0; d < this.embeddingDim; d++) {
            for (let h = 0; h < this.headDim; h++) {
                inputGradients[i][d] += qGrad[i][h] * this.weightsQ[d][h];
                weightsQGrad[d][h] += qGrad[i][h] * this.cachedInputs[i][d];
            }
        }
        
        // K path
        for (let d = 0; d < this.embeddingDim; d++) {
            for (let h = 0; h < this.headDim; h++) {
                inputGradients[i][d] += kGrad[i][h] * this.weightsK[d][h];
                weightsKGrad[d][h] += kGrad[i][h] * this.cachedInputs[i][d];
            }
        }
        
        // V path
        for (let d = 0; d < this.embeddingDim; d++) {
            for (let h = 0; h < this.headDim; h++) {
                inputGradients[i][d] += vGrad[i][h] * this.weightsV[d][h];
                weightsVGrad[d][h] += vGrad[i][h] * this.cachedInputs[i][d];
            }
        }
    }
    
    // Update weights
    for (let d = 0; d < this.embeddingDim; d++) {
        for (let h = 0; h < this.headDim; h++) {
            this.weightsQ[d][h] -= learningRate * weightsQGrad[d][h];
            this.weightsK[d][h] -= learningRate * weightsKGrad[d][h];
            this.weightsV[d][h] -= learningRate * weightsVGrad[d][h];
        }
    }
    
    return inputGradients;  // [seqLen][embeddingDim]
}
```

The input receives gradients from three paths. This is why we accumulate with `+=`. Each path contributes its portion of the error signal. Key points about the backward pass:

1. **Step 4 backward**: The weighted sum has two contributors - the attention weights and the values. Both receive gradients.

2. **Step 3 backward**: Softmax gradient is tricky because each output depends on all inputs (through normalization). The diagonal terms use `w * (1 - w)`, off-diagonal use `-w[j] * w[k]`.

3. **Step 2 backward**: The dot product gradient flows to both Q and K.

4. **Step 1 backward**: Three paths (Q, K, V) all came from the same input, so we sum all three contributions to `inputGradients`.

# The MultiHeadAttentionLayer Class

This class orchestrates multiple attention heads, concatenates their outputs, and applies an output projection. 

```javascript
class MultiHeadAttentionLayer {
    heads = null;
    outputProj = null;  // DenseLayer: embeddingDim → embeddingDim
    
    numHeads = 0;
    headDim = 0;
    embeddingDim = 0;
    
    // Caches
    cachedHeadOutputs = null;
    cachedConcatenated = null;

    constructor(embeddingDim, numHeads) {
        if (embeddingDim % numHeads !== 0) {
            throw new Error("embeddingDim must be divisible by numHeads");
        }
        
        this.embeddingDim = embeddingDim;
        this.numHeads = numHeads;
        this.headDim = embeddingDim / numHeads;
        
        // Create attention heads
        this.heads = new Array(numHeads);
        for (let h = 0; h < numHeads; h++) {
            this.heads[h] = new AttentionHead(embeddingDim, this.headDim);
        }
        
        // Output projection 
        this.outputProj = new DenseLayer(embeddingDim, embeddingDim);
    }
}
```

Each head projects the full embedding dimension down to `headDim`. With 4 heads and 64 embedding dim, each head works with 16 dimensions. After concatenation, we're back to 64. The output projection then mixes information across head dimensions using our familiar `DenseLayer`.

## MultiHeadAttentionLayer Forward Pass

```javascript
forward(inputs) {
    // inputs: [seqLen][embeddingDim]
    const seqLen = inputs.length;
    
    // Run each head
    this.cachedHeadOutputs = new Array(this.numHeads);
    for (let h = 0; h < this.numHeads; h++) {
        this.cachedHeadOutputs[h] = this.heads[h].forward(inputs);
        // Each head output: [seqLen][headDim]
    }
    
    // Concatenate head outputs
    this.cachedConcatenated = new Array(seqLen);
    for (let i = 0; i < seqLen; i++) {
        this.cachedConcatenated[i] = new Array(this.embeddingDim);
        
        for (let h = 0; h < this.numHeads; h++) {
            for (let d = 0; d < this.headDim; d++) {
                this.cachedConcatenated[i][h * this.headDim + d] = this.cachedHeadOutputs[h][i][d];
            }
        }
    }
    
    // Output projection using DenseLayer
    const outputs = new Array(seqLen);
    for (let i = 0; i < seqLen; i++) {
        outputs[i] = this.outputProj.forward(this.cachedConcatenated[i]);
    }
    
    return outputs;  // [seqLen][embeddingDim]
}
```

The forward pass:
1. **Run all heads**: Each head receives the same input and produces `[seqLen][headDim]` output
2. **Concatenate**: Stack head outputs side-by-side to get `[seqLen][embeddingDim]`
3. **Output projection**: Our `DenseLayer` transforms each position

The concatenation places head 0's output in positions `[0..headDim-1]`, head 1's in `[headDim..2*headDim-1]`, etc.

## MultiHeadAttentionLayer Backward Pass

```javascript
backward(outputGradients, learningRate) {
    // outputGradients: [seqLen][embeddingDim]
    const seqLen = outputGradients.length;
    
    // ====== Output projection backward using DenseLayer ======
    const concatenatedGrad = new Array(seqLen);
    
    for (let i = 0; i < seqLen; i++) {
        // Re-forward to set up DenseLayer cache for this position
        this.outputProj.forward(this.cachedConcatenated[i]);
        // Backward through the dense layer
        concatenatedGrad[i] = this.outputProj.backward(outputGradients[i], learningRate);
    }
    
    // ====== Concatenation backward: split gradients to each head ======
    const headGradients = new Array(this.numHeads);
    for (let h = 0; h < this.numHeads; h++) {
        headGradients[h] = new Array(seqLen);
        for (let i = 0; i < seqLen; i++) {
            headGradients[h][i] = new Array(this.headDim);
            for (let d = 0; d < this.headDim; d++) {
                headGradients[h][i][d] = concatenatedGrad[i][h * this.headDim + d];
            }
        }
    }
    
    // ====== Run backward on each head and sum input gradients ======
    const inputGradients = new Array(seqLen);
    for (let i = 0; i < seqLen; i++) {
        inputGradients[i] = new Array(this.embeddingDim);
        for (let d = 0; d < this.embeddingDim; d++) {
            inputGradients[i][d] = 0;
        }
    }
    
    for (let h = 0; h < this.numHeads; h++) {
        const headInputGrad = this.heads[h].backward(headGradients[h], learningRate);
        
        // Sum input gradients from all heads
        for (let i = 0; i < seqLen; i++) {
            for (let d = 0; d < this.embeddingDim; d++) {
                inputGradients[i][d] += headInputGrad[i][d];
            }
        }
    }
    
    return inputGradients;  // [seqLen][embeddingDim]
}
```

The backward pass:
1. **Output projection backward**: `DenseLayer` handles its own gradient computation and weight updates
2. **Concatenation backward**: This is just slicing—each head's gradient is its slice of `concatenatedGrad`
3. **Head backward**: Run backward on each head, which updates its Q/K/V weights and returns input gradients
4. **Sum input gradients**: All heads received the same input, so their gradients sum

> Each head independently updates its own Q/K/V weights during its backward pass. The `MultiHeadAttentionLayer` just needs to route gradients appropriately—`DenseLayer` handles the output projection weights automatically.

# Attention recap

Self-attention is a **learned, differentiable lookup mechanism**:

1. Each token creates a **Query** ("what am I looking for?")
2. Each token creates a **Key** ("what do I contain?")
3. Each token creates a **Value** ("here's my information")
5. Scores are **masked** (for causal models) and **softmaxed** to get weights
6. Each token takes a **weighted sum** of all values it can see
7. An **output projection** combines information across heads

**Insights and architecture:**
- `AttentionHead` handles one set of Q/K/V projections and computes attention in a reduced dimension
- `MultiHeadAttentionLayer` contains N `AttentionHead` instances, concatenates their outputs, and applies an output projection
- Single-head attention is just `MultiHeadAttentionLayer` with `numHeads = 1`
- Attention is O(n²) in sequence length - every token looks at every other token
- Multi-head attention lets the network learn multiple types of relationships
- More heads = more types of relationships the model can learn.
- All heads receive the same input, so their input gradients sum during backprop

This mechanism, combined with layer normalization and feed-forward networks, forms the transformer block - the building block of GPT and most modern language models.


# The MLP Block

After attention gathers information from across the sequence, the MLP (Multi-Layer Perceptron) block processes each position independently. The MLP block is simple: two linear transformations with a non-linearity in between. The first layer expands the representation (typically 4x the embedding dimension), applies an activation function, then the second layer projects back down.

```
┌─────────────────────────────────────────────────────────────────┐
│                           MLP Block                             │
│                                                                 │
│   Input ──► Dense1 ──► GELU ──► Dense2 ──► Output              │
│   [64]       [256]      [256]     [64]      [64]                │
│                                                                 │
│   embeddingDim  →  4×embeddingDim  →  embeddingDim              │
└─────────────────────────────────────────────────────────────────┘
```

## Adding GELU to ActivationLayer

First, we need to extend the `ActivationLayer` class to support GELU (Gaussian Error Linear Unit). GELU smoothly gates inputs based on their value: large positive values pass through mostly unchanged, large negative values are suppressed, and values near zero get partially dampened.

```javascript
// Add to ActivationLayer from Part 1:
    #geluActivation(x) {
        // Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        const c = Math.sqrt(2 / Math.PI);
        return 0.5 * x * (1 + Math.tanh(c * (x + 0.044715 * x * x * x)));
    }

    #geluDerivative(x) {
        const c = Math.sqrt(2 / Math.PI);
        const x3 = x * x * x;
        const inner = c * (x + 0.044715 * x3);
        const tanhInner = Math.tanh(inner);
        const sech2 = 1 - tanhInner * tanhInner;
        const innerDeriv = c * (1 + 3 * 0.044715 * x * x);
        return 0.5 * (1 + tanhInner) + 0.5 * x * sech2 * innerDeriv;
    }
```

> Why GELU over ReLU? ReLU has a hard cutoff at zero—anything negative becomes exactly zero, killing gradients. GELU's smooth curve means gradients always flow, even for slightly negative inputs. This helps with training stability.

## MLP Block Implementation

We can implement MLPBlock by composing existing layers. Each position in the sequence is processed independently through the same dense layers. We cache all intermediate values during the forward pass so we don't need to recompute them during backward:

```javascript
class MLPBlock {
    dense1 = null;      // embeddingDim → hiddenDim
    activation = null;  // GELU
    dense2 = null;      // hiddenDim → embeddingDim
    
    // Caches for backpropagation
    cachedInputs = null;
    cachedHidden1 = null;   // After dense1
    cachedHidden2 = null;   // After activation

    constructor(embeddingDim, expansionFactor = 4) {
        const hiddenDim = embeddingDim * expansionFactor;
        
        // Reuse our DenseLayer and ActivationLayer from Part 1!
        this.dense1 = new DenseLayer(embeddingDim, hiddenDim);
        this.activation = new ActivationLayer("gelu");
        this.dense2 = new DenseLayer(hiddenDim, embeddingDim);
    }
}
```

### MLP Forward Pass

The forward pass processes each position through our layer stack, caching all intermediate results:

```javascript
// ... class MLPBlock
    forward(inputs) {
        // inputs: [seqLen][embeddingDim]
        const seqLen = inputs.length;
        
        // Cache everything we'll need for backward
        this.cachedInputs = inputs;
        this.cachedHidden1 = new Array(seqLen);
        this.cachedHidden2 = new Array(seqLen);
        
        const outputs = new Array(seqLen);
        
        for (let t = 0; t < seqLen; t++) {
            // Each position goes through: Dense1 → GELU → Dense2
            this.cachedHidden1[t] = this.dense1.forward(inputs[t]);
            this.cachedHidden2[t] = this.activation.forward(this.cachedHidden1[t]);
            outputs[t] = this.dense2.forward(this.cachedHidden2[t]);
        }
        
        return outputs;  // [seqLen][embeddingDim]
    }
```

The MLP treats each position identically, unlike attention which mixes information across positions. By caching `cachedHidden1` and `cachedHidden2` for every position, we have all the intermediate values needed for backpropagation.

### MLP Backward Pass

The backward pass propagates gradients back through each position using the cached values:

```javascript
// ... class MLPBlock
    backward(outputGradients, learningRate) {
        // outputGradients: [seqLen][embeddingDim]
        const seqLen = outputGradients.length;
        const inputGradients = new Array(seqLen);
        
        for (let t = 0; t < seqLen; t++) {
            // Restore layer caches for this position
            this.dense1.cachedInput = this.cachedInputs[t];
            this.activation.cachedInput = this.cachedHidden1[t];
            this.dense2.cachedInput = this.cachedHidden2[t];
            
            // Backward through the layers
            let grad = this.dense2.backward(outputGradients[t], learningRate);
            grad = this.activation.backward(grad);
            inputGradients[t] = this.dense1.backward(grad, learningRate);
        }
        
        return inputGradients;  // [seqLen][embeddingDim]
    }
```

> The weights update after processing each position. This is slightly different from accumulating all gradients first, but converges to the same solution.

# The Transformer Block

Now we can assemble the complete transformer block. Each block contains layer normalization, multi-head attention, another layer normalization, and an MLP, all connected with residual connections.

```
┌────────────────────────────────────────────────────────────────────────┐
│                         Transformer Block                              │
│                                                                        │
│   Input ──┬──► LayerNorm ──► Attention ──┬──► LayerNorm ──► MLP ──┬──► Output
│           │                              │                        │
│           └──────────── + ◄──────────────┘          + ◄───────────┘
│                     (residual)                  (residual)              │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

The residual connections are crucial. They let gradients flow directly backward without passing through the attention or MLP transformations, solving the vanishing gradient problem. They also let the network learn "refinements" rather than complete transformations—the attention and MLP blocks learn what to *add* to the representation.

```javascript
class TransformerBlock {
    layerNorm1 = null;
    attention = null;
    layerNorm2 = null;
    mlp = null;
    
    embeddingDim = 0;
    
    // Caches for residual connections
    cachedInput = null;
    cachedPostAttention = null;

    constructor(embeddingDim, numHeads, mlpExpansion = 4) {
        this.embeddingDim = embeddingDim;
        
        this.layerNorm1 = new LayerNormalization(embeddingDim);
        this.attention = new MultiHeadAttentionLayer(embeddingDim, numHeads);
        this.layerNorm2 = new LayerNormalization(embeddingDim);
        this.mlp = new MLPBlock(embeddingDim, mlpExpansion);
    }
}
```

The constructor creates all four sub-components. The `numHeads` parameter controls how many attention heads to use, and `mlpExpansion` (defaulting to 4) controls the MLP's hidden dimension.

### Transformer Block Forward Pass

```javascript
// ... class TransformerBlock
    forward(inputs) {
        // inputs: [seqLen][embeddingDim]
        const seqLen = inputs.length;
        
        // Cache input for residual
        this.cachedInput = inputs;
        
        // Pre-norm attention block
        const normalized1 = this.layerNorm1.forward(inputs);
        const attended = this.attention.forward(normalized1);
        
        // First residual connection: input + attention output
        this.cachedPostAttention = new Array(seqLen);
        for (let t = 0; t < seqLen; t++) {
            this.cachedPostAttention[t] = new Array(this.embeddingDim);
            for (let d = 0; d < this.embeddingDim; d++) {
                this.cachedPostAttention[t][d] = inputs[t][d] + attended[t][d];
            }
        }
        
        // Pre-norm MLP block
        const normalized2 = this.layerNorm2.forward(this.cachedPostAttention);
        const mlpOutput = this.mlp.forward(normalized2);
        
        // Second residual connection: post-attention + MLP output
        const outputs = new Array(seqLen);
        for (let t = 0; t < seqLen; t++) {
            outputs[t] = new Array(this.embeddingDim);
            for (let d = 0; d < this.embeddingDim; d++) {
                outputs[t][d] = this.cachedPostAttention[t][d] + mlpOutput[t][d];
            }
        }
        
        return outputs;  // [seqLen][embeddingDim]
    }
```

The forward pass follows this path:
1. LayerNorm → Attention → Add residual from input
2. LayerNorm → MLP → Add residual from step 1

We cache intermediate values at the residual connection points because backward needs them.

### Transformer Block Backward Pass

```javascript
// ... class TransformerBlock
    backward(outputGradients, learningRate) {
        // outputGradients: [seqLen][embeddingDim]
        const seqLen = outputGradients.length;
        
        // ====== Second residual backward ======
        // Output = postAttention + mlpOutput
        // Gradient flows to both paths
        const mlpOutputGrad = outputGradients;
        const postAttentionGrad1 = outputGradients;  // Direct path through residual
        
        // ====== MLP backward ======
        const normalized2Grad = this.mlp.backward(mlpOutputGrad, learningRate);
        const postAttentionGrad2 = this.layerNorm2.backward(normalized2Grad, learningRate);
        
        // Combine gradients at post-attention point
        const postAttentionGrad = new Array(seqLen);
        for (let t = 0; t < seqLen; t++) {
            postAttentionGrad[t] = new Array(this.embeddingDim);
            for (let d = 0; d < this.embeddingDim; d++) {
                postAttentionGrad[t][d] = postAttentionGrad1[t][d] + postAttentionGrad2[t][d];
            }
        }
        
        // ====== First residual backward ======
        // PostAttention = input + attended
        const attendedGrad = postAttentionGrad;
        const inputGrad1 = postAttentionGrad;  // Direct path through residual
        
        // ====== Attention backward ======
        const normalized1Grad = this.attention.backward(attendedGrad, learningRate);
        const inputGrad2 = this.layerNorm1.backward(normalized1Grad, learningRate);
        
        // Combine gradients at input point
        const inputGradients = new Array(seqLen);
        for (let t = 0; t < seqLen; t++) {
            inputGradients[t] = new Array(this.embeddingDim);
            for (let d = 0; d < this.embeddingDim; d++) {
                inputGradients[t][d] = inputGrad1[t][d] + inputGrad2[t][d];
            }
        }
        
        return inputGradients;
    }
```

The key insight in the backward pass is that residual connections split the gradient, it flows through both the transformation path (attention/MLP) and directly through the skip connection. At each residual point, we sum the gradients from both paths.

This is why residual connections help with gradient flow: even if the attention or MLP gradients vanish, the direct path always carries the full gradient signal.

# The Complete Model: ChatGPT

Now we can assemble all components into a complete GPT-style language model. The architecture stacks multiple transformer blocks between the embedding layer and output layer:

```javascript
class ChatGPT {
    embedding = null;       // Positional + token embeddings
    blocks = null;          // Array of transformer blocks
    finalNorm = null;       // Final layer normalization
    output = null;          // Output projection to vocabulary
    
    vocabSize = 0;
    embeddingDim = 0;
    numBlocks = 0;

    constructor(vocabSize, embeddingDim, numHeads, numBlocks, maxSeqLength) {
        this.vocabSize = vocabSize;
        this.embeddingDim = embeddingDim;
        this.numBlocks = numBlocks;
        
        // Embedding layer (token + positional)
        this.embedding = new PositionalEmbeddingLayer(vocabSize, embeddingDim, maxSeqLength);
        
        // Stack of transformer blocks
        this.blocks = new Array(numBlocks);
        for (let i = 0; i < numBlocks; i++) {
            this.blocks[i] = new TransformerBlock(embeddingDim, numHeads);
        }
        
        // Final layer normalization (GPT-2 style)
        this.finalNorm = new LayerNormalization(embeddingDim);
        
        // Output projection to vocabulary
        this.output = new OutputLayer(embeddingDim, vocabSize);
    }
}
```

`OutputLayer` projects hidden states to vocabulary size and applies softmax.

### ChatGPT Forward Pass

```javascript
// ... class ChatGPT
    forward(inputTokens) {
        // inputTokens: array of token IDs
        
        // Embed tokens with positional information
        let hidden = this.embedding.forward(inputTokens);
        
        // Pass through each transformer block
        for (let i = 0; i < this.numBlocks; i++) {
            hidden = this.blocks[i].forward(hidden);
        }
        
        // Final normalization
        hidden = this.finalNorm.forward(hidden);
        
        // Project to vocabulary probabilities
        const probabilities = this.output.forward(hidden);
        
        return probabilities;  // [seqLen][vocabSize]
    }
```

embeddings → transformer blocks → normalization → output probabilities. Each transformer block refines the representation, with attention gathering context and MLP processing it.

### ChatGPT Backward Pass

```javascript
// ... class ChatGPT
    backward(targetTokens, learningRate) {
        // Backward through output layer
        let gradients = this.output.backward(targetTokens, learningRate);
        
        // Backward through final normalization
        gradients = this.finalNorm.backward(gradients, learningRate);
        
        // Backward through transformer blocks (in reverse order)
        for (let i = this.numBlocks - 1; i >= 0; i--) {
            gradients = this.blocks[i].backward(gradients, learningRate);
        }
        
        // Backward through embeddings
        this.embedding.backward(gradients, learningRate);
    }
```

Backpropagation flows in reverse through the network. Each layer updates its own weights and passes gradients to the previous layer. The transformer blocks process in reverse order—last block first, first block last.

### Training Method

```javascript
// ... class ChatGPT
    train(inputTokens, targetTokens, learningRate) {
        // Forward pass
        const predictions = this.forward(inputTokens);
        
        // Compute loss
        const loss = Loss.crossEntropy(predictions, targetTokens);
        
        // Backward pass
        this.backward(targetTokens, learningRate);
        
        return loss;
    }
```

Training follows the same pattern as ChatRNN: forward pass to get predictions, compute cross-entropy loss, backward pass to update weights. The `Loss.crossEntropy` function works unchanged.

### Text Generation

```javascript
// ... class ChatGPT
    generate(startTokens, maxLength) {
        const generated = [];
        for (let i = 0; i < startTokens.length; i++) {
            generated.push(startTokens[i]);
        }
        
        for (let i = 0; i < maxLength; i++) {
            // Get predictions for current sequence
            const probs = this.forward(generated);
            const lastProbs = probs[probs.length - 1];
            
            // Sample next token from the probability distribution
            const nextToken = this.#sampleFromDistribution(lastProbs);
            generated.push(nextToken);
        }
        
        return generated;
    }
    
    #sampleFromDistribution(probs) {
        const random = Math.random();
        let cumulative = 0;
        
        for (let i = 0; i < probs.length; i++) {
            cumulative += probs[i];
            if (random < cumulative) {
                return i;
            }
        }
        
        return probs.length - 1;
    }
```

To generate text, run forward on the current sequence, get the probability distribution for the next token, sample from it, append the result, and repeat. The `#sampleFromDistribution` method performs weighted random selection—tokens with higher probability are more likely to be chosen.

# Putting It All Together

Let's train a small transformer on some text. We'll use modest dimensions to keep training fast—this is meant to be a proof of concept you can run and experiment with.

```javascript
// ==========================================
// Training a Mini ChatGPT
// ==========================================

// Training data - a small corpus of text
const trainingText = `
The cat sat on the mat. The dog sat on the log.
The cat chased the mouse. The dog chased the cat.
The mouse ran away. The cat ran after the mouse.
The dog watched the cat chase the mouse.
In the morning the sun rises in the east.
In the evening the sun sets in the west.
The birds sing in the morning. The owls hoot at night.
Stars twinkle in the night sky. The moon glows bright.
The quick brown fox jumps over the lazy dog.
The lazy dog sleeps all day. The quick fox hunts at night.
`;

// Create and train tokenizer
const tokenizer = new Tokenizer();
tokenizer.train(trainingText, 50);  // Small vocabulary for demo
const vocabSize = tokenizer.getVocabSize();
console.log(`Vocabulary size: ${vocabSize}`);

// Tokenize training data
const tokens = tokenizer.encode(trainingText);
console.log(`Training tokens: ${tokens.length}`);

// Model hyperparameters
const embeddingDim = 32;    // Small embedding dimension
const numHeads = 4;         // 4 attention heads (8 dims each)
const numBlocks = 2;        // 2 transformer blocks
const maxSeqLength = 64;    // Maximum sequence length

// Create model
const model = new ChatGPT(vocabSize, embeddingDim, numHeads, numBlocks, maxSeqLength);

// Count parameters
let totalParams = 0;
// Embedding: vocabSize * embeddingDim + maxSeqLength * embeddingDim
totalParams += vocabSize * embeddingDim + maxSeqLength * embeddingDim;
// Each transformer block: attention + MLP + layer norms
const attentionParams = 4 * embeddingDim * embeddingDim;  // Q, K, V, O projections
const mlpParams = 2 * embeddingDim * (4 * embeddingDim) + embeddingDim + 4 * embeddingDim;
const layerNormParams = 4 * embeddingDim;  // 2 norms * (gamma + beta)
totalParams += numBlocks * (attentionParams + mlpParams + layerNormParams);
// Output layer
totalParams += embeddingDim * vocabSize + vocabSize;
console.log(`Approximate parameters: ${totalParams.toLocaleString()}`);

// Training parameters
const learningRate = 0.01;
const epochs = 200;
const sequenceLength = 16;

// Training loop
console.log("\n=== Training ===");
for (let epoch = 0; epoch < epochs; epoch++) {
    let totalLoss = 0;
    let batchCount = 0;
    
    // Slide window across training data
    for (let start = 0; start < tokens.length - sequenceLength - 1; start += 8) {
        const inputTokens = tokens.slice(start, start + sequenceLength);
        const targetTokens = tokens.slice(start + 1, start + sequenceLength + 1);
        
        const loss = model.train(inputTokens, targetTokens, learningRate);
        totalLoss += loss;
        batchCount++;
    }
    
    if (epoch % 20 === 0) {
        const avgLoss = totalLoss / batchCount;
        console.log(`Epoch ${epoch}: Loss = ${avgLoss.toFixed(4)}`);
        
        // Generate sample
        const prompt = tokenizer.encode("The ");
        const generated = model.generate(prompt, 10);
        const text = tokenizer.decode(generated);
        console.log(`  Sample: "${text}"`);
    }
}

// Final generation
console.log("\n=== Generation Examples ===");
const prompts = ["The cat ", "In the ", "The quick "];
for (const promptText of prompts) {
    const prompt = tokenizer.encode(promptText);
    const generated = model.generate(prompt, 15);
    const text = tokenizer.decode(generated);
    console.log(`"${promptText}" → "${text}"`);
}
```

This creates a transformer with approximately 100,000 parameters—tiny by modern standards, but enough to learn patterns in our small training corpus.

## Understanding the Model Size

Let's break down where those parameters come from:

| Component | Calculation | Parameters |
|-----------|-------------|------------|
| Token embeddings | vocabSize × embeddingDim | ~10,000 |
| Position embeddings | maxSeqLength × embeddingDim | 2,048 |
| Attention (per block) | 4 × embeddingDim² | 4,096 |
| MLP (per block) | 2 × embeddingDim × 4×embeddingDim | 8,192 |
| Layer norms (per block) | 4 × embeddingDim | 128 |
| Output projection | embeddingDim × vocabSize | ~10,000 |

With 2 blocks, this totals roughly 100,000 parameters. GPT-2 Small has 124 million parameters (1,200× larger), and GPT-3 has 175 billion (1.75 million times larger).

## What You Should See

When you run this code:

1. **Loss should decrease** from around 4-5 (random guessing) toward 1-2 (learning patterns)
2. **Generated text should improve** from random tokens to word fragments to coherent phrases
3. **Training completes in minutes** on a modern computer—attention is O(n²) but with short sequences it's manageable

The model won't produce Shakespeare, but it should learn basic patterns like "The cat..." often leading to animal actions, or "In the..." often leading to time words.

# What's Next?

We've built a working GPT-style transformer from scratch. The core architecture matches what powers ChatGPT, Claude, and other modern language models. But production systems have many enhancements:

**Training improvements:**
- **Batch processing**: Train on multiple sequences simultaneously for better GPU utilization
- **Adam optimizer**: Adaptive learning rates per parameter, much faster than vanilla SGD
- **Learning rate scheduling**: Warm up then decay the learning rate
- **Gradient clipping**: Prevent exploding gradients

**Architecture improvements:**
- **Dropout**: Randomly zero activations during training to prevent overfitting
- **Tied embeddings**: Share weights between input and output embeddings
- **Rotary Position Embeddings (RoPE)**: Better position encoding for long sequences
- **Flash Attention**: Memory-efficient attention computation

**Scale:**
- Real models train on trillions of tokens
- Embedding dimensions of 4096+ with dozens of layers
- Thousands of GPUs training for months

The principles remain the same though. It's still embeddings, attention, feed-forward networks, residual connections, and gradient descent. The complexity comes from scale and optimization tricks.

You now understand the fundamental architecture powering the AI revolution. Every conversation with ChatGPT follows this same pattern: tokens in, embeddings, attention blocks, probabilities out, sample next token, repeat.