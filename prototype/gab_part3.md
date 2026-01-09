# Let's Build a ChatGPT,Part 3 / 4: Recurrent Neural Networks

Up to this point, we've built the foundations of neural networks: trainable neurons, dense layers, activation functions, and a BPE tokenizer. And we were able to train a network to solve XOR. But language is different. It flows. One word follows another, each building on what came before. "The cat sat on the mat" makes sense because each word carries forward context from previous words. Our dense networks see inputs as isolated snapshots, they have no concept of sequence or flow.

In this part, we'll build a Recurrent Neural Network (RNN) that maintains memory across a sequence. We'll combine it with our tokenizer to create ChatRNN. ChatRNN will chain together several components to create a language model:

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  Token IDs  │ ──▶ │  Embedding  │ ──▶ │ RNN Layers  │ ──▶ │   Output    │ ──▶ │Probabilities│
└─────────────┘      └─────────────┘      └─────────────┘      └─────────────┘      └─────────────┘
       │                    │                    │                    │                    │
       ▼                    ▼                    ▼                    ▼                    ▼
   [42, 17]            Vectors for          Process with        Next-token              Softmax
                       each token         sequential memory     predictions             output
```

We already have dense layers and activation functions. Now we need:
- **Embedding Layer**: Converts discrete token IDs into continuous vectors
- **RNN Layer**: Processes sequences while maintaining memory (using our existing `ActivationLayer`)
- **Output Layer**: Converts hidden states to vocabulary probabilities (built on `DenseLayer`)

## The Embedding Layer

Our tokenizer produces integers,token ID 42, token ID 17. But neural networks work with continuous numbers that can be differentiated. We can't compute the gradient of "token 42."

The embedding layer bridges this gap. It's a lookup table: each token ID maps to a unique vector of floating-point numbers. These vectors start random, but through training, tokens appearing in similar contexts develop similar vectors.

```javascript
class EmbeddingLayer {
    weights = null;
    vocabSize = 0;
    embeddingDim = 0;
    cachedInputTokens = null;

    constructor(vocabSize, embeddingDim) {
        this.vocabSize = vocabSize;
        this.embeddingDim = embeddingDim;
        
        // Xavier initialization for stable gradients
        const scale = Math.sqrt(1.0 / embeddingDim);
        
        this.weights = new Array(vocabSize);
        for (let i = 0; i < vocabSize; i++) {
            this.weights[i] = new Array(embeddingDim);
            for (let j = 0; j < embeddingDim; j++) {
                this.weights[i][j] = (Math.random() * 2 - 1) * scale;
            }
        }
    }
}
```

The constructor creates our embedding matrix,one row per token in the vocabulary, each row containing `embeddingDim` numbers. We scale by `1/sqrt(embeddingDim)` to keep variance reasonable regardless of dimension size.

Visually, that 64 dimension embedding matrix would look like this:
```
                        Embedding Weights Matrix
                        (vocabSize × embeddingDim)
                        
         dim 0    dim 1    dim 2    dim 3    dim 4   ...  dim 63
       ┌────────┬────────┬────────┬────────┬────────┬───┬────────┐
     0 │  0.12  │ -0.34  │  0.07  │  0.91  │ -0.23  │...│  0.45  │  "the"
       ├────────┼────────┼────────┼────────┼────────┼───┼────────┤
     1 │ -0.56  │  0.23  │  0.44  │ -0.18  │  0.67  │...│ -0.31  │  "cat"
       ├────────┼────────┼────────┼────────┼────────┼───┼────────┤
     2 │  0.89  │  0.02  │ -0.61  │  0.33  │ -0.45  │...│  0.12  │  "dog"
       ├────────┼────────┼────────┼────────┼────────┼───┼────────┤
   ... │  ...   │  ...   │  ...   │  ...   │  ...   │...│  ...   │
       ├────────┼────────┼────────┼────────┼────────┼───┼────────┤
    42 │  0.34  │ -0.78  │  0.15  │  0.52  │ -0.09  │...│  0.67  │  "sat"
       └────────┴────────┴────────┴────────┴────────┴───┴────────┘
Token                                                              Token
 ID                                                                Text
```

### Embedding Forward Pass

The forward pass converts a sequence of token IDs into a sequence of vectors. For each token, we grab its row from the weight matrix.

```javascript
// ... class EmbeddingLayer
    forward(inputTokens) {
        // Cache for backprop,we need to know which tokens were looked up
        this.cachedInputTokens = new Array(inputTokens.length);
        for (let i = 0; i < inputTokens.length; i++) {
            this.cachedInputTokens[i] = inputTokens[i];
        }
        
        // Look up each token's embedding vector
        const output = new Array(inputTokens.length);
        for (let i = 0; i < inputTokens.length; i++) {
            const tokenId = inputTokens[i];
            output[i] = new Array(this.embeddingDim);
            for (let j = 0; j < this.embeddingDim; j++) {
                output[i][j] = this.weights[tokenId][j];
            }
        }
        
        return output;  // Shape: [sequenceLength][embeddingDim]
    }
```

If we input `[42, 17, 8]` with 64-dimensional embeddings, we get back three 64-element vectors. These are what the RNN will process.

```
    Input: token ID 42 ("sat")
                │
                ▼
    ┌───────────────────────────────┐
    │ Look up row 42 in the matrix  │
    └───────────────────────────────┘
                │
                ▼
    Output: [0.34, -0.78, 0.15, 0.52, -0.09, ..., 0.67]
            └──────────────────────────────────────────┘
                 embeddingDim - dimensional vector
```

### Embedding Backward Pass

During backpropagation, we receive gradients telling us how each embedding value should change. Unlike dense layers where every weight affects every output, embeddings are sparse: only tokens that appeared in the input get updated.

```javascript
// ... class EmbeddingLayer
    backward(outputGradients, learningRate) {
        // Only update embeddings for tokens that were actually used
        for (let i = 0; i < this.cachedInputTokens.length; i++) {
            const tokenId = this.cachedInputTokens[i];
            const gradient = outputGradients[i];

            for (let j = 0; j < this.embeddingDim; j++) {
                this.weights[tokenId][j] -= learningRate * gradient[j];
            }
        }
        
        // First layer,no gradients to pass back
        return null;
    }
```

> If a token appears multiple times, its embedding gets updated multiple times, each occurrence contributed to the loss.

## The RNN Cell

The heart of recurrence is the RNN cell. Unlike our `Neuron` class from Part 1 that processes inputs in isolation, an RNN cell takes TWO inputs: the current data AND a hidden state from the previous timestep. In this context, "time" and "timestep" doesn't mean clock time, it means position in sequence. For the sentence "The cat sat":

* Timestep 0 → "The"
* Timestep 1 → "cat"
* Timestep 2 → "sat"

> The terminology comes from RNNs originally being designed for time-series data (audio signals, stock prices, sensor readings) where each step literally was a moment in time. The naming stuck even when applying RNNs to non-temporal sequences like text.

The RNN cell looks like this:

```
                                    [RNN Cell]
                    ┌─────────────────────────────────────────┐
                    │                                         │
                    │   ┌────────┐                            │
   Current Input ──►│──►│   Wx   │──┐                         │
        (x_t)       │   └────────┘  │   ┌───────┐             │
                    │               ├──►│ + bias│──► preAct ──┼──► (to tanh)
                    │   ┌────────┐  │   └───────┘             │
 Previous Hidden ──►│──►│   Wh   │──┘                         │
      (h_t-1)       │   └────────┘                            │
                    │                                         │
                    └─────────────────────────────────────────┘

              preActivation = Wx · x_t  +  Wh · h_t-1  +  bias
```

The **hidden state** is a vector of floats — the network's "memory." If `hiddenSize = 128`, each hidden state is an array of 128 numbers:

```
After "The":     [0.23, -0.87, 0.45, ..., 0.12]    ← 128 floats
After "The cat": [-0.34, 0.91, -0.22, ..., 0.67]   ← 128 floats 
```

It's a compressed representation of everything the network has seen so far in the sequence. Each new token transforms the hidden state, blending new information with the accumulated context:

```
[0, 0, 0, ...] → "The" → [0.23, -0.87, ...] → "cat" → [-0.34, 0.91, ...] → "sat" → ...
    initial                after 1 token              after 2 tokens
```

> What do these numbers mean? The network learns that through training. They might encode patterns like "we're mid-sentence" or "an animal was just mentioned", though in practice the representations aren't human-interpretable. The hidden size is a hyperparameter: bigger means more capacity to remember, but more parameters to train.

Notice that we compute the pre-activation (before tanh) in the RNN cell. The tanh activation will be applied separately using the existing `ActivationLayer`.

```javascript
class RNNCell {
    weightsX = null;    // Input transformation. Shape: [inputSize][hiddenSize]
    weightsH = null;    // Hidden state. Shape: [hiddenSize][hiddenSize]
    bias = null;
    hiddenSize = 0;
    inputSize = 0;

    constructor(inputSize, hiddenSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        
        // Xavier initialization for both weight matrices
        const scaleX = Math.sqrt(2.0 / (inputSize + hiddenSize));
        const scaleH = Math.sqrt(2.0 / (hiddenSize + hiddenSize));
        
        this.weightsX = new Array(inputSize);
        for (let i = 0; i < inputSize; i++) {
            this.weightsX[i] = new Array(hiddenSize);
            for (let j = 0; j < hiddenSize; j++) {
                this.weightsX[i][j] = (Math.random() * 2 - 1) * scaleX;
            }
        }
        
        this.weightsH = new Array(hiddenSize);
        for (let i = 0; i < hiddenSize; i++) {
            this.weightsH[i] = new Array(hiddenSize);
            for (let j = 0; j < hiddenSize; j++) {
                this.weightsH[i][j] = (Math.random() * 2 - 1) * scaleH;
            }
        }
        
        this.bias = new Array(hiddenSize);
        for (let i = 0; i < hiddenSize; i++) {
            this.bias[i] = 0;
        }
    }

    forward(input, prevHidden) {
        // Compute pre-activation only,activation applied separately
        const preActivation = new Array(this.hiddenSize);
        
        for (let h = 0; h < this.hiddenSize; h++) {
            let sum = this.bias[h];
            
            // Add weighted input
            for (let i = 0; i < this.inputSize; i++) {
                sum += input[i] * this.weightsX[i][h];
            }
            
            // Add weighted previous hidden state
            for (let i = 0; i < this.hiddenSize; i++) {
                sum += prevHidden[i] * this.weightsH[i][h];
            }
            
            preActivation[h] = sum;
        }
        
        return preActivation;
    }
}
```

The forward pass combines both inputs through their respective weight matrices and adds them together with a bias. The tanh activation will be applied by a separate layer to keep values between -1 and 1.

## The RNN Layer

A single RNN cell processes one timestep. To handle sequences, we wrap it in a layer that "unrolls" across time, meaning it processes each position in the sequence one after another. We'll use an ```ActivationLayer``` for the tanh activation:

```
Time 0          Time 1          Time 2          Time 3
  │               │               │               │
  ▼               ▼               ▼               ▼
┌─────┐        ┌─────┐        ┌─────┐        ┌─────┐
│Cell │        │Cell │        │Cell │        │Cell │
└──┬──┘        └──┬──┘        └──┬──┘        └──┬──┘
   │              │              │              │
   ▼              ▼              ▼              ▼
┌─────┐        ┌─────┐        ┌─────┐        ┌─────┐
│tanh │──h0───►│tanh │──h1───►│tanh │──h2───►│tanh │──h3──►
└─────┘        └─────┘        └─────┘        └─────┘
   │              │              │              │
   ▼              ▼              ▼              ▼
Output 0      Output 1      Output 2      Output 3
```

Each "cell" is the same cell reused. The hidden state flows left to right, carrying context forward. Notice how the tanh activation is a separate step.

```javascript
class RNNLayer {
    cell = null;
    activation = null;  // ActivationLayer
    hiddenSize = 0;
    
    // Caches for backpropagation through time
    cachedInputs = null;            // Shape [sequenceLength]
    cachedHiddens = null;           // Shape [sequenceLength + 1][hiddenSize]
    cachedPreActivations = null;    // Shape [sequenceLength]

    constructor(inputSize, hiddenSize) {
        this.hiddenSize = hiddenSize;
        this.cell = new RNNCell(inputSize, hiddenSize);
        
        this.activation = new ActivationLayer("tanh");
    }
}
```

# The RNN Forward Pass: A Detailed Walkthrough

The RNN forward pass is where sequence processing happens. Unlike a dense layer that sees all inputs simultaneously, the RNN processes one token at a time, maintaining a "memory" that flows from one timestep to the next. Imagine reading a sentence word by word—after each word, your understanding of the sentence updates. The RNN works the same way: "The" becomes a hidden state h₀, then "The cat" becomes h₁, then "The cat sat" becomes h₂, and so on.

The `inputSequence` parameter is an array of embedding vectors. If we tokenized "The cat sat" and looked up embeddings, we'd have three 64-dimensional vectors, one per token. The `sequenceLength` is simply the length of this array—in this case, 3. Each embedding vector represents one token's position in the learned semantic space.

We create three caches before processing begins. The `cachedInputs` array stores the input embedding at each timestep. The `cachedHiddens` array stores hidden states, and it has length `sequenceLength + 1` because we need to store both the initial hidden state (all zeros, before any input) and the hidden state after each timestep. The `cachedPreActivations` array stores the values before the tanh activation is applied. We cache all of this because backpropagation through time needs to revisit every timestep in reverse order.

```javascript
forward(inputSequence) {
    const sequenceLength = inputSequence.length;
    
    // Initialize caches
    this.cachedInputs = new Array(sequenceLength);
    this.cachedHiddens = new Array(sequenceLength + 1);
    this.cachedPreActivations = new Array(sequenceLength);
    
    // Start with zero hidden state
    this.cachedHiddens[0] = new Array(this.hiddenSize);
    for (let i = 0; i < this.hiddenSize; i++) {
        this.cachedHiddens[0][i] = 0;
    }
    
    const outputs = new Array(sequenceLength);
    
    for (let t = 0; t < sequenceLength; t++) {
        this.cachedInputs[t] = inputSequence[t];
        
        // Compute pre-activation using RNN cell
        const preActivation = this.cell.forward(
            inputSequence[t], 
            this.cachedHiddens[t]
        );
        this.cachedPreActivations[t] = preActivation;
        
        // Apply tanh 
        const newHidden = this.activation.forward(preActivation);
        
        this.cachedHiddens[t + 1] = newHidden;
        outputs[t] = newHidden;
    }
    
    return outputs;  // Hidden state at each timestep
}
```

The main loop processes tokens left to right, one at a time. At each timestep `t`, we cache the current input embedding, then call `this.cell.forward()` which computes `Wₓ · input + Wₕ · prevHidden + bias`. At timestep 0, the previous hidden state is all zeros, so only the input term contributes. At subsequent timesteps, both the current input and the accumulated context from previous tokens influence the result. We then pass this pre-activation through tanh to get the new hidden state, store it at `cachedHiddens[t + 1]`, and add it to outputs.

The returned `outputs` array contains the hidden state after processing each token. Each hidden state encodes cumulative context: h₁ knows about "The", h₂ knows about "The cat", h₃ knows about "The cat sat". The same weights `Wₓ`, `Wₕ`, and `bias` are reused at every timestep,this weight sharing is what allows RNNs to handle variable-length sequences with a fixed number of parameters.

### RNN Backward Pass: Backpropagation Through Time

This is where things get interesting. We need to propagate gradients backward through time — from the last timestep to the first. This algorithm is called **Backpropagation Through Time (BPTT)**.

Weights are shared across all timesteps. The same `weightsX`, `weightsH`, and `bias` are used at every position in the sequence. So we accumulate gradients from every timestep before making a single update.

```javascript
// ... class RNNLayer
    backward(outputGradients, learningRate) {
        const sequenceLength = this.cachedInputs.length;
        
        // Accumulate gradients across all timesteps. Initialize to 0
        const weightsXGrad = this.#createZeroMatrix(this.cell.inputSize, this.hiddenSize);
        const weightsHGrad = this.#createZeroMatrix(this.hiddenSize, this.hiddenSize);
        const biasGrad = new Array(this.hiddenSize);
        for (let i = 0; i < this.hiddenSize; i++) {
            biasGrad[i] = 0;
        }

        // Gradient flowing back from future timesteps
        let hiddenGradient = new Array(this.hiddenSize);
        for (let i = 0; i < this.hiddenSize; i++) {
            hiddenGradient[i] = 0;
        }
        
        const inputGradients = new Array(sequenceLength);
```

`hiddenGradient` carries error from "the future", timesteps we've already processed (remember, we're going backward). At the last timestep there is no future, so it starts as zeros. `inputGradients` will hold gradients to pass back to the embedding layer.


Next, we process the sequence in reverse: last token first, first token last. This lets error flow backward through the chain of hidden states. Each hidden state receives error from two sources:
1. `outputGradients[t]` — how wrong was our prediction at this position?
2. `hiddenGradient` — how did this hidden state hurt predictions at *future* positions?

We sum them because this hidden state contributed to both.

```javascript
        // Process timesteps in reverse order
        for (let t = sequenceLength - 1; t >= 0; t--) {
            // Total gradient = from output layer + from future timesteps
            const totalHiddenGrad = new Array(this.hiddenSize);
            for (let h = 0; h < this.hiddenSize; h++) {
                totalHiddenGrad[h] = outputGradients[t][h] + hiddenGradient[h];
            }

            // Backprop through tanh using ActivationLayer
            this.activation.cachedInputs = this.cachedPreActivations[t];
            const preActivationGrad = this.activation.backward(totalHiddenGrad);
```

The gradient so far is with respect to the tanh *output*. We need the gradient with respect to the tanh *input* (the pre-activation). The `ActivationLayer` handles this, it uses the cached pre-activation values to compute the tanh derivative.

Next we compute how each parameter contributed to the error at this timestep:
- `weightsXGrad` — gradient for input-to-hidden weights, using the input at this timestep
- `weightsHGrad` — gradient for hidden-to-hidden weights, using the *previous* hidden state
- `biasGrad` — gradient for biases

```javascript
            // Accumulate weight gradients
            const input = this.cachedInputs[t];
            const prevHidden = this.cachedHiddens[t];
            
            for (let h = 0; h < this.hiddenSize; h++) {
                for (let i = 0; i < this.cell.inputSize; i++) {
                    weightsXGrad[i][h] += preActivationGrad[h] * input[i];
                }
                
                for (let i = 0; i < this.hiddenSize; i++) {
                    weightsHGrad[i][h] += preActivationGrad[h] * prevHidden[i];
                }
                
                biasGrad[h] += preActivationGrad[h];
            }
```

The embedding layer needs to know: "how should each input value have been different?" We compute this by multiplying the pre-activation gradient by `weightsX` — the same weights that transformed the input during the forward pass.

```javascript
            // Gradient to pass to embedding layer
            inputGradients[t] = new Array(this.cell.inputSize);
            for (let i = 0; i < this.cell.inputSize; i++) {
                let grad = 0;
                for (let h = 0; h < this.hiddenSize; h++) {
                    grad += preActivationGrad[h] * this.cell.weightsX[i][h];
                }
                inputGradients[t][i] = grad;
            }
```

This is the "recurrent" part of backpropagation. We compute how the *previous* hidden state contributed to error, using `weightsH`. This becomes `hiddenGradient` for the next iteration (which processes the previous timestep).

```javascript
            // Gradient to pass to previous timestep
            hiddenGradient = new Array(this.hiddenSize);
            for (let i = 0; i < this.hiddenSize; i++) {
                let grad = 0;
                for (let h = 0; h < this.hiddenSize; h++) {
                    grad += preActivationGrad[h] * this.cell.weightsH[i][h];
                }
                hiddenGradient[i] = grad;
            }
        }
```

After processing all timesteps, we finally update the weights. Each parameter is adjusted based on its total contribution to error across the entire sequence.

```javascript
        // Apply accumulated gradients
        for (let i = 0; i < this.cell.inputSize; i++) {
            for (let h = 0; h < this.hiddenSize; h++) {
                this.cell.weightsX[i][h] -= learningRate * weightsXGrad[i][h];
            }
        }
        
        for (let i = 0; i < this.hiddenSize; i++) {
            for (let h = 0; h < this.hiddenSize; h++) {
                this.cell.weightsH[i][h] -= learningRate * weightsHGrad[i][h];
            }
        }
        
        for (let h = 0; h < this.hiddenSize; h++) {
            this.cell.bias[h] -= learningRate * biasGrad[h];
        }
        
        return inputGradients;
    }
```

We also need to add a helper function to create a zero matrix, as this is going to be a common operation to perform.

```javascript
    #createZeroMatrix(rows, cols) {
        const matrix = new Array(rows);
        for (let i = 0; i < rows; i++) {
            matrix[i] = new Array(cols);
            for (let j = 0; j < cols; j++) {
                matrix[i][j] = 0;
            }
        }
        return matrix;
    }
```

## The Output Layer

The RNN produces hidden states. We need predictions: probability distributions over the vocabulary. The output layer projects hidden states to vocabulary size, then applies softmax.

> The Softmax activation function is a mathematical function that converts a vector of raw numbers (called logits) into a vector of probabilities. All probabilities sum to 1.

This is similar to the `DenseLayer`, but specialized for sequence classification. It computes a linear transformation followed by softmax, and integrates cross-entropy loss for cleaner gradients.

```javascript
class OutputLayer {
    weights = null;
    bias = null;
    hiddenSize = 0;
    vocabSize = 0;
    cachedInputs = null;
    cachedOutputs = null;

    constructor(hiddenSize, vocabSize) {
        this.hiddenSize = hiddenSize;
        this.vocabSize = vocabSize;
        
        const scale = Math.sqrt(2.0 / (hiddenSize + vocabSize));
        
        this.weights = new Array(hiddenSize);
        for (let i = 0; i < hiddenSize; i++) {
            this.weights[i] = new Array(vocabSize);
            for (let j = 0; j < vocabSize; j++) {
                this.weights[i][j] = (Math.random() * 2 - 1) * scale;
            }
        }
        
        this.bias = new Array(vocabSize);
        for (let i = 0; i < vocabSize; i++) {
            this.bias[i] = 0;
        }
    }
}
```

### Softmax

The RNN produces a hidden state at every step. This hidden state is a vector (e.g., 128 numbers) representing the "memory" or "context" of the sentence so far. But we don't want memory; we want a prediction. If our vocabulary has 1,000 words, we want to know: "Which of these 1,000 words comes next?"

The Output Layer bridges this gap. It has two distinct jobs:

Project to Vocabulary (The Logits): It transforms the 128-dimension hidden state into a 1,000-dimension vector (one number for every word in our dictionary). These raw numbers are called logits.

* A high logit (e.g., 15.5) means the network thinks this word is very likely.
* A low or negative logit (e.g., -5.0) means the network thinks this word is unlikely.

> Note: These are just raw scores. They don't sum to 1, and they aren't percentages yet.

Convert to Probabilities (Softmax): We need to turn those raw scores into probabilities so we can say, "There is a 90% chance the next word is 'cat' and a 1% chance it is 'dog'."

> Softmax could be it's own activation layer, but there is mathematical conveniance in undling the softmax activation inside the output layer

```javascript
// ... class OutputLayer
    #softmax(logits) {
        // Find max for numerical stability
        let maxLogit = logits[0];
        for (let i = 1; i < logits.length; i++) {
            if (logits[i] > maxLogit) {
                maxLogit = logits[i];
            }
        }
        
        // Compute exp(logit - max) and sum
        const expValues = new Array(logits.length);
        let expSum = 0;
        for (let i = 0; i < logits.length; i++) {
            expValues[i] = Math.exp(logits[i] - maxLogit);
            expSum += expValues[i];
        }
        
        // Normalize
        const probabilities = new Array(logits.length);
        for (let i = 0; i < logits.length; i++) {
            probabilities[i] = expValues[i] / expSum;
        }
        
        return probabilities;
    }
```

### Output Forward Pass

The RNN gives us hidden states—one per timestep, each a vector of `hiddenSize` floats encoding the context so far. But we need predictions: which token comes next? The output layer's job is to convert each hidden state into a probability distribution over the entire vocabulary.

This happens in two stages. First, we project the hidden state to vocabulary size—if `hiddenSize` is 128 and `vocabSize` is 1000, we're going from 128 numbers to 1000 numbers. This projection produces raw scores called logits. Second, we run softmax to convert those logits into probabilities that sum to 1.

```javascript
// ... class OutputLayer
    forward(hiddenSequence) {
        // Cache inputs for backpropagation
        this.cachedInputs = new Array(hiddenSequence.length);
        for (let t = 0; t < hiddenSequence.length; t++) {
            this.cachedInputs[t] = new Array(this.hiddenSize);
            for (let h = 0; h < this.hiddenSize; h++) {
                this.cachedInputs[t][h] = hiddenSequence[t][h];
            }
        }
        
        this.cachedOutputs = new Array(hiddenSequence.length);
        const outputs = new Array(hiddenSequence.length);
```

We cache the inputs because backpropagation will need them—to compute weight gradients, we need to know what hidden values were multiplied by those weights during the forward pass.

Now we process each timestep. The hidden state is a compressed representation of the sequence so far. We need to "decode" it into a score for every possible next token.

```javascript
        for (let t = 0; t < hiddenSequence.length; t++) {
            const hidden = hiddenSequence[t];
            
            // Compute logits: one score per vocabulary token
            const logits = new Array(this.vocabSize);
            for (let v = 0; v < this.vocabSize; v++) {
                let sum = this.bias[v];
                for (let h = 0; h < this.hiddenSize; h++) {
                    sum += hidden[h] * this.weights[h][v];
                }
                logits[v] = sum;
            }
```

Each logit is a weighted sum of the hidden state values, plus a bias. The weight matrix acts as a lookup in reverse: instead of asking "what does token 42 mean?" (embedding), we're asking "how much does this hidden state look like it should predict token 42?" High logit means the network thinks that token is likely. Low or negative logit means unlikely.

The weights are learned during training. If certain hidden state patterns consistently precede token 42, the weights connecting those patterns to logit 42 will grow larger.

```javascript
            // Convert logits to probabilities
            outputs[t] = this.#softmax(logits);
            this.cachedOutputs[t] = outputs[t];
        }
        
        return outputs;  // Shape: [sequenceLength][vocabSize]
    }
```

Softmax exponentiates each logit and normalizes so the results sum to 1. This gives us a proper probability distribution—we can now say "70% chance the next token is 'cat', 20% chance it's 'dog', etc." We cache these probabilities too; the backward pass needs them to compute gradients.

The returned `outputs` array contains one probability distribution per timestep. During training, we compare each distribution against the actual next token to compute loss. During generation, we sample from the final distribution to pick the next token.

### Output Backward Pass

The backward pass needs to answer two questions: how should the weights change to make better predictions, and what gradients should flow back to the RNN?

When softmax and cross-entropy loss are combined, the gradient simplifies to `predicted - target`. If the network predicted 0.7 probability for the correct token, the gradient is `0.7 - 1 = -0.3`. For every wrong token with probability 0.1, the gradient is `0.1 - 0 = 0.1`. This works because cross-entropy loss is defined specifically to pair with softmax—the logarithm in cross-entropy cancels the exponential in softmax during differentiation.

```javascript
// ... class OutputLayer
    backward(targetTokens, learningRate) {
        const sequenceLength = this.cachedInputs.length;
        const hiddenGradients = new Array(sequenceLength);
        
        // Accumulate weight gradients across all timesteps
        const weightsGrad = this.#createZeroMatrix(this.hiddenSize, this.vocabSize);
        const biasGrad = new Array(this.vocabSize);
        for (let i = 0; i < this.vocabSize; i++) {
            biasGrad[i] = 0;
        }
```

Like the RNN layer, we accumulate gradients before updating. The same weights transform every hidden state in the sequence, so each timestep contributes to how those weights should change.

```javascript
        for (let t = 0; t < sequenceLength; t++) {
            const hidden = this.cachedInputs[t];
            const probs = this.cachedOutputs[t];
            const targetToken = targetTokens[t];
```

We retrieve three things from the forward pass: the hidden state that was input to this layer, the probability distribution the network produced, and the token that should have been predicted at this position.

The target is conceptually a one-hot vector—a vector of all zeros with a single 1 at the correct token's index. If the vocabulary has 1000 tokens and the correct answer is token 42, the target is `[0, 0, 0, ..., 1, ..., 0]` with the 1 at position 42. We want the network's output to match this: probability 1 for the correct token, probability 0 for everything else.

The gradient is `predicted - target`. For the correct token, that's `probs[v] - 1`. If the network predicted 0.8, the gradient is -0.2, meaning "you were 0.2 too low." For wrong tokens, the target is 0, so the gradient is just `probs[v] - 0 = probs[v]`. If the network gave a wrong token 0.15 probability, the gradient is +0.15, meaning "you were 0.15 too high."

```javascript
            // Softmax + cross-entropy gradient: predicted - target
            const outputGrad = new Array(this.vocabSize);
            for (let v = 0; v < this.vocabSize; v++) {
                if (v === targetToken) {
                    outputGrad[v] = probs[v] - 1;  // Target was 1
                } else {
                    outputGrad[v] = probs[v];      // Target was 0
                }
            }
```

Now we need to figure out which weights were responsible for the error. During the forward pass, each logit was computed as `sum of (hidden[h] * weights[h][v])`. If a logit was too high (positive gradient), we want to reduce the weights that contributed to it. If too low (negative gradient), increase them. How much? Proportional to how active the hidden unit was—if `hidden[h]` was large, that weight had more influence and needs a larger correction.

```javascript
            // Weight gradients
            for (let h = 0; h < this.hiddenSize; h++) {
                for (let v = 0; v < this.vocabSize; v++) {
                    weightsGrad[h][v] += outputGrad[v] * hidden[h];
                }
            }
            
            // Bias gradients
            for (let v = 0; v < this.vocabSize; v++) {
                biasGrad[v] += outputGrad[v];
            }
```

The RNN needs to know: "how should your hidden states have been different?" We're passing blame backward. If a vocabulary logit `v` was wrong, every hidden unit that connected to it shares some responsibility—proportional to how strong that connection was. A hidden unit with a large positive weight to an overconfident wrong answer needs to be smaller next time.

```javascript
            // Gradient to pass back to the RNN layer
            hiddenGradients[t] = new Array(this.hiddenSize);
            for (let h = 0; h < this.hiddenSize; h++) {
                let grad = 0;
                for (let v = 0; v < this.vocabSize; v++) {
                    grad += outputGrad[v] * this.weights[h][v];
                }
                hiddenGradients[t][h] = grad;
            }
        }
```

After accumulating gradients from every timestep, we update the parameters:

```javascript
        // Apply accumulated gradients to weights
        for (let h = 0; h < this.hiddenSize; h++) {
            for (let v = 0; v < this.vocabSize; v++) {
                this.weights[h][v] -= learningRate * weightsGrad[h][v];
            }
        }
        
        // Apply accumulated gradients to biases
        for (let v = 0; v < this.vocabSize; v++) {
            this.bias[v] -= learningRate * biasGrad[v];
        }
        
        return hiddenGradients;
    }
```

The returned `hiddenGradients` array has one gradient vector per timestep. These flow into the RNN's backward pass, driving updates through the recurrent weights, and eventually back to the embedding layer.

We also need the helper function to create zero-initialized matrices:

```javascript
// ... class OutputLayer
    #createZeroMatrix(rows, cols) {
        const matrix = new Array(rows);
        for (let i = 0; i < rows; i++) {
            matrix[i] = new Array(cols);
            for (let j = 0; j < cols; j++) {
                matrix[i][j] = 0;
            }
        }
        return matrix;
    }
```

## Cross-Entropy Loss

We need a way to measure how wrong our predictions are. The output layer produces probability distributions over the vocabulary—one distribution per timestep. Cross-entropy loss compares each distribution against the actual next token by computing `-log(probability assigned to the correct token)`. If the network assigns 90% probability to the correct token, loss is low (0.1). If it assigns 1%, loss is high (4.6). This punishes confident wrong answers harshly, which is exactly what we want during training.

The implementation loops through each timestep, looks up the probability the network assigned to the correct token, and accumulates `-log(prob)`. We add a tiny `epsilon` to prevent `log(0)` when the network assigns near-zero probability to the correct answer. The return value is a single number: the average loss across all positions in the sequence. We use this to monitor training progress—watching it decrease over epochs tells us the network is learning.

```javascript
class Loss {
    // ... existing methods: meanSquaredError, meanAbsoluteError, binaryCrossEntropy ...
    
    // Cross-entropy for multi-class classification (like next-token prediction)
    // predictions: array of probability distributions, one per timestep
    // targetTokens: array of correct token IDs, one per timestep
    static crossEntropy(predictions, targetTokens) {
        let totalLoss = 0;
        
        for (let t = 0; t < predictions.length; t++) {
            const probs = predictions[t];
            const targetToken = targetTokens[t];
            
            // -log(probability of correct token)
            const epsilon = 1e-10;  // Prevent log(0)
            totalLoss += -Math.log(probs[targetToken] + epsilon);
        }
        
        return totalLoss / predictions.length;
    }
}
```

You might notice we don't have a separate `crossEntropyDerivative` method. When cross-entropy loss is combined with softmax, the derivative simplifies to `predicted - target`—for the correct token, `prob - 1`; for wrong tokens, `prob - 0`. We compute this directly in `OutputLayer.backward()` rather than as a separate loss function.

## ChatRNN: Putting It All Together

With all our components built, can now assemble them into a language model. The `ChatRNN` class chains these layers together: token IDs flow into the embedding layer to become vectors, those vectors pass through the RNN which maintains sequential memory, and finally the output layer converts hidden states into probability distributions over our vocabulary. Training works by running a forward pass to get predictions, computing cross-entropy loss against the actual next tokens, then backpropagating gradients through all three layers in reverse order. 

```javascript
class ChatRNN {
    embedding = null;
    rnn = null;
    output = null;
    vocabSize = 0;

    constructor(vocabSize, embeddingDim, hiddenSize) {
        this.vocabSize = vocabSize;
        this.embedding = new EmbeddingLayer(vocabSize, embeddingDim);
        this.rnn = new RNNLayer(embeddingDim, hiddenSize);
        this.output = new OutputLayer(hiddenSize, vocabSize);
    }

    #forward(inputTokens) {
        const embedded = this.embedding.forward(inputTokens);
        const hiddenStates = this.rnn.forward(embedded);
        const probabilities = this.output.forward(hiddenStates);
        return probabilities;
    }

    #backward(targetTokens, learningRate) {
        const hiddenGradients = this.output.backward(targetTokens, learningRate);
        const embeddingGradients = this.rnn.backward(hiddenGradients, learningRate);
        this.embedding.backward(embeddingGradients, learningRate);
    }

    train(inputTokens, targetTokens, learningRate) {
        const predictions = this.#forward(inputTokens);
        const loss = Loss.crossEntropy(predictions, targetTokens);
        this.#backward(targetTokens, learningRate);
        return loss;
    }
}
```

## Text Generation

Once trained, the model generates text by predicting one token at a time, then feeding its prediction back as input. We start with a seed sequence (even just a single token), run it through the network to get a probability distribution over the vocabulary, sample a token from that distribution, append it to our sequence, and repeat. Each iteration, the RNN's hidden state accumulates more context, so later predictions are informed by everything generated so far.

```javascript
// ... class ChatRNN
    generate(startTokens, maxLength) {
        const generated = [];
        for (let i = 0; i < startTokens.length; i++) {
            generated.push(startTokens[i]);
        }
        
        for (let i = 0; i < maxLength; i++) {
            const probs = this.#forward(generated);
            const lastProbs = probs[probs.length - 1];
            const nextToken = this.#sampleFromDistribution(lastProbs);
            generated.push(nextToken);
        }
        
        return generated;
    }
```

The `#sampleFromDistribution` method performs weighted random selection. Given an array of probabilities (summing to 1), it returns an index where higher-probability entries are proportionally more likely to be chosen. The algorithm uses the inverse transform method. Imagine laying out the probabilities as segments on a number line from 0 to 1. If token 0 has probability 0.1, it occupies (0, 0.1). If token 1 has probability 0.25, it occupies [0.1, 0.35]. And so on. We generate a uniform random number between 0 and 1, then find which segment it lands in.

```javascript
// ... class ChatRNN
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

> Why sample instead of just picking the highest probability token? Sampling introduces controlled randomness. If the model thinks "cat" has 40% probability and "dog" has 35%, always picking "cat" would make generation deterministic and repetitive. Sampling lets "dog" win sometimes, producing more varied and interesting output. The probabilities still matter—"cat" will be chosen more often than "dog"—but rare tokens get a chance too.

## Training with the BPE Tokenizer

Now we can use the `Tokenizer` class from to train our language model. The tokenizer handles converting text to tokens and back:

```javascript
// Load training text
const trainingText = `Hello world. Hello there. Hello friend. 
The quick brown fox jumps over the lazy dog.
The dog was lazy but the fox was quick.`;

// Create and train tokenizer
const tokenizer = new Tokenizer();
tokenizer.train(trainingText, 100);

console.log(`Vocabulary size: ${tokenizer.getVocabSize()}`);

// Tokenize training data
const tokens = tokenizer.encode(trainingText);
console.log(`Training tokens: ${tokens.length}`);

// Create model
const embeddingDim = 64;
const hiddenSize = 128;
const model = new ChatRNN(tokenizer.getVocabSize(), embeddingDim, hiddenSize);

// Training parameters
const learningRate = 0.1;
const epochs = 500;
const sequenceLength = 32;
const slideStep = 16;

// Training loop
for (let epoch = 0; epoch < epochs; epoch++) {
    let totalLoss = 0;
    let batchCount = 0;
    
    // Slide window across training data
    for (let start = 0; start < tokens.length - sequenceLength - 1; start += slideStep) {
        const inputTokens = tokens.slice(start, start + sequenceLength);
        const targetTokens = tokens.slice(start + 1, start + sequenceLength + 1);
        
        const loss = model.train(inputTokens, targetTokens, learningRate);
        totalLoss += loss;
        batchCount++;
    }
    
    if (epoch % 50 === 0) {
        const avgLoss = totalLoss / batchCount;
        console.log(`Epoch ${epoch}: Loss = ${avgLoss.toFixed(4)}`);
        
        // Generate sample
        const prompt = tokenizer.encode("Hello");
        const generated = model.generate(prompt, 20, 0.8);
        const text = tokenizer.decode(generated);
        console.log(`  Sample: "${text}"`);
    }
}

// Final generation
console.log("\n=== Generation ===");
const prompt = tokenizer.encode("The ");
const generated = model.generate(prompt, 50, 0.8);
console.log(tokenizer.decode(generated));
```

The training loop:
1. Slides a window across tokenized text
2. Input is positions 0 to N-1, target is positions 1 to N (next token prediction)
3. Runs forward pass, computes loss, runs backward pass
4. Periodically generates samples to monitor progress

## Stacking RNN Layers

A single RNN layer captures patterns, but stacking layers creates hierarchy,early layers learn simple patterns, deeper layers combine them into complex concepts.

### RNN Layer with Residual Connections

Deep networks face a problem: gradients must travel backward through every layer during training, and with each layer the signal can shrink or distort. Residual connections solve this by creating a shortcut, we add the original input directly to the output, giving gradients a "highway" that bypasses the complex transformation. This only works when input and output dimensions match.

```javascript
class RNNLayerWithResidual {
    rnn = null;
    inputSize = 0;
    hiddenSize = 0;
    canUseResidual = false;

    constructor(inputSize, hiddenSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.rnn = new RNNLayer(inputSize, hiddenSize);
        this.canUseResidual = (inputSize === hiddenSize);
    }
```

The forward pass runs the input through the RNN normally, then adds the original input back to the output when dimensions allow. Each output value becomes the sum of the RNN's transformation plus the raw input signal.

```javascript
// ... class RNNLayerWithResidual
    forward(inputSequence) {
        const rnnOutput = this.rnn.forward(inputSequence);
        
        if (this.canUseResidual) {
            for (let t = 0; t < rnnOutput.length; t++) {
                for (let h = 0; h < this.hiddenSize; h++) {
                    rnnOutput[t][h] += inputSequence[t][h];
                }
            }
        }
        
        return rnnOutput;
    }
```

The backward pass mirrors this. We backpropagate through the RNN to get input gradients, then add the output gradients directly to them. This reflects that during forward, the input contributed through two paths—the RNN transformation and the direct addition—so gradients must flow back through both.

```javascript
// ... class RNNLayerWithResidual
    backward(outputGradients, learningRate) {
        const inputGradients = this.rnn.backward(outputGradients, learningRate);
        
        if (this.canUseResidual) {
            for (let t = 0; t < outputGradients.length; t++) {
                for (let h = 0; h < this.hiddenSize; h++) {
                    inputGradients[t][h] += outputGradients[t][h];
                }
            }
        }
        
        return inputGradients;
    }
}
```

### Multi-Layer ChatRNN

To build a deeper ChatRNN, you'd stack multiple `RNNLayerWithResidual` instances between the embedding and output layers. The first layer transforms from `embeddingDim` to `hiddenSize`, then subsequent layers maintain `hiddenSize` throughout, enabling residual connections at every level.

```javascript
class ChatRNN {
    embedding = null;
    rnnLayers = null;
    output = null;
    vocabSize = 0;
    numLayers = 0;

    constructor(vocabSize, embeddingDim, hiddenSize, numLayers = 1) {
        this.vocabSize = vocabSize;
        this.numLayers = numLayers;
        
        this.embedding = new EmbeddingLayer(vocabSize, embeddingDim);
        
        // Stack RNN layers
        this.rnnLayers = new Array(numLayers);
        for (let i = 0; i < numLayers; i++) {
            const inputSize = (i === 0) ? embeddingDim : hiddenSize;
            this.rnnLayers[i] = new RNNLayerWithResidual(inputSize, hiddenSize);
        }
        
        this.output = new OutputLayer(hiddenSize, vocabSize);
    }

    forward(inputTokens) {
        let layerInput = this.embedding.forward(inputTokens);
        
        for (let i = 0; i < this.rnnLayers.length; i++) {
            layerInput = this.rnnLayers[i].forward(layerInput);
        }
        
        return this.output.forward(layerInput);
    }

    backward(targetTokens, learningRate) {
        let gradients = this.output.backward(targetTokens, learningRate);
        
        for (let i = this.rnnLayers.length - 1; i >= 0; i--) {
            gradients = this.rnnLayers[i].backward(gradients, learningRate);
        }
        
        this.embedding.backward(gradients, learningRate);
    }

    // ... train() and generate() same as before
}
```

Now you can create deeper models:

```javascript
// 3-layer RNN
const model = new ChatRNN(vocabSize, 64, 128, 3);
```

## The Vanishing Gradient Problem

There's a fundamental limitation with vanilla RNNs. During backpropagation through time, gradients get multiplied at each step. If multipliers are consistently < 1, gradients shrink exponentially:

- After 10 steps: 0.9^10 ≈ 0.35
- After 50 steps: 0.9^50 ≈ 0.005  
- After 100 steps: 0.9^100 ≈ 0.00003

The gradient essentially vanishes. Early timesteps receive almost no learning signal,they stop learning while later timesteps train normally. This is why vanilla RNNs struggle with sequences longer than ~20-30 tokens. Solutions include:
- **Gradient clipping**: Cap gradient magnitude
- **Residual connections**: Provide gradient highways (what we implemented)
- **LSTM/GRU**: Explicit gating mechanisms
- **Transformers**: Attention bypasses sequential propagation 
