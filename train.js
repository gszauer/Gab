class Loss {
    // Mean Squared Error (MSE) - most common for regression
    static meanSquaredError(predictions, targets) {
        // (1/n) * Σ(prediction - target)²
        let sum = 0;
        for (let i = 0, size = predictions.length; i < size; i++) {
            const diff = predictions[i] - targets[i];
            sum += diff * diff;
        }
        return sum / predictions.length;
    }

    // Derivative of MSE with respect to predictions
    static meanSquaredErrorDerivative(predictions, targets) {
        // ∂MSE/∂prediction[i] = (2/n) * (prediction[i] - target[i])
        const derivatives = new Array(predictions.length);
        for (let i = 0; i < predictions.length; i++) {
            derivatives[i] = 2 * (predictions[i] - targets[i]) / predictions.length;
        }
        return derivatives;
    }

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

class Neuron {
    weights = null;
    bias = null;

    constructor(numberOfInputs) {
        this.weights = new Array(numberOfInputs);
        for (let i = 0; i < numberOfInputs; i++) {
            this.weights[i] = Math.random() * 2 - 1; // [-1, 1]
        }
        this.bias = Math.random() * 2 - 1;
    }

    forward(inputs) {
        let sum = 0;
        for (let i = 0; i < inputs.length; i++) {
            sum += this.weights[i] * inputs[i];
        }
        return sum + this.bias;
    }

    backward(inputs, /*1) gradient: */ neuronGradient, learningRate) {
        // 2) Calculate paramater gradients
        const parameterGradients = this.#calculateParameterGradients(inputs, neuronGradient);

        // 3) Calculate INPUT gradients (errors to pass backward)
        const inputGradients = this.#calculateInputGradients(neuronGradient);

        // 4) Update weights using parameter gradients
        this.#updateWeights(parameterGradients, learningRate);

        // 5) What the errors coming in from the last layer are. Pass them back.
        return inputGradients;
    }

    #calculateParameterGradients(inputs, neuronGradient) {
        const biasGradient = neuronGradient;

        // The gradient for weights depends on their input intensity
        const weightGradients = new Array(inputs.length);
        for (let i = 0; i < inputs.length; i++) {
            weightGradients[i] = neuronGradient * inputs[i];
        }

        return {
            bias: biasGradient,
            weights: weightGradients
        };
    }

    #calculateInputGradients(neuronGradient) {
        // These gradients are the errors we pass to the previous layer
        const inputGradients = new Array(this.weights.length);

        for (let i = 0; i < this.weights.length; i++) {
            inputGradients[i] = neuronGradient * this.weights[i];
        }

        return inputGradients;
    }

    #updateWeights(gradients, learningRate) {
        // Update each weight using its stored gradient
        for (let i = 0; i < this.weights.length; i++) {
            this.weights[i] -= learningRate * gradients.weights[i];
        }
        // Update bias using its stored gradient
        this.bias -= learningRate * gradients.bias;
    }
}

class DenseLayer {
    neurons = null;
    cachedInputs = null; // NEW

    constructor(numberOfInputs, numberOfOutputs) {
        this.neurons = new Array(numberOfOutputs);
        for (let i = 0; i < numberOfOutputs; ++i) {
            this.neurons[i] = new Neuron(numberOfInputs);
        }
        this.cachedInputs = new Array(numberOfInputs); // NEW
    }

    forward(inputs) {
        // Cache inputs for the backwards pass
        for (let i = 0, size = inputs.length; i < size; ++i) {
            this.cachedInputs[i] = inputs[i];
        }

        const outputs = new Array(this.neurons.length);
        for (let i = 0, size = this.neurons.length;  i < size; ++i) {
            outputs[i] = this.neurons[i].forward(inputs);
        }
        return outputs;
    }

    backward(outputGradients, learningRate) {
        // Start with zero error for each input
        const inputGradients = new Array(this.cachedInputs.length);
        for (let i = 0; i < inputGradients.length; i++) {
            inputGradients[i] = 0;  // Start at zero!
        }

        // Each neuron will ADD its blame to the inputs
        for (let neuronIdx = 0; neuronIdx < this.neurons.length; neuronIdx++) {
            const neuron = this.neurons[neuronIdx];

            // This neuron calculates: "how much did each input contribute to MY error?"
            const neuronsInputGradients = neuron.backward(
                this.cachedInputs,
                outputGradients[neuronIdx],  // This specific neuron's error
                learningRate
            );

            // ADD this neuron's blame to our running total
            for (let i = 0; i < neuronsInputGradients.length; i++) {
                inputGradients[i] += neuronsInputGradients[i];  // += not =
            }
        }

        return inputGradients;
    }
}

class ActivationLayer {
    type = "relu";
    cachedInputs = null; // NEW

    constructor(layerType = "relu") {
        this.type = layerType;
    }

    #reluActivation(x) {
        return Math.max(0, x);
    }

    #sigmoidActivation(x) {
        return 1 / (1 + Math.exp(-x));
    }

    #tanhActivation(x) {
        return Math.tanh(x);
    }

    #reluDerivative(x) {
        return x > 0 ? 1 : 0;
    }

    #sigmoidDerivative(x) {
        const sig = this.#sigmoidActivation(x);
        return sig * (1 - sig);
    }

    #tanhDerivative(x) {
        const t = Math.tanh(x);
        return 1 - t * t;
    }

    forward(inputs) {
         if (this.cachedInputs == null || this.cachedInputs.length !== inputs.length) {
            this.cachedInputs = new Array(inputs.length);
        }
        for (let i = 0, size = inputs.length; i < size; ++i) {
            this.cachedInputs[i] = inputs[i];
        }

        const output = new Array(inputs.length);

        if (this.type === "relu") {
            for (let i = 0, size = inputs.length; i < size; ++i) {
                output[i] = this.#reluActivation(inputs[i]);
            }
        }
        else  if (this.type === "sigmoid") {
            for (let i = 0, size = inputs.length; i < size; ++i) {
                output[i] = this.#sigmoidActivation(inputs[i]);
            }
        }
        else  if (this.type === "tanh") {
            for (let i = 0, size = inputs.length; i < size; ++i) {
                output[i] = this.#tanhActivation(inputs[i]);
            }
        }
        else {
            return null;
        }

        return output;
    }

    backward(outputGradients) {
        // outputGradients = error signal coming from the next layer
        // We need to figure out what the input gradient was

        const inputGradients = new Array(outputGradients.length);

        if (this.type === "relu") {
            for (let i = 0; i < outputGradients.length; i++) {
                inputGradients[i] = outputGradients[i] * this.#reluDerivative(this.cachedInputs[i]);
            }
        }
        else if (this.type === "sigmoid") {
            for (let i = 0; i < outputGradients.length; i++) {
                inputGradients[i] = outputGradients[i] * this.#sigmoidDerivative(this.cachedInputs[i]);
            }
        }
        else if (this.type === "tanh") {
            for (let i = 0; i < outputGradients.length; i++) {
                inputGradients[i] = outputGradients[i] * this.#tanhDerivative(this.cachedInputs[i]);
            }
        }

        return inputGradients;
    }
}

class Tokenizer {
    merges = new Map();      // Maps byte pairs to new token IDs (key is a string, value is bytes array)
    vocabulary = new Map();  // Maps token IDs to their byte sequences (key is an integer, value is bytes array)
    nextTokenId = 256;       // Start after single-byte tokens (0-255)

    constructor() {
        // Initialize vocabulary with single-byte tokens
        for (let i = 0; i < 256; i++) {
            this.vocabulary.set(i, [i]);
        }
    }

    #stringToBytes(text) {
        const encoder = new TextEncoder();
        const uint8Array = encoder.encode(text);
        const bytes = [];
        for (let i = 0; i < uint8Array.length; i++) {
            bytes.push(uint8Array[i]);
        }
        return bytes;
    }

    #bytesToString(bytes) {
        return new TextDecoder().decode(new Uint8Array(bytes));
    }

    #makeMerge(token1, token2) {
        // Check if this merge already exists
        const mergeKey = `${token1},${token2}`;
        const existingMerge = this.merges.get(mergeKey);
        if (existingMerge !== undefined) {
            return existingMerge;
        }

        // Create new token ID
        const newTokenId = this.nextTokenId++;

        // Store the merge rule
        this.merges.set(mergeKey, newTokenId);

        // Store what this new token represents
        const token1Bytes = this.vocabulary.get(token1);
        const token2Bytes = this.vocabulary.get(token2);
        const newTokenBytes = [...token1Bytes, ...token2Bytes];
        this.vocabulary.set(newTokenId, newTokenBytes);

        return newTokenId;
    }

    #applyMerge(tokens, token1, token2, mergedTokenId) {
        const result = [];
        let i = 0;

        while (i < tokens.length) {
            if (i < tokens.length - 1 &&
                tokens[i] === token1 &&
                tokens[i + 1] === token2) {
                // Found the pair - replace with merged token
                result.push(mergedTokenId);
                i += 2;
            } else {
                // No merge here, keep original token
                result.push(tokens[i]);
                i += 1;
            }
        }

        return result;
    }

    #findMostFrequentPair(tokensList) {
        const pairCounts = new Map();

        // Count all adjacent pairs
        for (let i = 0; i < tokensList.length - 1; i++) {
            const pair = `${tokensList[i]},${tokensList[i + 1]}`;
            const currentCount = pairCounts.get(pair) || 0;
            pairCounts.set(pair, currentCount + 1);
        }

        // Find the most frequent pair
        let maxCount = 0;
        let mostFrequentPair = null;

        for (const [pair, count] of pairCounts) {
            if (count > maxCount) {
                maxCount = count;
                mostFrequentPair = pair;
            }
        }

        // Return as array of token IDs, or null if no pairs exist
        if (mostFrequentPair && maxCount > 1) {
            const tokens = mostFrequentPair.split(',');
            return [parseInt(tokens[0]), parseInt(tokens[1])];
        }

        return null;
    }

    train(trainingText, numMerges) {
        let tokens = this.#stringToBytes(trainingText);

        console.log(`Starting training with ${tokens.length} bytes`);

        for (let mergeNum = 0; mergeNum < numMerges; mergeNum++) {
            // Find the most frequent pair
            const pair = this.#findMostFrequentPair(tokens);

            if (!pair) {
                console.log(`Training stopped early at merge ${mergeNum} - no more frequent pairs`);
                break;
            }

            const [token1, token2] = pair;

            // Create the merge (or get existing one if it already exists)
            const newTokenId = this.#makeMerge(token1, token2);

            // Apply the merge to all occurrences in the training data
            tokens = this.#applyMerge(tokens, token1, token2, newTokenId);

            // Log progress periodically
            if (mergeNum % 100 === 0 || mergeNum < 10) {
                const originalLength = this.#stringToBytes(trainingText).length;
                const compression = ((1 - tokens.length / originalLength) * 100).toFixed(1);
                console.log(`Merge ${mergeNum}: [${token1}, ${token2}] -> ${newTokenId}, compression: ${compression}%`);
            }
        }

        console.log(`Training complete. Vocabulary size: ${this.vocabulary.size}`);
        return this.merges.size;
    }

    encode(textToEncode) {
        let tokens = this.#stringToBytes(textToEncode);

        // Apply merges in the order they were learned
        // JavaScript Maps maintain insertion order, so iterating gives us merges
        // in the same order they were added during training
        for (const [mergeKey, mergedToken] of this.merges) {
            // Parse the merge key to get the two tokens to merge
            const [token1, token2] = mergeKey.split(',').map(Number);

            // Apply this merge wherever it appears in the sequence
            tokens = this.#applyMerge(tokens, token1, token2, mergedToken);
        }

        return tokens;
    }

    decode(tokensToDecode) {
        const bytes = [];

        for (let i = 0; i < tokensToDecode.length; i++) {
            const token = tokensToDecode[i];

            // Look up the token's byte sequence
            const tokenBytes = this.vocabulary.get(token);
            if (tokenBytes) {
                for (let j = 0; j < tokenBytes.length; j++) {
                    bytes.push(tokenBytes[j]);
                }
            } else {
                throw new Error(`Unknown token: ${token}`);
            }
        }

        return this.#bytesToString(bytes);
    }

    reserveToken(specialTokenString) {
        // Convert the special token to bytes
        const bytes = this.#stringToBytes(specialTokenString);
        let tokens = [...bytes]; // Copy the array

        // Apply all existing merges first to get the current token sequence
        for (const [mergeKey, mergedToken] of this.merges) {
            const [token1, token2] = mergeKey.split(',').map(Number);
            tokens = this.#applyMerge(tokens, token1, token2, mergedToken);
        }

        // Create merge chain to combine all remaining tokens into a single token
        while (tokens.length > 1) {
            // Create a merge for the first two tokens
            const newTokenId = this.#makeMerge(tokens[0], tokens[1]);

            // Apply it to our token array
            tokens = this.#applyMerge(tokens, tokens[0], tokens[1], newTokenId);
        }

        const finalTokenId = tokens[0];
        return finalTokenId;
    }

    /**
     * Serializes the tokenizer to a binary format
     *
     * Binary format structure:
     * - Magic number (4 bytes): 0x42504531 ("BPE1")
     * - Version (4 bytes): Currently 1
     * - nextTokenId (4 bytes)
     * - Number of merges (4 bytes)
     * - For each merge:
     *   - token1 (4 bytes)
     *   - token2 (4 bytes)
     *   - mergedTokenId (4 bytes)
     * - Number of vocabulary entries (4 bytes)
     * - For each vocabulary entry:
     *   - tokenId (4 bytes)
     *   - length of byte sequence (4 bytes)
     *   - byte sequence (variable length)
     *
     * @returns {Uint8Array} Binary representation of the tokenizer
     */
    serialize() {
        // Calculate required buffer size
        let bufferSize = 16; // magic (4) + version (4) + nextTokenId (4) + numMerges (4)
        bufferSize += this.merges.size * 12; // each merge: token1(4) + token2(4) + mergedId(4)
        bufferSize += 4; // numVocabulary (4)

        // Calculate vocabulary size
        for (const [tokenId, bytes] of this.vocabulary) {
            bufferSize += 8 + bytes.length; // tokenId(4) + length(4) + bytes
        }

        const buffer = new ArrayBuffer(bufferSize);
        const view = new DataView(buffer);
        const bytes = new Uint8Array(buffer);
        let offset = 0;

        // Write header
        view.setUint32(offset, 0x42504531, true); // Magic: "BPE1" in hex
        offset += 4;
        view.setUint32(offset, 1, true); // Version 1
        offset += 4;
        view.setUint32(offset, this.nextTokenId, true);
        offset += 4;

        // Write merges
        view.setUint32(offset, this.merges.size, true);
        offset += 4;

        for (const [mergeKey, mergedToken] of this.merges) {
            const [token1, token2] = mergeKey.split(',').map(Number);
            view.setUint32(offset, token1, true);
            offset += 4;
            view.setUint32(offset, token2, true);
            offset += 4;
            view.setUint32(offset, mergedToken, true);
            offset += 4;
        }

        // Write vocabulary
        view.setUint32(offset, this.vocabulary.size, true);
        offset += 4;

        for (const [tokenId, tokenBytes] of this.vocabulary) {
            view.setUint32(offset, tokenId, true);
            offset += 4;
            view.setUint32(offset, tokenBytes.length, true);
            offset += 4;
            for (let i = 0; i < tokenBytes.length; i++) {
                bytes[offset++] = tokenBytes[i];
            }
        }

        return bytes;
    }

    /**
     * Deserializes a tokenizer from binary format
     * @param {Uint8Array} data - Binary tokenizer data
     * @returns {Tokenizer} New tokenizer instance with loaded state
     */
    static deserialize(data) {
        const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
        let offset = 0;

        // Read and verify header
        const magic = view.getUint32(offset, true);
        offset += 4;
        if (magic !== 0x42504531) {
            throw new Error('Invalid tokenizer file format');
        }

        const version = view.getUint32(offset, true);
        offset += 4;
        if (version !== 1) {
            throw new Error(`Unsupported tokenizer version: ${version}`);
        }

        const tokenizer = new Tokenizer();

        // Read nextTokenId
        tokenizer.nextTokenId = view.getUint32(offset, true);
        offset += 4;

        // Read merges
        const numMerges = view.getUint32(offset, true);
        offset += 4;

        tokenizer.merges.clear();
        for (let i = 0; i < numMerges; i++) {
            const token1 = view.getUint32(offset, true);
            offset += 4;
            const token2 = view.getUint32(offset, true);
            offset += 4;
            const mergedToken = view.getUint32(offset, true);
            offset += 4;

            const mergeKey = `${token1},${token2}`;
            tokenizer.merges.set(mergeKey, mergedToken);
        }

        // Read vocabulary
        const numVocab = view.getUint32(offset, true);
        offset += 4;

        tokenizer.vocabulary.clear();
        for (let i = 0; i < numVocab; i++) {
            const tokenId = view.getUint32(offset, true);
            offset += 4;
            const length = view.getUint32(offset, true);
            offset += 4;

            const bytes = [];
            for (let j = 0; j < length; j++) {
                bytes.push(data[offset++]);
            }

            tokenizer.vocabulary.set(tokenId, bytes);
        }

        return tokenizer;
    }
}

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
}

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

            // Gradient to pass to embedding layer
            inputGradients[t] = new Array(this.cell.inputSize);
            for (let i = 0; i < this.cell.inputSize; i++) {
                let grad = 0;
                for (let h = 0; h < this.hiddenSize; h++) {
                    grad += preActivationGrad[h] * this.cell.weightsX[i][h];
                }
                inputGradients[t][i] = grad;
            }

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
}

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
                    // Convert logits to probabilities
            outputs[t] = this.#softmax(logits);
            this.cachedOutputs[t] = outputs[t];
        }
        
        return outputs;  // Shape: [sequenceLength][vocabSize]
    }

    backward(targetTokens, learningRate) {
        const sequenceLength = this.cachedInputs.length;
        const hiddenGradients = new Array(sequenceLength);
        
        // Accumulate weight gradients across all timesteps
        const weightsGrad = this.#createZeroMatrix(this.hiddenSize, this.vocabSize);
        const biasGrad = new Array(this.vocabSize);
        for (let i = 0; i < this.vocabSize; i++) {
            biasGrad[i] = 0;
        }

        for (let t = 0; t < sequenceLength; t++) {
            const hidden = this.cachedInputs[t];
            const probs = this.cachedOutputs[t];
            const targetToken = targetTokens[t];

            // Softmax + cross-entropy gradient: predicted - target
            const outputGrad = new Array(this.vocabSize);
            for (let v = 0; v < this.vocabSize; v++) {
                if (v === targetToken) {
                    outputGrad[v] = probs[v] - 1;  // Target was 1
                } else {
                    outputGrad[v] = probs[v];      // Target was 0
                }
            }

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
}





///

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

    #forward(inputTokens) {
        let layerInput = this.embedding.forward(inputTokens);
        
        for (let i = 0; i < this.rnnLayers.length; i++) {
            layerInput = this.rnnLayers[i].forward(layerInput);
        }
        
        return this.output.forward(layerInput);
    }

    #backward(targetTokens, learningRate) {
        let gradients = this.output.backward(targetTokens, learningRate);
        
        for (let i = this.rnnLayers.length - 1; i >= 0; i--) {
            gradients = this.rnnLayers[i].backward(gradients, learningRate);
        }
        
        this.embedding.backward(gradients, learningRate);
    }

    train(inputTokens, targetTokens, learningRate) {
        const predictions = this.#forward(inputTokens);
        const loss = Loss.crossEntropy(predictions, targetTokens);
        this.#backward(targetTokens, learningRate);
        return loss;
    }

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

    /**
     * Serializes the model weights to a binary format
     *
     * Binary format:
     * - Magic number (4 bytes): 0x52 0x4E 0x4E 0x31 ("RNN1")
     * - Version (4 bytes): u32
     * - vocabSize (4 bytes): u32
     * - embeddingDim (4 bytes): u32
     * - hiddenSize (4 bytes): u32
     * - numLayers (4 bytes): u32
     * - Embedding weights: vocabSize * embeddingDim * f32
     * - For each RNN layer:
     *   - inputSize (4 bytes): u32
     *   - weightsX: inputSize * hiddenSize * f32
     *   - weightsH: hiddenSize * hiddenSize * f32
     *   - bias: hiddenSize * f32
     * - Output weights: hiddenSize * vocabSize * f32
     * - Output bias: vocabSize * f32
     *
     * @returns {Uint8Array} Binary representation of model weights
     */
    serialize() {
        const embeddingDim = this.embedding.embeddingDim;
        const hiddenSize = this.output.hiddenSize;

        // Calculate buffer size
        let size = 24; // header: magic(4) + version(4) + vocabSize(4) + embeddingDim(4) + hiddenSize(4) + numLayers(4)

        // Embedding weights
        size += this.vocabSize * embeddingDim * 4;

        // RNN layers
        for (let i = 0; i < this.rnnLayers.length; i++) {
            const layer = this.rnnLayers[i];
            const inputSize = layer.inputSize;
            size += 4; // inputSize
            size += inputSize * hiddenSize * 4;      // weightsX
            size += hiddenSize * hiddenSize * 4;     // weightsH
            size += hiddenSize * 4;                   // bias
        }

        // Output layer
        size += hiddenSize * this.vocabSize * 4;  // weights
        size += this.vocabSize * 4;                // bias

        const buffer = new ArrayBuffer(size);
        const view = new DataView(buffer);
        let offset = 0;

        // Write header
        view.setUint32(offset, 0x524E4E31, true); // "RNN1"
        offset += 4;
        view.setUint32(offset, 1, true); // version
        offset += 4;
        view.setUint32(offset, this.vocabSize, true);
        offset += 4;
        view.setUint32(offset, embeddingDim, true);
        offset += 4;
        view.setUint32(offset, hiddenSize, true);
        offset += 4;
        view.setUint32(offset, this.numLayers, true);
        offset += 4;

        // Write embedding weights
        for (let i = 0; i < this.vocabSize; i++) {
            for (let j = 0; j < embeddingDim; j++) {
                view.setFloat32(offset, this.embedding.weights[i][j], true);
                offset += 4;
            }
        }

        // Write RNN layers
        for (let l = 0; l < this.rnnLayers.length; l++) {
            const cell = this.rnnLayers[l].rnn.cell;
            const inputSize = this.rnnLayers[l].inputSize;

            view.setUint32(offset, inputSize, true);
            offset += 4;

            // weightsX [inputSize][hiddenSize]
            for (let i = 0; i < inputSize; i++) {
                for (let j = 0; j < hiddenSize; j++) {
                    view.setFloat32(offset, cell.weightsX[i][j], true);
                    offset += 4;
                }
            }

            // weightsH [hiddenSize][hiddenSize]
            for (let i = 0; i < hiddenSize; i++) {
                for (let j = 0; j < hiddenSize; j++) {
                    view.setFloat32(offset, cell.weightsH[i][j], true);
                    offset += 4;
                }
            }

            // bias [hiddenSize]
            for (let i = 0; i < hiddenSize; i++) {
                view.setFloat32(offset, cell.bias[i], true);
                offset += 4;
            }
        }

        // Write output layer weights [hiddenSize][vocabSize]
        for (let i = 0; i < hiddenSize; i++) {
            for (let j = 0; j < this.vocabSize; j++) {
                view.setFloat32(offset, this.output.weights[i][j], true);
                offset += 4;
            }
        }

        // Write output layer bias [vocabSize]
        for (let i = 0; i < this.vocabSize; i++) {
            view.setFloat32(offset, this.output.bias[i], true);
            offset += 4;
        }

        return new Uint8Array(buffer);
    }

    /**
     * Deserializes model weights from binary format
     * @param {Uint8Array} data - Binary model data
     * @returns {ChatRNN} New model instance with loaded weights
     */
    static deserialize(data) {
        const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
        let offset = 0;

        // Read and verify header
        const magic = view.getUint32(offset, true);
        offset += 4;
        if (magic !== 0x524E4E31) {
            throw new Error('Invalid model file format');
        }

        const version = view.getUint32(offset, true);
        offset += 4;
        if (version !== 1) {
            throw new Error(`Unsupported model version: ${version}`);
        }

        const vocabSize = view.getUint32(offset, true);
        offset += 4;
        const embeddingDim = view.getUint32(offset, true);
        offset += 4;
        const hiddenSize = view.getUint32(offset, true);
        offset += 4;
        const numLayers = view.getUint32(offset, true);
        offset += 4;

        // Create model with same architecture
        const model = new ChatRNN(vocabSize, embeddingDim, hiddenSize, numLayers);

        // Read embedding weights
        for (let i = 0; i < vocabSize; i++) {
            for (let j = 0; j < embeddingDim; j++) {
                model.embedding.weights[i][j] = view.getFloat32(offset, true);
                offset += 4;
            }
        }

        // Read RNN layers
        for (let l = 0; l < numLayers; l++) {
            const inputSize = view.getUint32(offset, true);
            offset += 4;

            const cell = model.rnnLayers[l].rnn.cell;

            // weightsX [inputSize][hiddenSize]
            for (let i = 0; i < inputSize; i++) {
                for (let j = 0; j < hiddenSize; j++) {
                    cell.weightsX[i][j] = view.getFloat32(offset, true);
                    offset += 4;
                }
            }

            // weightsH [hiddenSize][hiddenSize]
            for (let i = 0; i < hiddenSize; i++) {
                for (let j = 0; j < hiddenSize; j++) {
                    cell.weightsH[i][j] = view.getFloat32(offset, true);
                    offset += 4;
                }
            }

            // bias [hiddenSize]
            for (let i = 0; i < hiddenSize; i++) {
                cell.bias[i] = view.getFloat32(offset, true);
                offset += 4;
            }
        }

        // Read output layer weights [hiddenSize][vocabSize]
        for (let i = 0; i < hiddenSize; i++) {
            for (let j = 0; j < vocabSize; j++) {
                model.output.weights[i][j] = view.getFloat32(offset, true);
                offset += 4;
            }
        }

        // Read output layer bias [vocabSize]
        for (let i = 0; i < vocabSize; i++) {
            model.output.bias[i] = view.getFloat32(offset, true);
            offset += 4;
        }

        return model;
    }
}


function TrainingDemo() {
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
    const model = new ChatRNN(tokenizer.getVocabSize(), embeddingDim, hiddenSize, 3);

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
}