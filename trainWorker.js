// Training Web Worker
importScripts('train.js');

let tokenizer = null;
let model = null;
let tokens = null;
let isPaused = true;
let currentEpoch = 0;
let trainingConfig = null;

self.onmessage = function(e) {
    const { type, data } = e.data;

    switch (type) {
        case 'init':
            initTraining(data);
            break;
        case 'start':
            isPaused = false;
            runTraining();
            break;
        case 'pause':
            isPaused = true;
            break;
        case 'generate':
            generate(data.prompt, data.length);
            break;
        case 'getState':
            sendState();
            break;
        case 'loadModel':
            loadModel(data.modelData, data.tokenizerData);
            break;
        case 'saveModel':
            saveModel();
            break;
    }
};

function initTraining(config) {
    trainingConfig = config;
    currentEpoch = 0;

    // Create tokenizer
    tokenizer = new Tokenizer();

    // Reserve special tokens first
    const reservedTokenIds = [];
    if (config.reservedTokens && config.reservedTokens.length > 0) {
        for (const token of config.reservedTokens) {
            if (token.trim()) {
                const id = tokenizer.reserveToken(token.trim());
                reservedTokenIds.push({ token: token.trim(), id });
            }
        }
    }

    // Train tokenizer
    self.postMessage({ type: 'log', message: 'Training tokenizer...' });
    tokenizer.train(config.trainingText, config.numMerges);

    self.postMessage({
        type: 'log',
        message: `Tokenizer ready. Vocabulary size: ${tokenizer.vocabulary.size}`
    });

    if (reservedTokenIds.length > 0) {
        self.postMessage({
            type: 'log',
            message: `Reserved tokens: ${reservedTokenIds.map(r => `"${r.token}"=${r.id}`).join(', ')}`
        });
    }

    // Tokenize training data
    tokens = tokenizer.encode(config.trainingText);
    self.postMessage({
        type: 'log',
        message: `Training data: ${tokens.length} tokens`
    });

    // Create model
    const vocabSize = tokenizer.vocabulary.size;
    model = new ChatRNN(vocabSize, config.embeddingDim, config.hiddenSize, config.numLayers);

    self.postMessage({
        type: 'log',
        message: `Model created: embedding=${config.embeddingDim}, hidden=${config.hiddenSize}, layers=${config.numLayers}`
    });

    self.postMessage({ type: 'ready', vocabSize });
}

async function runTraining() {
    if (!model || !tokens || !trainingConfig) {
        self.postMessage({ type: 'error', message: 'Training not initialized' });
        return;
    }

    const { learningRate, epochs, sequenceLength, slideStep } = trainingConfig;

    while (currentEpoch < epochs && !isPaused) {
        let totalLoss = 0;
        let batchCount = 0;

        // Slide window across training data
        for (let start = 0; start < tokens.length - sequenceLength - 1 && !isPaused; start += slideStep) {
            const inputTokens = tokens.slice(start, start + sequenceLength);
            const targetTokens = tokens.slice(start + 1, start + sequenceLength + 1);

            const loss = model.train(inputTokens, targetTokens, learningRate);
            totalLoss += loss;
            batchCount++;
        }

        if (!isPaused) {
            const avgLoss = batchCount > 0 ? totalLoss / batchCount : 0;

            self.postMessage({
                type: 'epoch',
                epoch: currentEpoch,
                loss: avgLoss,
                totalEpochs: epochs
            });

            currentEpoch++;

            // Yield to allow message processing
            await new Promise(r => setTimeout(r, 0));
        }
    }

    if (currentEpoch >= epochs) {
        self.postMessage({ type: 'complete' });
    } else {
        self.postMessage({ type: 'paused', epoch: currentEpoch });
    }
}

function generate(prompt, length) {
    if (!model || !tokenizer) {
        self.postMessage({ type: 'error', message: 'Model not ready' });
        return;
    }

    try {
        const promptTokens = tokenizer.encode(prompt);
        const generated = model.generate(promptTokens, length);
        const text = tokenizer.decode(generated);

        self.postMessage({ type: 'generated', text });
    } catch (err) {
        self.postMessage({ type: 'error', message: `Generation error: ${err.message}` });
    }
}

function sendState() {
    self.postMessage({
        type: 'state',
        hasModel: model !== null,
        hasTokenizer: tokenizer !== null,
        currentEpoch,
        isPaused,
        vocabSize: tokenizer ? tokenizer.vocabulary.size : 0
    });
}

function loadModel(modelData, tokenizerData) {
    try {
        tokenizer = Tokenizer.deserialize(new Uint8Array(tokenizerData));
        model = ChatRNN.deserialize(new Uint8Array(modelData));
        currentEpoch = 0;
        isPaused = true;

        self.postMessage({
            type: 'log',
            message: `Model loaded. Vocabulary size: ${tokenizer.vocabulary.size}`
        });
        self.postMessage({ type: 'modelLoaded', vocabSize: tokenizer.vocabulary.size });
    } catch (err) {
        self.postMessage({ type: 'error', message: `Load error: ${err.message}` });
    }
}

function saveModel() {
    if (!model || !tokenizer) {
        self.postMessage({ type: 'error', message: 'No model to save' });
        return;
    }

    const modelData = model.serialize();
    const tokenizerData = tokenizer.serialize();

    self.postMessage({
        type: 'modelData',
        modelData: modelData.buffer,
        tokenizerData: tokenizerData.buffer
    }, [modelData.buffer, tokenizerData.buffer]);
}
