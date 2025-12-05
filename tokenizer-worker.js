/* global self, Tokenizer */
// Worker responsible for running tokenizer training without blocking the UI thread.

importScripts('tokenizer.js');

let tokenizer = new Tokenizer();
let jobCounter = 0;

self.addEventListener('message', event => {
    const data = event.data || {};
    const type = data.type;

    switch (type) {
        case 'train':
            handleTrainingJob(data);
            break;
        default:
            postMessage({ type: 'training-error', message: `Unknown worker message: ${String(type)}` });
    }
});

function handleTrainingJob({ jobId, corpus, targetVocab, reservedTokens = [], corpusName = 'tokenizer' }) {
    const currentJobId = jobId ?? ++jobCounter;

    if (typeof corpus !== 'string') {
        postMessage({
            type: 'training-error',
            jobId: currentJobId,
            message: 'Training corpus was missing or invalid.'
        });
        return;
    }

    const numericTarget = Number(targetVocab);
    if (!Number.isFinite(numericTarget)) {
        postMessage({
            type: 'training-error',
            jobId: currentJobId,
            message: 'Target vocabulary must be a finite number.'
        });
        return;
    }

    try {
        tokenizer = new Tokenizer();
        let reservedCount = 0;

        for (const token of reservedTokens) {
            try {
                tokenizer.reserveToken(token);
                reservedCount += 1;
            } catch (error) {
                postMessage({
                    type: 'log',
                    jobId: currentJobId,
                    message: `Unable to reserve "${token}": ${error?.message || 'unknown error'}`
                });
            }
        }

        const mergesNeeded = Math.max(0, numericTarget - tokenizer.vocabulary.size);

        postMessage({
            type: 'log',
            jobId: currentJobId,
            message: `Training for ${mergesNeeded.toLocaleString()} merges to approach ${numericTarget.toLocaleString()} tokens.`
        });
        postMessage({ type: 'status', jobId: currentJobId, message: 'Training tokenizer…', tone: 'busy' });
        postMessage({
            type: 'progress',
            jobId: currentJobId,
            merge: 0,
            totalMerges: mergesNeeded,
            compression: 0
        });

        const progressInterval = 20;
        const performed = tokenizer.train(corpus, mergesNeeded, progress => {
            if (!progress) {
                return;
            }
            const shouldReport =
                progress.mergeNumber <= progressInterval ||
                progress.mergeNumber % progressInterval === 0 ||
                progress.mergeNumber === progress.totalMerges;
            if (!shouldReport) {
                return;
            }
            postMessage({
                type: 'progress',
                jobId: currentJobId,
                merge: progress.mergeNumber,
                totalMerges: progress.totalMerges,
                pair: progress.pair,
                newTokenId: progress.newTokenId,
                compression: progress.compression,
                tokenCount: progress.tokenCount
            });
        });

        const serialized = tokenizer.serialize();
        postMessage({
            type: 'trained',
            jobId: currentJobId,
            reservedCount,
            mergesPerformed: performed,
            vocabSize: tokenizer.vocabulary.size,
            mergesRequested: mergesNeeded,
            corpusName,
            tokenizerBytes: serialized.buffer
        }, [serialized.buffer]);
    } catch (error) {
        postMessage({
            type: 'training-error',
            jobId: currentJobId,
            message: error?.message || 'Tokenizer worker failed.',
            stack: error?.stack || null
        });
    }
}
