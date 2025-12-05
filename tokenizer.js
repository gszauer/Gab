class Tokenizer {
    merges = new Map();           
    vocabulary = new Map();       
    nextTokenId = 256;            
    
    constructor() {
        // Initialize vocabulary with single-byte tokens
        for (let i = 0; i < 256; i++) {
            this.vocabulary.set(i, [i]);
        }
    }
    
    encode(textToEncode) {
        let tokens = this.#stringToBytes(textToEncode);
        
        // Apply merges in the order they were learned
        // JavaScript Maps maintain insertion order, so iterating gives us merges
        // in the same order they were added during training
        for (const [mergeKey, mergedToken] of this.merges) {
            const [token1, token2] = mergeKey.split(',').map(Number);
            tokens = this.#applyMerge(tokens, token1, token2, mergedToken);
        }
        
        return tokens;
    }
    
    decode(tokensToDecode) {
        const bytes = [];
        
        for (let i = 0; i < tokensToDecode.length; i++) {
            const token = tokensToDecode[i];
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
        const bytes = this.#stringToBytes(specialTokenString);
        let tokens = [...bytes];
        
        while (tokens.length > 1) {
            const newTokenId = this.#makeMerge(tokens[0], tokens[1]);
            tokens = this.#applyMerge(tokens, tokens[0], tokens[1], newTokenId);
        }
        
        const finalTokenId = tokens[0];
        console.log(`Reserved special token "${specialTokenString}" as ID ${finalTokenId}`);
        return finalTokenId;
    }
    
    train(trainingText, numMerges, progressCallback) {
        let tokens = this.#stringToBytes(trainingText);
        const originalLength = tokens.length || 1;
        let mergesPerformed = 0;
        
        console.log(`Starting training with ${tokens.length} bytes`);
        
        for (let mergeNum = 0; mergeNum < numMerges; mergeNum++) {
            const pair = this.#findMostFrequentPair(tokens);
            
            if (!pair) {
                console.log(`Training stopped early at merge ${mergeNum} - no more frequent pairs`);
                break;
            }
            
            const [token1, token2] = pair;
            const newTokenId = this.#makeMerge(token1, token2);
            tokens = this.#applyMerge(tokens, token1, token2, newTokenId);
            mergesPerformed += 1;
            
            const compressionRatio = ((1 - tokens.length / originalLength) * 100);

            if (mergeNum % 100 === 0 || mergeNum < 10) {
                const readableCompression = compressionRatio.toFixed(1);
                console.log(`Merge ${mergeNum}: [${token1}, ${token2}] -> ${newTokenId}, compression: ${readableCompression}%`);
            }

            if (typeof progressCallback === 'function') {
                try {
                    progressCallback({
                        mergeNumber: mergeNum + 1,
                        totalMerges: numMerges,
                        pair: [token1, token2],
                        newTokenId,
                        compression: compressionRatio,
                        tokenCount: tokens.length
                    });
                } catch (error) {
                    console.warn('Tokenizer progress callback failed', error);
                }
            }
        }
        
        console.log(`Training complete. Vocabulary size: ${this.vocabulary.size}`);
        return mergesPerformed;
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
        const mergeKey = `${token1},${token2}`;
        const existingMerge = this.merges.get(mergeKey);
        
        if (existingMerge !== undefined) {
            return existingMerge;
        }
        
        const newTokenId = this.nextTokenId++;
        this.merges.set(mergeKey, newTokenId);
        
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
                result.push(mergedTokenId);
                i += 2;
            } else {
                result.push(tokens[i]);
                i += 1;
            }
        }
        
        return result;
    }
    
    #findMostFrequentPair(tokensList) {
        const pairCounts = new Map();
        
        for (let i = 0; i < tokensList.length - 1; i++) {
            const pair = `${tokensList[i]},${tokensList[i + 1]}`;
            const currentCount = pairCounts.get(pair) || 0;
            pairCounts.set(pair, currentCount + 1);
        }
        
        let maxCount = 0;
        let mostFrequentPair = null;
        
        for (const [pair, count] of pairCounts) {
            if (count > maxCount) {
                maxCount = count;
                mostFrequentPair = pair;
            }
        }
        
        if (mostFrequentPair && maxCount > 1) {
            const tokens = mostFrequentPair.split(',');
            return [parseInt(tokens[0]), parseInt(tokens[1])];
        }
        
        return null;
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
