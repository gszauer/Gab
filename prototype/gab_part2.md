# Building a Byte Pair Encoding Tokenizer from Scratch

Language models don't see text the way we do. They see sequences of tokens, numerical representations of text or characters. This tutorial walks you through building a complete BPE tokenizer using plain JavaScript. We'll implement each component step by step.

# Understanding Byte Pair Encoding

At its core, BPE is a compression algorithm that merges frequent byte pairs into single tokens. It starts with individual bytes as the base vocabulary, then iteratively merges the most frequent adjacent pairs until reaching a target vocabulary size. This simple process creates an effective tokenization scheme that balances vocabulary size with representation efficiency.

```
Initial:    [72, 101, 108, 108, 111, 32, 87, 111, 114, 108, 100]  // "Hello World" as bytes
After BPE:  [256, 257, 258]  // Merged frequent pairs into new tokens
```

> BPE learns which character sequences appear frequently together and creates shortcuts for them. The word "the" appears so often in English that it becomes a single token rather than three separate byte tokens.

# The Tokenizer Architecture

Our tokenizer needs to handle several responsibilities:
1. **Convert between text and bytes** - All text ultimately becomes bytes
2. **Learn merges from training data** - Find patterns in the training corpus
3. **Encode text to tokens** - Apply learned merges to new text
4. **Decode tokens back to text** - Reverse the encoding process
5. **Reserve special tokens** - Pre-create merge chains for special sequences

Let's translate this system into code:

```javascript
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
}
```

The tokenizer starts with 256 base tokens - one for each possible byte value. Why bytes instead of characters? **Unicode safety**. By working at the byte level, we can handle any text encoding without worrying about character boundaries or encoding issues.

## Converting Between Text and Bytes

Before we can tokenize, we need robust conversion between strings and bytes. JavaScript's TextEncoder/TextDecoder handle UTF-8 encoding:

```javascript
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
```

The `#stringToBytes` method converts a string into an array of byte values (0-255). We explicitly create a regular JavaScript array rather than returning the Uint8Array directly because our token arrays need to hold values larger than 255 - after merging, we'll have token IDs like 256, 257, and beyond. A Uint8Array is limited to values 0-255, but a regular array can hold any JavaScript number.

The `#bytesToString` reverses the process, converting byte arrays back to text. We wrap the input in `new Uint8Array()` to handle both regular arrays and typed arrays.

## The Core Helper Functions

The heart of our BPE implementation lies in two helper functions that handle merge creation and application. These functions do the heavy lifting and are used throughout the tokenizer:

### Creating Merges

This function creates a new token that represents the merging of two existing tokens:

```javascript
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
```


1. **Checks for existing merges** - If we've already merged these tokens, reuse the existing ID. This is crucial for special tokens that might create overlapping merges.
2. **Creates a merge rule** - The key `"token1,token2"` maps to the new token ID, telling the encoder "when you see these two tokens together, replace them with this new token."
3. **Updates the vocabulary** - We store what bytes the new token represents by concatenating the byte sequences of its components.
4. **Manages token IDs** - Auto-increments `nextTokenId` to ensure unique IDs.
5. **Preserves priority order** - Because JavaScript Maps maintain insertion order, merges added first (more frequent patterns) will be applied first during encoding.

> The merge key format (`"token1,token2"`) uses string concatenation for simplicity. In production, you might use a more efficient structure. Maps preserve insertion order is critical, it means our merge priority is implicitly stored without needing a separate priority field.

### Applying Merges

This function scans through a token sequence and replaces all occurrences of a specific pair:

```javascript
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
```

This function implements a single-pass merge operation. It:
1. **Scans left to right** - Processes tokens in order, ensuring deterministic merging
2. **Applies greedily** - When it finds a matching pair, it immediately merges and skips both tokens
3. **Preserves non-matching tokens** - Tokens that don't match the merge pattern pass through unchanged
4. **Returns a new array** - Never mutates the input, following functional programming principles

The greedy approach is important: if we have tokens `[1, 2, 2]` and we're merging `(2, 2)`, we'll get `[1, merged]`, not `[merged, 2]`. The first match wins.

## Finding Frequent Pairs

The training process needs to identify which byte pairs appear most frequently:

```javascript
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
```

This method scans through the token list, counting every adjacent pair. It uses the same string key format (`"token1,token2"`) as our merge rules for consistency. The method returns the most frequent pair as an array of two token IDs, or `null` if no pair appears more than once.

> Why require `maxCount > 1`? A pair that appears only once isn't worth merging - it won't save space or improve efficiency.

## Training the Tokenizer

Training builds the merge rules by iteratively finding and merging the most frequent pairs. With our helper functions, the training logic becomes beautifully clear:

```javascript
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
```

The training algorithm follows these 4 steps:
1. **Find the most common pair** using `#findMostFrequentPair`
2. **Create a merge rule** using `#makeMerge`
3. **Apply it everywhere** using `#applyMerge`
4. **Repeat** until we've created enough merges

## Encoding Text to Tokens

Encoding applies learned merge rules to new text. Merges should be applied in the order they were learned during training:

```javascript
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
```

The encoding process:
1. **Converts text to bytes** - Starts with the raw byte representation
2. **Applies merges in learned order** - Iterates through the Map of merges, which preserves insertion order
3. **Each merge is applied globally** - When we apply a merge, it replaces ALL occurrences in one pass

> Why does order matter? Consider if we learned merges for both "th" and "he". If "th" was more frequent (learned first), we want "the" to become ["th", "e"], not ["t", "he"]. By applying merges in learned order, more frequent patterns get priority.

## Decoding Tokens Back to Text

Decoding is the simplest operation - we just look up what each token represents:

```javascript
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
```

Each token ID maps to a sequence of bytes through our vocabulary. We concatenate all the byte sequences and convert back to a string. The process is deterministic and lossless - we can always recover the exact original text.

Notice there's no special handling needed - every token, whether it's a single byte, a merged pair, or a special token, is just an entry in the vocabulary.

## Handling Special Tokens

Special tokens like `<|endoftext|>` or `<|pad|>` need to remain intact during tokenization. We achieve this by pre-creating merge chains:

```javascript
reserveToken(specialTokenString) {
    // Convert the special token to bytes
    const bytes = this.#stringToBytes(specialTokenString);
    let tokens = [...bytes]; // Copy the array

    // Apply all existing merges first
    // This ensures shared prefixes like <|im_ reuse existing merge chains
    for (const [mergeKey, mergedToken] of this.merges) {
        const [token1, token2] = mergeKey.split(',').map(Number);
        tokens = this.#applyMerge(tokens, token1, token2, mergedToken);
    }

    // Create merge chain to combine remaining tokens into a single token
    while (tokens.length > 1) {
        // Create a merge for the first two tokens
        const newTokenId = this.#makeMerge(tokens[0], tokens[1]);

        // Apply it to our token array
        tokens = this.#applyMerge(tokens, tokens[0], tokens[1], newTokenId);
    }

    const finalTokenId = tokens[0];
    return finalTokenId;
}
```

For example, reserving `<|pad|>`:
- Start with bytes: `[60, 124, 112, 97, 100, 124, 62]`
- Merge `(60,124)` -> 256
- Now we have: `[256, 112, 97, 100, 124, 62]`
- Merge `(256,112)` -> 257
- ... Continue until: `[261]` - a single token!

These merges become part of the global merge rules. When encoding encounters `<|pad|>` in any text, the normal merge process naturally combines it into token 261. If you reserve the same special token twice, it will simply create the same merge chain (or reuse existing merges), returning the same final token ID.

# Using the Tokenizer

Let's see our tokenizer in action with a practical example:

```javascript
// Create and train a tokenizer
const tokenizer = new Tokenizer();

// Reserve special tokens before training
// This creates merge chains so they'll always be single tokens
tokenizer.reserveToken("<|endoftext|>");
tokenizer.reserveToken("<|pad|>");

// Train on sample text
const trainingData = `
The quick brown fox jumps over the lazy dog.
The dog was lazy but the fox was quick.
Quick foxes and lazy dogs don't mix well.
The the the - common words should merge.
`;

tokenizer.train(trainingData, 100);

// Test encoding and decoding
const testText = "The quick fox";
const encoded = tokenizer.encode(testText);
console.log(`"${testText}" -> [${encoded.join(", ")}]`);

const decoded = tokenizer.decode(encoded);
console.log(`[${encoded.join(", ")}] -> "${decoded}"`);

// Test that special tokens work correctly
const specialText = "Hello <|endoftext|> World";
const specialEncoded = tokenizer.encode(specialText);
console.log(`Special tokens remain intact: "${specialText}"`);
console.log(`Encoded as: [${specialEncoded.join(", ")}]`);

// The <|endoftext|> token appears as a single token ID in the sequence
// because we created the merge chain before training

// Verify round-trip conversion
const specialDecoded = tokenizer.decode(specialEncoded);
console.log(`Round-trip successful: ${specialText === specialDecoded}`);
```

The tokenizer learns patterns specific to your training data. Common words and phrases become single tokens, dramatically reducing sequence length. Special tokens like `<|endoftext|>` are guaranteed to remain as single tokens because we pre-created their merge chains.

# Saving / Loading

You will probably want to re-use pre-made tokenizers. After all, tokenization takes time. For this reason, we should include serialize / deserialize functions in the tokenizer. The encode function returns a uint8 array of bytes. The decode function takesa uint8 array of bytes. 

```javascript
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
```

# Performance Characteristics

BPE tokenization has it's trade-offs:

**Vocabulary Size vs Sequence Length**: More merges mean larger vocabulary but shorter sequences. Most modern models use 30,000-50,000 tokens as a sweet spot.

**Training Time**: O(n * m) where n is training text length and m is number of merges. Each merge requires scanning the entire token sequence.

**Encoding Time**: O(m * n) where m is the number of merges and n is the average text length after applying previous merges. We apply each merge rule exactly once in order, with each merge potentially scanning the entire sequence. This is more predictable than the iterative approach and guarantees the same tokenization as the training data.

**Memory Usage**: Vocabulary grows linearly with merges. Each merge adds one vocabulary entry and one merge rule. JavaScript Maps efficiently maintain insertion order without additional memory overhead.

**Special Token Handling**: O(s) where s is the length of the special token string. Creating merge chains is a one-time cost during reservation, after which special tokens encode with zero additional overhead - they use the same merge logic as regular text.

