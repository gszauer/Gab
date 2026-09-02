export class Tensor {
    // When false, operations skip building the graph:
    static gradEnabled = true;

    constructor(rows, columns) {
        this.rows = rows;
        this.columns = columns;
        this.data = new Float32Array(rows * columns); // zero-filled
        this.grad = null;             // allocated on demand by backward()
        this._inputs = [];            // tensors this one was built from
        this._backward = () => {};    // leaves do nothing
    }

    get(row, column) { // returns a plain number now, not a Value
        return this.data[row * this.columns + column];
    }

    set(row, column, value) {
        this.data[row * this.columns + column] = value;
    }

    // --- Initialization: these create the model's leaf parameters. ---

    initToZeroes() {
        for (let i = 0; i < this.data.length; ++i) {
            this.data[i] = 0.0;
        }
    }

    initToOnes() {
        for (let i = 0; i < this.data.length; ++i) {
            this.data[i] = 1.0;
        }
    }

    initToSmallRandom() {
        const scale = 1 / Math.sqrt(this.rows);
        for (let i = 0; i < this.data.length; ++i) {
            this.data[i] = (Math.random() - 0.5) * 2 * scale; // uniform in [-scale, +scale)
        }
    }

    zeroGrad() {
        if (this.grad !== null) this.grad.fill(0);
    }

    parameters() {
        return [this]; // a tensor is one parameter block
    }

    // --- Element-wise operations (both tensors share the same shape). ---

    add(other) {
        const out = new Tensor(this.rows, this.columns);
        for (let i = 0; i < this.data.length; ++i) {
            out.data[i] = this.data[i] + other.data[i];
        }
        if (Tensor.gradEnabled) {
            out._inputs = [this, other];
            out._backward = () => {
                for (let i = 0; i < this.data.length; ++i) {
                    this.grad[i]  += out.grad[i];
                    other.grad[i] += out.grad[i];
                }
            };
        }
        return out;
    }

    sub(other) {
        const out = new Tensor(this.rows, this.columns);
        for (let i = 0; i < this.data.length; ++i) {
            out.data[i] = this.data[i] - other.data[i];
        }
        if (Tensor.gradEnabled) {
            out._inputs = [this, other];
            out._backward = () => {
                for (let i = 0; i < this.data.length; ++i) {
                    this.grad[i]  += out.grad[i];
                    other.grad[i] -= out.grad[i];
                }
            };
        }
        return out;
    }

    mul(other) { // element-wise multiply (matching numpy/pytorch); matrix multiply is matmul
        const out = new Tensor(this.rows, this.columns);
        for (let i = 0; i < this.data.length; ++i) {
            out.data[i] = this.data[i] * other.data[i];
        }
        if (Tensor.gradEnabled) {
            out._inputs = [this, other];
            out._backward = () => {
                for (let i = 0; i < this.data.length; ++i) {
                    this.grad[i]  += other.data[i] * out.grad[i];
                    other.grad[i] += this.data[i]  * out.grad[i];
                }
            };
        }
        return out;
    }

    scale(scalar) { // multiply every value by one plain number
        const out = new Tensor(this.rows, this.columns);
        for (let i = 0; i < this.data.length; ++i) {
            out.data[i] = this.data[i] * scalar;
        }
        if (Tensor.gradEnabled) {
            out._inputs = [this];
            out._backward = () => {
                for (let i = 0; i < this.data.length; ++i) {
                    this.grad[i] += scalar * out.grad[i];
                }
            };
        }
        return out;
    }

    relu() {
        const out = new Tensor(this.rows, this.columns);
        for (let i = 0; i < this.data.length; ++i) {
            out.data[i] = this.data[i] > 0 ? this.data[i] : 0;
        }
        if (Tensor.gradEnabled) {
            out._inputs = [this];
            out._backward = () => {
                for (let i = 0; i < this.data.length; ++i) {
                    this.grad[i] += (this.data[i] > 0 ? 1 : 0) * out.grad[i];
                }
            };
        }
        return out;
    }

    gelu() {
        // Exact GELU: x * Phi(x), where Phi is the standard normal CDF.
        // Phi(x) = 0.5 * (1 + erf(x / sqrt(2))), and JS has no erf, so it's inlined.
        const SQRT_2 = Math.sqrt(2);
        const INV_SQRT_2PI = 1 / Math.sqrt(2 * Math.PI);

        // Abramowitz & Stegun 7.1.26 — erf to ~1e-7.
        const erf = (x) => {
            const sign = x < 0 ? -1 : 1;
            const ax = Math.abs(x);
            const t = 1 / (1 + 0.3275911 * ax);
            const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-ax * ax);
            return sign * y;
        };

        const out = new Tensor(this.rows, this.columns);
        for (let i = 0; i < this.data.length; ++i) {
            const x = this.data[i];
            out.data[i] = x * 0.5 * (1 + erf(x / SQRT_2));
        }

        if (Tensor.gradEnabled) {
            out._inputs = [this];
            out._backward = () => {
                for (let i = 0; i < this.data.length; ++i) {
                    const x = this.data[i];
                    const cdf = 0.5 * (1 + erf(x / SQRT_2));
                    const pdf = INV_SQRT_2PI * Math.exp(-0.5 * x * x);
                    // d/dx [x * Phi(x)] = Phi(x) + x * phi(x)
                    this.grad[i] += (cdf + x * pdf) * out.grad[i];
                }
            };
        }
        return out;
    }

    // --- Matrix multiplication: the one node that replaces the most scalars. ---

    matmul(other) { // (this.rows x this.columns) * (other.rows x other.columns)
        const m = this.rows;
        const k = this.columns; // shared inner dimension
        const n = other.columns;
        const out = new Tensor(m, n);

        for (let row = 0; row < m; ++row) {
            for (let col = 0; col < n; ++col) {
                let sum = 0.0; // accumulate in double, store as float
                for (let element = 0; element < k; ++element) {
                    sum += this.data[row * k + element] * other.data[element * n + col];
                }
                out.data[row * n + col] = sum;
            }
        }

        if (Tensor.gradEnabled) {
            out._inputs = [this, other];
            out._backward = () => {
                // dThis = dOut * other^T
                for (let row = 0; row < m; ++row) {
                    for (let p = 0; p < k; ++p) {
                        let sum = 0.0;
                        for (let col = 0; col < n; ++col) {
                            sum += out.grad[row * n + col] * other.data[p * n + col];
                        }
                        this.grad[row * k + p] += sum;
                    }
                }
                // dOther = this^T * dOut
                for (let p = 0; p < k; ++p) {
                    for (let col = 0; col < n; ++col) {
                        let sum = 0.0;
                        for (let row = 0; row < m; ++row) {
                            sum += this.data[row * k + p] * out.grad[row * n + col];
                        }
                        other.grad[p * n + col] += sum;
                    }
                }
            };
        }
        return out;
    }

    transposed() {
        const out = new Tensor(this.columns, this.rows);
        for (let row = 0; row < this.rows; ++row) {
            for (let col = 0; col < this.columns; ++col) {
                out.data[col * this.rows + row] = this.data[row * this.columns + col];
            }
        }
        if (Tensor.gradEnabled) {
            out._inputs = [this];
            out._backward = () => {
                for (let row = 0; row < this.rows; ++row) {
                    for (let col = 0; col < this.columns; ++col) {
                        this.grad[row * this.columns + col] += out.grad[col * this.rows + row];
                    }
                }
            };
        }
        return out;
    }

    // --- Fused operations. Each is one node with a single analytic backward
    //     rule instead of being assembled from many smaller nodes, which is
    //     both faster and far lighter on memory. These are general array
    //     operations, so they live here. RMS normalization and the
    //     cross-entropy loss are also fused, but each belongs to one component
    //     (the norm layer, the trainer), so they are defined there using the
    //     same _inputs/_backward protocol. ---

    softMaxedRows() {
        const rows = this.rows;
        const cols = this.columns;
        const out = new Tensor(rows, cols);

        for (let row = 0; row < rows; ++row) {
            const base = row * cols;

            let rowMax = this.data[base];
            for (let col = 1; col < cols; ++col) {
                const v = this.data[base + col];
                if (v > rowMax) rowMax = v;
            }

            let rowSum = 0.0;
            for (let col = 0; col < cols; ++col) {
                const e = Math.exp(this.data[base + col] - rowMax);
                out.data[base + col] = e;
                rowSum += e;
            }
            for (let col = 0; col < cols; ++col) {
                out.data[base + col] /= rowSum;
            }
        }

        if (Tensor.gradEnabled) {
            out._inputs = [this];
            out._backward = () => {
                // For each row: dx_j = p_j * (g_j - sum_k g_k p_k)
                for (let row = 0; row < rows; ++row) {
                    const base = row * cols;
                    let dot = 0.0;
                    for (let col = 0; col < cols; ++col) {
                        dot += out.grad[base + col] * out.data[base + col];
                    }
                    for (let col = 0; col < cols; ++col) {
                        const p = out.data[base + col];
                        this.grad[base + col] += p * (out.grad[base + col] - dot);
                    }
                }
            };
        }
        return out;
    }

    causalMasked() {
        const rows = this.rows;
        const cols = this.columns;
        const out = new Tensor(rows, cols);

        for (let row = 0; row < rows; ++row) {
            for (let col = 0; col < cols; ++col) {
                const index = row * cols + col;
                out.data[index] = col > row ? -Infinity : this.data[index];
            }
        }

        if (Tensor.gradEnabled) {
            out._inputs = [this];
            out._backward = () => {
                // Masked cells are constants, so gradient only flows where col <= row.
                for (let row = 0; row < rows; ++row) {
                    for (let col = 0; col <= row; ++col) {
                        const index = row * cols + col;
                        this.grad[index] += out.grad[index];
                    }
                }
            };
        }
        return out;
    }

    // Embedding lookup: copy the rows named by `indices` into a new tensor.
    // The gradient scatters back, accumulating when an index repeats.
    gatherRows(indices) {
        const out = new Tensor(indices.length, this.columns);
        for (let r = 0; r < indices.length; ++r) {
            const src = indices[r] * this.columns;
            const dst = r * this.columns;
            for (let col = 0; col < this.columns; ++col) {
                out.data[dst + col] = this.data[src + col];
            }
        }
        if (Tensor.gradEnabled) {
            out._inputs = [this];
            out._backward = () => {
                for (let r = 0; r < indices.length; ++r) {
                    const src = indices[r] * this.columns;
                    const dst = r * this.columns;
                    for (let col = 0; col < this.columns; ++col) {
                        this.grad[src + col] += out.grad[dst + col];
                    }
                }
            };
        }
        return out;
    }

    // Stack another tensor's rows below this one's, returning a new, taller
    // tensor. Both must share the same column count. The KV cache uses this to
    // grow one row per generated token. The gradient of the top rows flows back
    // to this, the bottom rows to other. Generation runs with the graph off, so
    // that rule never fires in practice, but it keeps appendRow a first-class op.
    appendRow(other) {
        // A new tensor: this tensor's rows on top, other's rows stacked below.
        // Both must share the same column count.
        const out = new Tensor(this.rows + other.rows, this.columns);
        out.data.set(this.data, 0);                 // top block
        out.data.set(other.data, this.data.length); // bottom block

        if (Tensor.gradEnabled) {
            out._inputs = [this, other];
            out._backward = () => {
                // Gradient of the top rows flows to this, the bottom rows to other.
                for (let i = 0; i < this.data.length; ++i) {
                    this.grad[i] += out.grad[i];
                }
                for (let i = 0; i < other.data.length; ++i) {
                    other.grad[i] += out.grad[this.data.length + i];
                }
            };
        }
        return out;
    }

    // RoPE: rotate each row's component pairs by an angle set by the row's
    // position. Query and key rows sit in sequence order, so the row index is
    // the position — plus positionOffset, which the KV cache uses to rotate a
    // new token by its true position rather than its row (offset zero, the
    // default, reproduces the original behavior). The angles are fixed functions
    // of position and frequency, not learned, so this op adds nothing to
    // parameters(); the gradient only routes back to the input. The cosine and
    // sine of each angle are cached, since the backward pass — a rotation by the
    // negated angle — reuses them.
    ropeRotated(thetaBase = 10000, positionOffset = 0) { // positionOffset is new
        const rows = this.rows;
        const cols = this.columns; // head_dim, must be even
        const halfCols = cols / 2;
        const out = new Tensor(rows, cols);

        // Frequency depends on the pair, not the row
        // Compute it once and cache
        const freqs = new Float32Array(halfCols);
        for (let pair = 0; pair < halfCols; ++pair) {
            freqs[pair] = 1.0 / Math.pow(thetaBase, (2 * pair) / cols);
        }

        // Cached for the backward pass.
        const cosTable = new Float32Array(rows * halfCols);
        const sinTable = new Float32Array(rows * halfCols);

        for (let row = 0; row < rows; ++row) {
            const base = row * cols;
            const trig = row * halfCols;
            for (let pair = 0; pair < halfCols; ++pair) {
                const angle = (positionOffset + row) * freqs[pair]; // true position = offset + row
                const cos = Math.cos(angle);
                const sin = Math.sin(angle);
                cosTable[trig + pair] = cos;
                sinTable[trig + pair] = sin;

                const a = this.data[base + 2 * pair];
                const b = this.data[base + 2 * pair + 1];
                out.data[base + 2 * pair]     = a * cos - b * sin;
                out.data[base + 2 * pair + 1] = a * sin + b * cos;
            }
        }

        if (Tensor.gradEnabled) {
            out._inputs = [this];
            out._backward = () => {
                // Rotation is linear, the gradient is the inverse rotation.
                for (let row = 0; row < rows; ++row) {
                    const base = row * cols;
                    const trig = row * halfCols;
                    for (let pair = 0; pair < halfCols; ++pair) {
                        const cos = cosTable[trig + pair];
                        const sin = sinTable[trig + pair];
                        const ga = out.grad[base + 2 * pair];
                        const gb = out.grad[base + 2 * pair + 1];
                        this.grad[base + 2 * pair]     +=  ga * cos + gb * sin;
                        this.grad[base + 2 * pair + 1] += -ga * sin + gb * cos;
                    }
                }
            };
        }
        return out;
    }

    backward() {
        // Topological sort: inputs come before the nodes built from them.
        const topo = [];
        const visited = new Set();

        const visit = (node) => {
            if (visited.has(node)) return;
            visited.add(node);
            for (let i = 0; i < node._inputs.length; ++i) {
                visit(node._inputs[i]);
            }
            topo.push(node);
        };
        visit(this);

        // Make sure every node in the graph has a gradient buffer. Freshly
        // built tensors start at null and get a zeroed buffer here. Parameter
        // tensors keep the buffer the trainer already zeroed.
        for (let i = 0; i < topo.length; ++i) {
            if (topo[i].grad === null) {
                topo[i].grad = new Float32Array(topo[i].data.length);
            }
        }

        // Seed: derivative of the output with respect to itself is 1.
        this.grad.fill(1);

        // Walk in reverse. Each node pushes gradient onto its inputs.
        for (let i = topo.length - 1; i >= 0; --i) {
            topo[i]._backward();
        }
    }
}

export class Tokenizer {
    constructor() {
        // Merge rules array.
        // Index = token id.
        // Value = the two token ids that merge into index.
        this.merges = new Array();

        // Reserved tokens, an array of strings
        this.reserved = new Array();

        // Seed first 256 tokens as placeholders.
        // These merge rules are not actually valid.
        for (let i = 0; i < 256; ++i) {
            this.merges.push([i, i]);
        }
    }

    vocabSize() {
        return this.merges.length;
    }

    encode(text) { // string -> Array<int>
        let bytes = new TextEncoder().encode(text);

        // Apply every merge rule, in order
        for (let rule = 256; rule < this.merges.length; ++rule) {
            bytes = this._merge(bytes, this.merges[rule][0], this.merges[rule][1], rule);
        }

        return bytes;
    }

    // Given a list of tokens, replace every instance of firstToken and secondToken with repalcementToken
    _merge(listOfTokens, firstToken, secondToken, replacementToken) {
        const result = new Array();

        for (let i = 0; i < listOfTokens.length; ++i) {
            if (listOfTokens[i] == firstToken) {
                if (i + 1 < listOfTokens.length) {
                    if (listOfTokens[i + 1] == secondToken) {
                        result.push(replacementToken);
                        i += 1;
                        continue;
                    }
                }
            }
            result.push(listOfTokens[i]);
        }

        return result;
    }

    decode(ids) { // Array<int> -> string
        const bytes = [];

        for (let i = 0; i < ids.length; ++i) {
            const stack = [ids[i]];

            while (stack.length > 0) {
                const id = stack.pop();

                if (id < 256) {
                    bytes.push(id);
                }
                else {
                    const pair = this.merges[id];
                    stack.push(pair[1]); // second half pushed FIRST,
                    stack.push(pair[0]); // so first half pops first
                }
            }
        }

        return new TextDecoder().decode(new Uint8Array(bytes));
    }

    reserve(text) { // string -> id; idempotent, call before train
        // 1. Apply every existing merge, in order. This is the dedupe: anything already learnable collapses now.
        let bytes = this.encode(text);

        // 2. Teach the leftovers, left to right. Each new rule's id is just its index.
        let id = bytes[0];
        for (let i = 1; i < bytes.length; ++i) {
            this.merges.push([id, bytes[i]]);
            id = this.merges.length - 1;
        }

        this.reserved.push(text);
        return id;
    }

    // inputText -> "Long text"
    // returns -> ["Long", " text"]
    _split(inputText) { // Splits long text into an array of text chunks
        // Sort this.reserved by reverse length, so _matchKeyword can match longest first
        this.reserved.sort((a, b) => b.length - a.length);

        const _matchKeyword = (text, position, keywords) => {
            for (const keyword of keywords) {
                if (text.startsWith(keyword, position)) {
                    return keyword;
                }
            }
            return null;
        }

        const _isLetter = (char) => {
            return char !== undefined && char.toLowerCase() !== char.toUpperCase();
        }

        const _isDigit = (char) => {
            return char >= '0' && char <= '9';
        }

        const chunks = [];
        let i = 0;

        while (i < inputText.length) {
            // Rule 1: keywords always win, they are atomic
            const keyword = _matchKeyword(inputText, i, this.reserved);
            if (keyword !== null) {
                chunks.push(keyword);
                i += keyword.length;
                continue;
            }

            const char = inputText[i];

            // Rule 2: a word, optionally carrying ONE leading space
            if (_isLetter(char) || (char === ' ' && _isLetter(inputText[i + 1]))) {
                let chunk = char;
                i++;
                while (i < inputText.length && _isLetter(inputText[i])) {
                    chunk += inputText[i];
                    i++;
                }
                chunks.push(chunk);
                continue;
            }

            // Rule 3: digits, grouped to at most 3
            if (_isDigit(char)) {
                let chunk = '';
                while (chunk.length < 3 && _isDigit(inputText[i])) {
                    chunk += inputText[i];
                    i++;
                }
                chunks.push(chunk);
                continue;
            }

            // Rule 4: anything else (punctuation, leftover spaces) is its own chunk
            chunks.push(char);
            i++;
        }

        return chunks;
    }

    train(text, targetVocabSize) { // learns merges
        const splitText = this._split(text);
        const chunks = new Array(splitText.length);
        for (let i = 0; i < splitText.length; ++i) {
            chunks[i] = this.encode(splitText[i]);
        }

        // while vocabSize < targetVocabSize
        while (this.merges.length < targetVocabSize) {
            const pairs = new Map();

            // Count pairs in all chunks
            for (let i = 0; i < chunks.length; ++i) {
                const chunk = chunks[i];
                for (let j = 0; j < chunk.length - 1; ++j) {
                    const key = chunk[j] + ',' + chunk[j + 1];

                    const entry = pairs.get(key);
                    if (entry) {
                        entry.count += 1;
                    }
                    else {
                        pairs.set(key, { count: 1, firstToken: chunk[j], secondToken: chunk[j + 1] });
                    }
                }
            }

            // Pick Best Pair
            let best = null;
            for (const entry of pairs.values()) {
                if (best === null || entry.count > best.count) {
                    best = entry;
                }
            }

            // Early out
            if (best == null) {
                break; // No best pair, all merged
            }
            else if (best.count < 2) {
                // If the best token was only seen once
                break;
            }

            // Record new rule
            this.merges.push([best.firstToken, best.secondToken]);
            const newToken = this.merges.length - 1;

            // Apply new rule to each chunk
            for (let i = 0; i < chunks.length; ++i) {
                chunks[i] = this._merge(chunks[i], best.firstToken, best.secondToken, newToken);
            }
        }
    }

    serializeToJSON() { // -> string
        return JSON.stringify({
            reserved: this.reserved,
            merges: this.merges.slice(256), // ids 0-255 are seeded, never saved
        });
    }

    deserializeFromJSON(json) {  // string -> void; replaces all state
        const data = JSON.parse(json);

        // Restore the freshly-constructed state: 256 byte tokens, nothing else.
        this.merges.length = 256;
        this.reserved = data.reserved;

        // Pushing the pairs back in order reproduces every id exactly
        for (let i = 0; i < data.merges.length; ++i) {
            this.merges.push(data.merges[i]);
        }
    }
}

// A key-value cache holds one key tensor and one value tensor for every block,
// for every head. The model reads and writes it but does not own it: the caller
// creates one and passes it in, so a single MiniGPT can serve many independent
// generations at once, each with its own cache.
export class KVCache {
    constructor(numBlocks, numHeads) {
        // One key tensor and one value tensor per block, per head. Each starts
        // null and becomes a growing tensor as tokens flow through.
        this.keys = [];
        this.values = [];
        for (let block = 0; block < numBlocks; ++block) {
            const blockKeys = [];
            const blockValues = [];
            for (let head = 0; head < numHeads; ++head) {
                blockKeys.push(null);
                blockValues.push(null);
            }
            this.keys.push(blockKeys);
            this.values.push(blockValues);
        }
    }

    // Tokens cached so far. Every block and head advances together, so the
    // first head's key tensor speaks for all of them.
    get length() {
        const first = this.keys[0][0];
        return first === null ? 0 : first.rows;
    }
}

export class RMSNorm {
    constructor(feature_dim) {
        // Learned per-feature scale, applied after normalization. Starts at one,
        // so the layer begins as a plain normalize until training moves it.
        this.gamma = new Tensor(1, feature_dim);
        this.gamma.initToOnes();
    }

    forward(x) {
        const rows = x.rows;
        const n = x.columns;
        const eps = 1e-5;
        const gamma = this.gamma;
        const out = new Tensor(rows, n);
        const invRmsByRow = new Float32Array(rows); // cached for the backward

        for (let row = 0; row < rows; ++row) {
            const base = row * n;

            let sumSq = 0.0;
            for (let col = 0; col < n; ++col) {
                const v = x.data[base + col];
                sumSq += v * v;
            }

            const invRms = 1 / Math.sqrt(sumSq / n + eps);
            invRmsByRow[row] = invRms;

            for (let col = 0; col < n; ++col) {
                // normalize, then scale by the per-feature gamma
                out.data[base + col] = x.data[base + col] * invRms * gamma.data[col];
            }
        }

        if (Tensor.gradEnabled) {
            out._inputs = [x, gamma];
            out._backward = () => {
                for (let row = 0; row < rows; ++row) {
                    const base = row * n;
                    const invRms = invRmsByRow[row];
                    const invRms3 = invRms * invRms * invRms;

                    // Upstream gradient passes through gamma before the RMS rule.
                    let dotGX = 0.0;
                    for (let col = 0; col < n; ++col) {
                        const gNorm = out.grad[base + col] * gamma.data[col];
                        dotGX += gNorm * x.data[base + col];
                    }
                    for (let col = 0; col < n; ++col) {
                        const g = out.grad[base + col];
                        const xv = x.data[base + col];
                        const gNorm = g * gamma.data[col];
                        x.grad[base + col] += gNorm * invRms - (xv * dotGX * invRms3) / n;
                        // gamma is shared across rows: sum the normalized input times upstream grad
                        gamma.grad[col] += xv * invRms * g;
                    }
                }
            };
        }
        return out;
    }

    parameters() {
        return [this.gamma];
    }
}

export class MultiHeadAttention {
    constructor(feature_dim, num_heads, rope_base) {
        this.num_heads = num_heads;
        this.head_dim = feature_dim / num_heads; // must divide evenly, and be even for RoPE
        this.rope_base = rope_base;

        // Each head owns Q, K, V down to head_dim, and an output projection
        // back up to feature_dim. The summed outputs equal a concat plus one
        // big output projection, so no concat operation is needed.
        this.heads = [];
        for (let i = 0; i < num_heads; ++i) {
            const learnedQ = new Tensor(feature_dim, this.head_dim);
            learnedQ.initToSmallRandom();

            const learnedK = new Tensor(feature_dim, this.head_dim);
            learnedK.initToSmallRandom();

            const learnedV = new Tensor(feature_dim, this.head_dim);
            learnedV.initToSmallRandom();

            const learnedO = new Tensor(this.head_dim, feature_dim);
            learnedO.initToSmallRandom();

            this.heads.push({ learnedQ, learnedK, learnedV, learnedO });
        }
    }

    forward(rms_norm_matrix, cache = null, layerIndex = 0) { // Shape: new_tokens, feature_dim
        const scale = 1.0 / Math.sqrt(this.head_dim);
        let output = null;

        for (let i = 0; i < this.num_heads; ++i) {
            const head = this.heads[i];

            let Q = rms_norm_matrix.matmul(head.learnedQ); // Shape: new_tokens, head_dim
            let K = rms_norm_matrix.matmul(head.learnedK); // Shape: new_tokens, head_dim
            let V = rms_norm_matrix.matmul(head.learnedV); // Shape: new_tokens, head_dim

            // The new tokens sit after whatever this head has already cached.
            const pastK = cache === null ? null : cache.keys[layerIndex][i];
            const pastV = cache === null ? null : cache.values[layerIndex][i];
            const positionOffset = pastK === null ? 0 : pastK.rows;

            // Rotate by true position. Cached keys keep the rotation they were
            // stored with; the new keys are rotated from the offset onward.
            Q = Q.ropeRotated(this.rope_base, positionOffset);
            K = K.ropeRotated(this.rope_base, positionOffset);

            // Grow the cache with the new keys and values, then attend over the
            // full history. Without a cache, K and V are just the new tokens.
            if (cache !== null) {
                K = pastK === null ? K : pastK.appendRow(K);
                V = pastV === null ? V : pastV.appendRow(V);
                cache.keys[layerIndex][i] = K;
                cache.values[layerIndex][i] = V;
            }

            let scores = Q.matmul(K.transposed()); // Shape: new_tokens, total_tokens
            scores = scores.scale(scale);

            // The prompt's first pass scores a square block and needs the mask.
            // Every later token sees only its past, so no mask is needed.
            if (positionOffset === 0) {
                scores = scores.causalMasked();
            }

            const probabilities = scores.softMaxedRows();
            const mixed = probabilities.matmul(V);         // Shape: new_tokens, head_dim

            // Project this head up to feature_dim and accumulate.
            const projected = mixed.matmul(head.learnedO); // Shape: new_tokens, feature_dim
            output = output === null ? projected : output.add(projected);
        }

        return output; // Shape: new_tokens, feature_dim
    }

    parameters() {
        const params = [];
        for (let i = 0; i < this.heads.length; ++i) {
            const head = this.heads[i];
            const matrices = [head.learnedQ, head.learnedK, head.learnedV, head.learnedO];
            for (let j = 0; j < matrices.length; ++j) {
                const matParams = matrices[j].parameters();
                for (let k = 0; k < matParams.length; ++k) {
                    params.push(matParams[k]);
                }
            }
        }
        return params;
    }
}

export class MLP {
    constructor(feature_dim_size, hidden_dim_size) {
        this.learnedUp = new Tensor(feature_dim_size, hidden_dim_size);
        this.learnedUp.initToSmallRandom();

        this.learnedDown = new Tensor(hidden_dim_size, feature_dim_size);
        this.learnedDown.initToSmallRandom();

        // Per-neuron bias on the hidden layer, added to each pre-activation
        // before GELU. Starts at zero, so the layer begins as it did without it.
        this.bias = new Tensor(1, hidden_dim_size);
        this.bias.initToZeroes();
    }

    forward(x) { // Shape: sequence_length, feature_dim
        const up = x.matmul(this.learnedUp); // Shape: sequence_length, hidden_dim

        // Shift each pre-activation by its neuron's bias, broadcast across
        // every row. Fused into one node so the shared bias gradient sums
        // down each column in a single pass.
        const bias = this.bias;
        const rows = up.rows;
        const cols = up.columns;
        const preActivation = new Tensor(rows, cols);
        for (let row = 0; row < rows; ++row) {
            const base = row * cols;
            for (let col = 0; col < cols; ++col) {
                preActivation.data[base + col] = up.data[base + col] + bias.data[col];
            }
        }

        if (Tensor.gradEnabled) {
            preActivation._inputs = [up, bias];
            preActivation._backward = () => {
                for (let row = 0; row < rows; ++row) {
                    const base = row * cols;
                    for (let col = 0; col < cols; ++col) {
                        const g = preActivation.grad[base + col];
                        up.grad[base + col] += g;
                        // bias is shared by every row, so its gradient sums down the column.
                        bias.grad[col] += g;
                    }
                }
            };
        }

        // Bend with GELU, then project back down.
        return preActivation.gelu().matmul(this.learnedDown); // Shape: sequence_length, feature_dim
    }

    parameters() {
        const params = [];
        const tensors = [this.learnedUp, this.learnedDown, this.bias];
        for (let i = 0; i < tensors.length; ++i) {
            const sub = tensors[i].parameters();
            for (let j = 0; j < sub.length; ++j) {
                params.push(sub[j]);
            }
        }
        return params;
    }
}

export class TransformerBlock {
    constructor(feature_dim, num_heads, rope_base) {
        // MLP usually expands to 4x the feature dimension.
        const hidden_dim = feature_dim * 4;

        this.attentionNorm = new RMSNorm(feature_dim);
        this.attention = new MultiHeadAttention(feature_dim, num_heads, rope_base);

        this.mlpNorm = new RMSNorm(feature_dim);
        this.mlp = new MLP(feature_dim, hidden_dim);
    }

    forward(x, cache = null, layerIndex = 0) { // cache and layerIndex are new
        // Attention sub-layer, with residual. The cache and layer index pass
        // straight through to attention; the MLP and norms never mix tokens,
        // so they have nothing to cache.
        const attentionInput = this.attentionNorm.forward(x);
        const attentionDelta = this.attention.forward(attentionInput, cache, layerIndex);
        const afterAttention = x.add(attentionDelta);

        // MLP sub-layer, with residual
        const mlpInput = this.mlpNorm.forward(afterAttention);
        const mlpDelta = this.mlp.forward(mlpInput);
        const afterMLP = afterAttention.add(mlpDelta);

        return afterMLP;
    }

    parameters() {
        const params = [];
        const components = [this.attentionNorm, this.attention, this.mlpNorm, this.mlp];
        for (let i = 0; i < components.length; ++i) {
            const sub = components[i].parameters();
            for (let j = 0; j < sub.length; ++j) {
                params.push(sub[j]);
            }
        }
        return params;
    }
}

export class MiniGPT {
    constructor(vocabSize, featureDim, numHeads, ropeBase, numBlocks) {
        this.vocabSize = vocabSize;
        this.featureDim = featureDim;
        this.numHeads = numHeads;
        this.ropeBase = ropeBase; // replaces maxContextLength
        this.numBlocks = numBlocks;

        // Token embedding table (position now enters through rotation in attention).
        this.tokenEmbeddings = new Tensor(vocabSize, featureDim);
        this.tokenEmbeddings.initToSmallRandom();

        // Stack of transformer blocks.
        this.blocks = [];
        for (let i = 0; i < numBlocks; ++i) {
            this.blocks.push(new TransformerBlock(featureDim, numHeads, ropeBase)); // heads and base handed down
        }

        // Final norm before the unembedding.
        this.finalNorm = new RMSNorm(featureDim);
    }

    embed(ids) {
        // Token embeddings only; position now enters through rotation in attention
        return this.tokenEmbeddings.gatherRows(ids);
    }

    forward(tokenIdArray, cache = null) { // cache is new
        // No maxContextLength check: there is no position table to index past.
        // A sequence beyond the trained length just runs slower and predicts worse.
        // With a cache, tokenIdArray holds only the new tokens; the block index
        // threads through so each layer reads and writes its own cached slot.
        let x = this.embed(tokenIdArray); // Shape: new_tokens, feature_dim
        for (let i = 0; i < this.blocks.length; ++i) {
            x = this.blocks[i].forward(x, cache, i); // Shape preserved.
        }
        x = this.finalNorm.forward(x); // Shape: new_tokens, feature_dim

        // Tied unembedding: (sequence_length, feature_dim) * (feature_dim, vocabSize)
        const logits = x.matmul(this.tokenEmbeddings.transposed()); // Shape: sequence_length, vocabSize
        return logits;
    }

    // Predict the next token. Temperature reshapes the distribution, top-k and
    // top-p rule out the unlikely tail, and an optional KV cache skips rerunning
    // tokens that were processed on an earlier call.
    predictNextToken(tokenIdArray, temperature = 1.0, topK = 50, topP = 0.9, cache = null) {
        // With a cache, the stored tokens do not run again. Only the tokens
        // past the cached length are new.
        const newTokens = cache === null
            ? tokenIdArray
            : tokenIdArray.slice(cache.length);

        const logits = this.forward(newTokens, cache); // was forward(tokenIdArray)

        // The last row holds the scores for the token after the sequence,
        // whether that is the last row of a full prompt or the only row of a
        // single cached step.
        const lastRow = logits.rows - 1;

        // Temperature zero means greedy decoding: the single highest logit.
        // Softmax is monotonic, so the argmax of the logits is the argmax of
        // the probabilities, and the softmax can be skipped here.
        if (temperature === 0) {
            let bestId = 0;
            let bestScore = logits.get(lastRow, 0);
            for (let col = 1; col < this.vocabSize; ++col) {
                const score = logits.get(lastRow, col);
                if (score > bestScore) {
                    bestScore = score;
                    bestId = col;
                }
            }
            return bestId;
        }

        // Reshape by temperature and turn the last row into probabilities.
        const scaled = logits.scale(1 / temperature);
        const probabilities = scaled.softMaxedRows();

        // Rank the tokens from most to least likely so the worst can be cut.
        const ranked = [];
        for (let col = 0; col < this.vocabSize; ++col) {
            ranked.push({ id: col, probability: probabilities.get(lastRow, col) });
        }
        ranked.sort((a, b) => b.probability - a.probability);

        // top-k: keep only the k most likely tokens.
        const capped = ranked.slice(0, topK);

        // top-p: from the top, keep tokens until their combined probability
        // crosses topP, then stop. Always keeps at least the single top token.
        const kept = [];
        let keptTotal = 0.0;
        for (let i = 0; i < capped.length; ++i) {
            kept.push(capped[i]);
            keptTotal += capped[i].probability;
            if (keptTotal >= topP) {
                break;
            }
        }

        // The survivors cover only part of the number line now, up to keptTotal.
        // Draw a random number up to that total, then walk them as before.
        const draw = Math.random() * keptTotal;
        let cumulative = 0.0;
        for (let i = 0; i < kept.length; ++i) {
            cumulative += kept[i].probability;
            if (draw < cumulative) {
                return kept[i].id;
            }
        }

        return kept[kept.length - 1].id; // guard against floating-point drift
    }

    parameters() {
        const params = [];

        // Token embedding table (also used tied as the unembedding).
        const tokenParams = this.tokenEmbeddings.parameters();
        for (let i = 0; i < tokenParams.length; ++i) {
            params.push(tokenParams[i]);
        }

        // Per-block parameters.
        for (let i = 0; i < this.blocks.length; ++i) {
            const blockParams = this.blocks[i].parameters();
            for (let j = 0; j < blockParams.length; ++j) {
                params.push(blockParams[j]);
            }
        }

        // Final norm now carries a learned scale (gamma), gathered like the rest.
        const normParams = this.finalNorm.parameters();
        for (let i = 0; i < normParams.length; ++i) {
            params.push(normParams[i]);
        }

        return params;
    }

    serializeToArrayBuffer() {
        const params = this.parameters();

        // Each parameter is a tensor with many values, so size the buffer to the
        // total number of values across all tensors.
        let total = 0;
        for (const t of params) {
            total += t.data.length;
        }

        const view = new Float32Array(total);
        let offset = 0;
        for (const t of params) {
            view.set(t.data, offset);
            offset += t.data.length;
        }
        return view.buffer;
    }

    deserializeFromArrayBuffer(buffer) {
        const params = this.parameters();
        const view = new Float32Array(buffer);
        let offset = 0;
        for (const t of params) {
            t.data.set(view.subarray(offset, offset + t.data.length));
            offset += t.data.length;
        }
    }
}

export class AdamWTrainer {
    constructor(model, {
        maxLearningRate = 3e-4,
        minLearningRate = 3e-5,
        warmupSteps = 100,
        totalSteps = 10000,
        beta1 = 0.9,
        beta2 = 0.999,
        epsilon = 1e-8,
        weightDecay = 0.01,
        gradientClip = 1.0,
    } = {}) {
        this.model = model;

        this.maxLearningRate = maxLearningRate;
        this.minLearningRate = minLearningRate;
        this.warmupSteps = warmupSteps;
        this.totalSteps = totalSteps;

        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.weightDecay = weightDecay;
        this.gradientClip = gradientClip;

        // Grab the parameter list once. The tensors inside are stable references,
        // so the moment buffers below stay matched to them for the whole run.
        this.params = model.parameters();

        // Adam keeps two running averages per parameter, each the same shape as
        // the parameter itself. A fresh Float32Array is already all zeros.
        this.firstMoments = [];
        this.secondMoments = [];
        for (let i = 0; i < this.params.length; ++i) {
            const length = this.params[i].data.length;
            this.firstMoments.push(new Float32Array(length));  // m
            this.secondMoments.push(new Float32Array(length)); // v
        }

        // Updates applied so far. Drives both bias correction and the schedule.
        this.step = 0;

        // The learning rate actually used on the last step, exposed for logging.
        this.lastLearningRate = 0;
    }

    // Fused mean cross-entropy. The forward pass uses the log-sum-exp form for
    // each row of logits, which avoids exponentiating large values directly:
    // for a row with correct token t, the loss is log(sum_j e^{x_j}) - x_t.
    // The softmax probabilities are computed along the way and cached, because
    // the backward rule reuses them.
    crossEntropyLoss(logits, targetIds, targetMask = null) { // mask: new argument
        const rows = logits.rows;
        const cols = logits.columns;
        const probs = new Float32Array(rows * cols); // cached for the backward
        let total = 0.0;
        let count = 0; // mask: unmasked rows — the divisor for the mean

        for (let row = 0; row < rows; ++row) {
            // mask: a context-only row contributes no loss and no gradient.
            // Skip it entirely; its probs are never read.
            if (targetMask !== null && !targetMask[row]) {
                continue;
            }

            const base = row * cols;

            let rowMax = logits.data[base];
            for (let col = 1; col < cols; ++col) {
                const v = logits.data[base + col];
                if (v > rowMax) rowMax = v;
            }

            let sum = 0.0;
            for (let col = 0; col < cols; ++col) {
                const e = Math.exp(logits.data[base + col] - rowMax);
                probs[base + col] = e;
                sum += e;
            }
            for (let col = 0; col < cols; ++col) {
                probs[base + col] /= sum;
            }

            const logSumExp = rowMax + Math.log(sum);
            total += logSumExp - logits.data[base + targetIds[row]];
            count += 1; // mask: this row counted toward the mean
        }

        if (count === 0) {
            throw new Error("No unmasked targets were provided");
        }

        const loss = new Tensor(1, 1);
        loss.data[0] = total / count; // mask: divide by unmasked count, not rows

        if (Tensor.gradEnabled) {
            loss._inputs = [logits];
            loss._backward = () => {
                const seed = loss.grad[0] / count; // mask: count, not rows
                for (let row = 0; row < rows; ++row) {
                    // mask: skipped rows got no probs, so they get no gradient.
                    if (targetMask !== null && !targetMask[row]) {
                        continue;
                    }
                    const base = row * cols;
                    for (let col = 0; col < cols; ++col) {
                        logits.grad[base + col] += seed * probs[base + col];
                    }
                    logits.grad[base + targetIds[row]] -= seed;
                }
            };
        }
        return loss;
    }

    learningRate() {
        const step = this.step;

        // Warmup: ramp linearly from zero to the peak over the first steps.
        if (step < this.warmupSteps) {
            return this.maxLearningRate * (step / this.warmupSteps);
        }

        // Past the planned end: hold at the floor.
        if (step >= this.totalSteps) {
            return this.minLearningRate;
        }

        // Cosine decay: ease from the peak down to the floor.
        const progress = (step - this.warmupSteps) / (this.totalSteps - this.warmupSteps);
        const cosine = 0.5 * (1 + Math.cos(Math.PI * progress));
        return this.minLearningRate + (this.maxLearningRate - this.minLearningRate) * cosine;
    }

    clipGradients() {
        // Global L2 norm across every gradient in the model.
        let sumSquares = 0.0;
        for (let i = 0; i < this.params.length; ++i) {
            const grad = this.params[i].grad;
            for (let j = 0; j < grad.length; ++j) {
                sumSquares += grad[j] * grad[j];
            }
        }
        const norm = Math.sqrt(sumSquares);

        // Only scale down when the norm is over the threshold. A smaller
        // update is left alone.
        if (norm > this.gradientClip) {
            const scale = this.gradientClip / norm;
            for (let i = 0; i < this.params.length; ++i) {
                const grad = this.params[i].grad;
                for (let j = 0; j < grad.length; ++j) {
                    grad[j] *= scale;
                }
            }
        }

        return norm; // useful for logging
    }

    train(tokenIds, tokenMask = null) {
        const inputIds  = tokenIds.slice(0, tokenIds.length - 1);
        const targetIds = tokenIds.slice(1);
        // The mask lines up with tokenIds, so shift it like the targets. The
        // first token is never a target, so its mask is dropped.
        const targetMask = tokenMask === null ? null : tokenMask.slice(1);

        // Clear last step's gradients.
        for (let i = 0; i < this.params.length; ++i) {
            this.params[i].zeroGrad();
        }

        // Forward, loss, backward — same as the SGD trainer.
        const logits = this.model.forward(inputIds);
        const loss = this.crossEntropyLoss(logits, targetIds, targetMask);
        loss.backward();

        // Rein in the occasional oversized gradient before it moves anything.
        this.clipGradients();

        // Advance the step, then read this step's scheduled learning rate.
        this.step += 1;
        const learningRate = this.learningRate();
        this.lastLearningRate = learningRate;

        // Cache hyperparameters as locals for the inner loop.
        const beta1 = this.beta1;
        const beta2 = this.beta2;
        const epsilon = this.epsilon;
        const weightDecay = this.weightDecay;

        // Bias-correction denominators for this step.
        const correction1 = 1 - Math.pow(beta1, this.step);
        const correction2 = 1 - Math.pow(beta2, this.step);

        // AdamW update, parameter by parameter.
        for (let i = 0; i < this.params.length; ++i) {
            const param = this.params[i];
            const m = this.firstMoments[i];
            const v = this.secondMoments[i];

            for (let j = 0; j < param.data.length; ++j) {
                const g = param.grad[j];
                const weight = param.data[j];

                // Update the running averages of the gradient and its square.
                m[j] = beta1 * m[j] + (1 - beta1) * g;
                v[j] = beta2 * v[j] + (1 - beta2) * g * g;

                // Undo the startup bias toward zero.
                const mHat = m[j] / correction1;
                const vHat = v[j] / correction2;

                // The Adam step: smoothed direction, per-parameter scale.
                const adamStep = mHat / (Math.sqrt(vHat) + epsilon);

                // Decoupled weight decay: pull the weight toward zero on its
                // own, not routed through the adaptive scaling above.
                const decayStep = weightDecay * weight;

                param.data[j] = weight - learningRate * (adamStep + decayStep);
            }
        }

        return loss.data[0];
    }
}