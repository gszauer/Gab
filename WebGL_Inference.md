# WebGL GPU Inference for Transformer Models

This document describes how to implement transformer (GPT-style) neural network inference using WebGL2 fragment shaders. ebGL provides several advantages for browser-based ML inference:

1. **Universal availability** - Works in all modern browsers without plugins
2. **GPU acceleration** - Leverages the graphics card for parallel computation
3. **No CORS issues** - Unlike WebGPU, WebGL works with local files
4. **Synchronous API** - No async/await complexity for basic operations

WebGL was designed for graphics, not compute. We're "abusing" the rendering pipeline to do math. While not as elegant as proper compute shaders, this approach works today in every browser and can run meaningful models (25M+ parameters) at usable speeds.

We will think of WebGL as a massively parallel compute device where:

- **Weights are stored as textures** (read-only)
- **Activations flow through textures** (read/write via framebuffers)
- **Fragment shaders perform matrix operations** (one output pixel = one output value)
- **The full-screen quad pattern** drives computation

### Model Configuration

This inference engine is designed to run [Gab](https://giftofgab.chat/). The following model config is hard coded.

```
Vocab size:        5,256 (padded to 8,192)
Embedding dim:     512
Attention heads:   8 (head dim = 64)
Transformer blocks: 6
Max sequence length: 2,048
MLP hidden dim:    2,048 (4x expansion)
Total parameters:  ~25M
```

## WebGL2 Requirements

Rather than [floats to rgba](https://aras-p.info/blog/2009/07/30/encoding-floats-to-rgba-the-final/), we're just going to use floating point textures, which are available in WebGL as an extension.

```javascript
const gl = canvas.getContext('webgl2');

// Required: allows R32F textures as render targets
const ext = gl.getExtension('EXT_color_buffer_float');
if (!ext) throw new Error('EXT_color_buffer_float not supported');
```

For inference without visible rendering, create an offscreen Canvas:

```javascript
let canvas;
if (typeof OffscreenCanvas !== 'undefined') {
    canvas = new OffscreenCanvas(1, 1);
} else {
    canvas = document.createElement('canvas');
}
const gl = canvas.getContext('webgl2');
```

## Texture Storage Strategy

### Format: R32F (Single-Channel Float32)

We use `R32F` format (one 32-bit float per texel) rather than `RGBA32F` because:

- Simpler indexing (no channel packing/unpacking)
- More intuitive mapping to weight matrices
- Sufficient precision for inference

```javascript
gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.R32F,        // Internal format
    width,
    height,
    0,
    gl.RED,         // Format
    gl.FLOAT,       // Type
    data            // Float32Array
);
```

### Texture Layout

All textures use power-of-two dimensions for compatibility. Gab was mostly designed to use power of two hyper paramaters, only the vocab needs to be padded:

| Texture | Dimensions | Purpose |
|---------|------------|---------|
| Token Embeddings | 512 × 8192 | vocab (padded) × embedding |
| Position Embeddings | 512 × 2048 | seq × embedding |
| QKV Weights | 512 × 512 | embedding × embedding |
| MLP1 Weights | 512 × 2048 | embedding × hidden |
| MLP2 Weights | 2048 × 512 | hidden × embedding |
| Attention Scores | 2048 × 2048 | seq × seq |
| Hidden States | 512 × 2048 | embedding × seq |

Data is stored row-major in Float32Arrays:

```
data[row * width + col] -> texelFetch(tex, ivec2(col, row), 0).r
```

For a weight matrix W[output][input]:
```javascript
// Store as: weightData[outputIdx * inputDim + inputIdx]
for (let o = 0; o < outputDim; o++) {
    for (let i = 0; i < inputDim; i++) {
        weightData[o * inputDim + i] = W[o][i];
    }
}
// Access as: texelFetch(weights, ivec2(inputIdx, outputIdx), 0).r
```

### Total GPU Memory

| Category | Size |
|----------|------|
| Weight textures (~50) | ~108 MB |
| Working textures (~12) | ~10 MB |
| **Total** | **~118 MB** |

## The Shader Pipeline

### Vertex Shader (Shared)

All operations use the same vertex shader that renders a full-screen quad:

```glsl
#version 300 es
in vec2 a_position;
out vec2 v_texCoord;

void main() {
    v_texCoord = a_position * 0.5 + 0.5;
    gl_Position = vec4(a_position, 0.0, 1.0);
}
```

The quad vertices are: `[-1,-1], [1,-1], [-1,1], [-1,1], [1,-1], [1,1]`

### Fragment Shader Pattern

Each fragment shader computes one output value based on `gl_FragCoord`:

```glsl
#version 300 es
precision highp float;
out float outValue;

uniform sampler2D u_input;
uniform int u_width;
uniform int u_height;

void main() {
    int x = int(gl_FragCoord.x);  // Output column
    int y = int(gl_FragCoord.y);  // Output row

    // Bounds check
    if (x >= u_width || y >= u_height) {
        outValue = 0.0;
        return;
    }

    // Compute output value...
    outValue = /* computation */;
}
```

### Core Shaders

#### 1. Embedding Lookup

```glsl
void main() {
    int d = int(gl_FragCoord.x);    // Embedding dimension
    int pos = int(gl_FragCoord.y);  // Sequence position

    float tokenId = texelFetch(u_tokens, ivec2(pos, 0), 0).r;
    float tokEmb = texelFetch(u_tokenEmb, ivec2(d, int(tokenId)), 0).r;
    float posEmb = texelFetch(u_posEmb, ivec2(d, pos), 0).r;

    outValue = tokEmb + posEmb;
}
```

#### 2. Layer Normalization

```glsl
void main() {
    int d = int(gl_FragCoord.x);
    int pos = int(gl_FragCoord.y);

    // Compute mean across embedding dimension
    float mean = 0.0;
    for (int i = 0; i < u_embDim; i++) {
        mean += texelFetch(u_input, ivec2(i, pos), 0).r;
    }
    mean /= float(u_embDim);

    // Compute variance
    float variance = 0.0;
    for (int i = 0; i < u_embDim; i++) {
        float diff = texelFetch(u_input, ivec2(i, pos), 0).r - mean;
        variance += diff * diff;
    }
    variance /= float(u_embDim);

    // Normalize, scale, shift
    float x = texelFetch(u_input, ivec2(d, pos), 0).r;
    float normalized = (x - mean) / sqrt(variance + 1e-5);
    float gamma = texelFetch(u_gamma, ivec2(d, 0), 0).r;
    float beta = texelFetch(u_beta, ivec2(d, 0), 0).r;

    outValue = gamma * normalized + beta;
}
```

#### 3. Matrix Multiplication

The workhorse operation. Computes `Output = Input @ Weight`:

```glsl
void main() {
    int outCol = int(gl_FragCoord.x);  // Output dimension
    int outRow = int(gl_FragCoord.y);  // Sequence position

    float sum = 0.0;
    for (int i = 0; i < u_inputDim; i++) {
        float inputVal = texelFetch(u_input, ivec2(i, outRow), 0).r;
        float weightVal = texelFetch(u_weight, ivec2(i, outCol), 0).r;
        sum += inputVal * weightVal;
    }

    outValue = sum;
}
```

**Critical**: Weight matrix must be stored transposed as `W[outputIdx][inputIdx]` for this indexing to work.

#### 4. Attention Scores

```glsl
void main() {
    int s = int(gl_FragCoord.x);  // Source position (key)
    int t = int(gl_FragCoord.y);  // Target position (query)

    // Causal mask
    if (s > t) {
        outValue = -1e9;
        return;
    }

    int headStart = u_headIdx * u_headDim;
    float score = 0.0;

    for (int d = 0; d < u_headDim; d++) {
        float q = texelFetch(u_Q, ivec2(headStart + d, t), 0).r;
        float k = texelFetch(u_K, ivec2(headStart + d, s), 0).r;
        score += q * k;
    }

    outValue = score * u_scale;  // scale = 1/sqrt(headDim)
}
```

#### 5. Softmax (Row-wise)

```glsl
void main() {
    int x = int(gl_FragCoord.x);
    int y = int(gl_FragCoord.y);

    // Find max for numerical stability
    float maxVal = -1e9;
    for (int i = 0; i <= y; i++) {  // Causal: only up to position y
        maxVal = max(maxVal, texelFetch(u_input, ivec2(i, y), 0).r);
    }

    // Compute exp sum
    float sumExp = 0.0;
    for (int i = 0; i <= y; i++) {
        sumExp += exp(texelFetch(u_input, ivec2(i, y), 0).r - maxVal);
    }

    // Output normalized probability
    if (x > y) {
        outValue = 0.0;  // Causal mask
    } else {
        float val = texelFetch(u_input, ivec2(x, y), 0).r;
        outValue = exp(val - maxVal) / sumExp;
    }
}
```

#### 6. GELU Activation

```glsl
// Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
float c = 0.7978845608;  // sqrt(2/pi)
outValue = 0.5 * x * (1.0 + tanh(c * (x + 0.044715 * x * x * x)));
```
## Implementation Patterns

### 1. Ping-Pong Buffers

When a computation needs to both read from and write to the same logical buffer, use two physical textures and alternate:

```javascript
// Hidden states use ping-pong
let currentHidden = 'hiddenA';
let otherHidden = 'hiddenB';

// After each operation that modifies hidden state:
[currentHidden, otherHidden] = [otherHidden, currentHidden];
```

We also use ping-pong for attention output accumulation:

```javascript
let attnRead = 'attnOutA';
let attnWrite = 'attnOutB';

for (let head = 0; head < numHeads; head++) {
    // Compute head output...

    // Copy head to accumulated output (ping-pong)
    runProgram(copyHeadShader, framebuffers[attnWrite], {
        u_headOutput: textures.headOut,
        u_accumulator: textures[attnRead],  // Read from A
        // ...
    });

    // Swap for next iteration
    [attnRead, attnWrite] = [attnWrite, attnRead];
}
// After loop: attnRead contains final result
```

### 2. Running a Shader Program

```javascript
function runProgram(program, outputFramebuffer, outputWidth, outputHeight, uniforms) {
    gl.useProgram(program);
    gl.bindFramebuffer(gl.FRAMEBUFFER, outputFramebuffer);
    gl.viewport(0, 0, outputWidth, outputHeight);

    // Set up quad vertex attribute
    const posLoc = gl.getAttribLocation(program, 'a_position');
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

    // Set uniforms
    let textureUnit = 0;
    for (const [name, value] of Object.entries(uniforms)) {
        const loc = gl.getUniformLocation(program, name);
        if (loc === null) continue;

        if (value instanceof WebGLTexture) {
            gl.activeTexture(gl.TEXTURE0 + textureUnit);
            gl.bindTexture(gl.TEXTURE_2D, value);
            gl.uniform1i(loc, textureUnit);
            textureUnit++;
        } else if (Number.isInteger(value)) {
            gl.uniform1i(loc, value);
        } else {
            gl.uniform1f(loc, value);
        }
    }

    // Draw full-screen quad (6 vertices = 2 triangles)
    gl.drawArrays(gl.TRIANGLES, 0, 6);
}
```

### 3. Weight Matrix Transposition

The matmul shader uses this access pattern:

```glsl
weightVal = texelFetch(u_weight, ivec2(inputIdx, outputIdx), 0).r;
```

This expects weights stored as `W[outputIdx][inputIdx]`. If your model stores weights as `W[inputIdx][outputIdx]`, transpose during loading:

```javascript
function transpose(data, rows, cols) {
    const result = new Float32Array(rows * cols);
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            result[c * rows + r] = data[r * cols + c];
        }
    }
    return result;
}

// Attention weights need transposition
const wq = transpose(wqRaw, embeddingDim, embeddingDim);
```

### 4. Reading Results Back to CPU

```javascript
// Bind the framebuffer containing results
gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffers.probs);

// Read pixels
const probsData = new Float32Array(8192);
gl.readPixels(0, 0, 8192, 1, gl.RED, gl.FLOAT, probsData);

// probsData now contains the output probabilities
```

## Draw Call Analysis

Each token generation requires the following draw calls:

### Per Forward Pass

| Operation | Draw Calls | Notes |
|-----------|------------|-------|
| Embedding lookup | 1 | |
| **Per block (×6):** | | |
| - LayerNorm1 | 1 | |
| - Q projection | 1 | |
| - K projection | 1 | |
| - V projection | 1 | |
| - Attention scores (×8 heads) | 8 | One per head |
| - Softmax (×8 heads) | 8 | One per head |
| - Attention output (×8 heads) | 8 | One per head |
| - Copy head (×8 heads) | 8 | One per head |
| - Output projection | 1 | |
| - LayerNorm2 | 1 | |
| - MLP dense1 + GELU | 1 | |
| - MLP dense2 + residual | 1 | |
| Final LayerNorm | 1 | |
| Output projection | 1 | |
| Final softmax | 1 | |

### Total Draw Calls

```
Per block:    1 + 3 + (8×4) + 1 + 1 + 1 + 1 = 40 draw calls
All blocks:   40 × 6 = 240 draw calls
Fixed ops:    1 (embed) + 1 (final LN) + 1 (output proj) + 1 (softmax) = 4
──────────────────────────────────────────────────────────────────────────
TOTAL:        244 draw calls per token
```

The attention computation dominates with 32 calls per block × 6 blocks = 192 calls, or 79% of total.

## Lessons Learned

### 1. The Feedback Loop Bug

**Problem**: Reading from and writing to the same texture produces undefined behavior.

**Symptom**: Model generates complete gibberish despite correct weight loading.

**Solution**: Use ping-pong buffers for any texture that needs to be both read and written in the same logical operation.

**Detection**: Browser console shows warnings (but only the first 32):
```
WebGL warning: Texture level 0 would be read by TEXTURE_2D unit 1,
but written by framebuffer attachment COLOR_ATTACHMENT0
```

### 2. Coordinate System

`texelFetch(texture, ivec2(x, y), 0)` uses:
- **x** = column (horizontal)
- **y** = row (vertical)

Data uploaded via `texImage2D` maps: `data[row * width + col]` -> `texelFetch(tex, ivec2(col, row), 0)`

### 3. Integer Uniforms

WebGL2 distinguishes between `uniform1i` and `uniform1f`. Using the wrong one silently fails:

```javascript
if (Number.isInteger(value)) {
    gl.uniform1i(loc, value);
} else {
    gl.uniform1f(loc, value);
}
```

### 4. Texture Filtering

For compute workloads, always use NEAREST filtering:

```javascript
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
```

LINEAR filtering interpolates between texels, which corrupts your data.

### 5. Debugging Texture Uploads

Verify data roundtrips correctly:

```javascript
function verifyTextureUpload(gl, texture, originalData, width, height, name) {
    const fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);

    const readBack = new Float32Array(width * height);
    gl.readPixels(0, 0, width, height, gl.RED, gl.FLOAT, readBack);

    gl.deleteFramebuffer(fb);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    let maxDiff = 0;
    for (let i = 0; i < originalData.length; i++) {
        maxDiff = Math.max(maxDiff, Math.abs(originalData[i] - readBack[i]));
    }

    console.log(`[${name}] Max diff: ${maxDiff}`);
    return maxDiff < 1e-5;
}
```