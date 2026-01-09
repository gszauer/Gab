# Let's build a Chat GPT - Part 1 / 4: Learning

Neural networks are the engines driving much of modern artificial intelligence. They power recommendation systems, image generators, and language models. While libraries like TensorFlow and PyTorch make it easy to build complex networks, understanding the underlying mechanics is important for genuine insight. 

This tutorial walks you through building and training a simple neural network from the ground up, using plain JavaScript. We'll implement the neural network step by step, explaining the *why* behind the *how* without getting lost in the math of it all.

# A Trainable Neuron

At the heart of every neural network lies the neuron, a simple computational unit. A neuron takes multiple inputs (for now an array of floating point numbers), performs a weighted sum of those inputs, and adds a bias term to produce output. It's a surprisingly powerful primitive.


```
				┌─────────────────┐                 
				│ Neuron          │                 
				├──────────┬──────┤                 
				│ Weight0  │      │                 
	 Input0  ───┼─────────►│ Bias │─────► Output      
				│ Weight1  │      │                      
	 Input1  ───┼─────────►│      │                 
				└──────────┴──────┘   
```

> A neuron with two inputs, `Input0` and `Input1`. This neuron holds two weights, `Weight0` and `Weight1`, and a bias term, `Bias`. The output is calculated as: ```Output = (Input0 * Weight0 + Input1 * Weight1) + Bias```.

Let's translate this into code. We'll start with a simple neuron that takes two inputs:

```javascript
// Single Neuron
let weights = [
	Math.random() * 2 - 1, 
	Math.random() * 2 - 1
];
let bias    = Math.random() * 2 - 1;

// Evaluating a single neuron
function forward(inputs) {
  return inputs[0] * weights[0] + inputs[1] * weights[1] + bias;
}
```

When a neuron is initialized, it is filled with random values in the range of -1 to 1. Each weight / bias is a paramater. But why random initialization? If all paramaters started at zero, every neuron would learn the same thing. Randomness breaks symmetry and lets each neuron specialize. 

Having paramaters with random values produces random output. To generate more predicatble output, we need to find optimal values for each paramater. We can train the network to  adjust those paramaters automatically so the network produces better output. How Do We Train? With  **Gradient Descent**. Gradient Descent is guided trial and error, defined by this loop:

1) **Predict**: Run the neuron on an input -> get an output.
2) **Measure error**: Figure out how far off is the prediction from the target.
3) **Compute gradients**: Figure out which way to nudge each parameter (weights and bias) to reduce that error.
4) **Update paramaters**: Take a tiny step in that direction.

Let's say we want our neuron to learn this mapping: 

```
Input: [2, 1]  ->  Target Output: 5
```
To teach the neuron, we will run 100 training steps: 

```javascript
const trainingInput = [2, 1];
const targetOutput = 5;

for (let i = 0; i < 100; i++) {
  // 1) Predict
  const prediction = forward(trainingInput);

  // 2) Compute error
  const error = prediction - targetOutput;

  // 3) Compute gradients — "which way should each knob turn?"
  const weightGradients = [
    error * trainingInput[0], 
    error * trainingInput[1] 
  ];
  const biasGradient = error;

  // 4) Update parameters (move *against* the error)
  const learningRate = 0.01;
  weights[0] -= learningRate * weightGradients[0];
  weights[1] -= learningRate * weightGradients[1];
  bias       -= learningRate * biasGradient;

  // Log progress occasionally
  if (i % 20 === 0) {
    console.log(
      `Iter ${i}: pred=${prediction.toFixed(2)}, err=${error.toFixed(2)}, ` +
      `w=[${weights.map(w => w.toFixed(2)).join(", ")}], b=${bias.toFixed(2)}`
    );
  }
}
```

Run this, and you'll see the prediction inch closer to 5 with each step. The magic isn't in complexity, it's in repetition and feedback.


# A trainable neural network

A single neuron is limited, it can only learn linear relationships. But stack them into layers, add non-linear activation functions, and suddenly you can learn anything! 

## Neuron

Let's formalize our neuron into a class. The neuron class has some number of weights and a bias. These are the neurons paramaters. We'll implement the forward pass as well: 

```javascript
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
}
```

## Dense Layers

In a Neural Network, Neurons are organized into layers. The most common type is the dense layer (also called a fully connected layer). Every neuron in a dense layer is connected to all neurons of the previous layer. If we have a layer with ```n``` neurons, followed by a layer with ```m``` neurons, then the following hold true for the second layer:
1) Each neuron holds ```n``` weights (one per input) + 1 bias.
2) The layer as a whole learns ```m``` different ways to interpret the input.
3) The output is an ```m```-dimensional vector. It is a richer, transformed representation that will be passed to the next layer.

This is where the network starts building hierarchical understanding: early layers might detect simple patterns, while deeper layers combine those into complex concepts. 

In code, a dense layer is implemented as an array of neurons:     

```javascript
class DenseLayer {
    neurons = null;

    constructor(numberOfInputs, numberOfOutputs) {
        this.neurons = new Array(numberOfOutputs);
        for (let i = 0; i < numberOfOutputs; i++) {
            this.neurons[i] = new Neuron(numberOfInputs);
        }
    }

    forward(inputs) {
        const result = new Array(inputs.length);
        for (let i = 0, size = inputs.length; i < size; ++i) {
            result[i] = this.neurons[i].forward(inputs);
        }
        return result;
    }
}
```

But wait, there's a catch... **Dense layers are linear**.
No matter how many dense layers you stack, the whole network remains a fancy linear function. 

## Activation Layers

Activation functions add non-linearity to neural networks. Without them, your network is just a very expensive straight line. With them? It can learn curves.

Instead of adding an activation function to each neuron, we're going to implement activation layers. The activation layer will apply it's activation function to each input value. Here's our activation layer: 

```javascript
class ActivationLayer {
    type = "relu";

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

    forward(inputs) {
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
}
```

Why are we implementing these three activation functions specifically?   

* **ReLU** (max(0, x)) - is fast, simple, and works great in hidden layers.  
* **tanh** squashes values to [-1, 1] - smooth and centered, perfect for hidden units.  
* **Sigmoid** squashes to [0, 1] - ideal for binary outputs (like XOR's 0 or 1).

By stacking dense layers followed by activation layers we can start to create simple neural networks.

## Evaluating an XOR function

Let's build a tiny network that learns the XOR function. Given two inputs, this function returns true only if one input is true and the other one is false. Consider this truth table:

| Input A | Input B | A XOR B | Description |
|---------|---------|---------|-------------|
| 0 | 0 | 0 | Both inputs false -> Output false |
| 0 | 1 | 1 | Different inputs -> Output true |
| 1 | 0 | 1 | Different inputs -> Output true |
| 1 | 1 | 0 | Both inputs true -> Output false |

Solving this XOR function is impossible for a single neuron, but easy for a network.To solve this problem, we will design the following network:
* Input: 2 numbers (A, B)
* Hidden layer: 4 neurons -> tanh activation
* Output layer: 1 neuron -> sigmoid activation
     
Let's translate that to codde:

```javascript
function xor_ai(left, right) {
    // Create network
    const layer1 = new DenseLayer(2, 4); 
    const activation1 = new ActivationLayer("tanh");
    const layer2 = new DenseLayer(4, 1);  
    const activation2 = new ActivationLayer("sigmoid");

    // Load Weights (we will learn how to generate these later)
    layer1.neurons[0].weights[0] = -2.483405330352288; 
    layer1.neurons[0].weights[1] = 3.746893395232311;
    layer1.neurons[0].bias = 0.8972583832821088; 
    layer1.neurons[1].weights[0] = -3.653234475692758; 
    layer1.neurons[1].weights[1] = 0.9955207401046027;
    layer1.neurons[1].bias = 0.5799612103320189; 
    layer1.neurons[2].weights[0] = -1.986455463777911; 
    layer1.neurons[2].weights[1] = -2.140729883658909;
    layer1.neurons[2].bias = 0.1773771186808191; 
    layer1.neurons[3].weights[0] = -3.315691022651759;
    layer1.neurons[3].weights[1] = -3.478943831512278; 
    layer1.neurons[3].bias = 1.0466264947557882;
    layer2.neurons[0].weights[0] = -5.891722647970969;
    layer2.neurons[0].weights[1] = 5.841764877092181;
    layer2.neurons[0].weights[2] = -2.167853628782214; 
    layer2.neurons[0].weights[3] = -4.926738682524884;
    layer2.neurons[0].bias = -0.9654887789785622;

    // Run network
    let output = layer1.forward([left, right]);
    output = activation1.forward(output);
    output = layer2.forward(output);
    output = activation2.forward(output);

    console.log(`Input: [${left}, ${right}] -> Output: ${output[0].toFixed(3)}`);

    return output[0];
}
```

The ```xor_ai``` function provides pre-trained paramaters. But without training, if the paramaters where random, the output would also be random noise. To make this network smart, we need to train it. 

## The Training loop

We've built our network and it can make predictions. Terrible ones at first, but predictions nonetheless. How do we teach it? Trough  backpropagation, the algorithm that makes deep learning possible.

> Backpropagation is the chain rule from calculus, applied backwards through the network. When you have nested functions like `f(g(h(x)))`, the chain rule tells us: ```df/dx = df/dg x dg/dh x dh/dx```. In neural networks: `x` is the input, each layer is a function transforming its input.

Back propogation figures out how much each paramater is responsible for the error of the output. That gives us a direction and magnitude to adjust the paramater. Each weight only needs to know its local contribution to the error. No weight needs to understand the entire network-just its immediate neighborhood.

### Measuring error

In our network's forward pass, data flows from input through layers to output. But how do we know if the output is any good? We need a way to measure error. Measuring the error of a network is done with a loss function. It's a function that tells us how wrong the network was.

The most common loss function for regression tasks is Mean Squared Error (MSE). The function is simple, take the difference between what you predicted and what you wanted, square that difference (to make all errors positive), and average across all outputs.

```javascript
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
}
```

Why do we square the error? 
1. **Makes all errors positive**: We don't want negative and positive errors canceling out.
2. **Punishes large errors more**: A prediction that's off by 2 gets 4x the penalty of being off by 1. This pushes the network to avoid big mistakes.
3. **It's differentiable**: We need smooth gradients for backpropagation (more on this soon).

We will need the derivitive soon. The derivitive of the loss function is our networks error.

### Back propogation trough a neuron

Now let's implement backpropagation for a single neuron. Remember, a neuron computes `output = sum(weights x inputs) + bias`. During backpropagation, we need to reverse this process: given an error in the output, figure out how to adjust our parameters and what error to pass back to our inputs.

In backpropagation, gradients flow backward while updates happen locally. Each neuron:
1. **Receives** gradient from the layer ahead (how wrong was my output)
2. **Calculates** parameter gradients (how to adjust weights and bias)
3. **Calculates** input gradients (error to pass backward, proportional to weights)
4. **Updates** its own parameters (weights and bias)
5. **Returns** input gradients to the previous layer

```javascript
// ... class Neuron
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
// ...
```

#### Calculating paramater gradients

We need to figure out how much each specific weight contributed to the error. We do this by looking at the input signal.Think of it as Sensitivity Analysis. The incoming ```neuronGradient``` tells us the direction and magnitude we want the output to move (e.g., "The output needs to be lower by 0.5"). ```inputs[i]``` acts as an amplifier for that request.

* **The Silent Input** (Input = 0): If the input is 0, the math is Weight * 0. No matter how much you change the weight, the result is still 0. The output is insensitive to this weight. Therefore, the gradient is 0. We don't waste compute updating a weight that has no effect.
* **The Loud Input** (Input = 5): If the input is 5, any tiny change to the weight is multiplied by 5 in the output. The output is highly sensitive to this weight. Therefore, we multiply the gradient by 5 so we prioritize updating the weights that give us the most "bang for our buck."

We are essentially scaling our update effort by how "active" the connection was. In code, this relationship is linear. The "blame" assigned to a weight is simply the error coming from above (neuronGradient) scaled by the strength of the input signal (inputs[i]).

```javascript
// ... class Neuron
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
// ...
```

#### Calculating Input Gradients

In this step we propagate the gradient back to the previous layer. Conceptually, the neuron receives a single scalar gradient for its output (```neuronGradient```) and needs to distribute that signal across its inputs. 

```javascript
// ... class Neuron
    #calculateInputGradients(neuronGradient) { 
        // These gradients are the errors we pass to the previous layer
        const inputGradients = new Array(this.weights.length);
        
        for (let i = 0; i < this.weights.length; i++) {
            inputGradients[i] = neuronGradient * this.weights[i];
        }
        
        return inputGradients;
    }
// ...
```

Remember, during the forward pass, each input is scaled by its weight before being added into the neuron's output: ```sum += this.weights[i] * inputs[i];```

During the backward pass, we're answering the question "If this neurons output was too high, how much is each input to blame?" When trying to distribute the error to any input, we have to consider that the neurons weight acts like a volume knob.   

* If a weight is large (say, 5), then even a small input has a big effect on the output. So if the output is wrong, that input deserves a lot of "blame", a strong error signal to correct it.  
* If a weight is tiny (say, 0.01), then the input barely moved the needle. Even if the output is way off, this input didn't contribute much, so it gets a tiny error signal.
     
So to divide responsibility fairly, we multiply the incoming error (```neuronGradient```) by the weight that connected that input to the output: ```inputGradients[i] = neuronGradient * this.weights[i];```
     
```input[i]```, was scaled by ```weight[i]``` on the way forward. On the way back, ```input[i]```s share of the error is scaled by that same amount.  

#### Updating weights

Finally, it's time to adjust othe neurons parameters. This is where learning actually happens. We need to move *against* the gradient by subtracting. If the gradient says "increasing this weight increases error," we decrease the weight to decrease the error

```javascript
// ... class Neuron
    #updateWeights(gradients, learningRate) {
        // Update each weight using its stored gradient
        for (let i = 0; i < this.weights.length; i++) {
            this.weights[i] -= learningRate * gradients.weights[i];
        }
        // Update bias using its stored gradient
        this.bias -= learningRate * gradients.bias;
    }
// ...
```

This code moves each parameter in the direction that reduces error. The ```learningRate``` controls step size, too large and training becomes unstable; too small and learning takes forever. Each weight and the bias are adjusted independently, using only local information: their own gradient and the global learning rate. No coordination or global knowledge is needed, this locality is what makes neural network training parallelizable and scalable. 

### Backwards trough a dense layer

As we've seen, back propogation trough a single neuron requires us to remember what the input values of that neuron where. To compute gradients, we need to cache the inputs of each layer. Let's add the caching to our ```DenseLayer``` object, which will be able to pass the same cached vector to each neuron:

```javascript
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
// ...
```

Both dense and activation layers need to do cache their inputs.

```javascript
class ActivationLayer {
    kind = "relu";
    cachedInputs = null; // NEW

    forward(inputs) {
        // Cache inputs for the backwards pass
        if (this.cachedInputs == null || this.cachedInputs.length !== inputs.length) {
            this.cachedInputs = new Array(inputs.length);
        }
        for (let i = 0, size = inputs.length; i < size; ++i) {
            this.cachedInputs[i] = inputs[i];
        }
// ...
```

Each neuron knows how to update itself and pass error signals backward, but we need to coordinate this process across an entire layer. A dense layer contains many neurons, each receiving the same input vector but producing a unique output. During backpropagation, each neuron will receive its own error (from the layer ahead) and independently compute how much blame to assign to each input.

```javascript
// ... class DenseLayer
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
// ...
```

Every neuron in the layer contributes its own "blame" for each input, and we must sum those contributions together. Why? Because each input value fed into all neurons in the layer, so if multiple neurons were wrong, the input is responsible for all of those errors combined.

The ```backward``` method in ```DenseLayer``` handles this by initializing an ```inputGradients``` vector to zero, then looping over every neuron. Each neuron runs its own ```backward``` pass using its specific output gradient (```outputGradients[neuronIdx]```) and returns how much error to assign to each input. These per-neuron input gradients are added into the shared inputGradients array. The result is a complete picture of how the entire layer's error depends on each input, exactly what the previous layer needs to continue backpropagation. 

### Backwards trough an activation layer

Activation layers don't have learnable parameters, but they still need to participate in backpropagation. During the forward pass, activation layers warped the signal: ReLU killed negative values, sigmoid squashed everything toward 0 or 1, tanh bent the curve into a smooth S. To reverse this warping during backpropagation, we need to know how sensitive the output was to the input at the exact point where we evaluated it. 

That's why we cached the original inputs during the forward pass: the derivative of an activation function depends on the input value, not the output. For example, ReLU's derivative is 1 when the input was positive (meaning small changes in input pass through unchanged) and 0 when the input was negative (meaning no change in output regardless of input). Without the original input, we couldn't compute the correct local slope. 

These are the derivitives of the three activation functions we have implemented:

```javascript
// ... class ActivationLayer
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
// ...
```

These derivitives have some interesting properties:
- **ReLU derivative**: Binary. Either the gradient flows (x > 0) or it doesn't (x <= 0). This can cause "dead neurons" that stop learning.
- **Sigmoid derivative**: Strongest in the middle, weak at extremes. This causes the "vanishing gradient" problem in deep networks.
- **Tanh derivative**: Similar to sigmoid but generally better behaved.


Now let's implement the backward pass of the Activation Layer. This pass propagates gradients through the activation function by scaling the incoming ```outputGradients``` with the derivative of the activation evaluated at the cached input. 

```javascript
// ... class ActivationLayer
    backward(outputGradients) {
        // outputGradients = error signal coming from the next layer
        // We need to figure out what the input gradient was
        
        const inputGradients = new Array(outputGradients.length);

        if (this.kind === "relu") {
            for (let i = 0; i < outputGradients.length; i++) {
                inputGradients[i] = outputGradients[i] * this.#reluDerivative(this.cachedInputs[i]);
            }
        }
        else if (this.kind === "sigmoid") {
            for (let i = 0; i < outputGradients.length; i++) {
                inputGradients[i] = outputGradients[i] * this.#sigmoidDerivative(this.cachedInputs[i]);
            }
        }
        else if (this.kind === "tanh") {
            for (let i = 0; i < outputGradients.length; i++) {
                inputGradients[i] = outputGradients[i] * this.#tanhDerivative(this.cachedInputs[i]);
            }
        }

        return inputGradients;
    }
// ...
```

> The activation layer doesn't have weights to update, it just modulates the gradient based on its derivative. Think of it as a gradient filter: ReLU is a gate (open or closed), while sigmoid and tanh are dimmers (gradual adjustment).

## Training the XOR network

Let's put everything together into a trainable network! XOR is the "Hello World" of neural networks, simple enough to understand, complex enough to require hidden layers. A linear model can't solve XOR, but our network can.


```javascript
// XOR truth table:
// 0, 0 -> 0
// 0, 1 -> 1  
// 1, 0 -> 1
// 1, 1 -> 0

// Training data
const trainingData = [
    { input: [0, 0], target: [0] },
    { input: [0, 1], target: [1] },
    { input: [1, 0], target: [1] },
    { input: [1, 1], target: [0] }
];

// Build the network architecture
// 2 inputs -> 4 hidden neurons -> 1 output
const layer1 = new DenseLayer(2, 4);  // 2 inputs, 4 hidden neurons
const activation1 = new ActivationLayer("tanh");
const layer2 = new DenseLayer(4, 1);  // 4 inputs, 1 hidden neuron
const activation2 = new ActivationLayer("sigmoid");

// Training parameters
const learningRate = 0.5;
const epochs = 10000;

// Training loop
for (let epoch = 0; epoch < epochs; epoch++) {
    let totalLoss = 0;
    
    // Train on each example
    for (const data of trainingData) {
        // Forward pass
        let output = layer1.forward(data.input);
        output = activation1.forward(output);
        output = layer2.forward(output);
        output = activation2.forward(output);
        
        // Calculate loss
        const loss = Loss.meanSquaredError(output, data.target);
        totalLoss += loss;
        
        // Calculate initial gradient from loss
        let gradients = Loss.meanSquaredErrorDerivative(output, data.target);
        
        // Backward pass
        gradients = activation2.backward(gradients);
        gradients = layer2.backward(gradients, learningRate);
        gradients = activation1.backward(gradients);
        gradients = layer1.backward(gradients, learningRate);
    }
    
    // Print progress every 1000 epochs
    if (epoch % 1000 === 0) {
        console.log(`Epoch ${epoch}: Average Loss = ${totalLoss / trainingData.length}`);
    }
}

// Test the trained network
console.log("\n=== Testing Trained Network ===");
for (const data of trainingData) {
    // Forward pass only
    let output = layer1.forward(data.input);
    output = activation1.forward(output);
    output = layer2.forward(output);
    output = activation2.forward(output);
    
    console.log(`Input: [${data.input}] -> Output: ${output[0].toFixed(3)} (Target: ${data.target[0]})`);
}
```

Run this code and watch the loss drop! At first, the network is guessing randomly. But gradually, it discovers the XOR pattern. By epoch 10000, it should be able to answer all four cases.

> What's actually happening in those hidden neurons? They're learning features! One might activate for "exactly one input is 1", another for "both inputs are the same". The output layer then combines these features to compute XOR. It's feature engineering, automated.

# What's Next?

We've built a neural network from scratch., but this is just the beginning. Real networks have:

- **Batch processing**: Training on multiple examples at once for stability
- **Optimizers**: Adam, RMSprop, smarter than vanilla gradient descent
- **Regularization**: Dropout, L2 penalties-preventing overfitting
- **Convolutional layers**: For images, detecting patterns regardless of position
- **Recurrent connections**: For sequences, remembering previous inputs

The core principles remain the same though. It's all forward passes, backward passes, and gradients flowing through differentiable functions. The complexity comes from scale.
