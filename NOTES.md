# Theory Notes: Deep Learning Phase 1

Notes for Day 1 to Day 5. Each section ends with the key points to remember.

---

## Day 1: Neuron, activations and the forward pass

### Neuron
A neuron takes inputs, multiplies each by a weight, adds a bias, and applies an activation function.

```
z = w1*x1 + w2*x2 + ... + wn*xn + b = w . x + b
a = activation(z)
```

- **Weights** control how much each input matters. They are learned during training.
- **Bias** shifts the neuron's output, so it does not depend on the inputs only.
- **Activation function** turns `z` into the neuron's output `a`.

### Activation functions

| Activation | Formula | Output range | Typical use |
|---|---|---|---|
| ReLU | max(0, z) | [0, infinity) | Default for hidden layers |
| Sigmoid | 1 / (1 + e^(-z)) | (0, 1) | Output layer for binary classification (probability) |
| Tanh | (e^z - e^(-z)) / (e^z + e^(-z)) | (-1, 1) | Hidden layers in older networks, zero-centered |

- Sigmoid and tanh flatten out for large positive or negative `z`, so their gradient becomes almost 0. This is the **vanishing gradient** problem. ReLU avoids it for positive inputs.

### Why non-linear activations are needed
A linear layer computes `X @ W + b`. Two linear layers in a row are still one linear function, so any number of linear layers is equivalent to a single linear layer. A non-linear activation between layers is what lets the network learn curved decision boundaries (for example the two moons).

### Layers and the forward pass
A layer is many neurons working in parallel, written as one matrix multiplication:

```
Z = X @ W + b
```

Shapes for a batch of `m` samples: `X` is (m, n_in), `W` is (n_in, n_out), `b` is (1, n_out), `Z` is (m, n_out).

Forward pass for a 2-layer network:

```
Z1 = X @ W1 + b1
A1 = relu(Z1)
Z2 = A1 @ W2 + b2
A2 = sigmoid(Z2)       # prediction
```

The forward pass only computes predictions. It does not learn anything.

**Key points:** neuron = weighted sum + bias + activation; non-linearity is essential; forward pass = matrix multiplications layer by layer; always check shapes.

---

## Day 2: Loss, gradient descent and backpropagation

### Loss functions
A loss function measures how wrong the predictions are with a single number. Training tries to make it small.

- **MSE** (regression): `mean((y_hat - y)^2)`
- **Binary cross-entropy** (classification): `-mean(y*log(y_hat) + (1 - y)*log(1 - y_hat))`
- **Categorical cross-entropy** (multi-class): `-mean(log(probability assigned to the true class))`

Cross-entropy is preferred for classification because it penalizes confident wrong predictions heavily and, combined with sigmoid or softmax, gives a simple, well-behaved gradient.

### Gradient descent
Move each weight a small step against its gradient:

```
w = w - lr * dL/dw
```

- **Learning rate (lr)** is the step size. Too large: the loss jumps around or diverges. Too small: training is very slow.

### Backpropagation
Backpropagation applies the chain rule from the output back to the input to get `dL/dw` for every weight.

For the network `X -> Linear -> ReLU -> Linear -> Sigmoid -> binary cross-entropy`:

```
dZ2 = (A2 - y) / m             # sigmoid + cross-entropy simplifies to this
dW2 = A1.T @ dZ2
db2 = sum(dZ2 over the batch)
dA1 = dZ2 @ W2.T
dZ1 = dA1 * (Z1 > 0)           # derivative of ReLU
dW1 = X.T @ dZ1
db1 = sum(dZ1 over the batch)
```

### Gradient check
To verify the backward pass, compare each analytic gradient with a numerical one:

```
numeric = (L(w + eps) - L(w - eps)) / (2 * eps)
relative error = |numeric - analytic| / (|numeric| + |analytic|)
```

A relative error around 1e-6 or smaller means the backward pass is correct. Gradient checking is for debugging only, because it is slow.

### Training loop (from scratch)
1. Forward pass, compute predictions.
2. Compute the loss.
3. Backward pass, compute all gradients.
4. Update the weights.
5. Repeat for many epochs and watch the loss curve go down.

**Key points:** loss says how wrong we are; gradient descent uses gradients to reduce it; backprop is the chain rule used efficiently; gradient check proves the implementation is right.

---

## Day 3: PyTorch and optimizers

### Tensors and autograd
- A **tensor** is like a NumPy array that can run on a GPU and track gradients.
- **Autograd** records operations in a computation graph. Calling `loss.backward()` computes the gradient of the loss with respect to every tensor created with `requires_grad=True`, stored in `.grad`.

### Building blocks
- `nn.Linear(n_in, n_out)` computes `X @ W + b`.
- `nn.Module` is the base class for models. Define layers in `__init__` and the forward pass in `forward`.
- `Dataset` holds the data. `DataLoader` serves it in batches and can shuffle it.
- `BCEWithLogitsLoss` combines sigmoid and binary cross-entropy in one numerically stable step, so the model outputs raw scores (logits) and no sigmoid layer is added at the end.

### The standard training loop

```
for each batch:
    logits = model(x)               # 1. forward
    loss = criterion(logits, y)     # 2. loss
    optimizer.zero_grad()           # 3. clear old gradients
    loss.backward()                 # 4. backward (autograd)
    optimizer.step()                # 5. update weights
```

`zero_grad()` is required because PyTorch accumulates gradients by default.

### Loss function vs optimizer
- The **loss function** measures how wrong the model is.
- The **optimizer** decides how to change the weights using the gradients of that loss.

### Optimizers
- **SGD:** `w = w - lr * g`
- **SGD with momentum:** keeps a running average of past gradients, so updates are smoother and faster along consistent directions.
- **Adam:** momentum plus a per-parameter adaptive learning rate:

```
m = beta1 * m + (1 - beta1) * g            # average of gradients
v = beta2 * v + (1 - beta2) * g^2          # average of squared gradients
w = w - lr * m_hat / (sqrt(v_hat) + eps)   # m_hat, v_hat are bias-corrected
```

Default values: beta1 = 0.9, beta2 = 0.999, eps = 1e-8. Adam usually trains faster than plain SGD because each weight gets its own step size and momentum smooths the path, so it needs less learning-rate tuning.

### Mini-batch vs full-batch
Full-batch uses all samples per update (smooth but slow per step). Mini-batch uses a small random subset (noisier but more updates per epoch, and standard in practice).

**Key points:** autograd replaces hand-written backprop; training loop = forward, loss, zero_grad, backward, step; Adam is usually faster than SGD.

---

## Day 4: Binary classification on tabular data

### Data splitting
- **Train** set: the model learns from it.
- **Validation** set: used to monitor training and tune choices.
- **Test** set: used once at the end for the final, unbiased estimate.
- **Stratified** splitting keeps the class ratio the same in every split, which matters for imbalanced data.

### Feature scaling and data leakage
Fit `StandardScaler` on the training set only, then apply `transform` to validation and test. Fitting on all data leaks information from the test set into training and makes results look better than they really are.

### Evaluation metrics
With a confusion matrix of TP, FP, FN, TN:

```
accuracy  = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)       # of predicted positives, how many are right
recall    = TP / (TP + FN)       # of actual positives, how many are found
F1        = 2 * precision * recall / (precision + recall)
```

Accuracy can hide problems, especially with imbalanced classes. In medical settings, missing a malignant case (a false negative for malignant) is the most costly error, so look at recall for that class too.

In scikit-learn's Breast Cancer dataset, class 0 is malignant and class 1 is benign.

### Reading learning curves
Plot train and validation loss per epoch. Both going down means the model is learning. A validation loss that rises while training loss keeps falling means overfitting (Day 5).

### Neural networks vs classical models on small tabular data
Neural networks do not always win. On small tabular datasets, models such as SVM, logistic regression and random forests often match or beat an MLP, because there is little data to train many parameters and the features are already informative. Neural networks shine with large datasets and unstructured data such as images, text and audio. Also, a small test set (86 samples here) makes one run noisy: one extra mistake changes accuracy by about 1.2%, so repeat with several seeds before concluding.

**Key points:** stratify, scale on train only, test once; report precision, recall, F1 and the confusion matrix, not just accuracy; compare honestly with simpler models.

---

## Day 5: Regularization and multi-class classification

### Overfitting and underfitting
- **Overfitting:** the model memorizes the training data. Training loss keeps falling, validation loss starts to rise, and the gap between them grows.
- **Underfitting:** the model is too simple or trained too little. Both training and validation loss stay high.

### Regularization techniques
- **Dropout:** during training, randomly set a fraction `p` of neuron outputs to zero in each step, so the network cannot rely on specific neurons. PyTorch scales the remaining outputs by `1 / (1 - p)` during training, and dropout is switched off at evaluation time. Use `model.train()` and `model.eval()` correctly.
- **Weight decay (L2):** penalizes large weights, which pushes the model toward simpler functions. In PyTorch: `torch.optim.Adam(..., weight_decay=1e-4)`. With Adam, `weight_decay` adds an L2 term to the gradient, while `AdamW` decouples the decay from the gradient update.
- **Early stopping:** monitor validation loss, stop when it has not improved for a set number of epochs (the patience), and restore the weights from the best epoch.

### Multi-class classification
- The output layer has one unit per class, so Iris uses 3 outputs.
- **Softmax** turns logits into probabilities that sum to 1:

```
softmax(z_i) = e^(z_i) / sum_j e^(z_j)
```

- In PyTorch, use `nn.CrossEntropyLoss` with raw logits and integer class labels (0, 1, 2). It applies log-softmax internally, so do not add a softmax layer before it. Apply `softmax` only when you need probabilities for display.
- Predicted class = `argmax` of the logits.
- Evaluate with a confusion matrix (3 x 3) and per-class precision and recall.

**Key points:** overfitting shows up as a growing gap between train and validation loss; dropout, weight decay and early stopping reduce it; multi-class uses softmax and cross-entropy.

---

## Checkpoint: concepts to be able to explain

**What are a neuron, weight, bias and activation function, and why do we need non-linear activations?**
A neuron computes `w . x + b` and applies an activation. Weights say how important each input is, and the bias shifts the result. Without non-linear activations, any stack of layers is equivalent to one linear layer and cannot learn curved patterns.

**What is the difference between a loss function and an optimizer?**
The loss function measures how wrong the predictions are. The optimizer uses the gradients of that loss to update the weights.

**What do forward and backward propagation do?**
Forward propagation passes the input through the layers to produce a prediction. Backward propagation uses the chain rule to compute the gradient of the loss for every weight.

**Why does Adam usually train faster than plain SGD?**
Adam combines momentum with a per-parameter adaptive learning rate, so it takes better-scaled steps for each weight and needs less learning-rate tuning.

**What are overfitting and underfitting, and how do dropout, weight decay and early stopping help?**
Overfitting is when training loss falls but validation loss rises; underfitting is when both stay high. Dropout prevents reliance on specific neurons, weight decay keeps weights small, and early stopping ends training before the model starts to memorize the training data.

**Why do neural networks not always beat simpler models on small tabular data?**
Small datasets give a large network little to learn from, and the features are often already informative, so SVMs, logistic regression and tree models reach similar or better accuracy with less tuning and less risk of overfitting.

---

## Glossary

| Term | Meaning |
|---|---|
| Epoch | One full pass over the training data |
| Batch | A subset of samples used for one weight update |
| Logit | Raw output score before sigmoid or softmax |
| Hyperparameter | A setting chosen by you, such as learning rate, hidden size or dropout rate |
| Data leakage | Information from validation or test data influencing training |
| Gradient | The derivative of the loss with respect to a weight |
