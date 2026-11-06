# Deep Learning Phase 1: Foundations of Neural Networks

A five-day project in which I built a fully connected neural network (MLP) from scratch in NumPy, rebuilt it in PyTorch, and used it for binary and multi-class classification on tabular data.

> **Status:** replace every `TODO` below with your own results from Colab before pushing.

## Goals

- Understand how a basic MLP works: neuron, activation, forward pass, loss, backpropagation.
- Implement forward, backward and update by hand in NumPy and verify the gradients numerically.
- Rebuild the same network in PyTorch and compare it with the NumPy version.
- Compare optimizers (SGD, SGD with momentum, Adam).
- Train a binary classifier on the Breast Cancer dataset and compare it honestly with classical models.
- Study overfitting and apply regularization (dropout, weight decay, early stopping).
- Train a multi-class classifier on Iris.

## Repository structure

```
.
├── README.md
├── NOTES.md                                  # theory notes, Day 1 to Day 5
├── requirements.txt
└── notebooks/
    ├── day1_neuron_forward_pass.ipynb        # NumPy: activations, neuron, forward pass
    ├── day2_backprop_mlp_from_scratch.ipynb  # NumPy: loss, backprop, MLP class, gradient check
    ├── day3_pytorch_notebook2.ipynb          # PyTorch rebuild, NumPy vs PyTorch, SGD vs Adam
    ├── day4_breast_cancer_mlp.ipynb          # Breast Cancer binary classifier
    └── day5_regularization_iris.ipynb        # TODO: add your Day 5 notebook (regularization + Iris)
```

## Setup

The notebooks run on Google Colab (no GPU needed). To run locally:

```
pip install -r requirements.txt
jupyter notebook
```

## Day-by-day summary

| Day | Topic | Notebook |
|---|---|---|
| 1 | Neuron, activations (ReLU, Sigmoid, Tanh), forward pass in NumPy | `day1_neuron_forward_pass.ipynb` |
| 2 | Loss functions, backpropagation, MLP class, training on `make_moons`, gradient check | `day2_backprop_mlp_from_scratch.ipynb` |
| 3 | PyTorch basics, NumPy vs PyTorch comparison, SGD vs Adam | `day3_pytorch_notebook2.ipynb` |
| 4 | Breast Cancer MLP, evaluation, comparison with classical models | `day4_breast_cancer_mlp.ipynb` |
| 5 | Regularization, early stopping, multi-class classifier on Iris | `day5_regularization_iris.ipynb` |

## Part 1: MLP from scratch (NumPy)

- **Dataset:** `make_moons`, 500 samples, noise 0.2, stratified 80/20 train/test split, random_state 42.
- **Model:** `2 -> 16 -> 1`, ReLU hidden layer, sigmoid output, binary cross-entropy loss.
- **Training:** full-batch gradient descent, learning rate 0.5, 2000 epochs.
- **Backpropagation** (written by hand):

```
dZ2 = (A2 - y) / m
dW2 = A1.T @ dZ2          db2 = sum(dZ2)
dZ1 = (dZ2 @ W2.T) * (Z1 > 0)
dW1 = X.T @ dZ1           db1 = sum(dZ1)
```

| Metric | Result |
|---|---|
| Train accuracy | 0.980 |
| Test accuracy | 0.990 |
| Gradient check (analytic vs numerical, relative error) | about 1e-10 |

The gradient check compares the backpropagation gradient with the central-difference estimate `(L(w + eps) - L(w - eps)) / (2 * eps)` and agrees to about 1e-10 for the weights and biases I checked.

## Part 2: PyTorch re-implementation and optimizers

The same `2 -> 16 -> 1` network was rebuilt with `nn.Module`, `nn.Linear`, `BCEWithLogitsLoss` and `DataLoader`.

**NumPy vs PyTorch** (same setup: full-batch SGD, lr 0.5, 2000 epochs):

| Version | Final loss | Test accuracy |
|---|---|---|
| NumPy (from scratch) | TODO | TODO |
| PyTorch | TODO | TODO |

**SGD vs SGD with momentum vs Adam** (same initial weights, full-batch, 300 epochs):

| Optimizer | Final loss | Test accuracy |
|---|---|---|
| SGD (lr 0.1) | TODO | TODO |
| SGD + momentum 0.9 (lr 0.1) | TODO | TODO |
| Adam (lr 0.01) | TODO | TODO |

TODO: add the NumPy vs PyTorch loss curve and the SGD vs Adam plot (save them as images in the repo and embed them here), and write one or two sentences on what you observed.

## Part 3: Binary classifier on the Breast Cancer dataset

- **Data:** scikit-learn Breast Cancer, 569 samples, 30 features (212 malignant = class 0, 357 benign = class 1).
- **Split:** stratified 70/15/15 (train/validation/test), `random_state=42`.
- **Preprocessing:** `StandardScaler` fitted on the training set only, to avoid data leakage.
- **Model:** PyTorch MLP `30 -> 32 -> 16 -> 1`, ReLU, `BCEWithLogitsLoss`, Adam (lr 0.001), batch size 32, 100 epochs.
- **Tracked per epoch:** train and validation loss and accuracy.

### Test-set results (86 samples)

The classical baselines below were trained on the same split and the same scaled data (Logistic Regression with `max_iter=1000`, SVM with RBF kernel and `C=1.0`, Random Forest with 200 trees).

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| MLP (PyTorch) | TODO | TODO | TODO | TODO |
| Logistic Regression | 0.965 | 0.981 | 0.963 | 0.972 |
| SVM (RBF) | 0.965 | 0.981 | 0.963 | 0.972 |
| Random Forest | 0.919 | 0.961 | 0.907 | 0.933 |

Precision, recall and F1 are computed with the benign class (label 1) as the positive class.

MLP over 5 seeds: test accuracy TODO +/- TODO.

TODO: add the confusion matrix and learning curves.

**Comparison with my Week 3 results.** TODO: state honestly whether the MLP beat the Week 3 models. My Week 3 SVM reached about 0.98 test accuracy on a different split, so the numbers are not directly comparable to the table above. On a small tabular dataset like this one, classical models such as SVM and logistic regression usually match or beat a neural network, because there is little data and the features are already informative.

## Part 4: Regularization

Starting from the Day 4 model, I added dropout, weight decay (L2) and early stopping on validation loss.

| Setup | Best val loss | Test accuracy | Epochs trained |
|---|---|---|---|
| Baseline (no regularization) | TODO | TODO | TODO |
| + Dropout | TODO | TODO | TODO |
| + Weight decay | TODO | TODO | TODO |
| + Dropout + weight decay + early stopping | TODO | TODO | TODO |

TODO: add before/after learning curves and describe what regularization changed (for example, how the gap between train and validation loss changed).

## Part 5: Multi-class classifier on Iris

- **Model:** MLP with 3 output units (one per class), `CrossEntropyLoss` on raw logits, softmax applied only when probabilities are needed.
- **Test accuracy:** TODO
- **Confusion matrix:** TODO (add image)

## What I learned

- A neuron computes a weighted sum plus bias and passes it through a non-linear activation. Without non-linearity, stacked layers collapse into a single linear layer.
- Forward propagation computes predictions; backpropagation uses the chain rule to compute the gradient of the loss for every weight; the optimizer uses those gradients to update the weights.
- A gradient check is a reliable way to verify a hand-written backward pass.
- PyTorch's autograd reproduces the same gradients automatically, and my from-scratch version matched it.
- Adam often converges faster than plain SGD because it combines momentum with a per-parameter adaptive learning rate.
- Scalers must be fitted on the training data only, and the test set must be used only once at the end.
- Accuracy alone can hide errors, so precision, recall, F1 and the confusion matrix matter, especially for medical data.
- Neural networks do not always beat simpler models on small tabular data.
- TODO: add what regularization taught you.

See [NOTES.md](NOTES.md) for the detailed theory notes.

## Tools

Python, NumPy, pandas, matplotlib, scikit-learn, PyTorch, Google Colab.
