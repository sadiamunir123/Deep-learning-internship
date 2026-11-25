# Phase 3 Revision Sheet (Checkpoint Quiz)

## 1. Sequential data: why order and memory matter
- Data where order matters: time series, text, audio, sensor signals. Length can vary.
- MLP/CNN on a flattened window: no memory, fixed input size.
- RNN: same cell and weights at every step, plus a hidden state that carries memory.

## 2. The RNN update and unrolling
- `h_t = tanh(W_x x_t + W_h h_{t-1} + b)`, `h_0 = 0`.
- Unrolling = drawing the same cell once per time step. It is one cell reused, not many cells.
- Patterns: many-to-one (sequence in, one answer out), many-to-many, one-to-many.

## 3. Vanishing gradient
- Backpropagation through time (BPTT): gradients flow back through every step.
- The same weights are multiplied at each step, so gradients shrink (vanishing) or blow up (exploding) over long sequences.
- Result: a plain RNN forgets the distant past and cannot learn long dependencies.
- Gradient clipping limits the gradient size and protects against exploding gradients (it does not fix vanishing).

## 4. LSTM gates, cell state, and how GRU differs
- LSTM: cell state (long-term memory) + 3 gates:
  - forget gate: what to erase,
  - input gate: what to write,
  - output gate: what to reveal as the hidden state.
- Cell state is updated by addition (`c_t = f * c_{t-1} + i * g`), so gradients flow back more easily.
- GRU: 2 gates (update, reset), no separate cell state, fewer parameters, often similar results.
- Parameter counts for the same hidden size: LSTM about 4x an RNN, GRU about 3x.
- PyTorch: `nn.LSTM` returns `(output, (h_n, c_n))`; `nn.RNN`/`nn.GRU` return `(output, h_n)`. With `batch_first=True` the input shape is `(batch, time, features)`.

## 5. Many-to-one vs many-to-many, forecasting vs classification
- Many-to-one: take the last hidden state and pass it to `nn.Linear`.
- Many-to-many: an output at every time step.
- Forecasting (a number): `MSELoss`, metrics MAE/RMSE.
- Classification (a class): `CrossEntropyLoss`, metrics accuracy and confusion matrix.

## 6. Time-series evaluation rules
- Split by time, never shuffle (train = past, test = future).
- Scale with mean/std from the training part only, otherwise future information leaks.
- Windows: no target may appear in two splits.
- Always compare with a naive baseline (tomorrow = today). A model that does not beat it has not learned anything useful.
- Stock prices are close to a random walk, so beating the naive rule is hard. Seasonal data is easier.
- Use early stopping on validation loss, choose settings on validation, and evaluate on the test set once.

## Likely quiz questions (practice answering out loud)
1. Why can't a plain MLP handle sequences well?
2. Write the RNN update equation and explain each term.
3. What is vanishing gradient and why does it happen?
4. What does each LSTM gate do? Why does the cell state help?
5. How is a GRU different from an LSTM?
6. Why do we never shuffle a time series before splitting?
7. Why scale with training statistics only?
8. What is a naive baseline and why is it important?
9. When do you use CrossEntropyLoss and when MSELoss?
10. Your LSTM does not beat the naive baseline. What do you report?
