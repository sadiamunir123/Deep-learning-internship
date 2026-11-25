# Deep Learning Phase 3: Sequential Models (RNN, LSTM, GRU)

This repository contains my work for Phase 3 of the deep learning internship roadmap: how neural networks handle data where **order matters** (time series, text, signals). It ends with a forecasting project that compares LSTM/GRU models against honest baselines.

## Notebooks

| # | Notebook | What it covers |
|---|----------|----------------|
| 1 | `notebook1_rnn_basics_and_data_prep.ipynb` | RNN cell written by hand in NumPy, sliding windows, time-based split, scaling, naive baseline |
| 2 | `notebook2_rnn_pytorch_vanishing_gradient.ipynb` | `nn.RNN` forecaster, gradient clipping, vanishing-gradient demo, a long-memory task where the RNN fails |
| 3 | `notebook3_rnn_lstm_gru_comparison.ipynb` | RNN vs LSTM vs GRU (parameters, accuracy, loss curves, time), LSTM gate visualisation |
| 4 | `notebook4_sequence_pattern_classifier.ipynb` | Sequence pattern classifier (sine, square, sawtooth, noise), confusion matrix, hidden-state plots |
| 5 | `notebook5_forecasting_data_and_baselines.ipynb` | Forecasting data, time split, leakage check, baseline scores |
| 6 | `notebook6_forecasting_lstm_gru.ipynb` | LSTM/GRU forecasters, early stopping, results table, predictions over time, multi-step forecast, discussion |

## What I learned

- **Sequential data**: order matters and length can vary, so an MLP on a flattened window is not enough (no memory, fixed size).
- **RNN**: `h_t = tanh(W_x x_t + W_h h_{t-1} + b)`, with the same weights at every time step.
- **Vanishing gradient**: gradients shrink over long sequences, so a plain RNN forgets the distant past (shown with a gradient plot in notebook 2).
- **LSTM and GRU**: gates and a cell state that updates by addition let gradients flow further back. A GRU is simpler (2 gates, no separate cell state).
- **Classification vs forecasting**: `CrossEntropyLoss` for classes, `MSELoss` for numbers.
- **Time-series rules**: split by time (never shuffle), scale with training statistics only, avoid leakage from overlapping windows, and always compare with a naive baseline.

## Results

Dataset used for the forecasting project: **[FILL IN: e.g. airline passengers / AAPL daily close]**, window size K = **[FILL IN]**.
Split: 70% train / 15% validation / 15% test (by time).

| Model | Val MAE | Val RMSE | Test MAE | Test RMSE |
|-------|---------|----------|----------|-----------|
| Naive (tomorrow = today) | [ ] | [ ] | [ ] | [ ] |
| Moving average | [ ] | [ ] | [ ] | [ ] |
| Linear (last K values) | [ ] | [ ] | [ ] | [ ] |
| LSTM ([layers], [hidden]) | [ ] | [ ] | [ ] | [ ] |
| GRU ([layers], [hidden]) | [ ] | [ ] | [ ] | [ ] |

(Copy these numbers from `results_table.csv`, produced by notebook 6.)

Other results:
- Sequence pattern classifier test accuracy: **[ ]**
- Long-memory task (length 50) accuracy: RNN **[ ]**, LSTM **[ ]**, GRU **[ ]**

## Explanation

**[WRITE IN YOUR OWN WORDS, 1 to 2 short paragraphs.]** Suggested points:
- What the project does and how the data was prepared.
- Did the best model beat the naive baseline? By how much, and is the difference larger than the noise (paired comparison, seed variation)?
- What the plot over time shows (for example predictions that lag the real series by one step).
- How the multi-step forecast error grows with the horizon.

## Key takeaways

1. **[WRITE IN YOUR OWN WORDS]** A plain RNN struggles with long dependencies; LSTM/GRU fix this.
2. **[WRITE IN YOUR OWN WORDS]** A model is only useful if it beats a simple baseline.
3. **[WRITE IN YOUR OWN WORDS]** Stock prices are close to a random walk, so beating the naive rule is hard; seasonal data is easier.
4. **[WRITE IN YOUR OWN WORDS]** Correct splitting and scaling matter as much as the model.

## How to run

1. Open the notebooks in Google Colab (a GPU is **not** needed, the CPU is fine).
2. Run them in order, 1 to 6. Notebook 5 saves `prepared_data.npz` and `baseline_scores.csv`, which notebook 6 loads, so run both in the same session (or upload those two files).
3. Locally: `pip install -r requirements.txt`, then `jupyter notebook`.

## Sources

- Karpathy, The Unreasonable Effectiveness of Recurrent Neural Networks: https://karpathy.github.io/2015/05/21/rnn-effectiveness/
- Colah, Understanding LSTM Networks: https://colah.github.io/posts/2015-08-Understanding-LSTMs/
- PyTorch `nn.RNN`: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
- PyTorch `nn.LSTM`: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
- Andrew Ng, Sequence Models (Coursera): https://www.coursera.org/learn/nlp-sequence-models
