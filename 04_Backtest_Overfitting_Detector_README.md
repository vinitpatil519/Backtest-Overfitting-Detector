# Backtest Overfitting Detector

Detect whether an impressive Sharpe ratio is real or statistical luck.

## Goal
Implement Deflated Sharpe Ratio, Probability of Backtest Overfitting and Monte Carlo validation.

## Workflow
```mermaid
flowchart LR
A[Generate 1000 Strategies]-->B[Backtest]
B-->C[Sharpe Distribution]
C-->D[Deflated Sharpe]
D-->E[Robustness Score]
```

## Metrics
- Sharpe
- Deflated Sharpe
- Skew
- Kurtosis
- PBO

## UI
- Histogram
- DSR meter
- Robustness report
- Monte Carlo simulator

## Outcome
Only statistically significant strategies pass.
