# PM2.5 Multi-Horizon Forecasting (Bangladesh)

Comparative study of XGBoost, LSTM, LSTM+Attention, LSTM+Transformer and
persistence baseline for PM2.5 forecasting at 1h, 3h and 6h ahead,
using hourly multi-station data.

## Dataset
`archive__1_` (6 stations: Dhaka, Chittagong, Sylhet, Rajshahi, Khulna, Barisal), hourly, 2022-08-04 to 2026-07-20.
Place `archive__1_.zip` under `data/` (not included in this repo).

## Split (chronological)
- Train: < 2025-01-01
- Validation: 2025
- Test: 2026 (hold-out)

## Models
- Persistence (naive t-1)
- XGBoost (lag windows, 24h lookback, weather + time features)
- LSTM, LSTM+Attention, LSTM+Transformer (24h lookback)

## Results
Mean ± std over 3 seeds (42, 43, 44): `results/summary.csv`.

## Run
```bash
python -m src.train 42
python -m src.train 43
python -m src.train 44
python -m src.aggregate
```

## Known limitations
- Deep model inputs are not scaled yet (planned fix).
- Dataset completeness (no missing values) to be verified against source.