# Property Value Prediction — Machine Learning Challenge

Individual project for a Machine Learning course at Tilburg University. The task: predict property values from real estate transaction and cadastral data (property type, location codes, surface area, room counts, transaction dates).

## Approach

- **Feature engineering**: date decomposition, area ratios, room-count aggregation, and leakage-safe K-Fold target encoding for geographic codes such as commune and cadastral section
- **Modeling strategy**: separate models trained per property category (house, apartment, land, etc.) where data was sufficient, plus a pooled global model for sparser categories
- **Algorithms**: gradient-boosted trees — XGBoost, LightGBM, and CatBoost
- **Model selection**: blend weights between models chosen via 5-fold cross-validation, evaluated on a held-out split to avoid overfitting to the CV folds themselves
- **Robustness**: seed bagging to reduce variance in final predictions

## Files

- `ML_Challenge.py` — final submission pipeline
- `ml_challenge_baseline.py` — baseline scaffold provided as part of the course assignment

## Tech stack

Python, pandas, scikit-learn, XGBoost, LightGBM, CatBoost
