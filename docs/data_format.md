# Input data formats

## Frozen HHG Full entry

For each regional subset, `--data-root` contains `<region>/fold-01/config.json`, `data/development.npz`, and `data/preprocessor.npz`, with corresponding fold-02 and fold-03 directories. These files must be constructed from the intended training and evaluation partition; changing that partition defines a new experiment.

The historical loader expects 185 features and training references first. `development.npz` contains `raw_features`, `x`, `y`, `coords`, `sample_ids`, `missing_mask`, `train_edges`, `eval_edges`, and scalar `n_train`. The preprocessor contains training-fitted feature median/mean/scale and target mean/scale. The base split seed is `20261014 + fold - 1`. The ecological CSV contains `sample_id` and `species`; the latter is the original ecological class field name and need not imply an individual botanical species.

Use each fold's frozen configuration: hidden dimensions 64 and 32, dropout 0.2, and KNN k=8. Other optimizer, early-stopping and preprocessing fields are consumed directly from `config.json`; do not fill unspecified values by guessing. The original loaders validate split consistency and unique sample IDs.

## Public-dataset interfaces

`evaluation/dataset_schema.py`, `prepare_grouped_tabular.py`, and `normalize_public_sources.py` define the tabular normalization contract. Generic experiments receive their own `--config`, `--data`, and output paths. Preserve original feature dimensions and ecological categories. Preparation uses training-side imputation and standardization and stores the source/reference count explicitly. Product-derived biomass references and field measurements must remain identified separately.

The release includes no measured data or frozen split files. Obtaining the same original inputs and partitions is necessary for numerical reproduction; merely using the same number of rows is insufficient.
