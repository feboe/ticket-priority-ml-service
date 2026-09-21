# Dataset Manifest And Evaluation Protocol

## Scope

This project uses synthetic customer-support tickets from the public Kaggle
dataset [Multilingual Customer Support Tickets](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets).
The CSV files remain local under `data/` and are excluded from Git. The upstream
dataset is licensed under `CC BY-NC 4.0`; any redistribution of data or derived
models should be checked against those terms.

This document freezes the training and holdout roles before further model work.
These roles are part of the evaluation design and must not be changed in
response to model results.

## Dataset Inventory

| Role | File | Rows | Columns | Size (bytes) | SHA-256 |
| --- | --- | ---: | ---: | ---: | --- |
| Training and model selection | `aa_dataset-tickets-multi-lang-5-2-50-version.csv` | 28,587 | 16 | 25,996,354 | `f187c090e59581c2bbf3aa1377c8db4dd647464ecf2ae51bf8966e42e0ed6bc0` |
| Frozen external synthetic holdout | `dataset-tickets-multi-lang3-4k.csv` | 4,000 | 17 | 6,873,542 | `9aae7120cf459fc27561febe29c7757c6d222bfebff50e8baa868991e57b87d1` |

The hash of the 28k file matches the artifact published by the dataset author.
The artifact was added in upstream commit
[`255c10d`](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets/commit/255c10d9cd3f9a3e3c128323efb1888e26077dd7).
The local identity of the 4k file is frozen by its hash above; a standalone
upstream revision for that filename has not been independently identified.

## Training Dataset

`aa_dataset-tickets-multi-lang-5-2-50-version.csv` is the only dataset used to
fit preprocessors and models and to select algorithms or hyperparameters.

- Inputs used by the models: `subject`, `body`, and `language` for
  language-aware normalization.
- Targets: `queue` and `priority`.
- Languages: English `16,338`; German `12,249`.
- Target cardinality: 10 queue classes and 3 priority classes.
- Missing model inputs: `3,838` missing subjects and no missing bodies. The
  preprocessing pipeline treats missing text as an empty string.
- Internal evaluation: shared 5-fold stratified cross-validation using the
  combined `queue` and `priority` labels.

Cross-validation results are in-distribution estimates for this fixed synthetic
dataset. They are used for model selection and must be reported separately from
external holdout results.

## Frozen Holdout

The holdout is a deterministic view of `dataset-tickets-multi-lang3-4k.csv`:

```text
language in {"en", "de"}
```

This filter is defined from the languages represented in the training data,
not from model performance. It retains `2,239` rows: `1,391` English and `848`
German tickets. Spanish (`812`), French (`476`), and Portuguese (`473`) rows
are outside the supported training scope and are excluded from primary holdout
metrics. They may only be reported separately as unsupported-language,
zero-shot stress tests.

The holdout has the same 10 queue labels and 3 priority labels as the training
dataset. It contains `468` missing subjects and one missing body; the existing
preprocessing rules handle both without fitting any holdout-derived state.

### Holdout rules

1. Use only the English and German rows for primary holdout metrics.
2. Do not fit preprocessing or model parameters on holdout rows.
3. Select and freeze candidates using only training-data cross-validation.
4. Report holdout results separately from cross-validation results.

If holdout results influence another model change, the dataset becomes
validation data and a new untouched test set is required for a strict final
estimate. Because both datasets are synthetic, the holdout measures transfer to
a separate synthetic generation, not production-ticket performance.
