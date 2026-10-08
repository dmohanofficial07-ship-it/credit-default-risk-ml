# Project 2: historical credit-card default risk

**Question:** can six months of repayment history help predict whether a
credit-card client defaults the next month?

This uses actual historical credit-card records published by researchers,
downloaded directly from the University of California, Irvine's repository.
It introduces class imbalance, model
comparison, probability estimates, and the tradeoff between missed defaults
and false alarms.

## Where the data comes from

| Item | Detail |
| --- | --- |
| Official host | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) |
| Dataset | Default of Credit Card Clients |
| Creator credited by UCI | I-Cheng Yeh |
| DOI | [10.24432/C55S3H](https://doi.org/10.24432/C55S3H) |
| Population | 30,000 credit-card clients in Taiwan |
| Repayment history | April through September 2005 |
| Target | Default payment next month: 1 = yes, 0 = no |
| Currency | New Taiwan dollars (NT$) |
| License | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Related research | Yeh and Lien (2009), *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients*, Expert Systems with Applications |

Attribution: Yeh, I. (2009). Default of Credit Card Clients [Dataset].
UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H.
We select and rename fields for modeling; the downloaded CSV is preserved.

The first run downloads UCI's CSV and records its source, retrieval timestamp,
and SHA-256 checksum in `data/source.json`. Subsequent runs reuse the local
copy and verify the checksum. This detects later changes to that copy; it is
not an independently published authenticity signature.

Open `data/uci_credit_default.csv` to see actual rows. It uses `X1` through
`X23` for features and `Y` for the target; `data_source.py` maps those names to
the published field meanings. The `ID` field is a record identifier.

## Run it

**Latest experiment:** the accuracy-focused model scored **81.87%** on the
6,000-record test benchmark. The reference model fitted on the same development
records scored **81.70%**. This small difference does not establish a reliable
generalization gain. See [the experiment guide](ACCURACY_EXPERIMENT.md) for the
comparison with the original 81.83% run and the changed evaluation procedure.

Display the saved experiment output without retraining:

```powershell
.\.venv\Scripts\python.exe .\ml-projects\credit-default\report_results.py --experiment
```

In VS Code, use **Terminal > Run Task > ML: show accuracy experiment results**.
On a new checkout, complete environment setup below and run `improve_accuracy.py`
first to generate those results. The original workflow follows below.

Open **Project_Fav** in VS Code and use **Terminal > New Terminal**.
The commands below use the existing workspace Python environment.

On a new checkout, first install Python 3.13. With that version available as
`python`, run these commands from the Project_Fav root to create the environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\ml-projects\credit-default\requirements.txt
```

The dependency file includes the exact versions used for the verified run.
Downloaded data and generated outputs are excluded from Git. The first training
run downloads the dataset from UCI and recreates the outputs.

```powershell
.\.venv\Scripts\python.exe .\ml-projects\credit-default\train.py
```

This trains three models, prints validation and test results, and saves a
chart to `ml-projects/credit-default/artifacts/evaluation.png`.
You can also use **Terminal > Run Task > ML: train credit default model**.

To display the saved accuracy, precision, recall, and error counts without
retraining, run:

```powershell
.\.venv\Scripts\python.exe .\ml-projects\credit-default\report_results.py
```

The table compares the baseline, standard threshold, and recall-focused
threshold. It labels ROC-AUC separately from accuracy.

Score a real held-out example saved by training:

```powershell
.\.venv\Scripts\python.exe .\ml-projects\credit-default\predict.py
```

To experiment, copy `artifacts/example_customer.json` to `my_example.json`
inside this project, change some measurements, and run:

```powershell
.\.venv\Scripts\python.exe .\ml-projects\credit-default\predict.py --input .\ml-projects\credit-default\my_example.json
```

Each JSON field has to remain present. Your edited record is then a hypothetical
scenario, not an original UCI observation. Every training run replaces the
files under `artifacts`, so keep experiments outside that directory.

## What the model uses

There are 19 selected predictors:

- `LIMIT_BAL`: given credit amount in NT$.
- `PAY_0`, `PAY_2`, ..., `PAY_6`: repayment-status categories, September back to April.
- `BILL_AMT1` through `BILL_AMT6`: bill statements, September back to April.
- `PAY_AMT1` through `PAY_AMT6`: amounts paid, September back to April.

The unusual jump from `PAY_0` to `PAY_2` is the source's naming convention.
UCI documents -1 as paid duly and positive values as months of delay. The
download also includes -2 and 0 codes, whose meanings are not specified in
the source page. We preserve them as separate categories instead of inventing
definitions or treating the codes as a continuous scale.

Negative bill statements can occur in the source and are retained. Payment
amounts and statement balances are different quantities.

`ID`, sex, education, marital status, and age are excluded from model inputs.
The outcome label is also never included as a predictor. Excluding demographic
fields does not prove fairness; other variables can still act as proxies.

## How we evaluate

We split records into three disjoint groups, preserving default prevalence:

```text
30,000 records
  18,000 training   -> fit preprocessing and model parameters
   6,000 validation -> select the model and the probability threshold
   6,000 test       -> evaluate those fixed choices
```

This is a random split within one historical cohort, not a chronological
forecasting test. The dataset does not provide multiple outcome periods for
a proper future-period evaluation. The fixed seed makes the split repeatable.

Models:

1. **Baseline:** assign everyone the training default frequency as a probability.
2. **Logistic regression:** a weighted scoring model with standardized amounts
   and separately encoded repayment-status categories.
3. **Gradient boosting:** a sequence of small decision trees that can learn
   nonlinear patterns. It uses the same preprocessing fitted on training data.

The highest validation **average precision** selects the model. Then we choose
a threshold by maximizing validation **F2**, which emphasizes recall more than
precision. This is an educational objective, not a bank's financial loss model.
The selected model stays fitted on the training set; we do not refit it after
choosing its threshold. No test outcome is used in either selection.

## Read the results

| Measure | What it tells you |
| --- | --- |
| Accuracy | Fraction of all labels correct; misleading on its own here |
| Precision | Of records flagged as default, the fraction that actually defaulted |
| Recall | Of actual defaults, the fraction the model caught |
| Average precision (AP) | Summarizes the precision-recall tradeoff across thresholds; higher is better |
| ROC-AUC | How well the score ranks a randomly chosen defaulter above a non-defaulter; 0.5 is chance ranking |
| Brier score | Mean squared error of probability estimates; lower is better |
| Calibration curve | Whether groups assigned a given probability default at roughly that frequency |

The chart compares the standard 0.50 threshold with the validation-selected
threshold. A lower threshold flags more records, typically catching more
defaults while creating more false alarms. The bottom panels show ranking
performance and how the probability estimates compare with observed rates.

Generated outputs:

- `artifacts/evaluation.png`: four-panel results chart.
- `artifacts/metrics.json`: baseline, validation, test, and provenance details.
- `artifacts/model.joblib`: fitted model, threshold, and expected feature order.
- `artifacts/test_predictions.csv`: source ID, actual label, estimated probability, predicted label.
- `artifacts/split_ids.json`: record IDs for auditing train/validation/test separation.
- `artifacts/example_customer.json`: the first held-out test example, without its target label.

## What the experiment can establish

This is a reproducible historical prediction study. It is not an approval,
rejection, pricing, or credit-limit system. The observations come from a
specific population in 2005; performance and probabilities do not automatically
transfer to current customers, another country, or another credit product.
Customers without existing credit-card histories are also outside this setup.
The dataset does not provide a full economic loss/recovery model.

The goal is to learn how to measure prediction quality and examine errors.
The source dataset is widely used; our threshold and calibration investigation
is the project angle, rather than a claim of a new research discovery.

## Check the project

After training:

```powershell
.\.venv\Scripts\python.exe .\ml-projects\credit-default\test_project.py
```

Tests verify disjoint split IDs, training-only preprocessing, that the saved
model reproduces saved test predictions, and that malformed inputs fail clearly.
To restore dependencies in a new environment, use the installation commands
under "Run it" above.

Your first experiment: open the evaluation chart and compare **missed defaults**
and **false alarms** at the two thresholds. Explain which error each threshold
reduces, and what it gives up in return.
