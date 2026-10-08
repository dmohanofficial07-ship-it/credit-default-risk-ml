# Accuracy experiment: financial features and grouped validation

This experiment tests whether financial-history features and a small parameter
search improve accuracy. It writes separate outputs under
`artifacts/accuracy_experiment/`; the original model and its recall-focused
threshold remain under `artifacts/`. Both use the same pinned dependencies.

## See the output

From the Project_Fav root, display the saved results:

```powershell
.\.venv\Scripts\python.exe .\ml-projects\credit-default\report_results.py --experiment
```

Open [the comparison chart](artifacts/accuracy_experiment/comparison.png) in VS Code.
The chart is generated locally and is excluded from Git. To regenerate the
entire experiment, including its data download if needed:

```powershell
.\.venv\Scripts\python.exe .\ml-projects\credit-default\improve_accuracy.py
```

You can also select **Terminal > Run Task > ML: run accuracy experiment** or
**ML: show accuracy experiment results**. The latter does not train anything.

## What changed

The experiment adds 17 per-record features to the original 19 financial inputs:

- Six monthly bill-to-credit-limit ratios.
- Number of months with positive delay codes, greatest delay, and latest change
  in nonnegative delay (September minus August).
- Mean payment, payment standard deviation, and September-minus-August payment change.
- Mean bill, bill standard deviation, and September-minus-August bill change.
- September-minus-April bill change divided by credit limit.
- Six-month total payments divided by credit limit.

Ratios require a positive credit limit. Negative bills remain valid inputs.
The original status codes remain categorical inputs; for the added delay
summaries, nonpositive codes contribute zero. No target labels or statistics
from other customers are used to calculate these features.

Six fixed histogram-gradient-boosting configurations are compared: the original
reference, the same parameters with added features, slower learning, shallower
trees with and without features, and more flexible trees with stronger
regularization. This is a bounded search, not an exhaustive optimization.
Exact configurations are defined in `CONFIGS` in `improve_accuracy.py`.

## How data separation works

We retain the original 6,000 test IDs. Before training, we exclude **608**
development records whose 19 financial input values exactly match a test
profile. This leaves **23,392 development records**.

This broader definition differs from the earlier audit of all 23 source
predictors, which included demographics. Matching financial profiles do not
necessarily identify the same person; grouping is a conservative way to keep
the same model inputs from appearing on both sides of an evaluation split.

Within development data, three-fold `StratifiedGroupKFold` keeps identical
financial profiles together while attempting to preserve class proportions.
Preprocessing and model fitting happen separately inside each fold.

1. Select the configuration with the highest mean validation accuracy at cutoff 0.50.
2. Select an accuracy-maximizing cutoff using that configuration's out-of-fold
   development probabilities. Each record's probability came from a model that
   did not train on its group.
3. Fit the selected configuration on all 23,392 development records.
4. Fit the reference configuration on those same records for comparison.
5. Evaluate the fixed models and cutoff on the same 6,000 test records.

No test labels select a configuration or threshold. However, we have already
looked at this test set in earlier work; it is a reused benchmark, not newly
collected external validation. The winning cross-validation score and the
tuned out-of-fold score are selection estimates, not independent performance
estimates. A nested evaluation would be needed to assess the whole selection
process without this selection optimism.

## Recorded results

The chosen configuration was `history_shallow`: 160 boosting iterations,
7 maximum leaves, learning rate 0.05, L2 regularization 3.0, and the added
financial features. Mean three-fold development accuracy was **82.58%**.
The accuracy threshold selected from development probabilities was about **0.506**.

| Same test records | Accuracy | Precision | Recall |
| --- | ---: | ---: | ---: |
| Reference, threshold 0.50 | 81.70% | 65.54% | 36.40% |
| Selected model, threshold 0.50 | 81.82% | 65.99% | 36.70% |
| Selected model, development-selected threshold | 81.87% | 66.35% | 36.55% |

The final model correctly classifies **4,912 of 6,000** records. The equally
trained reference gets **4,902** correct, a gain of ten predictions or about
**0.17 percentage points**. This small difference has not been established as
statistically reliable.

The original project's model scored **81.83%** (4,910 correct). Comparing that
with 81.87% gives only two additional correct predictions. Its training set
was different: 18,000 records, without the new profile exclusion. Therefore,
that comparison alone cannot isolate the benefit of new features or tuning.

The raw-feature shallow model was almost tied in cross-validation at **82.57%**.
The evidence does not establish that the engineered features provide a robust
benefit. We retain the original model rather than silently replacing it.
`predict.py` continues to score using that original model; the experiment model
is saved separately for inspection and further research.

## Saved artifacts and checks

Under `artifacts/accuracy_experiment/`:

- `summary.txt`: readable results without running Python.
- `metrics.json`: configurations, fold scores, test metrics, provenance, and limitations.
- `comparison.png`: cross-validation comparison and test accuracy/precision/recall.
- `model.joblib`: selected fitted model and cutoff.
- `split_ids.json`: development, excluded, test, and fold membership.
- `development_predictions.csv`: out-of-fold probabilities used to choose the cutoff.
- `test_predictions.csv`: reference and selected model probabilities on test records.

Run the complete test suite after generating both original and experiment artifacts:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s .\ml-projects\credit-default -p "test_*.py" -v
```

Checks cover feature calculations, preservation of negative bills, independence
between records, group separation, out-of-fold coverage, threshold correctness
(including tied scores), saved-model reproduction, and the original workflow.
