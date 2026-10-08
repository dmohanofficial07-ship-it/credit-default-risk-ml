"""Run a bounded, group-aware accuracy experiment alongside the original model."""

import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'artifacts' / 'accuracy_experiment'
os.environ.setdefault('MPLCONFIGDIR', str(OUTPUT / '.matplotlib'))
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_source import FEATURES, load_data
from financial_features import DERIVED_NAMES, FinancialFeatures
from report_results import format_experiment_results
from train import score

CONFIGS = [
    dict(name='reference', history=False, iterations=120, leaves=15, rate=0.1, l2=1.0),
    dict(name='history_default', history=True, iterations=120, leaves=15, rate=0.1, l2=1.0),
    dict(name='history_slow', history=True, iterations=180, leaves=15, rate=0.05, l2=3.0),
    dict(name='history_shallow', history=True, iterations=160, leaves=7, rate=0.05, l2=3.0),
    dict(name='raw_shallow', history=False, iterations=160, leaves=7, rate=0.05, l2=3.0),
    dict(name='history_flexible', history=True, iterations=200, leaves=23, rate=0.05, l2=10.0),
]


def build_model(config):
    width = len(FEATURES) + (len(DERIVED_NAMES) if config['history'] else 0)
    preprocess = ColumnTransformer([
        ('amounts', StandardScaler(), [0, *range(7, width)]),
        ('status', OneHotEncoder(handle_unknown='ignore', sparse_output=False), list(range(1, 7))),
    ])
    steps = [FinancialFeatures()] if config['history'] else []
    return make_pipeline(*steps, preprocess, HistGradientBoostingClassifier(
        max_iter=config['iterations'], max_leaf_nodes=config['leaves'],
        learning_rate=config['rate'], l2_regularization=config['l2'],
        early_stopping=False, random_state=42,
    ))


def make_splits(X, y):
    # Preserve the original test IDs for comparison; purge matching input profiles
    # from development. Grouping uses features only, never outcome labels.
    _, groups = np.unique(X, axis=0, return_inverse=True)
    development, test = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)
    overlaps = np.isin(groups[development], groups[test])
    purged = development[overlaps]
    development = development[~overlaps]
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    folds = list(splitter.split(X[development], y[development], groups[development]))
    return development, test, purged, groups, folds


def accuracy_threshold(y, probability):
    candidates = np.unique(np.r_[0.5, probability, np.nextafter(np.max(probability), np.inf)])
    # Sort probabilities once; include tied scores together under the >= rule.
    order = np.argsort(probability, kind='stable')
    sorted_y = np.asarray(y)[order]
    prefix_positive = np.r_[0, np.cumsum(sorted_y)]
    positions = np.searchsorted(np.asarray(probability)[order], candidates, side='left')
    correct = positions - prefix_positive[positions] + prefix_positive[-1] - prefix_positive[positions]
    best = candidates[correct == correct.max()]
    return float(best[np.argmin(np.abs(best - 0.5))])


def main():
    X, y, ids, provenance = load_data()
    development, test, purged, groups, folds = make_splits(X, y)
    print(f'Development: {len(development):,} | Test: {len(test):,} | Purged overlapping profiles: {len(purged)}', flush=True)
    print('Selecting among six fixed configurations using three grouped development folds.', flush=True)
    candidates, oof = [], {}
    for config in CONFIGS:
        template = build_model(config)
        probabilities = np.empty(len(development))
        accuracies = []
        for number, (fit, valid) in enumerate(folds, 1):
            model = clone(template).fit(X[development[fit]], y[development[fit]])
            p = model.predict_proba(X[development[valid]])[:, 1]
            probabilities[valid] = p
            accuracies.append(float(accuracy_score(y[development[valid]], p >= 0.5)))
            print(f"  {config['name']} fold {number}/3: {accuracies[-1]:.2%}", flush=True)
        result = dict(config=config, fold_accuracy=accuracies,
                      cv_accuracy_mean=float(np.mean(accuracies)), cv_accuracy_std=float(np.std(accuracies)))
        candidates.append(result)
        oof[config['name']] = probabilities
    # Finalize all choices before evaluating either model on the test outcomes.
    winner = max(candidates, key=lambda item: item['cv_accuracy_mean'])
    name = winner['config']['name']
    cutoff = accuracy_threshold(y[development], oof[name])
    chosen = build_model(winner['config']).fit(X[development], y[development])
    reference = chosen if name == 'reference' else build_model(CONFIGS[0]).fit(X[development], y[development])
    p_chosen = chosen.predict_proba(X[test])[:, 1]
    p_reference = reference.predict_proba(X[test])[:, 1]
    evaluations = {
        'reference_0_5': score(y[test], p_reference, 0.5),
        'selected_0_5': score(y[test], p_chosen, 0.5),
        'selected_accuracy_threshold': score(y[test], p_chosen, cutoff),
    }
    report = dict(
        dataset=provenance, scikit_learn_version=sklearn.__version__,
        development_records=len(development), test_records=len(test), purged_records=len(purged),
        grouping='Exact matches across the 19 original financial input fields',
        selection='Highest mean 3-fold grouped development accuracy at threshold 0.5',
        threshold_selection='Highest accuracy on selected model out-of-fold development probabilities',
        selected_model=name, threshold=cutoff, candidates=candidates, test=evaluations,
        limitations=[
            'The selected CV score and tuned OOF score are model-selection estimates, not independent test scores.',
            'Test records were previously inspected in the original project; this is a reused benchmark, not new external validation.',
            'Reference and selected models use the same purged development records; the original 81.83% used fewer training records.',
        ],
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    joblib.dump(dict(model=chosen, features=FEATURES, threshold=cutoff, source_sha256=provenance['sha256']), OUTPUT / 'model.joblib')
    audit = dict(development=ids[development].tolist(), test=ids[test].tolist(), purged=ids[purged].tolist(),
                 folds=[dict(train=ids[development[fit]].tolist(), validation=ids[development[valid]].tolist()) for fit, valid in folds])
    (OUTPUT / 'split_ids.json').write_text(json.dumps(audit), encoding='utf-8')
    with (OUTPUT / 'development_predictions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['source_id', 'observed_default', 'out_of_fold_probability'])
        writer.writerows(zip(ids[development], y[development], oof[name]))
    with (OUTPUT / 'test_predictions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['source_id', 'observed_default', 'reference_probability', 'selected_probability'])
        writer.writerows(zip(ids[test], y[test], p_reference, p_chosen))
    summary = format_experiment_results(report)
    (OUTPUT / 'summary.txt').write_text(summary + '\n', encoding='utf-8')
    print('\n' + summary)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    labels = [item['config']['name'].replace('_', '\n') for item in candidates]
    axes[0].bar(labels, [100 * item['cv_accuracy_mean'] for item in candidates], color='#2866a8')
    axes[0].set(title='Development cross-validation (model selection)', ylabel='Accuracy (%)', ylim=(0, 100))
    for i, item in enumerate(candidates):
        axes[0].text(i, 100 * item['cv_accuracy_mean'] + 1, f"{item['cv_accuracy_mean']:.2%}", ha='center', fontsize=8)
    positions = np.arange(3)
    for offset, metric, color in [(-0.25, 'accuracy', '#2866a8'), (0, 'precision', '#d58428'), (0.25, 'recall', '#36966e')]:
        axes[1].bar(positions + offset, [100 * item[metric] for item in evaluations.values()], width=.25, label=metric, color=color)
    axes[1].set(xticks=positions, xticklabels=['Reference\n0.50', 'Selected\n0.50', 'Selected\ntuned cutoff'],
                title='Same 6,000 test records (reused benchmark)', ylabel='Percent', ylim=(0, 100))
    axes[1].legend()
    fig.savefig(OUTPUT / 'comparison.png', dpi=150)
    plt.close(fig)
    print(f'\nExperiment outputs: {OUTPUT}')


if __name__ == '__main__':
    main()
