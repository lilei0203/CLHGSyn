"""Independently recompute reported metrics and audit all saved CV partitions."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from clhgsyn.metrics import classification_metrics


def verify(directory):
    summary = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
    samples = pd.read_csv(directory / 'samples.csv')
    mapping = json.loads((directory / 'node_mapping.json').read_text(encoding='utf-8'))
    dm = {v: i for i, v in enumerate(mapping['drugs'])}
    cm = {v: len(dm) + i for i, v in enumerate(mapping['cells'])}
    triples = np.column_stack([samples.drug_a.map(dm), samples.drug_b.map(dm), samples.cell.map(cm)])
    collected = []
    fold_metrics = []
    for fold in range(1, 6):
        folder = directory / f'fold_{fold}'
        split = np.load(folder / 'split_indices.npz', allow_pickle=False)
        keys = ['train', 'validation', 'test', 'independent']
        groups = [set(samples.iloc[split[k]].group_id) for k in keys]
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                assert groups[i].isdisjoint(groups[j])
        joined = np.concatenate([split[k] for k in keys])
        assert len(np.unique(joined)) == len(samples) == len(joined)
        expected_edges = np.unique(triples[split['train']][samples.label.to_numpy()[split['train']] == 1], axis=0)
        np.testing.assert_array_equal(np.load(folder / 'training_positive_hyperedges.npy'), expected_edges)
        predictions = pd.read_csv(folder / 'test_predictions.csv')
        np.testing.assert_array_equal(predictions.row_id, split['test'])
        assert predictions.probability.between(0, 1).all()
        np.testing.assert_array_equal(predictions.label, samples.iloc[split['test']].label)
        computed = classification_metrics(predictions.label, predictions.probability)
        stored = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))
        for metric, value in computed.items():
            np.testing.assert_allclose(value, stored[metric], atol=1e-12)
        history = pd.read_csv(folder / 'history.csv')
        assert stored['best_epoch'] in history.epoch.to_list()
        assert (folder / 'best.pt').stat().st_size > 0
        fold_metrics.append(computed)
        collected.append(predictions)
    merged = pd.concat(collected).sort_values('row_id').reset_index(drop=True)
    oof = pd.read_csv(directory / 'oof_predictions.csv')
    assert oof.row_id.is_unique and len(oof) == summary['n_development']
    pd.testing.assert_frame_equal(merged, oof, check_dtype=False)
    metrics = pd.DataFrame(fold_metrics)
    for key in metrics:
        np.testing.assert_allclose(metrics[key].mean(), summary['fold_mean'][key], atol=1e-12)
        np.testing.assert_allclose(metrics[key].std(ddof=1), summary['fold_std_ddof1'][key], atol=1e-12)
    if summary['n_independent']:
        ensemble = pd.read_csv(directory / 'independent_ensemble_predictions.csv')
        assert len(ensemble) == summary['n_independent']
        assert set(ensemble.row_id).isdisjoint(oof.row_id)
        for key, value in classification_metrics(ensemble.label, ensemble.probability).items():
            np.testing.assert_allclose(value, summary['independent_ensemble'][key], atol=1e-12)
    return {'dataset': directory.name, 'status': 'PASS', 'folds': 5,
            'checks': ['group_disjointness', 'partition_coverage', 'training_only_hyperedges',
                       'prediction_alignment', 'metric_recalculation', 'oof_coverage', 'checkpoints']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default='results')
    args = parser.parse_args()
    reports = [verify(p) for p in Path(args.results).iterdir() if p.is_dir() and (p / 'summary.json').exists()]
    assert reports, 'No completed results found'
    target = Path(args.results) / 'verification.json'
    target.write_text(json.dumps(reports, indent=2), encoding='utf-8')
    print(json.dumps(reports, indent=2))
