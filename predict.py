"""Predict known drug/cell entities with a saved fold model."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from clhgsyn.model import CLHGSyn
from clhgsyn.train import predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-dir', required=True, help='e.g. results/OncologyScreen')
    parser.add_argument('--fold', type=int, choices=range(1, 6), default=1)
    parser.add_argument('--input', required=True, help='CSV columns: drug_a,drug_b,cell (DrugBank IDs)')
    parser.add_argument('--output', default='predictions.csv')
    args = parser.parse_args()
    torch.set_num_threads(4)
    directory = Path(args.result_dir)
    mapping = json.loads((directory / 'node_mapping.json').read_text(encoding='utf-8'))
    arrays = np.load(directory / 'protein_neighbors.npz', allow_pickle=False)
    data = SimpleNamespace(drug_names=mapping['drugs'], cell_names=mapping['cells'],
                           protein_names=mapping['proteins'], neighbors=arrays['neighbors'], neighbor_mask=arrays['mask'])
    checkpoint = torch.load(directory / f'fold_{args.fold}' / 'best.pt', map_location='cpu', weights_only=False)
    config = checkpoint['config']
    model = CLHGSyn(data, checkpoint['model_state_dict']['operator'], config['dim'],
                   config['layers'], config['dropout'], config['mixing'])
    model.load_state_dict(checkpoint['model_state_dict'])
    frame = pd.read_csv(args.input)
    dm = {name: i for i, name in enumerate(data.drug_names)}
    cm = {name: i + len(dm) for i, name in enumerate(data.cell_names)}
    encoded = pd.DataFrame({'a': frame.drug_a.map(dm), 'b': frame.drug_b.map(dm), 'c': frame.cell.map(cm)})
    if encoded.isna().any().any():
        raise ValueError('Unknown or missing entity IDs: this embedding model predicts only known entities')
    if frame.empty:
        raise ValueError('Input contains no samples')
    frame['probability'] = predict(model, torch.tensor(encoded.to_numpy(), dtype=torch.long), config['batch_size'])
    frame['prediction'] = (frame.probability >= 0.5).astype(int)
    frame.to_csv(args.output, index=False)
    print(f'Saved {len(frame)} predictions to {args.output}')


if __name__ == '__main__':
    main()
