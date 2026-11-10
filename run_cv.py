"""Run both benchmarks: python run_cv.py --data-zip path/to/data.zip."""
import argparse
from pathlib import Path
import torch
from clhgsyn.data import extract_data
from clhgsyn.train import run_dataset


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='CLHGSyn five-fold reproduction')
    parser.add_argument('--data-zip', type=str)
    parser.add_argument('--data-dir', default=str(root / 'data'))
    parser.add_argument('--output-dir', default=str(root / 'results'))
    parser.add_argument('--dataset', choices=['all', 'DrugCombDB', 'OncologyScreen'], default='all')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--min-epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--min-delta', type=float, default=0.0001)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--mixing', type=float, default=0.4)
    parser.add_argument('--eta', type=float, default=0.3)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--neighbors', type=int, default=32)
    parser.add_argument('--hops', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--weight-decay', type=float, default=0.)
    parser.add_argument('--holdout', type=float, default=0.1)
    parser.add_argument('--validation', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.dim < 1 or args.hops < 1 or args.neighbors < 1 or args.epochs < 1 or args.batch_size < 1:
        parser.error('dimensions, hops, neighbors, epochs and batch size must be positive')
    if not 0 <= args.holdout < 0.5 or not 0 < args.validation < 0.5:
        parser.error('holdout must be in [0, .5), validation in (0, .5)')
    if args.temperature <= 0 or not 0 <= args.mixing <= 1:
        parser.error('temperature must be positive and mixing in [0, 1]')
    torch.set_num_threads(args.threads)
    if args.data_zip:
        extract_data(args.data_zip, args.data_dir)
    for dataset in (['OncologyScreen', 'DrugCombDB'] if args.dataset == 'all' else [args.dataset]):
        run_dataset(args, dataset)


if __name__ == '__main__':
    main()
