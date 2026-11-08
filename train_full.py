"""Train the frozen Full/A7 model; data and ecology metadata are required.

This launcher changes no model statement. It delegates to historical_case.run with arm=A7.
Example:
python train_full.py --data-root <three-fold-root> --dataset hunan --fold 1 --seed 0 --ecology <ecology.csv> --output <new-output-directory>
"""
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'experiment' / 'agb_node_reduction'))
import historical_case as full


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True,
                        help='Contains <region>/fold-01/{config.json,data/development.npz,data/preprocessor.npz}.')
    parser.add_argument('--dataset', required=True, choices=('hunan', 'hubei', 'guizhou'))
    parser.add_argument('--fold', type=int, required=True, choices=(1, 2, 3))
    parser.add_argument('--seed', type=int, required=True, choices=(0, 1000, 2000))
    parser.add_argument('--ecology', type=Path, required=True,
                        help='CSV with sample_id and species columns covering this fold.')
    parser.add_argument('--output', type=Path, required=True, help='New output directory; never overwritten.')
    args = parser.parse_args()
    if not args.data_root.is_dir():
        parser.error('--data-root does not exist or is not a directory')
    if not args.ecology.is_file():
        parser.error('--ecology does not exist or is not a CSV file')
    if args.output.exists():
        parser.error('--output already exists; select a new directory')
    full.baseline.r.SOURCE = args.data_root.resolve()
    full.run(argparse.Namespace(dataset=args.dataset, fold=args.fold, seed=args.seed,
                                arm='A7', ecology=args.ecology.resolve(),
                                output=args.output.resolve(), small=False))


if __name__ == '__main__':
    main()
