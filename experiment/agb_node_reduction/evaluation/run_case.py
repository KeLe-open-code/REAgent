import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from evaluation.dataset_schema import sha256
from evaluation.training import run_external_model, run_project_case
from evaluation.versioning import evaluation_code_fingerprint


def main(args):
    assert args.output and not args.output.exists(), 'Every case requires a new output directory'
    with np.load(args.data, allow_pickle=False) as saved:
        data = {key: saved[key].copy() for key in saved.files}
    selection_path = args.data.parent/'selection.npz'
    if selection_path.exists():
        with np.load(selection_path, allow_pickle=False) as saved:
            data['selection'] = {key: saved[key].copy() for key in saved.files}
    config = json.loads(args.config.read_text(encoding='utf-8'))
    assert int(data['n_train']) > 0 and len(data['y']) > int(data['n_train'])
    assert args.fold == config['fold'] and args.dataset == config['dataset']
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if args.external_model:
        summary = run_external_model(args.external_model, data, config, args.seed, args.output,
                                     device, args.epochs)
    else:
        summary = run_project_case(data, config, args.arm, args.seed, args.output,
                                   device, args.epochs, args.agent_episodes)
    execution = {'status': 'COMPLETE', 'dataset': args.dataset, 'fold': args.fold, 'seed': args.seed,
                 'arm': args.external_model or args.arm, 'data_sha256': sha256(args.data),
                 'config_sha256': sha256(args.config), 'runner_sha256': evaluation_code_fingerprint(),
                 'epochs_override': args.epochs, 'agent_episodes': args.agent_episodes,
                 'device': device, 'final_holdout_access': False}
    (args.output/'execution.json').write_text(json.dumps(execution, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--fold', type=int, required=True, choices=(1, 2, 3))
    parser.add_argument('--seed', type=int, required=True, choices=(0, 1000, 2000))
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--arm', default='A7', choices=('A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7'))
    parser.add_argument('--external-model', choices=('linear_regression', 'random_forest', 'xgboost', 'mlp', 'gcn', 'gat', 'semi_arid_transformer'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'))
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--agent-episodes', type=int, default=4)
    main(parser.parse_args())
