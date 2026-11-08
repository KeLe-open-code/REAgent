import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def collect(inputs):
    prediction_rows, compression_rows, ablation_rows = [], [], []
    for root in inputs:
        for summary_path in Path(root).rglob('case_summary.json'):
            summary = json.loads(summary_path.read_text(encoding='utf-8'))
            metrics = summary.get('metrics', {})
            execution_path = summary_path.parent/'execution.json'
            execution = json.loads(execution_path.read_text(encoding='utf-8')) if execution_path.exists() else {}
            row = {'path': str(summary_path.parent), 'dataset': execution.get('dataset'),
                   'fold': execution.get('fold'), 'seed': summary.get('seed', execution.get('seed')),
                   'model': summary.get('model'), 'method': summary.get('method'), 'arm': summary.get('arm'),
                   **{key: metrics.get(key) for key in ('n', 'RMSE', 'MAE', 'R2', 'Bias', 'CR_percent',
                       'full_RMSE', 'delta_RMSE', 'relative_RMSE_improvement_percent', 'query_count')}}
            if summary.get('arm') in ('A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7'):
                ablation_rows.append(row)
            elif summary.get('method'):
                compression_rows.append(row)
            else:
                prediction_rows.append(row)
    return prediction_rows, compression_rows, ablation_rows


def ablation_contributions(rows):
    grouped = {}
    for row in rows:
        key = (row['dataset'], row['fold'], row['seed'])
        assert row['arm'] not in grouped.setdefault(key, {}), f'Duplicate ablation arm: {key} {row["arm"]}'
        grouped[key][row['arm']] = row
    output = []
    for key, arms in grouped.items():
        if set(arms) != {f'A{index}' for index in range(8)}:
            continue
        rmse = {arm: values['RMSE'] for arm, values in arms.items()}
        factors = {
            'RP': [(('A1', 'A0')), (('A4', 'A2')), (('A5', 'A3')), (('A7', 'A6'))],
            'Agent': [(('A2', 'A0')), (('A4', 'A1')), (('A6', 'A3')), (('A7', 'A5'))],
            'ECRC': [(('A3', 'A0')), (('A5', 'A1')), (('A6', 'A2')), (('A7', 'A4'))],
        }
        for factor, pairs in factors.items():
            marginal = [rmse[with_factor]-rmse[without] for with_factor, without in pairs]
            output.append({'case': key[0], 'fold': key[1], 'seed': key[2], 'factor': factor,
                           'mean_marginal_RMSE': float(np.mean(marginal)),
                           'utility_negative_RMSE': float(-np.mean(marginal)),
                           'conditional_marginals': marginal})
    return output


def main(inputs, output):
    output.mkdir(parents=True, exist_ok=False)
    predictions, compression, ablation = collect(inputs)
    for name, rows in (('prediction_metrics.csv', predictions), ('compression_metrics.csv', compression),
                       ('ablation_metrics.csv', ablation)):
        if rows:
            pd.DataFrame(rows).to_csv(output/name, index=False)
    contributions = ablation_contributions(ablation)
    if contributions:
        pd.DataFrame(contributions).to_csv(output/'ablation_contributions.csv', index=False)
    manifest = {'prediction_cases': len(predictions), 'compression_cases': len(compression),
                'ablation_cases': len(ablation), 'complete_ablation_cases': len(contributions)//3 if contributions else 0,
                'input_roots': [str(path) for path in inputs],
                'input_summary_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                         for root in inputs for path in Path(root).rglob('case_summary.json')}}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({key: value for key, value in manifest.items() if key != 'input_summary_sha256'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.inputs, args.output)
