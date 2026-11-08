import json
from pathlib import Path
import sys

import numpy as np

import gpu_exact_case


def main():
    config_path = Path(sys.argv[sys.argv.index('--config')+1])
    output = Path(sys.argv[sys.argv.index('--output')+1])
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if 'dataset' not in config:
        config['dataset'] = config['region']
    normalized = output.with_suffix('.config.json')
    normalized.parent.mkdir(parents=True, exist_ok=True)
    normalized.write_text(json.dumps(config, indent=2), encoding='utf-8')
    sys.argv[sys.argv.index('--config')+1] = str(normalized)
    data_path = Path(sys.argv[sys.argv.index('--data')+1])
    with np.load(data_path, allow_pickle=False) as saved:
        if 'ecology' not in saved.files:
            data = {key: saved[key].copy() for key in saved.files}
            data['ecology'] = np.full(len(data['sample_ids']), 'not_used_by_external_model')
            normalized_data = output.with_suffix('.data.npz')
            np.savez_compressed(normalized_data, **data)
            sys.argv[sys.argv.index('--data')+1] = str(normalized_data)
    gpu_exact_case.main()


if __name__ == '__main__':
    main()
