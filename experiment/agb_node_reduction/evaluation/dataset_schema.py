import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def parse_dms(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    match = re.fullmatch(r'(\d+(?:\.\d+)?)°\s*(\d+(?:\.\d+)?)?[′\']?\s*(\d+(?:\.\d+)?)?[″\"]?\s*([NSEW])', text, re.I)
    if not match:
        return float(text)
    degrees, minutes, seconds, direction = match.groups()
    result = float(degrees)+float(minutes or 0)/60+float(seconds or 0)/3600
    return -result if direction.upper() in ('S', 'W') else result


def read_table(path, sheet_name=0):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == '.csv':
        return pd.read_csv(path)
    if suffix in ('.xlsx', '.xls'):
        return pd.read_excel(path, sheet_name=sheet_name)
    if suffix == '.parquet':
        return pd.read_parquet(path)
    raise ValueError(f'Unsupported tabular input: {path}')


def audit_frame(frame, mapping, source_path):
    required = ('sample_id', 'longitude', 'latitude', 'agb', 'ecology', 'features')
    assert set(mapping) == set(required)
    assert set(mapping['features']).isdisjoint({mapping[key] for key in required[:-1]})
    missing_columns = [mapping[key] for key in required[:-1] if mapping[key] not in frame]
    missing_columns += [value for value in mapping['features'] if value not in frame]
    assert not missing_columns, f'Missing declared columns: {missing_columns}'
    sample_ids = frame[mapping['sample_id']].astype(str)
    longitude = pd.to_numeric(frame[mapping['longitude']], errors='coerce').to_numpy(float)
    latitude = pd.to_numeric(frame[mapping['latitude']], errors='coerce').to_numpy(float)
    agb = pd.to_numeric(frame[mapping['agb']], errors='coerce').to_numpy(float)
    ecology = frame[mapping['ecology']].astype(str)
    features = frame[mapping['features']].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    valid = np.isfinite(longitude)&np.isfinite(latitude)&np.isfinite(agb)
    valid &= (longitude >= -180)&(longitude <= 180)&(latitude >= -90)&(latitude <= 90)
    valid &= np.isfinite(features).any(axis=1)
    counts = ecology[valid].value_counts()
    report = {
        'source_path': str(source_path), 'source_sha256': sha256(source_path),
        'rows': int(len(frame)), 'valid_coordinate_agb_feature_rows': int(valid.sum()),
        'unique_sample_ids': int(sample_ids.nunique()), 'duplicate_sample_ids': int(sample_ids.duplicated().sum()),
        'duplicate_coordinate_rows': int(pd.DataFrame({'lon': longitude, 'lat': latitude}).duplicated().sum()),
        'agb_min': float(np.nanmin(agb)), 'agb_max': float(np.nanmax(agb)),
        'feature_count': len(mapping['features']),
        'missing_by_feature': {name: int(frame[name].isna().sum()) for name in mapping['features']},
        'ecology_categories': int(len(counts)), 'ecology_support': {str(k): int(v) for k, v in counts.items()},
        'categories_with_at_least_20_samples': int((counts >= 20).sum()),
        'coordinate_crs_assumed': 'EPSG:4326; must be confirmed by dataset documentation',
    }
    assert len(set(sample_ids)) == len(frame), 'sample_id must be unique before splitting'
    assert valid.all(), 'Invalid coordinates, labels or fully missing feature rows require explicit cleaning'
    assert (counts >= 2).sum() >= 2, 'Full requires at least two supported ecological categories'
    return report
