import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from dataset_schema import sha256

ROOT = Path(__file__).resolve().parents[3]
PUBLIC = ROOT/'01_数据集/public'


def spatial_groups(frame, size=0.25):
    x = np.floor(frame.longitude.to_numpy(float)/size).astype(int)
    y = np.floor(frame.latitude.to_numpy(float)/size).astype(int)
    return np.asarray([f'{size:g}_{a}_{b}' for a, b in zip(x, y)])


def finish(frame, dataset, features, metadata, groups=None):
    frame = frame.copy()
    frame['spatial_group'] = spatial_groups(frame) if groups is None else np.asarray(groups, dtype=str)
    ordered = ['sample_id', 'longitude', 'latitude', 'agb', 'ecology', 'spatial_group']+features
    frame = frame[ordered]
    directory = PUBLIC/dataset
    output = directory/'normalized.csv'
    frame.to_csv(output, index=False)
    mapping = {
        'dataset': dataset,
        'columns': {
            'sample_id': 'sample_id', 'longitude': 'longitude', 'latitude': 'latitude',
            'agb': 'agb', 'ecology': 'ecology', 'features': features,
        },
        'group': 'spatial_group',
        **metadata,
    }
    (directory/'mapping.json').write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding='utf-8')
    report = {
        'dataset': dataset, 'normalized_file': str(output), 'normalized_sha256': sha256(output),
        'rows': len(frame), 'feature_count': len(features), 'unique_sample_ids': int(frame.sample_id.nunique()),
        'spatial_groups': int(frame.spatial_group.nunique()), 'ecology_support': frame.ecology.value_counts().to_dict(),
        'missing': {column: int(frame[column].isna().sum()) for column in ordered},
        'agb_min': float(frame.agb.min()), 'agb_max': float(frame.agb.max()),
    }
    (directory/'normalization_audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    return report


def mexico():
    source = PUBLIC/'mexico/data_biomass.xlsx'
    frames = []
    for sheet in pd.ExcelFile(source).sheet_names:
        current = pd.read_excel(source, sheet_name=sheet)
        current['source_sheet'] = sheet
        frames.append(current)
    source_frame = pd.concat(frames, ignore_index=True)
    features = [column for column in source_frame if '_glcm_' in column]
    frame = pd.DataFrame({
        'sample_id': [f'mexico_{index:06d}' for index in range(len(source_frame))],
        'longitude': source_frame.Easting_Geo,
        'latitude': source_frame.Northing_Geo,
        'agb': source_frame.Biomass,
        'ecology': source_frame.Vegetation_type.astype(str),
    })
    for column in features:
        frame[column] = source_frame[column]
    return finish(frame, 'mexico', features, {
        'label_source': 'field plots and allometric equations', 'label_unit': 'Mg/ha',
        'ecology_source': 'native Vegetation_type', 'spatial_scale': 'see Plot_size and source publication',
        'coordinate_crs': 'EPSG:4326', 'measurement_year': 'see Yr_bio',
    })


def west_kalimantan():
    directory = PUBLIC/'west_kalimantan'
    source_frame = pd.read_csv(directory/'raw/agb_samples_full.csv')
    coordinates = source_frame['.geo'].map(json.loads).map(lambda value: value['coordinates'])
    frame = pd.DataFrame({
        'sample_id': source_frame['system:index'].map(lambda value: f'west_{int(value):06d}'),
        'longitude': coordinates.map(lambda value: value[0]),
        'latitude': coordinates.map(lambda value: value[1]),
        'agb': source_frame.agbd,
    })
    features = [column for column in source_frame if column not in ('system:index', 'agbd', '.geo')]
    for column in features:
        frame[column] = source_frame[column]
    height = pd.read_csv(directory/'raw/ch_samples_full.csv')
    height_coordinates = height['.geo'].map(json.loads).map(lambda value: value['coordinates'])
    height_xy = np.asarray(height_coordinates.map(lambda value: [value[0], value[1]]).tolist())
    query_xy = frame[['longitude', 'latitude']].to_numpy(float)
    distances, indices = cKDTree(height_xy).query(query_xy, k=1)
    rh98 = height.rh98.to_numpy(float)[indices]
    groups = spatial_groups(frame)
    group_medians = pd.DataFrame({'group': groups, 'rh98': rh98}).groupby('group').rh98.median()
    assert len(group_medians) == len(set(groups))
    frame['ecology'] = [('mangrove_low_rh98' if group_medians[group] < 10 else
                         'mangrove_mid_rh98' if group_medians[group] < 20 else 'mangrove_high_rh98')
                        for group in groups]
    report = finish(frame, 'west_kalimantan', features, {
        'label_source': 'GEDI L4A AGBD product; not independent field measurements', 'label_unit': 'Mg/ha',
        'ecology_source': '0.25-degree spatial-group median GEDI L2A RH98: low <10 m, mid 10–20 m, high >=20 m',
        'spatial_scale': 'approximately 25 m GEDI footprint/pixel', 'coordinate_crs': 'EPSG:4326',
        'measurement_year': 'GEDI 2019–2025; Sentinel-2 composite 2025',
    })
    report['canopy_height_groups'] = {
        'covered_groups': int(len(group_medians)), 'max_nearest_distance_degrees': float(distances.max()),
        'group_rh98_min': float(group_medians.min()), 'group_rh98_max': float(group_medians.max()),
    }
    (directory/'normalization_audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    return report


def belgium():
    source = PUBLIC/'belgium/field_data_v1.xlsx'
    plots = pd.read_excel(source, sheet_name='plot_data')
    frame = pd.DataFrame({
        'sample_id': plots.patch_ID.astype(str)+'_'+plots.plot_ID.astype(str),
        'longitude': plots.long,
        'latitude': plots.lat,
        'agb': plots[['AGB_per_ha_conv_A', 'AGB_per_ha_conv_B', 'AGB_per_ha_conv_C']].sum(axis=1),
        'ecology': plots.patch_ID.str.replace(r'_\d+$', '', regex=True),
    })
    return finish(frame, 'belgium', [], {
        'label_source': 'field plots and allometric equations; A+B+C per-ha components summed', 'label_unit': 'Mg/ha',
        'ecology_source': 'patch-prefix management/forest series labels', 'spatial_scale': '18 m plot buffer in author analysis',
        'coordinate_crs': 'EPSG:4326', 'measurement_year': '2023-08-07 to 2023-10-15',
    }, groups=plots.patch_ID)


def main(dataset):
    reports = {'mexico': mexico, 'west_kalimantan': west_kalimantan, 'belgium': belgium}
    selected = reports if dataset == 'all' else {dataset: reports[dataset]}
    for name, function in selected.items():
        report = function()
        print(json.dumps({'dataset': name, 'rows': report['rows'], 'features': report['feature_count'],
                          'ecology_support': report['ecology_support']}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=('all', 'mexico', 'west_kalimantan', 'belgium'), default='all')
    main(parser.parse_args().dataset)
