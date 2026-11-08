import numpy as np


def regression_metrics(target, prediction):
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    assert target.shape == prediction.shape and target.ndim == 1 and len(target) > 0
    assert np.isfinite(target).all() and np.isfinite(prediction).all()
    residual = prediction-target
    sse = float(residual@residual)
    centered = target-target.mean()
    sst = float(centered@centered)
    return {
        'n': int(len(target)),
        'RMSE': float(np.sqrt(sse/len(target))),
        'MAE': float(np.mean(np.abs(residual))),
        'R2': float(1-sse/sst) if sst > 0 else None,
        'Bias': float(np.mean(residual)),
        'SSE': sse,
    }


def compression_metrics(target, full_prediction, compressed_prediction, original_references, retained_references):
    assert 0 < retained_references <= original_references
    result = regression_metrics(target, compressed_prediction)
    full = regression_metrics(target, full_prediction)
    result.update({
        'full_RMSE': full['RMSE'],
        'delta_RMSE': result['RMSE']-full['RMSE'],
        'relative_RMSE_improvement_percent': 100*(full['RMSE']-result['RMSE'])/full['RMSE'],
        'original_references': int(original_references),
        'retained_references': int(retained_references),
        'CR_percent': 100*(1-retained_references/original_references),
        'query_count': int(len(target)),
    })
    return result


def aggregate_paired(cases):
    assert cases and all(case['query_count'] > 0 for case in cases)
    total_n = sum(case['query_count'] for case in cases)
    return {
        'cases': len(cases),
        'micro_RMSE': float(np.sqrt(sum(case['SSE'] for case in cases)/total_n)),
        'macro_RMSE': float(np.mean([case['RMSE'] for case in cases])),
        'macro_MAE': float(np.mean([case['MAE'] for case in cases])),
        'macro_R2': float(np.mean([case['R2'] for case in cases if case['R2'] is not None])),
        'macro_Bias': float(np.mean([case['Bias'] for case in cases])),
        'macro_CR_percent': float(np.mean([case['CR_percent'] for case in cases])),
        'macro_relative_RMSE_improvement_percent': float(np.mean([
            case['relative_RMSE_improvement_percent'] for case in cases])),
    }
