from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EcologicalOOFCalibrator:
    coefficients: dict
    center: float
    scale: float
    minimum_support: int = 20

    @classmethod
    def fit(cls, ecology, residual, oof_prediction, minimum_support=20):
        ecology = np.asarray(ecology).astype(str)
        residual = np.asarray(residual, dtype=np.float64)
        prediction = np.asarray(oof_prediction, dtype=np.float64)
        assert len(ecology) == len(residual) == len(prediction)
        assert np.isfinite(residual).all() and np.isfinite(prediction).all()
        center = float(prediction.mean())
        scale = max(float(prediction.std()), 1e-12)
        z = (prediction-center)/scale
        global_design = np.column_stack((np.ones(len(z)), z))
        global_coefficient = np.linalg.solve(global_design.T@global_design+np.diag([10., 20.]),
                                             global_design.T@residual)
        coefficients = {'__GLOBAL__': global_coefficient}
        for category in np.unique(ecology):
            selected = ecology == category
            if selected.sum() < minimum_support:
                continue
            design = np.column_stack((np.ones(selected.sum()), z[selected]))
            coefficients[category] = np.linalg.solve(design.T@design+np.diag([10., 20.]),
                                                     design.T@residual[selected])
        return cls(coefficients, center, scale, int(minimum_support))

    def transform(self, ecology, compressed_prediction):
        ecology = np.asarray(ecology).astype(str)
        prediction = np.asarray(compressed_prediction, dtype=np.float64)
        assert len(ecology) == len(prediction) and np.isfinite(prediction).all()
        z = (prediction-self.center)/self.scale
        coefficients = np.array([self.coefficients.get(value, self.coefficients['__GLOBAL__']) for value in ecology])
        offset = coefficients[:, 0]+coefficients[:, 1]*z
        result = prediction+offset
        assert np.isfinite(result).all()
        return result, offset

    def shuffled_control(self, ecology, compressed_prediction, seed):
        shuffled = np.asarray(ecology).astype(str).copy()
        np.random.default_rng(seed).shuffle(shuffled)
        return self.transform(shuffled, compressed_prediction)
