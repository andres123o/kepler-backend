"""
Fase 3: Validación temporal — Walk-Forward.
Genera splits (train_idx, test_idx) respetando orden temporal.
"""

import logging
from typing import Generator

import numpy as np

from ml_pipeline.config import INITIAL_TRAIN_WEEKS

logger = logging.getLogger("kepler.ml.validation")


def walk_forward_splits(
    n_samples: int,
    initial_train_size: int,
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generador de (train_indices, test_indices) en orden temporal.
    Entrena 1..initial_train_size, predice initial_train_size+1;
    luego 1..initial_train_size+1, predice initial_train_size+2; etc.
    """
    if initial_train_size < 10:
        raise ValueError("initial_train_size debe ser al menos 10")
    if n_samples <= initial_train_size:
        logger.warning("walk_forward_splits: n_samples (%d) <= initial_train_size (%d); sin folds.", n_samples, initial_train_size)
        return

    n_folds = 0
    for test_start in range(initial_train_size, n_samples):
        train_idx = np.arange(0, test_start)
        test_idx = np.array([test_start])
        n_folds += 1
        logger.debug("Walk-forward fold %d: train size=%d, test index=%d.", n_folds, len(train_idx), test_start)
        yield train_idx, test_idx

    logger.info("Walk-forward: %d folds (train inicial=%d, último test=%d).", n_folds, initial_train_size, n_samples - 1)
