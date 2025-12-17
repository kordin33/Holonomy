"""
test_utils.py - Wspólne utilities dla testów (STABILNE CV!)

Używaj tych funkcji zamiast pisać własne train_test_split/GridSearchCV!
"""

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score
from typing import Dict, Any

# STAŁE - używaj zawsze te same!
RANDOM_STATE = 42
N_SPLITS = 5
TEST_SIZE = 0.3


def get_stable_cv():
    """Zwraca stabilny CV z seed."""
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def get_stable_pipeline():
    """Zwraca stabilny pipeline."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, random_state=RANDOM_STATE))
    ])


def get_param_grid():
    """Zwraca standardowy param_grid."""
    return {
        'svm__C': [0.1, 0.3, 1, 3, 10],
        'svm__gamma': ['scale', 0.001, 0.01, 0.1]
    }


def test_with_stable_pipeline(features: np.ndarray, labels: np.ndarray, name: str) -> Dict[str, Any]:
    """
    Testuje cechy z STABILNYM pipeline (deterministyczne wyniki).
    
    UŻYWAJ TEJ FUNKCJI zamiast pisać własną!
    """
    # Handle NaN/Inf
    if np.isnan(features).any() or np.isinf(features).any():
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # STABILNY split
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, 
        stratify=labels
    )
    
    # STABILNY pipeline
    pipe = get_stable_pipeline()
    
    # STABILNY CV
    cv = get_stable_cv()
    
    # GridSearch z n_jobs=1 dla pełnej powtarzalności
    grid = GridSearchCV(
        pipe, 
        get_param_grid(), 
        cv=cv, 
        scoring='roc_auc', 
        n_jobs=1  # n_jobs=1 dla deterministyczności!
    )
    grid.fit(X_train, y_train)
    
    # Predict
    y_prob = grid.best_estimator_.predict_proba(X_test)[:, 1]
    y_pred = grid.best_estimator_.predict(X_test)
    
    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    
    return {
        'name': name,
        'auc': auc,
        'acc': acc,
        'shape': features.shape,
        'best_params': grid.best_params_,
        'cv_mean': grid.cv_results_['mean_test_score'][grid.best_index_],
        'cv_std': grid.cv_results_['std_test_score'][grid.best_index_],
    }


def verify_reproducibility(features: np.ndarray, labels: np.ndarray, n_runs: int = 3) -> bool:
    """Weryfikuje czy wyniki są powtarzalne."""
    results = []
    for i in range(n_runs):
        res = test_with_stable_pipeline(features, labels, f"run_{i}")
        results.append(res['auc'])
    
    # Sprawdź czy wszystkie identyczne
    is_stable = all(abs(r - results[0]) < 1e-6 for r in results)
    
    if is_stable:
        print(f"  ✅ Wyniki stabilne: AUC = {results[0]:.4f} (identyczne w {n_runs} runach)")
    else:
        print(f"  ❌ Wyniki niestabilne: {results}")
    
    return is_stable
