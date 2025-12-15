import numpy as np
from pathlib import Path

from preprocessing import load_subject_raw, preprocess
from features import extract_features
from preprocessing import BASE_DIR

# DATASET GENERATION
def build_dataset(state):
    """
    Builds full dataset for a given state ('Eyes_open' or 'Eyes_closed').

    Returns:
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of 0/1 labels (0 = Healthy, 1 = AD)
    """

    Xs = []
    ys = []

    # Loop through both classes
    for group, label in [("AD", 1), ("Healthy", 0)]:

        folder = BASE_DIR / group / state
        if not folder.exists():
            continue

        # Each patient folder inside AD/state or Healthy/state
        for subj in folder.glob("*"):
            if not subj.is_dir():
                continue

            raw = load_subject_raw(group, state, subj.name)
            if raw is None:
                continue

            # Preprocess and extract features
            Xp = preprocess(raw)
            fv = extract_features(Xp)

            Xs.append(fv)
            ys.append(label)

    return np.vstack(Xs), np.array(ys)
