import numpy as np
from utils_eeg import load_subject_raw
from utils_eeg import preprocess
from features import extract_features
from utils_eeg import BASE_DIR

def build_dataset(state):
    Xs, ys = [], []

    for group, label in [("AD", 1), ("Healthy", 0)]:
        folder = BASE_DIR / group / state
        if not folder.exists():
            continue

        for subj in folder.glob("*"):
            if not subj.is_dir():
                continue

            raw = load_subject_raw(group, state, subj.name)
            if raw is None:
                continue

            Xp = preprocess(raw)
            fv = extract_features(Xp)

            Xs.append(fv)
            ys.append(label)

    return np.vstack(Xs), np.array(ys)
