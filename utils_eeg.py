import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt, welch
from config import MAX_CHANNELS, FS

def find_eeg_base():
    candidates = []
    for p in Path.cwd().rglob("*"):
        if p.is_dir() and p.name.lower() == "eeg_data":
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError("EEG_data folder not found.")

    candidates = sorted(candidates, key=lambda x: len(str(x)), reverse=True)

    for c in candidates:
        if (c/"AD").exists() and (c/"Healthy").exists():
            print("[DEBUG] EEG Base:", c)
            return c

    return candidates[0]

BASE_DIR = find_eeg_base()


def safe_load_txt(path):
    try:
        return np.loadtxt(path).flatten()
    except:
        return None


def load_subject_raw(group, state, paciente):
    folder = BASE_DIR / group / state / paciente
    if not folder.exists():
        return None

    txt_files = sorted(folder.glob("*.txt"))
    if not txt_files:
        return None

    data = []
    for t in txt_files:
        x = safe_load_txt(t)
        if x is not None:
            data.append(x)

    if not data:
        return None

    L = min(map(len, data))
    data = [d[:L] for d in data]

    while len(data) < MAX_CHANNELS:
        data.append(np.zeros(L))

    return np.vstack(data[:MAX_CHANNELS])


def band_filter(x):
    b, a = butter(4, [0.5/(FS/2), 45/(FS/2)], btype="band")
    return filtfilt(b, a, x)


def preprocess(X):
    out = []
    for ch in X:
        y = ch - np.mean(ch)
        y = band_filter(y)
        y = y / (np.std(y)+1e-12)
        out.append(y)
    return np.vstack(out)
