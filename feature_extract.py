import numpy as np
from scipy.signal import welch
from scipy.stats import entropy

FS = 256
BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45)
}

MAX_CHANNELS = 10


def extract_features(X):
    feats = []

    # ============================
    # 1. Relative Band Powers
    rel = []
    for ch in X:
        f, Pxx = welch(ch, FS)
        total = np.trapz(Pxx[(f >= 0.5) & (f <= 45)],
                        f[(f >= 0.5) & (f <= 45)]) + 1e-9

        vals = []
        for lo, hi in BANDS.values():
            idx = (f >= lo) & (f <= hi)
            p = np.trapz(Pxx[idx], f[idx]) if np.any(idx) else 0
            vals.append(p / total)

        rel.append(vals)

    rel = np.array(rel)
    feats.extend(rel.mean(axis=0))
    feats.extend(rel.std(axis=0))

    # ============================
    # 2. Entropy + Hjorth Params
    ent, mob, comp = [], [], []

    for ch in X:
        f, Pxx = welch(ch, FS)
        psd = Pxx[(f >= 0.5) & (f <= 45)]
        s = psd.sum()
        en = entropy(psd / s, base=2) if s > 0 else 0
        ent.append(en)

        dx = np.diff(ch)
        v0 = np.var(ch)
        v1 = np.var(dx)

        m = np.sqrt(v1 / (v0 + 1e-12))
        c = np.sqrt(np.var(np.diff(dx)) / (v1 + 1e-12)) / (m + 1e-12)

        mob.append(m)
        comp.append(c)

    feats.extend([np.mean(ent), np.std(ent)])
    feats.extend([np.mean(mob), np.std(mob)])
    feats.extend([np.mean(comp), np.std(comp)])

    # ============================
    # 3. Connectivity (Correlation)
    C = np.corrcoef(X)
    iu = np.triu_indices_from(C, 1)
    feats.append(np.mean(np.abs(C[iu])))

    return np.array(feats)
