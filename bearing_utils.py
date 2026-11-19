"""
Fonctions communes aux notebooks : chargement des signaux, normalisation,
images GASF/GADF (avec cache disque) et générateur de lots Keras.
"""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

CLASS_NAMES = ["Normal", "Inner Race Fault", "Outer Race Fault"]
N_SAMPLES = 5000

# Jeu de démonstration inclus dans le dépôt (100 signaux par classe)
DEMO_DIR = Path(__file__).resolve().parent / "data" / "demo"
DEMO_FILES = {
    "Normal": "normal.csv",
    "Inner Race Fault": "inner_race_fault.csv",
    "Outer Race Fault": "outer_race_fault.csv",
}


# ─────────────────────────────────────────────────────────────
# Chargement
# ─────────────────────────────────────────────────────────────

def natural_key(name):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def dataset_available(dataset_dir):
    """Vrai si le dossier contient les 3 sous-dossiers de classes du dataset SUBF."""
    dataset_dir = Path(dataset_dir)
    return all((dataset_dir / c).is_dir() for c in CLASS_NAMES)


def _fit_length(signal, n_samples=N_SAMPLES):
    signal = np.asarray(signal, dtype="float32").ravel()
    if len(signal) < n_samples:
        return np.pad(signal, (0, n_samples - len(signal)))
    return signal[:n_samples]


def load_signals(dataset_dir=None, max_per_class=None, n_samples=N_SAMPLES):
    """
    Charge les signaux bruts.

    - Si `dataset_dir` contient le dataset complet (Normal/, Inner Race Fault/,
      Outer Race Fault/ avec un CSV par signal), il est utilisé.
    - Sinon, le jeu de démonstration de `data/demo/` est utilisé.

    Retourne X (n_signaux, n_samples), y (n_signaux,) et la source utilisée.
    """
    use_full = dataset_dir is not None and dataset_available(dataset_dir)
    X_parts, y_parts = [], []

    for label, class_name in enumerate(CLASS_NAMES):
        if use_full:
            folder = Path(dataset_dir) / class_name
            files = sorted((f for f in os.listdir(folder) if f.endswith(".csv")), key=natural_key)
            files = files[:max_per_class] if max_per_class else files
            signals = [
                _fit_length(pd.read_csv(folder / f, header=None).values, n_samples)
                for f in files
            ]
        else:
            df = pd.read_csv(DEMO_DIR / DEMO_FILES[class_name])
            rows = df.values[:max_per_class] if max_per_class else df.values
            signals = [_fit_length(r, n_samples) for r in rows]

        X_parts.append(np.stack(signals))
        y_parts.append(np.full(len(signals), label, dtype="int32"))

    source = f"dataset complet ({dataset_dir})" if use_full else "jeu de démonstration (data/demo)"
    return np.concatenate(X_parts).astype("float32"), np.concatenate(y_parts), source


# ─────────────────────────────────────────────────────────────
# Normalisation (ligne par ligne, chaque signal indépendamment)
# ─────────────────────────────────────────────────────────────

def minmax(X):
    """Min-Max [0, 1] — utilisée avant la transformation GAF."""
    mn = X.min(axis=1, keepdims=True)
    mx = X.max(axis=1, keepdims=True)
    return (X - mn) / (mx - mn + 1e-8)


def zscore(X):
    """Standardisation (moyenne 0, écart-type 1) — utilisée pour le Conv1D."""
    return (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)


# ─────────────────────────────────────────────────────────────
# Images GAF
# ─────────────────────────────────────────────────────────────

def gaf_images(X_minmax, method, image_size, cache_dir="cache", tag=""):
    """
    Transforme chaque signal en image GASF (method="summation") ou
    GADF (method="difference") de taille image_size x image_size.

    Les images sont stockées dans un fichier memmap float16 pour ne pas saturer
    la RAM (1024x1024 x 6480 signaux ≈ 13 Go par méthode).
    """
    from pyts.image import GramianAngularField

    os.makedirs(cache_dir, exist_ok=True)
    shape = (len(X_minmax), image_size, image_size, 1)
    path = Path(cache_dir) / f"{method}_{image_size}_{len(X_minmax)}{tag}.dat"

    if not path.exists():
        print(f"Génération des images {method} {image_size}x{image_size} → {path}")
        gaf = GramianAngularField(image_size=image_size, method=method)
        images = np.memmap(path, dtype="float16", mode="w+", shape=shape)
        for i in range(len(X_minmax)):
            images[i, :, :, 0] = gaf.fit_transform(X_minmax[i:i + 1])[0].astype("float16")
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(X_minmax)}")
        images.flush()
        del images
    else:
        print(f"Cache trouvé : {path}")

    return np.memmap(path, dtype="float16", mode="r", shape=shape)


def image_sequence(images, indices, labels, batch_size, shuffle, seed=0):
    """Générateur de lots qui lit les images depuis le memmap au fur et à mesure."""
    import tensorflow as tf

    class _Seq(tf.keras.utils.Sequence):
        def __init__(self):
            super().__init__()
            self.rng = np.random.default_rng(seed)
            self.order = np.arange(len(indices))
            self.on_epoch_end()

        def __len__(self):
            return int(np.ceil(len(indices) / batch_size))

        def __getitem__(self, b):
            ids = self.order[b * batch_size:(b + 1) * batch_size]
            # lecture triée = accès disque plus rapide
            rows = np.asarray(indices)[ids]
            sort = np.argsort(rows)
            X = np.asarray(images[rows[sort]], dtype="float32")
            unsort = np.argsort(sort)
            return X[unsort], np.asarray(labels)[ids]

        def on_epoch_end(self):
            if shuffle:
                self.rng.shuffle(self.order)

    return _Seq()
