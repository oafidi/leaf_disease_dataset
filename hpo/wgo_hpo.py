"""
Hyperparameter Optimisation (HPO) for EfficientNetB0 fine-tuning on the
Apple Leaf Disease dataset using Grey Wolf Optimizer (GWO).

Adapted from the PSO version. The full infrastructure (cache, checkpointing,
logging, visualisations) is preserved unchanged; only the optimisation core
is replaced.

GWO — biologically inspired metaheuristic (Mirjalili et al., 2014)
───────────────────────────────────────────────────────────────────
The pack hierarchy mimics grey wolves: α (best), β (2nd best), δ (3rd best),
ω (remaining wolves). Every ω wolf updates its position guided by the three
leaders.

Update equations (for wolf i, dimension d, iteration t)
────────────────────────────────────────────────────────
For each leader ℓ ∈ {α, β, δ}:
  A_ℓ = 2·a·r1_ℓ - a              (r1 ~ U(0,1) per dimension)
  C_ℓ = 2·r2_ℓ                    (r2 ~ U(0,1) per dimension)
  D_ℓ = |C_ℓ · x_ℓ - x_i|        (distance to leader)
  X_ℓ = x_ℓ - A_ℓ · D_ℓ          (proposed step toward leader)

  x_i(t+1) = (X_α + X_β + X_δ) / 3   (average of three leader guidance)

  a decays linearly from 2 → 0 over N_GENERATIONS iterations:
      a(t) = 2 - 2·(t / N_GENERATIONS)

When |A| < 1  → wolves converge on the prey  (exploitation)
When |A| > 1  → wolves diverge to explore    (exploration)

Fault-tolerance design
──────────────────────
Checkpoint is written atomically (tmp → rename) after every single wolf
evaluation. The checkpoint stores:
  • positions        : current location of every wolf
  • fitnesses        : f(x_i) at the current generation
  • alpha/beta/delta : the three leaders (position + fitness)
  • generation       : which generation we are in
  • wolf_index       : how many wolves were processed this generation
  • rng_state        : numpy rng bit-generator state for exact reproducibility
  • evaluated_hashes : set of already-evaluated config hashes
  • trial_counter    : global trial counter
  • best_val_acc / best_hyperparams : global best

On resume the script locates the exact wolf it stopped at inside the current
generation and continues without re-evaluating any previous wolf.
"""

import os
import json
import csv
import time
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import signal
import sys


# ─── CONFIGURATION ────────────────────────────────────────────────────────────

TRAIN_DIR   = "results_coreset_selection/coresets/hdbscan_kmeans"
VAL_DIR     = "apple_dataset/val"
TEST_DIR    = "apple_dataset/test"

CLASS_NAMES = [
    "apple_frogeye_leaf_spot", "apple_leaf_healthy",      "apple_mosaic_leaf",
    "apple_powdery_mildew_leaf", "apple_rust_leaf",       "apple_scab_leaf",
]

IMG_SIZE    = (224, 224)
NUM_CLASSES = len(CLASS_NAMES)

EPOCHS      = 40
PATIENCE    = 10

SEEDS           = [0, 1, 2, 3, 4]
MAX_EVALUATIONS = 100
POP_SIZE        = 10          # pack size (number of wolves including leaders)
N_GENERATIONS   = 10          # number of hunting iterations after initialisation

RESULTS_DIR = Path("results_gwo")

EXTERNAL_SUMMARY_CSVS = [
    "results_ga/summary.csv",
    "results_de/summary.csv",
    "results_pso/summary.csv",
]


# ─── DISCRETE SEARCH SPACE ────────────────────────────────────────────────────

SEARCH_SPACE = {
    "freezing_ratio": [0.70, 0.80, 0.85, 0.90, 0.95, 0.99],
    "learning_rate":  [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
    "dropout_rate":   [0.0, 0.2, 0.3, 0.4, 0.5, 0.6],
    "l2_reg":         [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1],
    "optimizer":      ["adam", "adamw", "sgd"],
    "batch_size":     [8, 16, 32],
}

HP_KEYS   = list(SEARCH_SPACE.keys())
HP_VALUES = [SEARCH_SPACE[k] for k in HP_KEYS]
DIM       = len(HP_KEYS)

# Upper bound per dimension (exclusive); wolves live in [0, BOUNDS).
BOUNDS    = np.array([len(v) for v in HP_VALUES], dtype=float)


# ─── GRACEFUL SHUTDOWN ────────────────────────────────────────────────────────

_shutdown_requested = False

def _signal_handler(sig, frame):
    global _shutdown_requested
    print("\n  [Signal] Ctrl+C / SIGTERM detected — finishing current trial "
          "then saving checkpoint...")
    _shutdown_requested = True

signal.signal(signal.SIGINT,  _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ─── VECTOR ↔ HYPERPARAMETER HELPERS ─────────────────────────────────────────

def vec_to_indices(vec: np.ndarray) -> np.ndarray:
    """Continuous vector → integer index vector (always within bounds)."""
    clipped = np.clip(vec, 0.0, BOUNDS - 1e-9)
    return np.floor(clipped).astype(int)


def indices_to_dict(idx: np.ndarray) -> dict:
    """Integer index vector → hyperparameter dict."""
    return {k: HP_VALUES[i][idx[i]] for i, k in enumerate(HP_KEYS)}


def vec_to_dict(vec: np.ndarray) -> dict:
    return indices_to_dict(vec_to_indices(vec))


def config_hash(hp_dict: dict) -> str:
    s = json.dumps(hp_dict, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:10]


def random_vector(rng_np: np.random.Generator) -> np.ndarray:
    """Uniform random position: gene i ~ U(0, BOUNDS[i])."""
    return rng_np.uniform(0.0, BOUNDS)


# ─── GWO CORE OPERATOR ────────────────────────────────────────────────────────

def gwo_compute_a(generation: int) -> float:
    """
    Linear decay of control parameter a from 2 → 0 over N_GENERATIONS.

    a drives the balance between exploration (|A|>1) and exploitation (|A|<1).
    At generation 0  → a = 2.0  (full exploration)
    At generation N  → a = 0.0  (pure exploitation around the prey)
    """
    return 2.0 - 2.0 * (generation / N_GENERATIONS)


def gwo_update_position(wolf_pos: np.ndarray,
                        alpha_pos: np.ndarray,
                        beta_pos:  np.ndarray,
                        delta_pos: np.ndarray,
                        a: float,
                        rng_np: np.random.Generator) -> np.ndarray:
    """
    Standard GWO position update: guided by α, β, δ leaders.

    For each leader ℓ:
        r1, r2 ~ U(0, 1)  (per-dimension random vectors)
        A_ℓ = 2a·r1 - a
        C_ℓ = 2·r2
        D_ℓ = |C_ℓ · x_ℓ - x_wolf|
        X_ℓ = x_ℓ - A_ℓ · D_ℓ

    New position = (X_α + X_β + X_δ) / 3
    Result clamped to [0, BOUNDS).
    """
    new_pos = np.zeros(DIM)

    for leader_pos in (alpha_pos, beta_pos, delta_pos):
        r1 = rng_np.random(DIM)
        r2 = rng_np.random(DIM)
        A  = 2.0 * a * r1 - a          # shape (DIM,)
        C  = 2.0 * r2                  # shape (DIM,)
        D  = np.abs(C * leader_pos - wolf_pos)
        X  = leader_pos - A * D
        new_pos += X

    new_pos /= 3.0
    return np.clip(new_pos, 0.0, BOUNDS - 1e-9)


# ─── FILE PATHS ───────────────────────────────────────────────────────────────

def seed_dir(seed: int) -> Path:
    return RESULTS_DIR / f"seed_{seed}"

def trial_path(seed: int, trial_id: int) -> Path:
    return seed_dir(seed) / "trials" / f"trial_{trial_id:03d}.json"

def checkpoint_path(seed: int) -> Path:
    return seed_dir(seed) / "gwo_checkpoint.json"

def summary_csv_path() -> Path:
    return RESULTS_DIR / "summary.csv"


# ─── DIRECTORY SETUP ──────────────────────────────────────────────────────────

def setup_dirs(seed: int):
    (seed_dir(seed) / "trials").mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ─── RESULTS CACHE ────────────────────────────────────────────────────────────

def _load_one_csv(path: Path, cache: dict, label: str) -> int:
    if not path.exists():
        print(f"  [Cache] Skipping '{path}' — file not found.")
        return 0

    added = 0
    try:
        df = pd.read_csv(path)
        available_keys = [k for k in HP_KEYS if k in df.columns]
        if len(available_keys) < len(HP_KEYS):
            missing = set(HP_KEYS) - set(available_keys)
            print(f"  [Cache] Warning: '{path}' is missing columns {missing}.")

        for _, row in df.iterrows():
            try:
                hp = {}
                for k in HP_KEYS:
                    val = row[k]
                    if pd.isna(val):
                        raise ValueError(f"NaN for key {k}")
                    expected = SEARCH_SPACE[k][0]
                    if isinstance(expected, int):
                        val = int(val)
                    elif isinstance(expected, float):
                        val = float(val)
                    hp[k] = val
            except (KeyError, ValueError):
                continue

            h        = config_hash(hp)
            val_acc  = float(row["val_accuracy"])  if pd.notna(row.get("val_accuracy"))  else None
            test_acc = float(row["test_accuracy"]) if pd.notna(row.get("test_accuracy")) else None

            if val_acc is None:
                continue

            if h not in cache or val_acc > cache[h]["val_accuracy"]:
                cache[h] = {
                    "val_accuracy":  val_acc,
                    "test_accuracy": test_acc,
                    "source":        label,
                }
                added += 1

    except Exception as e:
        print(f"  [Cache] Warning: could not load '{path}' — {e}")

    return added


def build_cache_from_csv() -> dict:
    """
    Build the unified result cache from external CSVs and GWO's own summary.
    """
    cache: dict = {}
    total_external = 0

    if EXTERNAL_SUMMARY_CSVS:
        print(f"  [Cache] Loading {len(EXTERNAL_SUMMARY_CSVS)} external "
              f"summary file(s)…")
        for raw_path in EXTERNAL_SUMMARY_CSVS:
            p = Path(raw_path)
            n = _load_one_csv(p, cache, label=p.name)
            total_external += n
            print(f"           {p}  →  {n} new entries")
    else:
        print("  [Cache] No external summary files configured.")

    own_p = summary_csv_path()
    n_own = _load_one_csv(own_p, cache, label="gwo_summary.csv")
    if n_own:
        print(f"  [Cache] GWO own summary ({own_p})  →  {n_own} new/updated entries")

    print(f"  [Cache] Total unique configs in cache: {len(cache)}  "
          f"(external: {total_external}, GWO own: {n_own})")
    return cache


# ─── TRIAL LOGGING ────────────────────────────────────────────────────────────

def save_trial(seed, trial_id, hyperparams, history_dict,
               val_acc, test_acc, best_epoch, total_epochs, elapsed,
               from_cache=False, wolf_role="omega"):
    log = {
        "method":                "grey_wolf_optimizer",
        "seed":                  seed,
        "trial_id":              trial_id,
        "wolf_role":             wolf_role,
        "hyperparams":           hyperparams,
        "val_accuracy":          float(val_acc),
        "test_accuracy":         float(test_acc) if test_acc is not None else None,
        "val_loss":              (float(min(history_dict["val_loss"]))
                                  if history_dict else None),
        "best_epoch":            int(best_epoch)   if best_epoch   is not None else None,
        "total_epochs":          int(total_epochs) if total_epochs is not None else None,
        "history":               history_dict,
        "training_time_seconds": round(elapsed, 1),
        "from_cache":            from_cache,
        "timestamp":             datetime.now().isoformat(),
    }
    with open(trial_path(seed, trial_id), "w") as f:
        json.dump(log, f, indent=2)

    csv_p        = summary_csv_path()
    write_header = not csv_p.exists()
    with open(csv_p, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "method", "seed", "trial_id", "wolf_role",
            *HP_KEYS,
            "val_accuracy", "test_accuracy", "best_epoch",
            "total_epochs", "training_time_seconds", "from_cache", "timestamp",
        ])
        if write_header:
            writer.writeheader()
        writer.writerow({
            "method":                "grey_wolf_optimizer",
            "seed":                  seed,
            "trial_id":              trial_id,
            "wolf_role":             wolf_role,
            **hyperparams,
            "val_accuracy":          log["val_accuracy"],
            "test_accuracy":         log["test_accuracy"],
            "best_epoch":            log["best_epoch"],
            "total_epochs":          log["total_epochs"],
            "training_time_seconds": round(elapsed, 1),
            "from_cache":            from_cache,
            "timestamp":             log["timestamp"],
        })


def load_trial(seed: int, trial_id: int):
    p = trial_path(seed, trial_id)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


# ─── CHECKPOINTING ────────────────────────────────────────────────────────────

def save_checkpoint(seed, generation, wolf_index,
                    positions, fitnesses,
                    alpha_pos, alpha_fit,
                    beta_pos,  beta_fit,
                    delta_pos, delta_fit,
                    evaluated_hashes, trial_counter,
                    best_val_acc, best_hyperparams,
                    rng_np: np.random.Generator):
    """
    Atomic checkpoint write (tmp → rename).

    What is persisted and why
    ─────────────────────────
    positions        : current location of every wolf — needed to continue
                       the GWO trajectory from exactly where we stopped.
    fitnesses        : f(x_i) at the current position — lets us reconstruct
                       which wolves were already evaluated this generation.
    wolf_index       : how many wolves were processed in the current
                       generation — the exact resume point within the loop.
    alpha/beta/delta : the three leaders (position + fitness) — these are
                       the GWO attractors; losing them collapses the search.
    rng_state        : numpy rng bit-generator state — guarantees that r1, r2
                       draws after resume are IDENTICAL to an uninterrupted run.
    evaluated_hashes : set of already-evaluated config hashes — prevents
                       re-evaluating configs on resume.
    trial_counter    : global trial counter — ensures trial IDs are never reused.
    best_val_acc     : global best fitness observed — used for reporting.
    best_hyperparams : configuration that achieved best_val_acc.
    generation       : current generation index — the outer loop cursor.

    NOTE: The leaders are stored explicitly (not recomputed) to guarantee
    consistency even if the checkpoint is written mid-generation while a new
    alpha candidate has just been found but not yet propagated to all wolves.
    """
    ckpt = {
        "generation":    generation,
        "wolf_index":    wolf_index,      # ← exact mid-generation resume point
        # ── Pack state ────────────────────────────────────────────────────
        "positions":     [v.tolist() for v in positions],
        "fitnesses":     list(fitnesses),
        # ── Leader memory (GWO attractors) ────────────────────────────────
        "alpha_pos":     alpha_pos.tolist(),
        "alpha_fit":     float(alpha_fit),
        "beta_pos":      beta_pos.tolist(),
        "beta_fit":      float(beta_fit),
        "delta_pos":     delta_pos.tolist(),
        "delta_fit":     float(delta_fit),
        # ── Bookkeeping ───────────────────────────────────────────────────
        "evaluated_hashes": list(evaluated_hashes),
        "trial_counter": trial_counter,
        "best_val_acc":  best_val_acc,
        "best_hyperparams": best_hyperparams,
        "rng_state":     rng_np.bit_generator.state,
        "timestamp":     datetime.now().isoformat(),
    }
    p   = checkpoint_path(seed)
    tmp = str(p) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ckpt, f, indent=2)
    os.replace(tmp, p)   # atomic on POSIX; near-atomic on Windows
    print(f"  [Checkpoint] seed={seed} gen={generation} "
          f"wolf={wolf_index}/{POP_SIZE} "
          f"trial={trial_counter}/{MAX_EVALUATIONS} "
          f"best={best_val_acc:.4f}")


def load_checkpoint(seed: int):
    """
    Load checkpoint and restore rng to the exact state at save time.

    Returns (ckpt_dict, rng) or (None, fresh_rng) if no valid checkpoint.

    Validation:
      - All mandatory keys must be present.
      - Arrays are converted back to numpy.
      - evaluated_hashes is restored as a set.
      - rng_state is injected into the bit-generator.

    If the checkpoint file exists but is corrupt (e.g. partial write of the
    .tmp that replaced the real file before atomic rename), the function
    falls back to a fresh start and prints a warning rather than crashing.
    """
    p = checkpoint_path(seed)
    if not p.exists():
        print(f"  [Checkpoint] No checkpoint found for seed {seed} — fresh start.")
        return None, np.random.default_rng(seed)

    REQUIRED_KEYS = {
        "generation", "wolf_index",
        "positions", "fitnesses",
        "alpha_pos", "alpha_fit",
        "beta_pos",  "beta_fit",
        "delta_pos", "delta_fit",
        "evaluated_hashes", "trial_counter",
        "best_val_acc", "best_hyperparams",
        "rng_state",
    }

    try:
        with open(p) as f:
            raw = f.read()
        ckpt = json.loads(raw)

        missing = REQUIRED_KEYS - set(ckpt.keys())
        if missing:
            raise KeyError(f"Missing keys in checkpoint: {missing}")

        # ── Deserialise arrays ────────────────────────────────────────────
        ckpt["positions"]  = [np.array(v) for v in ckpt["positions"]]
        ckpt["fitnesses"]  = list(ckpt["fitnesses"])
        ckpt["alpha_pos"]  = np.array(ckpt["alpha_pos"])
        ckpt["beta_pos"]   = np.array(ckpt["beta_pos"])
        ckpt["delta_pos"]  = np.array(ckpt["delta_pos"])
        ckpt["evaluated_hashes"] = set(ckpt["evaluated_hashes"])

        # Sanity: positions length must match POP_SIZE
        if len(ckpt["positions"]) != POP_SIZE:
            raise ValueError(
                f"positions length {len(ckpt['positions'])} ≠ POP_SIZE {POP_SIZE}"
            )

        # ── Restore rng ───────────────────────────────────────────────────
        rng_np = np.random.default_rng(seed)
        rng_np.bit_generator.state = ckpt["rng_state"]

        print(f"  [Checkpoint loaded] seed={seed} "
              f"gen={ckpt['generation']} "
              f"wolf_index={ckpt['wolf_index']}/{POP_SIZE} "
              f"trials={ckpt['trial_counter']}/{MAX_EVALUATIONS} "
              f"best={ckpt['best_val_acc']:.4f} "
              f"best_hp={ckpt.get('best_hyperparams')}")
        return ckpt, rng_np

    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        print(f"  [Checkpoint] WARNING: corrupt checkpoint for seed {seed} — {exc}")
        print(f"  [Checkpoint] Renaming bad file and starting fresh.")
        bad = str(p) + f".bad_{int(time.time())}"
        try:
            os.rename(p, bad)
        except OSError:
            pass
        return None, np.random.default_rng(seed)


# ─── DATA LOADING ─────────────────────────────────────────────────────────────

def load_datasets(batch_size: int):
    import tensorflow as tf

    def make_ds(directory, shuffle):
        return tf.keras.utils.image_dataset_from_directory(
            directory,
            labels="inferred",
            label_mode="int",
            class_names=CLASS_NAMES,
            image_size=IMG_SIZE,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=42,
        )

    train_ds = make_ds(TRAIN_DIR, shuffle=True)
    val_ds   = make_ds(VAL_DIR,   shuffle=False)
    test_ds  = make_ds(TEST_DIR,  shuffle=False)

    preprocess = tf.keras.applications.efficientnet.preprocess_input
    AUTOTUNE   = tf.data.AUTOTUNE

    train_ds = (train_ds
                .map(lambda x, y: (preprocess(x), y),
                     num_parallel_calls=AUTOTUNE)
                .prefetch(AUTOTUNE))
    val_ds   = (val_ds
                .map(lambda x, y: (preprocess(x), y),
                     num_parallel_calls=AUTOTUNE)
                .prefetch(AUTOTUNE))
    test_ds  = (test_ds
                .map(lambda x, y: (preprocess(x), y),
                     num_parallel_calls=AUTOTUNE)
                .prefetch(AUTOTUNE))
    return train_ds, val_ds, test_ds


# ─── MODEL CONSTRUCTION ───────────────────────────────────────────────────────

def build_model(freezing_ratio: float, dropout_rate: float,
                l2_reg: float, num_classes: int = NUM_CLASSES):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    base_model = keras.applications.EfficientNetB0(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )

    total_layers = len(base_model.layers)
    fine_tune_at = int(freezing_ratio * total_layers)

    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    inputs  = keras.Input(shape=(*IMG_SIZE, 3))
    x       = base_model(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=regularizers.l2(l2_reg),
    )(x)
    return keras.Model(inputs, outputs, name="AppleLeaf_EfficientNetB0")


# ─── TRAINING & EVALUATION ────────────────────────────────────────────────────

def train_and_eval(hyperparams: dict, seed: int, trial_id: int) -> tuple:
    """
    Train EfficientNetB0 with the given hyperparams.
    Returns (val_acc, test_acc, history_dict, best_epoch, total_ep, elapsed).
    """
    import tensorflow as tf
    from tensorflow import keras

    trial_seed = seed * 1000 + trial_id
    tf.random.set_seed(trial_seed)
    np.random.seed(trial_seed)

    hp = hyperparams
    t0 = time.time()

    train_ds_hp, val_ds_hp, test_ds_hp = load_datasets(hp["batch_size"])

    model = build_model(
        freezing_ratio=hp["freezing_ratio"],
        dropout_rate=hp["dropout_rate"],
        l2_reg=hp["l2_reg"],
    )

    total_steps = EPOCHS * len(train_ds_hp)
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=hp["learning_rate"],
        decay_steps=total_steps,
        alpha=1e-6,
    )

    if hp["optimizer"] == "adam":
        opt = keras.optimizers.Adam(learning_rate=lr_schedule)
    elif hp["optimizer"] == "adamw":
        opt = keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=hp["l2_reg"],
        )
    else:
        opt = keras.optimizers.SGD(
            learning_rate=lr_schedule,
            momentum=0.9,
            nesterov=True,
        )

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=0,
        ),
    ]

    history = model.fit(
        train_ds_hp,
        validation_data=val_ds_hp,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=0,
    )

    val_acc    = float(max(history.history["val_accuracy"]))
    best_epoch = int(np.argmax(history.history["val_accuracy"]))
    total_ep   = len(history.history["val_accuracy"])
    elapsed    = time.time() - t0

    _, test_acc = model.evaluate(test_ds_hp, verbose=0)

    history_dict = {
        "train_accuracy": [float(x) for x in history.history["accuracy"]],
        "val_accuracy":   [float(x) for x in history.history["val_accuracy"]],
        "train_loss":     [float(x) for x in history.history["loss"]],
        "val_loss":       [float(x) for x in history.history["val_loss"]],
    }

    del model
    tf.keras.backend.clear_session()

    return val_acc, float(test_acc), history_dict, best_epoch, total_ep, elapsed


def evaluate(hyperparams: dict, seed: int, trial_id: int,
             result_cache: dict, wolf_role: str = "omega") -> float:
    """
    Evaluate a hyperparameter config.

    Cache hits return instantly without training.  New configs are trained
    and immediately added to the cache so subsequent seeds / wolves benefit.
    """
    h = config_hash(hyperparams)

    if h in result_cache:
        cached   = result_cache[h]
        val_acc  = cached["val_accuracy"]
        test_acc = cached["test_accuracy"]
        source   = cached.get("source", "unknown")
        test_str = f"{test_acc:.4f}" if test_acc is not None else "N/A"
        print(f"    Trial {trial_id:03d} [{wolf_role.upper():5}] "
              f"[CACHE:{source}] | "
              f"val={val_acc:.4f} test={test_str} | {hyperparams}")
        save_trial(seed, trial_id, hyperparams,
                   history_dict=None,
                   val_acc=val_acc, test_acc=test_acc,
                   best_epoch=None, total_epochs=None,
                   elapsed=0.0, from_cache=True,
                   wolf_role=wolf_role)
        return val_acc

    val_acc, test_acc, history_dict, best_epoch, total_ep, elapsed = \
        train_and_eval(hyperparams, seed, trial_id)

    save_trial(seed, trial_id, hyperparams, history_dict,
               val_acc, test_acc, best_epoch, total_ep, elapsed,
               from_cache=False, wolf_role=wolf_role)

    result_cache[h] = {
        "val_accuracy":  val_acc,
        "test_accuracy": test_acc,
        "source":        "gwo_summary.csv",
    }

    print(f"    Trial {trial_id:03d} [{wolf_role.upper():5}] | "
          f"val={val_acc:.4f} test={test_acc:.4f} "
          f"| ep={total_ep} | {elapsed:.0f}s | {hyperparams}")
    return val_acc


# ─── GREY WOLF OPTIMISER — MAIN LOOP ─────────────────────────────────────────

def _wolf_role(i: int) -> str:
    """Return the role label for wolf at index i (used in logging only)."""
    return {0: "alpha", 1: "beta", 2: "delta"}.get(i, "omega")


def run_gwo_seed(seed: int, result_cache: dict) -> float:
    """
    Run GWO for one seed.

    Pack representation
    ───────────────────
    Each wolf has:
      position  – continuous vector in [0, BOUNDS), mapped to discrete HPs
                  via floor(clip(.)).
      fitness   – f(position), higher is better.

    The three leaders (α, β, δ) are tracked explicitly and updated after
    each wolf evaluation.  All other wolves are ω (omega).

    Fault-tolerance design
    ──────────────────────
    A checkpoint is written atomically (tmp → rename) after every single wolf
    evaluation.  On resume:
      • positions, fitnesses        → full pack state at the last save
      • wolf_index                  → exact mid-generation resume point
      • alpha/beta/delta pos+fit    → GWO attractors are intact
      • rng_state                   → r1/r2 draws after resume are identical
                                      to an uninterrupted run
      • evaluated_hashes, trial_counter → no config is evaluated twice
      • best_val_acc, best_hyperparams  → global optimum is preserved

    Duplicate-config handling
    ─────────────────────────
    No deduplication filter is applied before evaluate().  The cache inside
    evaluate() returns instantly for already-seen configs, so no budget slot
    is wasted and every wolf always has a valid fitness for the leader update.
    """
    global _shutdown_requested

    setup_dirs(seed)

    print(f"\n{'='*60}")
    print(f"  GREY WOLF OPTIMIZER — seed {seed}")
    print(f"{'='*60}")

    # ── Resume or initialise ──────────────────────────────────────────────────
    ckpt, rng_np = load_checkpoint(seed)

    if ckpt is not None:
        generation       = ckpt["generation"]
        wolf_index       = ckpt["wolf_index"]
        positions        = ckpt["positions"]          # list of np.ndarray
        fitnesses        = ckpt["fitnesses"]          # list of float
        alpha_pos        = ckpt["alpha_pos"]
        alpha_fit        = ckpt["alpha_fit"]
        beta_pos         = ckpt["beta_pos"]
        beta_fit         = ckpt["beta_fit"]
        delta_pos        = ckpt["delta_pos"]
        delta_fit        = ckpt["delta_fit"]
        evaluated_hashes = ckpt["evaluated_hashes"]
        trial_counter    = ckpt["trial_counter"]
        best_val_acc     = ckpt["best_val_acc"]
        best_hyperparams = ckpt.get("best_hyperparams", None)
        print(f"  Resuming from generation {generation}, "
              f"wolf {wolf_index}/{POP_SIZE} "
              f"({len(positions)}/{POP_SIZE} wolves initialised)")
    else:
        generation       = 0
        wolf_index       = 0
        positions        = []
        fitnesses        = []
        # Initialise leaders with -inf so any real evaluation beats them
        alpha_pos        = np.zeros(DIM)
        alpha_fit        = -np.inf
        beta_pos         = np.zeros(DIM)
        beta_fit         = -np.inf
        delta_pos        = np.zeros(DIM)
        delta_fit        = -np.inf
        evaluated_hashes = set()
        trial_counter    = 0
        best_val_acc     = 0.0
        best_hyperparams = None

    # ── Centralised checkpoint helper ─────────────────────────────────────────
    def _save(gen: int, w_idx: int):
        save_checkpoint(
            seed, gen, w_idx,
            positions, fitnesses,
            alpha_pos, alpha_fit,
            beta_pos,  beta_fit,
            delta_pos, delta_fit,
            evaluated_hashes, trial_counter,
            best_val_acc, best_hyperparams,
            rng_np,
        )

    # ── Leader update helper ──────────────────────────────────────────────────
    def _update_leaders(val_acc: float, pos: np.ndarray):
        """
        Compare val_acc with the three leaders and demote as needed.

        α > β > δ:  if val_acc > α → shift α→β→δ, promote new wolf to α
                    if val_acc > β → shift β→δ,    promote new wolf to β
                    if val_acc > δ → replace δ with new wolf
        """
        nonlocal alpha_pos, alpha_fit, beta_pos, beta_fit, delta_pos, delta_fit

        if val_acc > alpha_fit:
            delta_pos, delta_fit = beta_pos.copy(),  beta_fit
            beta_pos,  beta_fit  = alpha_pos.copy(), alpha_fit
            alpha_pos, alpha_fit = pos.copy(),        val_acc
        elif val_acc > beta_fit:
            delta_pos, delta_fit = beta_pos.copy(), beta_fit
            beta_pos,  beta_fit  = pos.copy(),       val_acc
        elif val_acc > delta_fit:
            delta_pos, delta_fit = pos.copy(), val_acc

    # ── Generation 0: random initialisation ───────────────────────────────────
    if generation == 0:
        print(f"\n  [Gen 0] Initialising pack ({POP_SIZE} wolves)...")
        already_done = len(positions)  # wolves already evaluated before crash

        for i in range(already_done, POP_SIZE):
            if trial_counter >= MAX_EVALUATIONS or _shutdown_requested:
                break

            pos    = random_vector(rng_np)
            hp     = vec_to_dict(pos)
            h      = config_hash(hp)
            role   = _wolf_role(i)

            val_acc = evaluate(hp, seed, trial_counter, result_cache, role)

            positions.append(pos)
            fitnesses.append(val_acc)
            evaluated_hashes.add(h)
            trial_counter += 1

            _update_leaders(val_acc, pos)

            if val_acc > best_val_acc:
                best_val_acc     = val_acc
                best_hyperparams = hp
                print(f"    ★ New best (gen 0, wolf {i}): {best_val_acc:.4f} "
                      f"| {best_hyperparams}")

            _save(gen=0, w_idx=i + 1)   # wolf_index = how many wolves done

            if _shutdown_requested:
                print("  [Shutdown] Checkpoint saved. Exiting cleanly.")
                sys.exit(0)

        if len(positions) == POP_SIZE:
            # Gen 0 complete — advance so resume skips initialisation.
            generation  = 1
            wolf_index  = 0
            _save(generation, 0)

    # ── Generations 1 … N_GENERATIONS ─────────────────────────────────────────
    for gen in range(generation, N_GENERATIONS + 1):
        if trial_counter >= MAX_EVALUATIONS or _shutdown_requested:
            print(f"\n  Budget exhausted or stop requested "
                  f"({trial_counter} trials).")
            break

        # Decay parameter for this generation.
        a = gwo_compute_a(gen)

        print(f"\n  [Gen {gen}] a={a:.4f}  best_so_far={best_val_acc:.4f} "
              f"trials={trial_counter}/{MAX_EVALUATIONS}")
        print(f"    α={alpha_fit:.4f}  β={beta_fit:.4f}  δ={delta_fit:.4f}")

        # Resume: determine how many wolves were already updated this generation.
        start_wolf = wolf_index if gen == generation else 0

        for i in range(start_wolf, POP_SIZE):
            if trial_counter >= MAX_EVALUATIONS or _shutdown_requested:
                break

            # ── GWO position update ──────────────────────────────────────
            new_pos = gwo_update_position(
                positions[i], alpha_pos, beta_pos, delta_pos, a, rng_np
            )
            hp  = vec_to_dict(new_pos)
            h   = config_hash(hp)
            role = _wolf_role(i)

            val_acc = evaluate(hp, seed, trial_counter, result_cache, role)

            # ── Update pack ──────────────────────────────────────────────
            positions[i] = new_pos
            fitnesses[i] = val_acc
            evaluated_hashes.add(h)
            trial_counter += 1

            _update_leaders(val_acc, new_pos)

            if val_acc > best_val_acc:
                best_val_acc     = val_acc
                best_hyperparams = hp
                print(f"    ★ New best (gen {gen}, wolf {i}): "
                      f"{best_val_acc:.4f} | {best_hyperparams}")

            # wolf_index = i + 1 (number of wolves done this gen)
            _save(gen=gen, w_idx=i + 1)

            if _shutdown_requested:
                print("  [Shutdown] Checkpoint saved. Exiting cleanly.")
                sys.exit(0)

        # ── End of generation: reset wolf_index for next gen ─────────────────
        if not _shutdown_requested and trial_counter < MAX_EVALUATIONS:
            wolf_index = 0
            _save(gen=gen + 1, w_idx=0)

    print(f"\n  [DONE] seed={seed} | best_val_acc={best_val_acc:.4f} "
          f"| best_hyperparams={best_hyperparams} "
          f"| total_trials={trial_counter}")
    return best_val_acc


# ─── VISUALISATION ────────────────────────────────────────────────────────────

def plot_results():
    import matplotlib.pyplot as plt
    import seaborn as sns

    csv_p = summary_csv_path()
    if not csv_p.exists():
        print("No results to plot yet.")
        return

    df     = pd.read_csv(csv_p)
    df_gwo = df[df["method"] == "grey_wolf_optimizer"]

    plots_dir = RESULTS_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    # ── 1. Convergence ───────────────────────────────────────────────────────
    fig, ax    = plt.subplots(figsize=(10, 5))
    all_curves = []

    for seed in SEEDS:
        seed_df = df_gwo[df_gwo["seed"] == seed].sort_values("trial_id")
        if len(seed_df) == 0:
            continue
        curve = seed_df["val_accuracy"].cummax().values
        all_curves.append(curve)
        ax.plot(range(1, len(curve) + 1), curve,
                alpha=0.3, color="teal", linewidth=1)

    if all_curves:
        max_len = max(len(c) for c in all_curves)
        padded  = np.array([
            np.pad(c, (0, max_len - len(c)), mode="edge")
            for c in all_curves
        ])
        mean = padded.mean(axis=0)
        std  = padded.std(axis=0)
        x    = np.arange(1, max_len + 1)
        ax.plot(x, mean, color="teal", linewidth=2.5,
                label=f"GWO mean (n={len(all_curves)} seeds)")
        ax.fill_between(x, mean - std, mean + std,
                        alpha=0.2, color="teal", label="± std")

    ax.set_xlabel("Number of trials")
    ax.set_ylabel("Best validation accuracy (so far)")
    ax.set_title("Grey Wolf Optimizer — Convergence")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "gwo_convergence.png", dpi=150)
    plt.close()

    # ── 2. Stability boxplot ─────────────────────────────────────────────────
    best_per_seed = df_gwo.groupby("seed")["val_accuracy"].max().values
    if len(best_per_seed) > 0:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.boxplot([best_per_seed], labels=["Grey Wolf"],
                   patch_artist=True,
                   boxprops=dict(facecolor="teal", alpha=0.6))
        ax.set_ylabel("Best validation accuracy")
        ax.set_title(
            f"Stability — GWO\n"
            f"mean={best_per_seed.mean():.4f}  std={best_per_seed.std():.4f}"
        )
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "gwo_stability_boxplot.png", dpi=150)
        plt.close()

    # ── 3. HP importance (correlation) ───────────────────────────────────────
    hp_df = df_gwo[HP_KEYS + ["val_accuracy"]].copy()
    hp_df["optimizer"] = hp_df["optimizer"].map(
        {"adam": 0, "adamw": 1, "sgd": 2}
    )
    hp_df = hp_df.apply(pd.to_numeric, errors="coerce").dropna()

    if len(hp_df) > 5:
        corr = hp_df.corr()[["val_accuracy"]].drop("val_accuracy")
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                    center=0, vmin=-1, vmax=1, ax=ax)
        ax.set_title("HP correlation with val_accuracy")
        plt.tight_layout()
        plt.savefig(plots_dir / "gwo_hp_importance.png", dpi=150)
        plt.close()

    # ── 4. All trials scatter ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    cached_mask  = (df_gwo["from_cache"] == True
                    if "from_cache" in df_gwo.columns
                    else pd.Series([False] * len(df_gwo)))
    trained_mask = ~cached_mask
    ax.scatter(df_gwo.loc[trained_mask, "trial_id"],
               df_gwo.loc[trained_mask, "val_accuracy"],
               alpha=0.4, s=20, color="teal", label="trained")
    if cached_mask.any():
        ax.scatter(df_gwo.loc[cached_mask, "trial_id"],
                   df_gwo.loc[cached_mask, "val_accuracy"],
                   alpha=0.6, s=20, color="darkorange", marker="x",
                   label="cache hit")
    ax.set_xlabel("Trial ID")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("GWO — All trials val_accuracy")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "gwo_all_trials.png", dpi=150)
    plt.close()

    # ── 5. Wolf-role distribution ─────────────────────────────────────────────
    if "wolf_role" in df_gwo.columns:
        role_acc = df_gwo.groupby("wolf_role")["val_accuracy"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = {"alpha": "gold", "beta": "silver", "delta": "#cd7f32", "omega": "teal"}
        for _, row in role_acc.iterrows():
            ax.bar(row["wolf_role"], row["val_accuracy"],
                   color=colors.get(row["wolf_role"], "grey"), alpha=0.8)
        ax.set_ylabel("Mean val_accuracy")
        ax.set_title("GWO — Mean accuracy per wolf role")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "gwo_wolf_roles.png", dpi=150)
        plt.close()

    print(f"  Plots saved in {plots_dir}")


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  GWO HPO — EfficientNetB0 — Apple Leaf Disease")
    print(f"  Seeds            : {SEEDS}")
    print(f"  Max evaluations  : {MAX_EVALUATIONS}  per seed")
    print(f"  Pack size        : {POP_SIZE}")
    print(f"  Generations      : {N_GENERATIONS}")
    print(f"  Budget (max)     : {MAX_EVALUATIONS} evals per seed "
          f"(gen 0: {POP_SIZE}, gens 1-{N_GENERATIONS}: up to "
          f"{N_GENERATIONS * POP_SIZE})")
    if EXTERNAL_SUMMARY_CSVS:
        print(f"  External caches  :")
        for p in EXTERNAL_SUMMARY_CSVS:
            print(f"    • {p}")
    else:
        print(f"  External caches  : none")
    print("=" * 60)

    result_cache = build_cache_from_csv()

    results = {}

    for seed in SEEDS:
        ckpt, _ = load_checkpoint(seed)
        if ckpt is not None and ckpt["trial_counter"] >= MAX_EVALUATIONS:
            print(f"\n  Seed {seed} already complete "
                  f"({ckpt['trial_counter']} trials). Skipping.")
            results[seed] = ckpt["best_val_acc"]
            continue

        best = run_gwo_seed(seed, result_cache)
        results[seed] = best

    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    accs = list(results.values())
    for seed, acc in results.items():
        print(f"  Seed {seed}: best_val_acc = {acc:.4f}")
    if accs:
        print(f"\n  Mean ± Std : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"  Min / Max  : {np.min(accs):.4f} / {np.max(accs):.4f}")

    print("\n" + "=" * 60)
    print("  BEST HYPERPARAMETERS PER SEED")
    print("=" * 60)
    for seed in SEEDS:
        ckpt, _ = load_checkpoint(seed)
        if ckpt:
            print(f"  Seed {seed}: val={ckpt['best_val_acc']:.4f} "
                  f"| α-fit={ckpt['alpha_fit']:.4f} "
                  f"| {ckpt['best_hyperparams']}")

    print("\nGenerating plots...")
    plot_results()
    print(f"\nDone. All results saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()