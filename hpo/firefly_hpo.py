"""
Hyperparameter Optimisation (HPO) for EfficientNetB0 fine-tuning on the
Apple Leaf Disease dataset using the Firefly Algorithm (FA).

Adapted from the PSO version. The full infrastructure (cache, checkpointing,
logging, visualisations) is preserved unchanged; only the optimisation core
is replaced.

Firefly Algorithm — update equations
─────────────────────────────────────
Each firefly i is attracted toward any brighter firefly j (higher fitness):

  r_ij    = ||x_i - x_j||₂          (Euclidean distance in continuous space)

  β(r_ij) = β₀ · exp(−γ · r_ij²)   (attractiveness, decays with distance)

  x_i(t+1) = x_i(t)
              + β(r_ij) · (x_j(t) − x_i(t))   (attraction toward j)
              + α · (rand() − 0.5)              (random perturbation)

  • If no brighter firefly exists → random walk only:
      x_i(t+1) = x_i(t) + α · (rand() − 0.5)

  • Result is clipped to [0, BOUNDS) after every update.

Parameters
──────────
  β₀  : initial attractiveness at distance 0   (typical: 1.0)
  γ   : light absorption coefficient           (typical: 0.01–1.0;
                                                scales with search space)
  α   : randomisation step size                (typical: 0.1–0.5 of range)
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
POP_SIZE        = 10
N_GENERATIONS   = 10

# ── Firefly Algorithm hyperparameters ─────────────────────────────────────────
FA_BETA0 = 1.0    # initial attractiveness at distance 0  (β₀)
FA_GAMMA = 0.1    # light absorption coefficient          (γ)
#   γ governs how quickly attractiveness fades with distance.
#   Low γ  → global attraction (all fireflies "see" each other equally)
#   High γ → local attraction  (only nearby fireflies pull strongly)
#   Typical rule of thumb: γ ≈ 1 / (typical_range²) where
#   typical_range is the expected inter-firefly distance in search space.
FA_ALPHA = 0.3    # random walk step size as fraction of each dimension's range
#   Actual per-dimension perturbation ≈ FA_ALPHA * BOUNDS[d]
#   Applied after BOUNDS is defined (see _fa_random_step below).

RESULTS_DIR = Path("results_fa")

EXTERNAL_SUMMARY_CSVS = [
    "results_ga/summary.csv",
    "results_de/summary.csv",
    "results_pso/summary.csv",
    "result_aco/summary.csv",
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


# ─── FIREFLY ALGORITHM OPERATORS ──────────────────────────────────────────────

def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Euclidean distance in the continuous search space.

    NOTE: Positions live in a heterogeneous space (dimensions have different
    ranges given by BOUNDS).  We normalise each dimension by its range so
    that every axis contributes equally to the distance measure.  Without
    normalisation a dimension with 7 values would dominate over one with 3.
    """
    normalised_diff = (a - b) / BOUNDS          # scale each dim to [0, 1)
    return float(np.sqrt(np.dot(normalised_diff, normalised_diff)))


def fa_attractiveness(r: float) -> float:
    """
    β(r) = β₀ · exp(−γ · r²)

    The attractiveness felt by firefly i toward firefly j at distance r.
    Monotonically decreasing: β(0) = β₀, β(∞) → 0.
    """
    return FA_BETA0 * np.exp(-FA_GAMMA * r * r)


def fa_move_toward(pos_i: np.ndarray,
                   pos_j: np.ndarray,
                   rng_np: np.random.Generator) -> np.ndarray:
    """
    Move firefly i one step toward brighter firefly j:

        x_i' = x_i  +  β(r_ij) · (x_j − x_i)  +  α · BOUNDS · (rand − 0.5)

    The random perturbation is scaled by BOUNDS so that α has the same
    geometric meaning regardless of dimension range.

    Result is clipped to [0, BOUNDS) to remain feasible.
    """
    r      = _euclidean_distance(pos_i, pos_j)
    beta   = fa_attractiveness(r)
    rand   = rng_np.random(DIM)                      # U(0,1) per dimension
    noise  = FA_ALPHA * BOUNDS * (rand - 0.5)        # scaled random walk

    new_pos = pos_i + beta * (pos_j - pos_i) + noise
    return np.clip(new_pos, 0.0, BOUNDS - 1e-9)


def fa_random_walk(pos: np.ndarray,
                   rng_np: np.random.Generator) -> np.ndarray:
    """
    No brighter firefly exists → pure random walk.

        x_i' = x_i  +  α · BOUNDS · (rand − 0.5)

    Used for the globally brightest firefly (or when the swarm has converged).
    """
    rand    = rng_np.random(DIM)
    noise   = FA_ALPHA * BOUNDS * (rand - 0.5)
    new_pos = pos + noise
    return np.clip(new_pos, 0.0, BOUNDS - 1e-9)


# ─── FILE PATHS ───────────────────────────────────────────────────────────────

def seed_dir(seed: int) -> Path:
    return RESULTS_DIR / f"seed_{seed}"

def trial_path(seed: int, trial_id: int) -> Path:
    return seed_dir(seed) / "trials" / f"trial_{trial_id:03d}.json"

def checkpoint_path(seed: int) -> Path:
    return seed_dir(seed) / "fa_checkpoint.json"

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
    Build the unified result cache from external CSVs and FA's own summary.
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
    n_own = _load_one_csv(own_p, cache, label="fa_summary.csv")
    if n_own:
        print(f"  [Cache] FA own summary ({own_p})  →  {n_own} new/updated entries")

    print(f"  [Cache] Total unique configs in cache: {len(cache)}  "
          f"(external: {total_external}, FA own: {n_own})")
    return cache


# ─── TRIAL LOGGING ────────────────────────────────────────────────────────────

def save_trial(seed, trial_id, hyperparams, history_dict,
               val_acc, test_acc, best_epoch, total_epochs, elapsed,
               from_cache=False):
    log = {
        "method":                "firefly_algorithm",
        "seed":                  seed,
        "trial_id":              trial_id,
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
            "method", "seed", "trial_id",
            *HP_KEYS,
            "val_accuracy", "test_accuracy", "best_epoch",
            "total_epochs", "training_time_seconds", "from_cache", "timestamp",
        ])
        if write_header:
            writer.writeheader()
        writer.writerow({
            "method":                "firefly_algorithm",
            "seed":                  seed,
            "trial_id":              trial_id,
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

def save_checkpoint(seed, generation,
                    positions, fitnesses,
                    evaluated_hashes, trial_counter,
                    best_val_acc, best_hyperparams,
                    rng_np: np.random.Generator):
    """
    Atomic checkpoint write (tmp → rename).

    What is persisted and why
    ─────────────────────────
    positions       : current location of each firefly — needed to continue
                      the swarm from exactly where it stopped.
    fitnesses       : f(x_i) at the current position — the "brightness" of
                      each firefly; required to determine which fireflies
                      attract which others on resume.

    NOTE: Unlike PSO, the Firefly Algorithm has no velocity or personal-best
    memory.  The full algorithm state is captured by (positions, fitnesses).

    rng_state       : numpy rng bit-generator state — guarantees that the
                      random perturbation draws after resume are IDENTICAL
                      to an uninterrupted run.
    """
    ckpt = {
        "generation":       generation,
        # ── Population state ──────────────────────────────────────────────
        "positions":        [v.tolist() for v in positions],
        "fitnesses":        list(fitnesses),
        # ── Bookkeeping ───────────────────────────────────────────────────
        "evaluated_hashes": list(evaluated_hashes),
        "trial_counter":    trial_counter,
        "best_val_acc":     best_val_acc,
        "best_hyperparams": best_hyperparams,
        "rng_state":        rng_np.bit_generator.state,
        "timestamp":        datetime.now().isoformat(),
    }
    p   = checkpoint_path(seed)
    tmp = str(p) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ckpt, f, indent=2)
    os.replace(tmp, p)
    print(f"  [Checkpoint] seed={seed} gen={generation} "
          f"trial={trial_counter}/{MAX_EVALUATIONS} "
          f"best={best_val_acc:.4f}")


def load_checkpoint(seed: int):
    """
    Load checkpoint and restore rng to the exact state at save time.
    Returns (ckpt_dict, rng) or (None, fresh_rng) if no checkpoint exists.
    """
    p = checkpoint_path(seed)
    if not p.exists():
        return None, np.random.default_rng(seed)

    with open(p) as f:
        ckpt = json.load(f)

    ckpt["positions"]        = [np.array(v) for v in ckpt["positions"]]
    ckpt["evaluated_hashes"] = set(ckpt["evaluated_hashes"])

    rng_np = np.random.default_rng(seed)
    rng_np.bit_generator.state = ckpt["rng_state"]

    print(f"  [Checkpoint loaded] seed={seed} "
          f"gen={ckpt['generation']} "
          f"trials={ckpt['trial_counter']}/{MAX_EVALUATIONS} "
          f"best={ckpt['best_val_acc']:.4f} "
          f"best_hp={ckpt.get('best_hyperparams')}")
    return ckpt, rng_np


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
             result_cache: dict) -> float:
    """
    Evaluate a hyperparameter config.

    Cache hits return instantly without training.  New configs are trained
    and immediately added to the cache so subsequent seeds / fireflies benefit.
    """
    h = config_hash(hyperparams)

    if h in result_cache:
        cached   = result_cache[h]
        val_acc  = cached["val_accuracy"]
        test_acc = cached["test_accuracy"]
        source   = cached.get("source", "unknown")
        test_str = f"{test_acc:.4f}" if test_acc is not None else "N/A"
        print(f"    Trial {trial_id:03d} [CACHE:{source}] | "
              f"val={val_acc:.4f} test={test_str} | {hyperparams}")
        save_trial(seed, trial_id, hyperparams,
                   history_dict=None,
                   val_acc=val_acc, test_acc=test_acc,
                   best_epoch=None, total_epochs=None,
                   elapsed=0.0, from_cache=True)
        return val_acc

    val_acc, test_acc, history_dict, best_epoch, total_ep, elapsed = \
        train_and_eval(hyperparams, seed, trial_id)

    save_trial(seed, trial_id, hyperparams, history_dict,
               val_acc, test_acc, best_epoch, total_ep, elapsed,
               from_cache=False)

    result_cache[h] = {
        "val_accuracy":  val_acc,
        "test_accuracy": test_acc,
        "source":        "fa_summary.csv",
    }

    print(f"    Trial {trial_id:03d} | val={val_acc:.4f} test={test_acc:.4f} "
          f"| ep={total_ep} | {elapsed:.0f}s | {hyperparams}")
    return val_acc


# ─── FIREFLY ALGORITHM — MAIN LOOP ───────────────────────────────────────────

def run_fa_seed(seed: int, result_cache: dict) -> float:
    """
    Run the Firefly Algorithm for one seed.

    Population representation
    ─────────────────────────
    Each firefly has:
      position  – continuous vector in [0, BOUNDS), mapped to discrete HPs
                  via floor(clip(.)).
      fitness   – f(x_i), the "brightness" / intensity of the firefly.
                  Higher is better (maximising val_accuracy).

    FA update logic (per generation)
    ─────────────────────────────────
    For each firefly i (sorted by ascending brightness so dimmer fireflies
    move first, consistent with the canonical FA):
      For each firefly j:
        if fitness[j] > fitness[i]:
          Move i toward j  (attraction + random perturbation)
          Re-evaluate i    (brightness may have changed)
          Update global best if improved

    Fireflies with no brighter neighbour perform a pure random walk.

    Fault-tolerance design
    ──────────────────────
    A checkpoint is written after every single firefly evaluation.
    On resume the mid-generation state is fully restored:
      • positions  → current locations (post-move for processed fireflies)
      • fitnesses  → brightness at those positions
      • rng_state  → random draws after resume are identical to uninterrupted run

    Duplicate-config handling
    ─────────────────────────
    No deduplication filter before evaluate().  The cache inside evaluate()
    returns instantly for already-seen configs, so no budget is wasted.
    """
    global _shutdown_requested

    setup_dirs(seed)

    print(f"\n{'='*60}")
    print(f"  FIREFLY ALGORITHM — seed {seed}")
    print(f"{'='*60}")

    # ── Resume or initialise ──────────────────────────────────────────────────
    ckpt, rng_np = load_checkpoint(seed)

    if ckpt is not None:
        generation       = ckpt["generation"]
        positions        = ckpt["positions"]
        fitnesses        = list(ckpt["fitnesses"])
        evaluated_hashes = ckpt["evaluated_hashes"]
        trial_counter    = ckpt["trial_counter"]
        best_val_acc     = ckpt["best_val_acc"]
        best_hyperparams = ckpt.get("best_hyperparams", None)
        print(f"  Resuming from generation {generation}, "
              f"trial {trial_counter}/{MAX_EVALUATIONS} "
              f"({len(positions)}/{POP_SIZE} fireflies initialised)")
    else:
        generation       = 0
        positions        = []
        fitnesses        = []
        evaluated_hashes = set()
        trial_counter    = 0
        best_val_acc     = 0.0
        best_hyperparams = None

    # ── Helper: one checkpoint call with all current state ───────────────────
    def _save(gen):
        save_checkpoint(
            seed, gen,
            positions, fitnesses,
            evaluated_hashes,
            trial_counter,
            best_val_acc, best_hyperparams,
            rng_np,
        )

    # ── Generation 0: random initialisation ──────────────────────────────────
    if generation == 0:
        print(f"\n  [Gen 0] Initialising swarm ({POP_SIZE} fireflies)...")
        already_done = len(positions)

        for i in range(already_done, POP_SIZE):
            if trial_counter >= MAX_EVALUATIONS or _shutdown_requested:
                break

            pos = random_vector(rng_np)
            hp  = vec_to_dict(pos)
            h   = config_hash(hp)

            val_acc = evaluate(hp, seed, trial_counter, result_cache)

            positions.append(pos)
            fitnesses.append(val_acc)
            evaluated_hashes.add(h)
            trial_counter += 1

            if val_acc > best_val_acc:
                best_val_acc     = val_acc
                best_hyperparams = hp
                print(f"    ★ New best (gen 0): {best_val_acc:.4f} "
                      f"| {best_hyperparams}")

            _save(0)

            if _shutdown_requested:
                print("  [Shutdown] Checkpoint saved. Exiting cleanly.")
                sys.exit(0)

        # Gen 0 complete — advance generation counter so resume skips gen 0.
        generation = 1
        _save(generation)

    # ── Generations 1 … N_GENERATIONS ────────────────────────────────────────
    for gen in range(generation, N_GENERATIONS + 1):
        if trial_counter >= MAX_EVALUATIONS or _shutdown_requested:
            print(f"\n  Budget exhausted or stop requested "
                  f"({trial_counter} trials).")
            break

        print(f"\n  [Gen {gen}] best_so_far={best_val_acc:.4f} "
              f"trials={trial_counter}/{MAX_EVALUATIONS}")

        # ── Sort fireflies by ascending fitness (dim → bright) ────────────
        # The canonical FA moves dimmer fireflies toward brighter ones.
        # Sorting ensures every firefly i only attracts those processed after
        # it (brighter), preserving the pairwise attraction semantics even
        # during a mid-generation resume.
        sort_order   = np.argsort(fitnesses)           # dim → bright
        new_positions = [None] * POP_SIZE
        new_fitnesses = [None] * POP_SIZE

        # On a mid-generation resume, some fireflies already have new positions.
        already_done = trial_counter % POP_SIZE if gen == generation else 0

        # Pre-fill already-processed slots (order restored from checkpoint).
        for rank in range(already_done):
            idx = sort_order[rank]
            new_positions[idx] = positions[idx].copy()
            new_fitnesses[idx] = fitnesses[idx]

        for rank in range(already_done, POP_SIZE):
            if trial_counter >= MAX_EVALUATIONS or _shutdown_requested:
                break

            idx = sort_order[rank]            # firefly to move (dimmest first)
            pos_i   = positions[idx]
            fit_i   = fitnesses[idx]

            # ── Find the brightest firefly that outshines firefly i ───────
            # We pick the single most attractive mover (highest fitness among
            # those brighter than i).  Alternative: move toward all brighter
            # ones sequentially — equally valid; single-best is faster.
            best_j     = -1
            best_fit_j = fit_i

            for rank_j in range(POP_SIZE):
                j = sort_order[rank_j]
                if fitnesses[j] > best_fit_j:
                    best_j     = j
                    best_fit_j = fitnesses[j]

            # ── Move firefly i ────────────────────────────────────────────
            if best_j >= 0:
                # Attraction toward brighter firefly j
                new_pos = fa_move_toward(pos_i, positions[best_j], rng_np)
                move_type = f"→ firefly {best_j}"
            else:
                # No brighter firefly exists → random walk
                new_pos = fa_random_walk(pos_i, rng_np)
                move_type = "random walk"

            hp  = vec_to_dict(new_pos)
            h   = config_hash(hp)
            val_acc = evaluate(hp, seed, trial_counter, result_cache)

            evaluated_hashes.add(h)
            trial_counter += 1

            new_positions[idx] = new_pos
            new_fitnesses[idx] = val_acc

            # ── Update global best ────────────────────────────────────────
            if val_acc > best_val_acc:
                best_val_acc     = val_acc
                best_hyperparams = hp
                print(f"    ★ New best (gen {gen}, {move_type}): "
                      f"{best_val_acc:.4f} | {best_hyperparams}")

            # ── Checkpoint after every firefly ────────────────────────────
            # Temporarily merge new and old positions so the checkpoint
            # captures the exact mid-generation state.
            for k in range(POP_SIZE):
                if new_positions[k] is not None:
                    positions[k] = new_positions[k]
                    fitnesses[k] = new_fitnesses[k]
            _save(gen)

            if _shutdown_requested:
                print("  [Shutdown] Checkpoint saved. Exiting cleanly.")
                sys.exit(0)

        # ── End of generation: commit fully updated state ─────────────────
        all_done = all(p is not None for p in new_positions)
        if all_done:
            positions  = new_positions
            fitnesses  = new_fitnesses
            generation = gen + 1
            _save(generation)

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

    df    = pd.read_csv(csv_p)
    df_fa = df[df["method"] == "firefly_algorithm"]

    plots_dir = RESULTS_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    # ── 1. Convergence ───────────────────────────────────────────────────────
    fig, ax    = plt.subplots(figsize=(10, 5))
    all_curves = []

    for seed in SEEDS:
        seed_df = df_fa[df_fa["seed"] == seed].sort_values("trial_id")
        if len(seed_df) == 0:
            continue
        curve = seed_df["val_accuracy"].cummax().values
        all_curves.append(curve)
        ax.plot(range(1, len(curve) + 1), curve,
                alpha=0.3, color="darkorange", linewidth=1)

    if all_curves:
        max_len = max(len(c) for c in all_curves)
        padded  = np.array([
            np.pad(c, (0, max_len - len(c)), mode="edge")
            for c in all_curves
        ])
        mean = padded.mean(axis=0)
        std  = padded.std(axis=0)
        x    = np.arange(1, max_len + 1)
        ax.plot(x, mean, color="darkorange", linewidth=2.5,
                label=f"FA mean (n={len(all_curves)} seeds)")
        ax.fill_between(x, mean - std, mean + std,
                        alpha=0.2, color="darkorange", label="± std")

    ax.set_xlabel("Number of trials")
    ax.set_ylabel("Best validation accuracy (so far)")
    ax.set_title("Firefly Algorithm — Convergence")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "fa_convergence.png", dpi=150)
    plt.close()

    # ── 2. Stability boxplot ─────────────────────────────────────────────────
    best_per_seed = df_fa.groupby("seed")["val_accuracy"].max().values
    if len(best_per_seed) > 0:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.boxplot([best_per_seed], labels=["Firefly Algorithm"],
                   patch_artist=True,
                   boxprops=dict(facecolor="darkorange", alpha=0.6))
        ax.set_ylabel("Best validation accuracy")
        ax.set_title(
            f"Stability — Firefly Algorithm\n"
            f"mean={best_per_seed.mean():.4f}  std={best_per_seed.std():.4f}"
        )
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "fa_stability_boxplot.png", dpi=150)
        plt.close()

    # ── 3. HP importance (correlation) ───────────────────────────────────────
    hp_df = df_fa[HP_KEYS + ["val_accuracy"]].copy()
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
        plt.savefig(plots_dir / "fa_hp_importance.png", dpi=150)
        plt.close()

    # ── 4. All trials scatter ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    cached_mask  = (df_fa["from_cache"] == True
                    if "from_cache" in df_fa.columns
                    else pd.Series([False] * len(df_fa)))
    trained_mask = ~cached_mask
    ax.scatter(df_fa.loc[trained_mask, "trial_id"],
               df_fa.loc[trained_mask, "val_accuracy"],
               alpha=0.4, s=20, color="darkorange", label="trained")
    if cached_mask.any():
        ax.scatter(df_fa.loc[cached_mask, "trial_id"],
                   df_fa.loc[cached_mask, "val_accuracy"],
                   alpha=0.6, s=20, color="steelblue", marker="x",
                   label="cache hit")
    ax.set_xlabel("Trial ID")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Firefly Algorithm — All trials val_accuracy")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "fa_all_trials.png", dpi=150)
    plt.close()

    print(f"  Plots saved in {plots_dir}")


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  Firefly Algorithm HPO — EfficientNetB0 — Apple Leaf Disease")
    print(f"  Seeds            : {SEEDS}")
    print(f"  Max evaluations  : {MAX_EVALUATIONS}  per seed")
    print(f"  Swarm size       : {POP_SIZE}")
    print(f"  Generations      : {N_GENERATIONS}")
    print(f"  FA β₀ / γ / α    : {FA_BETA0} / {FA_GAMMA} / {FA_ALPHA}")
    print(f"  Budget (max)     : {MAX_EVALUATIONS} evaluations per seed "
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

        best = run_fa_seed(seed, result_cache)
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
                  f"| {ckpt['best_hyperparams']}")

    print("\nGenerating plots...")
    plot_results()
    print(f"\nDone. All results saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()