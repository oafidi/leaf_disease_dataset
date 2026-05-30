"""
Hyperparameter Optimisation (HPO) for EfficientNetB0 fine-tuning on the
Apple Leaf Disease dataset using Ant Colony Optimisation (ACO).

ACO update equations
────────────────────
  Construction (per ant, per dimension d):
      P(j | d) = τ[d][j]^α · η[d][j]^β
                 ─────────────────────────
                 Σ_k τ[d][k]^α · η[d][k]^β

  Evaporation + deposit (end of iteration):
      τ[d][j] ← clip( (1-ρ) · τ[d][j] + Δτ[d][j],  τ_min, τ_max )

  η = 1 (uniform heuristic — no natural distance between HP values)
  Elitist/MMAS: only the global best deposits

Fault-tolerance design
──────────────────────
  • Signal handler (SIGINT/SIGTERM) sets a flag ONLY — never calls sys.exit()
    directly. The main loop checks the flag at safe points (between atomic
    operations), then performs an orderly shutdown: flush logs → write
    checkpoint → exit.

  • The checkpoint write is atomic: write to .tmp → os.replace() → done.
    A crash during the write leaves a .tmp orphan that is cleaned on the
    next startup, never a corrupt main checkpoint.

  • Every ant evaluation is wrapped in try/except so that a TensorFlow
    OOM or CUDA error triggers a checkpoint save and a clean exit instead
    of an unhandled traceback that loses all state.

  • current_gen_done tracks mid-iteration progress explicitly so a resume
    always continues from exactly the right ant index.

  • rng_state is saved so random draws after resume are identical to an
    uninterrupted run.

  • best_solution / best_hyperparams are always non-None after the first
    ant: we initialise best_solution to the first solution sampled, so the
    checkpoint never carries a None best.

  • The pheromone update runs only after a complete iteration. Partial
    iterations do not corrupt τ — correct ACO behaviour.

  • On resume, iter_solutions and iter_fitnesses are empty for the current
    (unfinished) iteration. Ants evaluated before the crash are skipped via
    the result cache; their individual results are already on disk.
    τ is updated only once the full iteration completes.
"""

import os
import json
import csv
import time
import hashlib
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import signal
import sys
import threading


# ─── LOGGING ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


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
N_ANTS          = 10
N_ITERATIONS    = 10

# ── ACO hyperparameters ────────────────────────────────────────────────────────
ACO_RHO     = 0.1
ACO_ALPHA   = 1.0
ACO_BETA    = 1.0    # η=1 everywhere so β has no effect; kept for clarity
ACO_TAU_MIN = 0.01
ACO_TAU_MAX = 1.0
ACO_ELITIST = True   # MMAS (only global best deposits)

RESULTS_DIR = Path("results_aco")

EXTERNAL_SUMMARY_CSVS = [
    "results_ga/summary.csv",
    "results_de/summary.csv",
    "results_pso/summary.csv",
]


# ─── SEARCH SPACE ─────────────────────────────────────────────────────────────

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
N_CHOICES = [len(v) for v in HP_VALUES]


# ─── GRACEFUL SHUTDOWN ────────────────────────────────────────────────────────
#
# Design rules:
#   1. The handler ONLY sets the flag. It never raises, never calls sys.exit().
#      Calling sys.exit() (which raises SystemExit) from a signal handler is
#      re-entrant: if the main thread is inside train_and_eval or a file write
#      when the signal arrives, the exception can corrupt state or skip the
#      final checkpoint.
#
#   2. The flag is a threading.Event so it is thread-safe and can be waited on.
#
#   3. The main loop checks the flag at exactly two "safe points":
#        a. After evaluate() returns and trial state is fully written to disk.
#        b. After the iteration-level checkpoint is written.
#      At both points no file is half-written and the checkpoint is consistent.
#
#   4. We register SIGTERM in addition to SIGINT so that `kill <pid>` and
#      container orchestrators (Docker, SLURM) trigger the same clean path.

_shutdown_event = threading.Event()

def _signal_handler(sig, frame):          # noqa: ARG001
    if not _shutdown_event.is_set():
        # Use os.write (async-signal-safe) instead of print/logging
        msg = b"\n  [Signal] Shutdown requested - finishing current trial...\n"
        os.write(sys.stdout.fileno(), msg)
        _shutdown_event.set()

signal.signal(signal.SIGINT,  _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def _shutdown_requested() -> bool:
    return _shutdown_event.is_set()


# ─── INDEX ↔ HYPERPARAMETER HELPERS ──────────────────────────────────────────

def indices_to_dict(idx: np.ndarray) -> dict:
    return {k: HP_VALUES[i][idx[i]] for i, k in enumerate(HP_KEYS)}


def config_hash(hp_dict: dict) -> str:
    s = json.dumps(hp_dict, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:10]


# ─── PHEROMONE MATRIX ─────────────────────────────────────────────────────────

def init_pheromones() -> list:
    """
    τ as a list-of-lists (variable row length per dimension).
    Stored as list-of-lists to avoid NaN-padding issues.
    """
    return [[ACO_TAU_MAX] * n for n in N_CHOICES]


def tau_to_json(tau: list) -> list:
    return [list(row) for row in tau]


def tau_from_json(raw: list) -> list:
    return [list(row) for row in raw]


# ─── ACO OPERATORS ────────────────────────────────────────────────────────────

def aco_probabilities(tau: list, d: int) -> np.ndarray:
    tau_vals = np.array(tau[d], dtype=float)
    # η = 1 everywhere → β exponent has no effect
    scores = tau_vals ** ACO_ALPHA
    total  = scores.sum()
    if total == 0.0:
        return np.ones(len(tau_vals)) / len(tau_vals)
    return scores / total


def aco_construct_solution(tau: list, rng_np: np.random.Generator) -> np.ndarray:
    solution = np.zeros(DIM, dtype=int)
    for d in range(DIM):
        probs       = aco_probabilities(tau, d)
        solution[d] = rng_np.choice(N_CHOICES[d], p=probs)
    return solution


def aco_update_pheromones(
    tau: list,
    best_solution: np.ndarray,
    best_fitness: float,
    iter_solutions: list,
    iter_fitnesses: list,
) -> list:
    for d in range(DIM):
        row = np.array(tau[d], dtype=float)

        # Evaporation
        row *= (1.0 - ACO_RHO)

        # Deposit
        if ACO_ELITIST:
            row[best_solution[d]] += best_fitness
        else:
            for sol, fit in zip(iter_solutions, iter_fitnesses):
                row[sol[d]] += fit

        row    = np.clip(row, ACO_TAU_MIN, ACO_TAU_MAX)
        tau[d] = row.tolist()

    return tau


# ─── FILE PATHS ───────────────────────────────────────────────────────────────

def seed_dir(seed: int) -> Path:
    return RESULTS_DIR / f"seed_{seed}"

def trial_path(seed: int, trial_id: int) -> Path:
    return seed_dir(seed) / "trials" / f"trial_{trial_id:03d}.json"

def checkpoint_path(seed: int) -> Path:
    return seed_dir(seed) / "aco_checkpoint.json"

def summary_csv_path() -> Path:
    return RESULTS_DIR / "summary.csv"


# ─── DIRECTORY SETUP ──────────────────────────────────────────────────────────

def setup_dirs(seed: int):
    (seed_dir(seed) / "trials").mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ─── ORPHAN .TMP CLEANUP ──────────────────────────────────────────────────────

def cleanup_tmp_checkpoint(seed: int):
    """
    Remove a leftover .tmp checkpoint file from a previous crash.

    os.replace() is atomic on POSIX (rename(2) syscall).  The only way a
    .tmp file survives a crash is if the process was killed between open()
    and os.replace().  In that case the main checkpoint file is intact
    (the replace never happened), so it is safe to delete the .tmp.
    """
    tmp = Path(str(checkpoint_path(seed)) + ".tmp")
    if tmp.exists():
        tmp.unlink()
        log.warning("Removed orphan tmp checkpoint: %s", tmp)


# ─── RESULTS CACHE ────────────────────────────────────────────────────────────

def _load_one_csv(path: Path, cache: dict, label: str) -> int:
    if not path.exists():
        log.info("  [Cache] Skipping '%s' — file not found.", path)
        return 0

    added = 0
    try:
        df = pd.read_csv(path)
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

    except Exception as exc:
        log.warning("  [Cache] Could not load '%s': %s", path, exc)

    return added


def build_cache_from_csv() -> dict:
    cache: dict = {}
    total_external = 0

    log.info("  [Cache] Loading external summary files…")
    for raw_path in EXTERNAL_SUMMARY_CSVS:
        p = Path(raw_path)
        n = _load_one_csv(p, cache, label=p.name)
        total_external += n
        log.info("           %s  →  %d new entries", p, n)

    own_p = summary_csv_path()
    n_own = _load_one_csv(own_p, cache, label="aco_summary.csv")
    if n_own:
        log.info("  [Cache] ACO own summary  →  %d new/updated entries", n_own)

    log.info(
        "  [Cache] Total unique configs: %d  (external: %d, ACO own: %d)",
        len(cache), total_external, n_own,
    )
    return cache


# ─── TRIAL LOGGING ────────────────────────────────────────────────────────────

def save_trial(seed, trial_id, hyperparams, history_dict,
               val_acc, test_acc, best_epoch, total_epochs,
               elapsed, from_cache=False):
    log_entry = {
        "method":                "ant_colony_optimisation",
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

    # Atomic write for the trial JSON as well
    tp      = trial_path(seed, trial_id)
    tp_tmp  = Path(str(tp) + ".tmp")
    with open(tp_tmp, "w") as f:
        json.dump(log_entry, f, indent=2)
    os.replace(tp_tmp, tp)

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
            "method":                "ant_colony_optimisation",
            "seed":                  seed,
            "trial_id":              trial_id,
            **hyperparams,
            "val_accuracy":          log_entry["val_accuracy"],
            "test_accuracy":         log_entry["test_accuracy"],
            "best_epoch":            log_entry["best_epoch"],
            "total_epochs":          log_entry["total_epochs"],
            "training_time_seconds": round(elapsed, 1),
            "from_cache":            from_cache,
            "timestamp":             log_entry["timestamp"],
        })


# ─── CHECKPOINTING ────────────────────────────────────────────────────────────

def save_checkpoint(
    seed,
    iteration,
    tau,
    best_solution: np.ndarray,
    best_hyperparams,
    evaluated_hashes,
    trial_counter,
    best_val_acc,
    rng_np: np.random.Generator,
    current_gen_done: int = 0,
):
    """
    Atomic checkpoint.  Write to .tmp then os.replace() (rename(2) on POSIX).

    Fields
    ──────
    tau               : pheromone matrix, list-of-lists, variable row length.
    best_solution     : integer index vector of the global best solution.
    best_hyperparams  : HP dict of the global best (never None after ant 0).
    current_gen_done  : number of ants evaluated in the CURRENT iteration.
                        Explicit field — avoids any inference from list lengths.
    rng_state         : numpy bit-generator state for exact reproducibility.

    NOTE: τ is saved AS-IS (state at the start of the current iteration,
    or at the end of the previous one).  On resume, ants in the current
    iteration sample from this τ — identical to what they would have sampled
    if the run had not been interrupted, because τ is updated only AFTER a
    complete iteration.
    """
    ckpt = {
        "iteration":         iteration,
        "current_gen_done":  current_gen_done,
        "tau":               tau_to_json(tau),
        "best_solution":     best_solution.tolist(),
        "best_hyperparams":  best_hyperparams,
        "evaluated_hashes":  list(evaluated_hashes),
        "trial_counter":     trial_counter,
        "best_val_acc":      best_val_acc,
        "rng_state":         rng_np.bit_generator.state,
        "timestamp":         datetime.now().isoformat(),
    }
    p   = checkpoint_path(seed)
    tmp = Path(str(p) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(ckpt, f, indent=2)
    os.replace(tmp, p)          # atomic on POSIX (rename syscall)

    log.info(
        "  [Checkpoint] seed=%d iter=%d iter_done=%d/%d trial=%d/%d best=%.4f",
        seed, iteration, current_gen_done, N_ANTS,
        trial_counter, MAX_EVALUATIONS, best_val_acc,
    )


def load_checkpoint(seed: int):
    """
    Load checkpoint and restore the RNG to the exact state at save time.
    Returns (ckpt_dict, rng) or (None, fresh_rng).

    Back-compat: 'current_gen_done' defaults to 0 if missing (old checkpoints
    restart the iteration — safe because cache hits are free).
    """
    p = checkpoint_path(seed)
    if not p.exists():
        return None, np.random.default_rng(seed)

    with open(p) as f:
        ckpt = json.load(f)

    ckpt["tau"]              = tau_from_json(ckpt["tau"])
    ckpt["best_solution"]    = np.array(ckpt["best_solution"], dtype=int)
    ckpt["evaluated_hashes"] = set(ckpt["evaluated_hashes"])
    ckpt.setdefault("current_gen_done", 0)
    ckpt.setdefault("best_hyperparams", None)

    rng_np = np.random.default_rng(seed)
    rng_np.bit_generator.state = ckpt["rng_state"]

    log.info(
        "  [Checkpoint loaded] seed=%d iter=%d iter_done=%d/%d "
        "trials=%d/%d best=%.4f best_hp=%s",
        seed, ckpt["iteration"], ckpt["current_gen_done"], N_ANTS,
        ckpt["trial_counter"], MAX_EVALUATIONS,
        ckpt["best_val_acc"], ckpt.get("best_hyperparams"),
    )
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
                .map(lambda x, y: (preprocess(x), y), num_parallel_calls=AUTOTUNE)
                .prefetch(AUTOTUNE))
    val_ds   = (val_ds
                .map(lambda x, y: (preprocess(x), y), num_parallel_calls=AUTOTUNE)
                .prefetch(AUTOTUNE))
    test_ds  = (test_ds
                .map(lambda x, y: (preprocess(x), y), num_parallel_calls=AUTOTUNE)
                .prefetch(AUTOTUNE))
    return train_ds, val_ds, test_ds


# ─── MODEL CONSTRUCTION ───────────────────────────────────────────────────────

def build_model(freezing_ratio, dropout_rate, l2_reg,
                num_classes=NUM_CLASSES):
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


def evaluate(
    hyperparams: dict,
    seed: int,
    trial_id: int,
    result_cache: dict,
) -> float:
    """
    Evaluate a configuration.  Cache hits are free; new configs are trained.
    Any training exception is caught, logged, and returns 0.0 (worst fitness)
    so the run continues rather than crashing and losing all state.
    """
    h = config_hash(hyperparams)

    if h in result_cache:
        cached   = result_cache[h]
        val_acc  = cached["val_accuracy"]
        test_acc = cached["test_accuracy"]
        source   = cached.get("source", "unknown")
        test_str = f"{test_acc:.4f}" if test_acc is not None else "N/A"
        log.info(
            "    Trial %03d [CACHE:%s] | val=%.4f test=%s | %s",
            trial_id, source, val_acc, test_str, hyperparams,
        )
        save_trial(
            seed, trial_id, hyperparams,
            history_dict=None,
            val_acc=val_acc, test_acc=test_acc,
            best_epoch=None, total_epochs=None,
            elapsed=0.0, from_cache=True,
        )
        return val_acc

    # ── New evaluation — wrap in try/except ───────────────────────────────────
    # If TensorFlow raises (OOM, CUDA error, corrupt batch, …) we log the
    # error, record the trial as failed (val_acc=0.0), and return 0.0.
    # The caller will checkpoint after this, so no state is lost.
    try:
        val_acc, test_acc, history_dict, best_epoch, total_ep, elapsed = \
            train_and_eval(hyperparams, seed, trial_id)
    except Exception as exc:             # noqa: BLE001
        log.error(
            "    Trial %03d FAILED: %s — recording val=0.0 and continuing.",
            trial_id, exc,
        )
        save_trial(
            seed, trial_id, hyperparams,
            history_dict=None,
            val_acc=0.0, test_acc=None,
            best_epoch=None, total_epochs=None,
            elapsed=0.0, from_cache=False,
        )
        result_cache[h] = {
            "val_accuracy":  0.0,
            "test_accuracy": None,
            "source":        "aco_failed",
        }
        return 0.0

    save_trial(
        seed, trial_id, hyperparams, history_dict,
        val_acc, test_acc, best_epoch, total_ep, elapsed,
        from_cache=False,
    )

    result_cache[h] = {
        "val_accuracy":  val_acc,
        "test_accuracy": test_acc,
        "source":        "aco_summary.csv",
    }

    log.info(
        "    Trial %03d | val=%.4f test=%.4f | ep=%d | %.0fs | %s",
        trial_id, val_acc, test_acc, total_ep, elapsed, hyperparams,
    )
    return val_acc


# ─── ANT COLONY OPTIMISATION — MAIN LOOP ─────────────────────────────────────

def run_aco_seed(seed: int, result_cache: dict) -> float:
    """
    Run the ACO for one seed with bulletproof fault-tolerance.

    Shutdown protocol
    ─────────────────
    When SIGINT or SIGTERM is received, _shutdown_event is set.
    The main loop checks _shutdown_requested() at two safe points:

      Point A — after evaluate() and save_trial() have returned,
                BEFORE updating best_val_acc and writing the ant-level
                checkpoint. At this point the trial is on disk but the
                in-memory state has not yet been updated.

                Correction: we first finish updating in-memory state
                (best_val_acc, best_solution), THEN checkpoint, THEN
                check the shutdown flag. This guarantees the checkpoint
                always reflects the result of the ant just evaluated.

      Point B — after the iteration-level checkpoint (τ updated) is
                written. Same guarantee.

    After the checkpoint is written at either point, we call sys.exit(0)
    — the only place sys.exit() appears in this file.
    """
    setup_dirs(seed)
    cleanup_tmp_checkpoint(seed)

    log.info("\n%s", "=" * 60)
    log.info("  ANT COLONY OPTIMISATION — seed %d", seed)
    log.info("%s", "=" * 60)

    # ── Resume or initialise ──────────────────────────────────────────────────
    ckpt, rng_np = load_checkpoint(seed)

    if ckpt is not None:
        iteration        = ckpt["iteration"]
        tau              = ckpt["tau"]
        best_solution    = ckpt["best_solution"]
        best_hyperparams = ckpt["best_hyperparams"]
        evaluated_hashes = ckpt["evaluated_hashes"]
        trial_counter    = ckpt["trial_counter"]
        best_val_acc     = ckpt["best_val_acc"]
        resume_iter_done = ckpt["current_gen_done"]
        log.info(
            "  Resuming from iteration %d, trial %d/%d, iter_done=%d/%d",
            iteration, trial_counter, MAX_EVALUATIONS,
            resume_iter_done, N_ANTS,
        )
    else:
        iteration        = 0
        tau              = init_pheromones()
        # Sample a first solution immediately so best_solution / best_hyperparams
        # are never None in the first checkpoint.
        best_solution    = aco_construct_solution(tau, rng_np)
        best_hyperparams = indices_to_dict(best_solution)
        evaluated_hashes = set()
        trial_counter    = 0
        best_val_acc     = 0.0
        resume_iter_done = 0

    # ── Helper: checkpoint with current state ─────────────────────────────────
    def _save(it: int, current_gen_done: int = 0):
        save_checkpoint(
            seed, it,
            tau, best_solution, best_hyperparams,
            evaluated_hashes, trial_counter,
            best_val_acc, rng_np,
            current_gen_done=current_gen_done,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Main ACO loop
    # ─────────────────────────────────────────────────────────────────────────
    for it in range(iteration, N_ITERATIONS + 1):

        if trial_counter >= MAX_EVALUATIONS:
            log.info("\n  Budget exhausted (%d trials).", trial_counter)
            break

        log.info(
            "\n  [Iter %d] best_so_far=%.4f trials=%d/%d",
            it, best_val_acc, trial_counter, MAX_EVALUATIONS,
        )

        # On a mid-iteration resume, skip ants already done.
        # For all other iterations, start from ant 0.
        already_done     = resume_iter_done if it == iteration else 0
        resume_iter_done = 0

        iter_solutions: list = []
        iter_fitnesses: list = []

        for ant in range(already_done, N_ANTS):

            if trial_counter >= MAX_EVALUATIONS:
                log.info("  Budget reached mid-iteration at ant %d.", ant)
                break

            # ── Construct & evaluate ──────────────────────────────────────
            solution = aco_construct_solution(tau, rng_np)
            hp       = indices_to_dict(solution)
            h        = config_hash(hp)

            # evaluate() never raises (all exceptions caught internally)
            val_acc = evaluate(hp, seed, trial_counter, result_cache)

            evaluated_hashes.add(h)
            trial_counter += 1

            iter_solutions.append(solution)
            iter_fitnesses.append(val_acc)

            # ── Update in-memory best BEFORE checkpointing ────────────────
            # This ensures the checkpoint always reflects the latest result.
            if val_acc > best_val_acc:
                best_val_acc     = val_acc
                best_hyperparams = hp
                best_solution    = solution.copy()
                log.info(
                    "    ★ New best (iter %d, ant %d): %.4f | %s",
                    it, ant, best_val_acc, best_hyperparams,
                )

            # ── Checkpoint after every ant (SAFE POINT A) ─────────────────
            # τ is NOT updated here; on resume, ants sample from the same
            # pheromone matrix as before the interruption.
            _save(it, current_gen_done=ant + 1)

            # ── Check shutdown flag at safe point A ───────────────────────
            # All state is on disk. Safe to exit.
            if _shutdown_requested():
                log.info(
                    "  [Shutdown] Checkpoint written at iter=%d ant=%d. "
                    "Exiting cleanly.",
                    it, ant,
                )
                sys.exit(0)

        # ── End of iteration: update pheromones ──────────────────────────────
        if len(iter_solutions) == N_ANTS:
            tau = aco_update_pheromones(
                tau,
                best_solution,
                best_val_acc,
                iter_solutions,
                iter_fitnesses,
            )
            tau_max = max(max(row) for row in tau)
            tau_min = min(min(row) for row in tau)
            log.info(
                "  [Iter %d] Pheromones updated. τ_max=%.4f τ_min=%.4f",
                it, tau_max, tau_min,
            )
            # Advance iteration; current_gen_done=0 signals a fresh start
            _save(it + 1, current_gen_done=0)
        else:
            # Partial iteration — budget or shutdown mid-iter.
            # τ is NOT updated: partial deposits would bias future ants.
            log.info(
                "  [Iter %d] Partial (%d/%d ants). Pheromones NOT updated.",
                it, len(iter_solutions), N_ANTS,
            )

        # ── Check shutdown flag at SAFE POINT B ──────────────────────────────
        # Iteration checkpoint (with or without τ update) is already written.
        if _shutdown_requested():
            log.info(
                "  [Shutdown] Checkpoint written after iter %d. "
                "Exiting cleanly.",
                it,
            )
            sys.exit(0)

    log.info(
        "\n  [DONE] seed=%d | best_val_acc=%.4f | best_hyperparams=%s "
        "| total_trials=%d",
        seed, best_val_acc, best_hyperparams, trial_counter,
    )
    return best_val_acc


# ─── VISUALISATION ────────────────────────────────────────────────────────────

def plot_results():
    import matplotlib.pyplot as plt
    import seaborn as sns

    csv_p = summary_csv_path()
    if not csv_p.exists():
        log.info("No results to plot yet.")
        return

    df     = pd.read_csv(csv_p)
    df_aco = df[df["method"] == "ant_colony_optimisation"]

    plots_dir = RESULTS_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    # ── 1. Convergence ───────────────────────────────────────────────────────
    fig, ax    = plt.subplots(figsize=(10, 5))
    all_curves = []

    for seed in SEEDS:
        seed_df = df_aco[df_aco["seed"] == seed].sort_values("trial_id")
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
                label=f"ACO mean (n={len(all_curves)} seeds)")
        ax.fill_between(x, mean - std, mean + std,
                        alpha=0.2, color="darkorange", label="± std")

    ax.set_xlabel("Number of trials")
    ax.set_ylabel("Best validation accuracy (so far)")
    ax.set_title("Ant Colony Optimisation — Convergence")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "aco_convergence.png", dpi=150)
    plt.close()

    # ── 2. Stability boxplot ─────────────────────────────────────────────────
    best_per_seed = df_aco.groupby("seed")["val_accuracy"].max().values
    if len(best_per_seed) > 0:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.boxplot([best_per_seed], labels=["Ant Colony"],
                   patch_artist=True,
                   boxprops=dict(facecolor="darkorange", alpha=0.6))
        ax.set_ylabel("Best validation accuracy")
        ax.set_title(
            f"Stability — ACO\n"
            f"mean={best_per_seed.mean():.4f}  std={best_per_seed.std():.4f}"
        )
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "aco_stability_boxplot.png", dpi=150)
        plt.close()

    # ── 3. HP importance (correlation) ───────────────────────────────────────
    hp_df = df_aco[HP_KEYS + ["val_accuracy"]].copy()
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
        plt.savefig(plots_dir / "aco_hp_importance.png", dpi=150)
        plt.close()

    # ── 4. All trials scatter ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    cached_mask  = (df_aco["from_cache"] == True
                    if "from_cache" in df_aco.columns
                    else pd.Series([False] * len(df_aco)))
    trained_mask = ~cached_mask
    ax.scatter(df_aco.loc[trained_mask, "trial_id"],
               df_aco.loc[trained_mask, "val_accuracy"],
               alpha=0.4, s=20, color="darkorange", label="trained")
    if cached_mask.any():
        ax.scatter(df_aco.loc[cached_mask, "trial_id"],
                   df_aco.loc[cached_mask, "val_accuracy"],
                   alpha=0.6, s=20, color="steelblue", marker="x",
                   label="cache hit")
    ax.set_xlabel("Trial ID")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("ACO — All trials val_accuracy")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "aco_all_trials.png", dpi=150)
    plt.close()

    # ── 5. Pheromone heatmap (last checkpoint per seed) ──────────────────────
    fig, axes = plt.subplots(1, len(SEEDS), figsize=(4 * len(SEEDS), 4),
                             sharey=True)
    if len(SEEDS) == 1:
        axes = [axes]

    for ax, seed in zip(axes, SEEDS):
        p = checkpoint_path(seed)
        if not p.exists():
            ax.set_title(f"Seed {seed}\n(no checkpoint)")
            continue
        with open(p) as f:
            ckpt = json.load(f)
        tau_raw = ckpt["tau"]

        max_n = max(len(row) for row in tau_raw)
        data  = np.full((DIM, max_n), np.nan)
        for d, row in enumerate(tau_raw):
            data[d, :len(row)] = row

        im = ax.imshow(data, aspect="auto", cmap="YlOrRd",
                       vmin=ACO_TAU_MIN, vmax=ACO_TAU_MAX)
        ax.set_yticks(range(DIM))
        ax.set_yticklabels(HP_KEYS, fontsize=8)
        ax.set_xlabel("Value index")
        ax.set_title(f"Seed {seed}\niter={ckpt['iteration']}")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Pheromone matrix τ per seed", y=1.02)
    plt.tight_layout()
    plt.savefig(plots_dir / "aco_pheromone_heatmap.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    log.info("  Plots saved in %s", plots_dir)


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

def main():
    log.info("\n%s", "=" * 60)
    log.info("  ACO HPO — EfficientNetB0 — Apple Leaf Disease")
    log.info("  Seeds            : %s", SEEDS)
    log.info("  Max evaluations  : %d  per seed", MAX_EVALUATIONS)
    log.info("  Colony size      : %d", N_ANTS)
    log.info("  Iterations       : %d", N_ITERATIONS)
    log.info("  ρ (evaporation)  : %s", ACO_RHO)
    log.info("  α (pheromone)    : %s", ACO_ALPHA)
    log.info("  τ_min / τ_max    : %s / %s", ACO_TAU_MIN, ACO_TAU_MAX)
    log.info("  Elitist deposit  : %s", ACO_ELITIST)
    if EXTERNAL_SUMMARY_CSVS:
        log.info("  External caches  :")
        for p in EXTERNAL_SUMMARY_CSVS:
            log.info("    • %s", p)
    else:
        log.info("  External caches  : none")
    log.info("%s", "=" * 60)

    result_cache = build_cache_from_csv()

    results = {}

    for seed in SEEDS:
        ckpt, _ = load_checkpoint(seed)
        if ckpt is not None and ckpt["trial_counter"] >= MAX_EVALUATIONS:
            log.info(
                "\n  Seed %d already complete (%d trials). Skipping.",
                seed, ckpt["trial_counter"],
            )
            results[seed] = ckpt["best_val_acc"]
            continue

        best = run_aco_seed(seed, result_cache)
        results[seed] = best

        # After each seed, check if a shutdown was requested.
        # We finish the seed fully before honoring it here.
        if _shutdown_requested():
            log.info(
                "  [Shutdown] Seed %d finished. "
                "Stopping before next seed as requested.",
                seed,
            )
            break

    log.info("\n%s", "=" * 60)
    log.info("  FINAL RESULTS")
    log.info("%s", "=" * 60)
    accs = list(results.values())
    for seed, acc in results.items():
        log.info("  Seed %d: best_val_acc = %.4f", seed, acc)
    if accs:
        log.info(
            "\n  Mean ± Std : %.4f ± %.4f", np.mean(accs), np.std(accs)
        )
        log.info(
            "  Min / Max  : %.4f / %.4f", np.min(accs), np.max(accs)
        )

    log.info("\n%s", "=" * 60)
    log.info("  BEST HYPERPARAMETERS PER SEED")
    log.info("%s", "=" * 60)
    for seed in SEEDS:
        ckpt, _ = load_checkpoint(seed)
        if ckpt:
            log.info(
                "  Seed %d: val=%.4f | %s",
                seed, ckpt["best_val_acc"], ckpt["best_hyperparams"],
            )

    log.info("\nGenerating plots...")
    plot_results()
    log.info("\nDone. All results saved in: %s", RESULTS_DIR)


if __name__ == "__main__":
    main()