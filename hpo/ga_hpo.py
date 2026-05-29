"""
Hyperparameter Optimisation (HPO) for EfficientNetB0 fine-tuning on the
Apple Leaf Disease dataset using a Genetic Algorithm (GA).

"""

import os
import json
import csv
import time
import random
import copy
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

CROSSOVER_PROB  = 0.7
MUTATION_PROB   = 0.2

RESULTS_DIR = Path("results_ga")


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


# ─── GRACEFUL SHUTDOWN ────────────────────────────────────────────────────────

_shutdown_requested = False

def _signal_handler(sig, frame):
    global _shutdown_requested
    print("\n  [Signal] Ctrl+C / SIGTERM detected — finishing current trial then saving checkpoint...")
    _shutdown_requested = True

signal.signal(signal.SIGINT,  _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ─── HELPER UTILITIES ─────────────────────────────────────────────────────────

def config_hash(individual_dict: dict) -> str:
    s = json.dumps(individual_dict, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:10]


def individual_to_dict(ind: list) -> dict:
    return {k: HP_VALUES[i][ind[i]] for i, k in enumerate(HP_KEYS)}


def random_individual(rng: random.Random) -> list:
    return [rng.randint(0, len(vals) - 1) for vals in HP_VALUES]


def crossover(ind1: list, ind2: list, rng: random.Random):
    child1 = copy.deepcopy(ind1)
    child2 = copy.deepcopy(ind2)
    for i in range(len(HP_KEYS)):
        if rng.random() < 0.5:
            child1[i], child2[i] = child2[i], child1[i]
    return child1, child2


def mutate(individual: list, rng: random.Random,
           prob: float = MUTATION_PROB) -> list:
    ind = copy.deepcopy(individual)
    for i in range(len(HP_KEYS)):
        if rng.random() < prob:
            ind[i] = rng.randint(0, len(HP_VALUES[i]) - 1)
    return ind


def tournament_select(population: list, fitnesses: list,
                      k: int = 3, rng: random.Random = None) -> list:
    # FIX 1: clamp k so we never request more candidates than individuals exist.
    safe_k     = min(k, len(population))
    candidates = rng.sample(range(len(population)), safe_k)
    best_idx   = max(candidates, key=lambda i: fitnesses[i])
    return copy.deepcopy(population[best_idx])


# ─── FILE PATHS ───────────────────────────────────────────────────────────────

def seed_dir(seed: int) -> Path:
    return RESULTS_DIR / f"seed_{seed}"

def trial_path(seed: int, trial_id: int) -> Path:
    return seed_dir(seed) / "trials" / f"trial_{trial_id:03d}.json"

def checkpoint_path(seed: int) -> Path:
    return seed_dir(seed) / "ga_checkpoint.json"

def summary_csv_path() -> Path:
    return RESULTS_DIR / "summary.csv"


# ─── DIRECTORY SETUP ──────────────────────────────────────────────────────────

def setup_dirs(seed: int):
    (seed_dir(seed) / "trials").mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ─── TRIAL LOGGING ────────────────────────────────────────────────────────────

def save_trial(seed, trial_id, hyperparams, history_dict,
               val_acc, test_acc, best_epoch, total_epochs, elapsed):
    log = {
        "method":                "genetic_algorithm",
        "seed":                  seed,
        "trial_id":              trial_id,
        "hyperparams":           hyperparams,
        "val_accuracy":          float(val_acc),
        "test_accuracy":         float(test_acc) if test_acc is not None else None,
        "val_loss":              float(min(history_dict["val_loss"])),
        "best_epoch":            int(best_epoch),
        "total_epochs":          int(total_epochs),
        "history":               history_dict,
        "training_time_seconds": round(elapsed, 1),
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
            "total_epochs", "training_time_seconds", "timestamp",
        ])
        if write_header:
            writer.writeheader()
        writer.writerow({
            "method":                "genetic_algorithm",
            "seed":                  seed,
            "trial_id":              trial_id,
            **hyperparams,
            "val_accuracy":          log["val_accuracy"],
            "test_accuracy":         log["test_accuracy"],
            "best_epoch":            best_epoch,
            "total_epochs":          total_epochs,
            "training_time_seconds": round(elapsed, 1),
            "timestamp":             log["timestamp"],
        })


def load_trial(seed: int, trial_id: int):
    p = trial_path(seed, trial_id)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


# ─── GA CHECKPOINTING ─────────────────────────────────────────────────────────

def save_checkpoint(seed, generation, population, fitnesses,
                    evaluated_hashes, trial_counter,
                    best_val_acc, best_hyperparams,
                    selection_pool=None, selection_fitnesses=None,
                    gen_elite=None, gen_elite_fitness=None):

    ckpt = {
        "generation":          generation,
        "population":          population,
        "fitnesses":           fitnesses,
        "selection_pool":      selection_pool      if selection_pool      is not None else population,
        "selection_fitnesses": selection_fitnesses if selection_fitnesses is not None else fitnesses,
        "gen_elite":           gen_elite,
        "gen_elite_fitness":   gen_elite_fitness,
        "evaluated_hashes":    list(evaluated_hashes),
        "trial_counter":       trial_counter,
        "best_val_acc":        best_val_acc,
        "best_hyperparams":    best_hyperparams,
        "timestamp":           datetime.now().isoformat(),
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
    p = checkpoint_path(seed)
    if not p.exists():
        return None
    with open(p) as f:
        ckpt = json.load(f)
    ckpt["evaluated_hashes"] = set(ckpt["evaluated_hashes"])
    if "selection_pool" not in ckpt:
        ckpt["selection_pool"]      = ckpt["population"]
        ckpt["selection_fitnesses"] = ckpt["fitnesses"]
    if "gen_elite" not in ckpt:
        ckpt["gen_elite"]           = None
        ckpt["gen_elite_fitness"]   = None
    print(f"  [Checkpoint loaded] seed={seed} "
          f"gen={ckpt['generation']} "
          f"trials={ckpt['trial_counter']}/{MAX_EVALUATIONS} "
          f"best={ckpt['best_val_acc']:.4f} "
          f"best_hp={ckpt.get('best_hyperparams')}")
    return ckpt


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


# ─── TRAINING AND EVALUATION ──────────────────────────────────────────────────

def evaluate(hyperparams: dict, seed: int, trial_id: int,
             train_ds, val_ds, test_ds) -> float:
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
    save_trial(seed, trial_id, hp, history_dict,
               val_acc, test_acc, best_epoch, total_ep, elapsed)

    print(f"    Trial {trial_id:03d} | val={val_acc:.4f} test={test_acc:.4f} "
          f"| ep={total_ep} | {elapsed:.0f}s | {hp}")

    del model
    tf.keras.backend.clear_session()

    return val_acc


# ─── GENETIC ALGORITHM — MAIN LOOP ────────────────────────────────────────────

def run_ga_seed(seed: int, train_ds, val_ds, test_ds) -> float:
    global _shutdown_requested

    setup_dirs(seed)
    rng = random.Random(seed)

    print(f"\n{'='*60}")
    print(f"  GENETIC ALGORITHM — seed {seed}")
    print(f"{'='*60}")

    # ── Resume from checkpoint ────────────────────────────────────────────────
    ckpt = load_checkpoint(seed)
    if ckpt is not None:
        generation          = ckpt["generation"]
        population          = ckpt["population"]
        fitnesses           = ckpt["fitnesses"]
        # FIX 2: restore the full selection pool (complete previous generation).
        selection_pool      = ckpt["selection_pool"]
        selection_fitnesses = ckpt["selection_fitnesses"]
        # FIX 4: restore the elite chosen at the start of this generation.
        gen_elite           = ckpt["gen_elite"]
        gen_elite_fitness   = ckpt["gen_elite_fitness"]
        evaluated_hashes    = ckpt["evaluated_hashes"]
        trial_counter       = ckpt["trial_counter"]
        best_val_acc        = ckpt["best_val_acc"]
        best_hyperparams    = ckpt.get("best_hyperparams", None)
        resuming_gen        = generation
        print(f"  Resuming from generation {generation}, "
              f"trial {trial_counter}/{MAX_EVALUATIONS}")
    else:
        generation          = 0
        population          = []
        fitnesses           = []
        selection_pool      = []
        selection_fitnesses = []
        gen_elite           = None
        gen_elite_fitness   = None
        evaluated_hashes    = set()
        trial_counter       = 0
        best_val_acc        = 0.0
        best_hyperparams    = None
        resuming_gen        = -1

    # ── Generation 0: random population initialisation ────────────────────────
    if generation == 0 and len(population) < POP_SIZE:
        print(f"\n  [Gen 0] Initialising population ({POP_SIZE} individuals)...")

        while len(population) < POP_SIZE:
            if trial_counter >= MAX_EVALUATIONS or _shutdown_requested:
                break

            ind = random_individual(rng)
            h   = config_hash(individual_to_dict(ind))
            if h in evaluated_hashes:
                continue

            hp      = individual_to_dict(ind)
            val_acc = evaluate(hp, seed, trial_counter,
                               train_ds, val_ds, test_ds)

            population.append(ind)
            fitnesses.append(val_acc)
            evaluated_hashes.add(h)
            trial_counter += 1

            if val_acc > best_val_acc:
                best_val_acc     = val_acc
                best_hyperparams = hp
                print(f"    ★ New best (gen 0): {best_val_acc:.4f} | {best_hyperparams}")

            # Gen-0: selection_pool grows in lockstep with population.
            save_checkpoint(seed, 0, population, fitnesses,
                            evaluated_hashes, trial_counter,
                            best_val_acc, best_hyperparams,
                            selection_pool=population,
                            selection_fitnesses=fitnesses,
                            gen_elite=None,
                            gen_elite_fitness=None)

            if _shutdown_requested:
                print("  [Shutdown] Checkpoint saved. Exiting cleanly.")
                sys.exit(0)

        generation = 1

    # ── Generations 1 … N_GENERATIONS ────────────────────────────────────────
    for gen in range(generation, N_GENERATIONS + 1):
        if trial_counter >= MAX_EVALUATIONS or _shutdown_requested:
            print(f"\n  Budget exhausted or stop requested ({trial_counter} trials).")
            break

        print(f"\n  [Gen {gen}] best_so_far={best_val_acc:.4f} "
              f"trials={trial_counter}/{MAX_EVALUATIONS}")

        if gen != resuming_gen:
            # ── Fresh generation: snapshot the full previous population ──────
            # FIX 2: tournament_select draws from this complete pool throughout
            # the entire generation, including after a future mid-gen resume.
            selection_pool      = copy.deepcopy(population)
            selection_fitnesses = copy.deepcopy(fitnesses)

            # FIX 4: pick the elite NOW from the full previous population and
            # save it so a resume always seeds new_population with the correct
            # individual, regardless of what fitnesses looks like mid-gen.
            best_idx          = int(np.argmax(fitnesses))
            gen_elite         = copy.deepcopy(population[best_idx])
            gen_elite_fitness = fitnesses[best_idx]
        # else: resuming into this gen — selection_pool, gen_elite already
        #       restored from checkpoint above; leave them untouched.

        # Always start new_population with the saved elite (FIX 4).
        new_population = [copy.deepcopy(gen_elite)]
        new_fitnesses  = [gen_elite_fitness]

        attempts = 0
        while len(new_population) < POP_SIZE and trial_counter < MAX_EVALUATIONS:
            if _shutdown_requested:
                break

            attempts += 1
            if attempts > 200:
                child    = random_individual(rng)
                attempts = 0
            else:
                # FIX 2: always select from the full previous generation.
                p1 = tournament_select(selection_pool, selection_fitnesses,
                                       k=3, rng=rng)
                p2 = tournament_select(selection_pool, selection_fitnesses,
                                       k=3, rng=rng)

                if rng.random() < CROSSOVER_PROB:
                    child, _ = crossover(p1, p2, rng)
                else:
                    child = copy.deepcopy(p1)

                child = mutate(child, rng, prob=MUTATION_PROB)

            h = config_hash(individual_to_dict(child))
            if h in evaluated_hashes:
                continue

            hp      = individual_to_dict(child)
            val_acc = evaluate(hp, seed, trial_counter,
                               train_ds, val_ds, test_ds)

            new_population.append(child)
            new_fitnesses.append(val_acc)
            evaluated_hashes.add(h)
            trial_counter += 1

            if val_acc > best_val_acc:
                best_val_acc     = val_acc
                best_hyperparams = hp
                print(f"    ★ New best (gen {gen}): {best_val_acc:.4f} | {best_hyperparams}")

            save_checkpoint(seed, gen, new_population, new_fitnesses,
                            evaluated_hashes, trial_counter,
                            best_val_acc, best_hyperparams,
                            selection_pool=selection_pool,
                            selection_fitnesses=selection_fitnesses,
                            gen_elite=gen_elite,
                            gen_elite_fitness=gen_elite_fitness)

            if _shutdown_requested:
                print("  [Shutdown] Checkpoint saved. Exiting cleanly.")
                sys.exit(0)

        population   = new_population
        fitnesses    = new_fitnesses
        resuming_gen = -1   # subsequent gens always take a fresh snapshot

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
    df_ga = df[df["method"] == "genetic_algorithm"]

    plots_dir = RESULTS_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    # ── 1. Convergence ──────────────────────────────────────────────────────
    fig, ax    = plt.subplots(figsize=(10, 5))
    all_curves = []

    for seed in SEEDS:
        seed_df = df_ga[df_ga["seed"] == seed].sort_values("trial_id")
        if len(seed_df) == 0:
            continue
        curve = seed_df["val_accuracy"].cummax().values
        all_curves.append(curve)
        ax.plot(range(1, len(curve) + 1), curve,
                alpha=0.3, color="steelblue", linewidth=1)

    if all_curves:
        max_len = max(len(c) for c in all_curves)
        padded  = np.array([
            np.pad(c, (0, max_len - len(c)), mode="edge")
            for c in all_curves
        ])
        mean = padded.mean(axis=0)
        std  = padded.std(axis=0)
        x    = np.arange(1, max_len + 1)
        ax.plot(x, mean, color="steelblue", linewidth=2.5,
                label=f"GA mean (n={len(all_curves)} seeds)")
        ax.fill_between(x, mean - std, mean + std,
                        alpha=0.2, color="steelblue", label="± std")

    ax.set_xlabel("Number of trials")
    ax.set_ylabel("Best validation accuracy (so far)")
    ax.set_title("Genetic Algorithm — Convergence")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "ga_convergence.png", dpi=150)
    plt.close()

    # ── 2. Stability boxplot ────────────────────────────────────────────────
    best_per_seed = df_ga.groupby("seed")["val_accuracy"].max().values
    if len(best_per_seed) > 0:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.boxplot([best_per_seed], labels=["Genetic Algorithm"],
                   patch_artist=True,
                   boxprops=dict(facecolor="steelblue", alpha=0.6))
        ax.set_ylabel("Best validation accuracy")
        ax.set_title(
            f"Stability — GA\n"
            f"mean={best_per_seed.mean():.4f}  std={best_per_seed.std():.4f}"
        )
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "ga_stability_boxplot.png", dpi=150)
        plt.close()

    # ── 3. HP importance ────────────────────────────────────────────────────
    hp_df = df_ga[HP_KEYS + ["val_accuracy"]].copy()
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
        plt.savefig(plots_dir / "ga_hp_importance.png", dpi=150)
        plt.close()

    # ── 4. All trials scatter ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(df_ga["trial_id"], df_ga["val_accuracy"],
               alpha=0.4, s=20, color="steelblue")
    ax.set_xlabel("Trial ID")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("GA — All trials val_accuracy")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "ga_all_trials.png", dpi=150)
    plt.close()

    print(f"  Plots saved in {plots_dir}")


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  GA HPO — EfficientNetB0 — Apple Leaf Disease")
    print(f"  Seeds            : {SEEDS}")
    print(f"  Max evaluations  : {MAX_EVALUATIONS}  per seed")
    print(f"  Population size  : {POP_SIZE}")
    print(f"  Generations      : {N_GENERATIONS}")
    print(f"  Budget check     : {POP_SIZE} + {N_GENERATIONS} × "
          f"{POP_SIZE - 1} = "
          f"{POP_SIZE + N_GENERATIONS * (POP_SIZE - 1)} CNN trainings")
    print("=" * 60)

    _, val_ds, test_ds = load_datasets(batch_size=16)

    results = {}

    for seed in SEEDS:
        ckpt = load_checkpoint(seed)
        if ckpt is not None and ckpt["trial_counter"] >= MAX_EVALUATIONS:
            print(f"\n  Seed {seed} already complete "
                  f"({ckpt['trial_counter']} trials). Skipping.")
            results[seed] = ckpt["best_val_acc"]
            continue

        train_ds, _, _ = load_datasets(batch_size=16)
        best = run_ga_seed(seed, train_ds, val_ds, test_ds)
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
        ckpt = load_checkpoint(seed)
        if ckpt:
            print(f"  Seed {seed}: val={ckpt['best_val_acc']:.4f} | {ckpt['best_hyperparams']}")

    print("\nGenerating plots...")
    plot_results()
    print(f"\nDone. All results saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()