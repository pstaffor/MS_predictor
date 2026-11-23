"""
train_ms_gbm.py
Friedman Gradient Boosting (sklearn GradientBoostingClassifier) trainer for the
MS peptide-array classifier. Parallel patient-level LOOCV + ONNX export.

Classifier: sklearn.ensemble.GradientBoostingClassifier (Friedman stochastic GBM).
Chosen as the deployment model for the MS panel; see gbm_params.json for knobs.

Dataset notes (MS):
  - Format: ID, class, then feature columns (e.g. X.4001.IgG ...). All non-{ID,class}
    columns are treated as features, so the SAME script trains the 50-, 60-, and
    336-feature panels — point --train_data at the matching file.
  - The MS cohort has one row per subject (no technical replicates), so
    leave-one-patient-out reduces to leave-one-sample-out. The replicate-aware
    logic still applies (n_replicates = 1 per patient).

Model export:
  - model.onnx via skl2onnx.convert_sklearn with options={'zipmap': False}.
    zipmap=False is REQUIRED: without it skl2onnx emits a ZipMap (sequence of
    dicts) and onnxruntime returns a list-of-dicts instead of a tensor, which
    breaks tensor-based inference. With it, the graph outputs [label, proba]
    where proba is a clean (n, 2) float tensor; column 1 is P(disease).
  - Inference dependency at deployment is onnxruntime only (no sklearn needed).
  - The companion predict script's ONNX helper already handles the two-output
    [label, proba] form, so prediction needs no changes for GBM vs XGBoost.

Parallel LOOCV:
  - Every fold is independent and seeded with random_state=(base_seed + fold_index),
    so results are order-independent and reproducible.
  - PRODUCTION-ENVIRONMENT LIMITATION: GradientBoostingClassifier.fit is
    sequential boosting and does not release the GIL, so the thread backend gives
    no speedup here (unlike the XGBoost pipeline). The LOOCV therefore defaults to
    the PROCESS backend (joblib/loky). X_all (~0.6 MB at 425x336 float32) is
    pickled to workers, which is negligible against per-fold fit cost. Friedman
    GBM fit is ~4x slower per fold than XGBoost, so parallelism matters more here.
  - Control with --n_jobs (default -1 = all cores) and --parallel_backend
    (default 'processes'; 'threads' is available but will not parallelize GBM).

Requirements:
  pip install scikit-learn skl2onnx onnxruntime joblib
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)


def _train_one_fold(fold_index, patient_id, X_all, y_all, id_arr,
                    sk_params, seed):
    """
    Train one leave-one-patient-out fold with Friedman GBM and score that
    patient's replicates.

    Pure function of its arguments (no shared mutable state) so it is safe to
    call from any joblib worker. Uses numpy boolean indexing on pre-extracted
    arrays — no DataFrame access inside the parallel region.

    Returns a per-patient detail dict (same schema as the XGBoost version).
    """
    test_mask  = id_arr == patient_id
    train_mask = ~test_mask

    y_test = y_all[test_mask]
    if np.unique(y_test).size > 1:
        raise ValueError(
            f"Patient {patient_id} has replicates with mixed class labels. "
            "Each patient must belong to exactly one class."
        )

    # random_state = seed + fold_index -> order-independent reproducibility
    fold_params = dict(sk_params)
    fold_params["random_state"] = seed + fold_index

    clf = GradientBoostingClassifier(**fold_params)
    clf.fit(X_all[train_mask], y_all[train_mask])

    probs_replicates = clf.predict_proba(X_all[test_mask])[:, 1]  # P(disease)
    return {
        "patient_id":      patient_id,
        "n_replicates":    int(len(probs_replicates)),
        "replicate_probs": probs_replicates.tolist(),
        "mean_prob":       float(np.mean(probs_replicates)),
        "true_class":      int(y_test[0]),
    }


def run_patient_level_loocv(df, feature_columns, sk_params, threshold,
                            seed=42, n_jobs=-1, backend="processes"):
    """
    Perform Leave-One-Patient-Out Cross-Validation with Friedman GBM, in parallel.

    Args:
        df:              DataFrame with 'ID', 'class_binary', and feature columns
        feature_columns: List of feature column names
        sk_params:       GradientBoostingClassifier kwargs
        threshold:       Classification threshold
        seed:            Base random seed; fold i uses random_state=(seed + i)
        n_jobs:          Worker count (-1 = all cores, 1 = serial)
        backend:         "processes" (default, required for real GBM speedup) or
                         "threads" (available but will NOT parallelize GBM)

    Returns:
        y_true_patient, y_prob_patient, y_pred_patient (np.arrays), patient_results (list)

    Notes:
        random_state=(seed + i) makes the joblib-preserved input order yield
        results independent of execution order. The PROCESS backend is the
        default because GradientBoostingClassifier.fit does not release the GIL;
        the thread backend would serialize. X_all is pickled to each worker
        (~0.6 MB at 425x336), negligible against per-fold fit cost.
    """
    patient_ids = df["ID"].unique()
    n_patients  = len(patient_ids)

    # Pre-extract arrays ONCE; workers index these.
    X_all  = df[feature_columns].values.astype(np.float32)
    y_all  = df["class_binary"].values
    id_arr = df["ID"].values

    if backend == "processes":
        prefer, jl_backend = None, "loky"
    else:
        prefer, jl_backend = "threads", None

    print(f"\nPatient-Level LOOCV (Friedman GBM, parallel):")
    print(f"  Unique patients : {n_patients}")
    print(f"  Total samples   : {len(df)}")
    print(f"  Avg replicates  : {len(df) / n_patients:.2f}")
    print(f"  n_jobs          : {n_jobs}   backend: {backend}")
    if backend == "threads":
        print("  WARNING: thread backend will NOT parallelize GBM (GIL-bound fit).")

    results = Parallel(n_jobs=n_jobs, prefer=prefer, backend=jl_backend, verbose=5)(
        delayed(_train_one_fold)(
            i, pid, X_all, y_all, id_arr, sk_params, seed
        )
        for i, pid in enumerate(patient_ids)
    )
    # joblib preserves submission order -> results align with patient_ids order.

    y_true_patient = np.array([r["true_class"] for r in results])
    y_prob_patient = np.array([r["mean_prob"]  for r in results])
    y_pred_patient = (y_prob_patient >= threshold).astype(int)

    return y_true_patient, y_prob_patient, y_pred_patient, results


def main():
    parser = argparse.ArgumentParser(
        description="Train XGBoost model with patient-level LOOCV and export as ONNX"
    )
    parser.add_argument("--train_data",     required=True,
                        help="Tab-delimited training data: ID, class, features")
    parser.add_argument("--output_dir",     default=".",
                        help="Directory for all output files")
    parser.add_argument("--cv_type",        default="LOOCV",
                        help="Cross-validation type (only LOOCV supported)")
    parser.add_argument("--threshold",      type=float, default=0.2,
                        help="Classification threshold (default: 0.2)")
    parser.add_argument("--gbm_param_path", required=True,
                        help="Path to GBM parameters JSON file (e.g. gbm_params.json)")
    parser.add_argument("--seed",           type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--n_jobs",         type=int, default=-1,
                        help="Parallel workers for LOOCV (-1 = all cores, 1 = serial)")
    parser.add_argument("--parallel_backend", default="processes",
                        choices=["processes", "threads"],
                        help="LOOCV backend. 'processes' (default, loky) is REQUIRED "
                             "for real speedup: GradientBoostingClassifier.fit is "
                             "GIL-bound, so 'threads' will NOT parallelize it.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # ------------------------------------------------------------------
    # Load training data
    # ------------------------------------------------------------------
    print(f"\nLoading training data: {args.train_data}")
    df = pd.read_csv(args.train_data, sep="\t")
    print(f"Shape: {df.shape}")

    if "ID" not in df.columns or "class" not in df.columns:
        raise ValueError("Training data must contain 'ID' and 'class' columns.")

    feature_columns = [c for c in df.columns if c not in ("ID", "class")]
    print(f"Features ({len(feature_columns)}): {feature_columns}")

    df["class_binary"] = (df["class"] == "disease").astype(int)

    patient_counts = df["ID"].value_counts()
    reps = patient_counts.values
    print(f"\nPatient summary:")
    print(f"  Unique patients : {len(patient_counts)}")
    print(f"  Total samples   : {len(df)}")
    print(f"  Replicates      : min={reps.min()}  max={reps.max()}  mean={reps.mean():.2f}")

    first_reps = df.drop_duplicates(subset="ID", keep="first")
    n_disease  = (first_reps["class"] == "disease").sum()
    n_healthy  = len(first_reps) - n_disease
    print(f"  Class balance   : {n_disease} disease  /  {n_healthy} healthy")

    # ------------------------------------------------------------------
    # Save feature list and threshold (consumed by predict v1.3)
    # ------------------------------------------------------------------
    features_file = output_dir / "features_used.txt"
    with open(features_file, "w") as fh:
        for feat in feature_columns:
            fh.write(feat + "\n")
    print(f"\nFeatures saved : {features_file}")

    threshold_file = output_dir / "threshold.txt"
    with open(threshold_file, "w") as fh:
        fh.write(str(args.threshold))
    print(f"Threshold saved: {threshold_file}")

    # ------------------------------------------------------------------
    # Load GBM parameters
    # ------------------------------------------------------------------
    print(f"\nLoading GBM parameters: {args.gbm_param_path}")
    with open(args.gbm_param_path) as fh:
        params_raw = json.load(fh)

    # Map JSON knobs to GradientBoostingClassifier kwargs. Accept a few aliases
    # so an XGBoost-style param file still works (eta->learning_rate, etc.).
    # max_features: JSON null -> Python None (use all features per split).
    max_features = params_raw.get("max_features", None)

    sk_params = {
        "learning_rate":     params_raw.get("learning_rate", params_raw.get("eta", 0.1)),
        "n_estimators":      params_raw.get("n_estimators",
                             params_raw.get("num_boost_round",
                             params_raw.get("nrounds", 100))),
        "max_depth":         params_raw.get("max_depth", 3),
        "subsample":         params_raw.get("subsample", 0.8),
        "min_samples_leaf":  params_raw.get("min_samples_leaf", 1),
        "min_samples_split": params_raw.get("min_samples_split", 2),
        "max_features":      max_features,
        # random_state is set per-fold (seed + fold_index) and once for the
        # final full-data model below.
    }
    verbosity = params_raw.get("verbosity", 0)

    print("GBM (GradientBoostingClassifier) parameters:")
    for k, v in sk_params.items():
        print(f"  {k}: {v}")

    # ------------------------------------------------------------------
    # LOOCV
    # ------------------------------------------------------------------
    if args.cv_type.upper() != "LOOCV":
        raise ValueError(f"Only LOOCV is supported. Got: {args.cv_type}")

    print("\n" + "=" * 70)
    print("Running Patient-Level Leave-One-Out Cross-Validation")
    print("=" * 70)

    y_true, y_prob, y_pred, patient_results = run_patient_level_loocv(
        df, feature_columns, sk_params, args.threshold, args.seed,
        n_jobs=args.n_jobs, backend=args.parallel_backend
    )

    results_df   = pd.DataFrame(patient_results)
    results_file = output_dir / "loocv_patient_results.tsv"
    results_df.to_csv(results_file, sep="\t", index=False)
    print(f"\nDetailed LOOCV results saved: {results_file}")

    # ------------------------------------------------------------------
    # Train final model on ALL data and export as ONNX
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Training final model on all data")
    print("=" * 70)

    X_all = df[feature_columns].values.astype(np.float32)
    y_all = df["class_binary"].values

    final_model = GradientBoostingClassifier(random_state=args.seed, **sk_params)
    final_model.fit(X_all, y_all)

    # --- ONNX export via skl2onnx -------------------------------------------
    # options={id(model): {"zipmap": False}} is REQUIRED. Without it, skl2onnx
    # appends a ZipMap node and onnxruntime returns a list-of-dicts instead of a
    # tensor. With zipmap disabled the graph outputs:
    #   output[0] label  -- int64 (n_samples,)
    #   output[1] proba  -- float32 (n_samples, 2); column 1 is P(disease)
    # The companion predict script's ONNX helper reads output[1][:, 1], so no
    # prediction-side change is needed relative to the XGBoost pipeline.
    # -----------------------------------------------------------------------
    model_file = output_dir / "model.onnx"
    initial_type = [("float_input", FloatTensorType([None, len(feature_columns)]))]
    onnx_model = convert_sklearn(
        final_model,
        initial_types=initial_type,
        options={id(final_model): {"zipmap": False}},
    )
    with open(model_file, "wb") as fh:
        fh.write(onnx_model.SerializeToString())
    print(f"Final model saved (ONNX via skl2onnx): {model_file}")

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Patient-Level LOOCV Performance Metrics")
    print("=" * 70)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob)
    cm   = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv  = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    metrics_lines = [
        "PATIENT-LEVEL LOOCV METRICS (MS classifier, Friedman GBM)",
        "=" * 50,
        f"Model               : train_ms_gbm (GradientBoostingClassifier + ONNX)",
        f"Features            : {len(feature_columns)}",
        f"Number of patients  : {len(y_true)}",
        f"Threshold           : {args.threshold}",
        f"Random seed         : {args.seed}",
        "",
        f"Accuracy            : {acc:.4f}",
        f"AUC                 : {auc:.4f}",
        f"Sensitivity (Recall): {rec:.4f}",
        f"Specificity         : {spec:.4f}",
        f"Precision (PPV)     : {prec:.4f}",
        f"NPV                 : {npv:.4f}",
        f"F1 Score            : {f1:.4f}",
        "",
        "Confusion Matrix:",
        "                  Predicted Negative  Predicted Positive",
        f"Actual Negative        {tn:6d}              {fp:6d}",
        f"Actual Positive        {fn:6d}              {tp:6d}",
    ]

    metrics_file = output_dir / "metrics.txt"
    with open(metrics_file, "w") as fh:
        fh.write("\n".join(metrics_lines) + "\n")

    print("\n".join(metrics_lines))
    print(f"\nMetrics saved: {metrics_file}")

    print("\n" + "=" * 70)
    print("Training complete!  Artifacts produced:")
    print(f"  model.onnx               — deploy with onnxruntime (no sklearn needed)")
    print(f"  features_used.txt        — feature order for inference")
    print(f"  threshold.txt            — classification threshold")
    print(f"  loocv_patient_results.tsv— per-patient LOOCV probabilities")
    print(f"  metrics.txt              — summary performance metrics")
    print("=" * 70)


if __name__ == "__main__":
    main()
