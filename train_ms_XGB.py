"""
train_ms.py
XGBoost patient-level LOOCV trainer for the MS peptide-array classifier.
Parallel LOOCV + ONNX export (merges both improvements from the Lyme v1.3 work).

Dataset notes (MS):
  - Format: ID, class, then feature columns (e.g. X.4001.IgG ...). The program
    reads ALL non-{ID,class} columns as features, so the SAME script trains the
    50-, 60-, and 336-feature panels — just point --train_data at the matching
    file.
  - The MS cohort has one row per subject (no technical replicates), so
    leave-one-patient-out reduces to leave-one-sample-out. The replicate-aware
    logic still applies transparently (n_replicates = 1 per patient).
  - Per-fold cost at 336 features is ~19x the Lyme case, so the parallel LOOCV
    speedup is large here (24 cores: minutes -> seconds).

Model export:
  - model.onnx via onnxmltools (real ONNX protobuf).
    NOTE: xgb.save_model("model.onnx") does NOT produce valid ONNX protobuf —
    it writes XGBoost's internal binary format and onnxruntime then fails with
    "Protobuf parsing failed". onnxmltools does the proper graph conversion.
  - Inference dependency at deployment is onnxruntime only (no XGBoost needed).

Parallel LOOCV:
  - Every fold is independent and seeded with (base_seed + fold_index), so
    results are bit-identical regardless of execution order (FDA-reproducible).
  - joblib dispatch; default backend THREADS because xgb.train() releases the
    GIL — true parallelism with NO copying of the 425x336 matrix to workers.
  - Each fold pinned to nthread=1 to avoid oversubscription against the outer
    loop. Control with --n_jobs (default -1 = all cores) and --parallel_backend.

Requirements:
  pip install xgboost onnxmltools skl2onnx onnxruntime joblib
"""

import argparse
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from joblib import Parallel, delayed
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)


def _train_one_fold(fold_index, patient_id, X_all, y_all, id_arr,
                    params, num_boost_round, seed):
    """
    Train one leave-one-patient-out fold and score that patient's replicates.

    Pure function of its arguments (no shared mutable state) so it is safe to
    call from any joblib worker. Uses numpy boolean indexing on pre-extracted
    arrays — no DataFrame access inside the parallel region.

    Returns a per-patient detail dict (same schema as the serial version).
    """
    test_mask  = id_arr == patient_id
    train_mask = ~test_mask

    y_test = y_all[test_mask]
    if np.unique(y_test).size > 1:
        raise ValueError(
            f"Patient {patient_id} has replicates with mixed class labels. "
            "Each patient must belong to exactly one class."
        )

    dtrain = xgb.DMatrix(X_all[train_mask], label=y_all[train_mask])
    dtest  = xgb.DMatrix(X_all[test_mask])

    fold_params = params.copy()
    fold_params["seed"]    = seed + fold_index  # order-independent reproducibility
    fold_params["nthread"] = 1                  # pin: outer loop provides the parallelism

    model = xgb.train(fold_params, dtrain, num_boost_round=num_boost_round)

    probs_replicates = model.predict(dtest)
    return {
        "patient_id":      patient_id,
        "n_replicates":    int(len(probs_replicates)),
        "replicate_probs": probs_replicates.tolist(),
        "mean_prob":       float(np.mean(probs_replicates)),
        "true_class":      int(y_test[0]),
    }


def run_patient_level_loocv(df, feature_columns, params, threshold,
                            seed=42, n_jobs=-1, backend="threads"):
    """
    Perform Leave-One-Patient-Out Cross-Validation with XGBoost, in parallel.

    Args:
        df:              DataFrame with 'ID', 'class_binary', and feature columns
        feature_columns: List of feature column names
        params:          XGBoost parameters dict (native format)
        threshold:       Classification threshold
        seed:            Base random seed; fold i uses (seed + i)
        n_jobs:          Worker count (-1 = all cores, 1 = serial)
        backend:         "threads" (recommended) or "processes"/"loky"

    Returns:
        y_true_patient, y_prob_patient, y_pred_patient (np.arrays), patient_results (list)

    Notes:
        Because each fold's seed is (seed + i), the joblib-preserved input order
        guarantees results bit-identical to a serial run. "threads" is preferred:
        xgb.train() releases the GIL, so folds run concurrently with no pickling
        of X_all into worker processes.
    """
    patient_ids = df["ID"].unique()
    n_patients  = len(patient_ids)
    num_boost_round = params.get("num_boost_round", 100)

    # Pre-extract arrays ONCE; workers index these (no per-fold DataFrame ops,
    # and with the thread backend these buffers are shared, not copied).
    X_all  = df[feature_columns].values.astype(np.float32)
    y_all  = df["class_binary"].values
    id_arr = df["ID"].values

    prefer = "threads" if backend == "threads" else None
    jl_backend = None if backend == "threads" else "loky"

    print(f"\nPatient-Level LOOCV (parallel):")
    print(f"  Unique patients : {n_patients}")
    print(f"  Total samples   : {len(df)}")
    print(f"  Avg replicates  : {len(df) / n_patients:.2f}")
    print(f"  n_jobs          : {n_jobs}   backend: {backend}")

    results = Parallel(n_jobs=n_jobs, prefer=prefer, backend=jl_backend, verbose=5)(
        delayed(_train_one_fold)(
            i, pid, X_all, y_all, id_arr, params, num_boost_round, seed
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
    parser.add_argument("--xgb_param_path", required=True,
                        help="Path to XGBoost parameters JSON file")
    parser.add_argument("--seed",           type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--n_jobs",         type=int, default=-1,
                        help="Parallel workers for LOOCV (-1 = all cores, 1 = serial)")
    parser.add_argument("--parallel_backend", default="threads",
                        choices=["threads", "processes"],
                        help="LOOCV backend. 'threads' (default) is fastest: "
                             "xgb.train releases the GIL, no data copying. Use "
                             "'processes' only if you hit BLAS/OpenMP contention.")
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
    # Load XGBoost parameters
    # ------------------------------------------------------------------
    print(f"\nLoading XGBoost parameters: {args.xgb_param_path}")
    with open(args.xgb_param_path) as fh:
        params_raw = json.load(fh)

    params = {
        "objective":        "binary:logistic",
        "eval_metric":      "auc",
        "eta":              params_raw.get("learning_rate", params_raw.get("eta", 0.1)),
        "max_depth":        params_raw.get("max_depth", 3),
        "gamma":            params_raw.get("gamma", 0),
        "min_child_weight": params_raw.get("min_child_weight", 1),
        "subsample":        params_raw.get("subsample", 0.8),
        "colsample_bytree": params_raw.get("colsample_bytree", 0.8),
        "seed":             args.seed,
        "verbosity":        params_raw.get("verbosity", 0),
        "num_boost_round":  params_raw.get("n_estimators",
                            params_raw.get("nrounds",
                            params_raw.get("num_boost_round", 100))),
    }

    print("XGBoost parameters:")
    for k, v in params.items():
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
        df, feature_columns, params, args.threshold, args.seed,
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

    dtrain_all  = xgb.DMatrix(X_all, label=y_all)
    final_model = xgb.train(
        params,
        dtrain_all,
        num_boost_round=params["num_boost_round"],
    )

    # --- ONNX export via onnxmltools ----------------------------------------
    # xgb.save_model("model.onnx") saves XGBoost's internal binary format,
    # NOT a valid ONNX protobuf -- onnxruntime raises "Protobuf parsing failed".
    # onnxmltools performs the real ONNX graph conversion.
    #   Input  : float_input  -- float32 (n_samples, n_features)
    #   Output : variable     -- float32 (n_samples,)  P(disease)
    # -----------------------------------------------------------------------
    model_file = output_dir / "model.onnx"
    initial_type = [("float_input", FloatTensorType([None, len(feature_columns)]))]
    onnx_model = convert_xgboost(final_model, initial_types=initial_type)
    with open(model_file, "wb") as fh:
        fh.write(onnx_model.SerializeToString())
    print(f"Final model saved (ONNX via onnxmltools): {model_file}")

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
        "PATIENT-LEVEL LOOCV METRICS (MS classifier)",
        "=" * 50,
        f"Model               : train_ms (parallel LOOCV + ONNX)",
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
    print(f"  model.onnx               — deploy with onnxruntime (no XGBoost needed)")
    print(f"  features_used.txt        — feature order for inference")
    print(f"  threshold.txt            — classification threshold")
    print(f"  loocv_patient_results.tsv— per-patient LOOCV probabilities")
    print(f"  metrics.txt              — summary performance metrics")
    print("=" * 70)


if __name__ == "__main__":
    main()
