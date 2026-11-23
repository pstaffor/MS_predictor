#!/bin/bash
python3 train_ms_XGB.py \
  --train_data "train425-336.txt" \
  --output_dir  "." \
  --cv_type "LOOCV" \
  --threshold 0.5 \
  --xgb_param_path "xgb_params.json"