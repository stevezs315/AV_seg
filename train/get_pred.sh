#!/usr/bin/env bash

# ============================================================
# RRWNet Inference Script
# ============================================================
# Run test.py to generate predictions from a trained model.
# Point --weights to a training output dir containing
# generator_best.pth and config.json.
# ============================================================

DATA=/mnt/nasv3/zs/RRWNet_zs/train/_Data
SAVE_ROOT="eval/_Evaluation_Data"

# ---- Active config ------------------------------------------------
WEIGHTS="__training/RITE_RRWNet_minmax_0.19_spCL_0.05"
IMAGES_PATH="$DATA/RITE/test/enhanced"
MASKS_PATH="$DATA/RITE/test/enhanced_masks"
TEST_NAME="RITE-test"
SAVE_PATH="RITE_RRWNet_ours_gamma_dark_debug"

# ---- RITE-test (other variants) -----------------------------------
# IMAGES_PATH="$DATA/RITE/test/enhanced"
# MASKS_PATH="$DATA/RITE/test/enhanced_masks"
# TEST_NAME="RITE-test"

# ---- LES-AV -------------------------------------------------------
# IMAGES_PATH="$DATA/LES-AV/test/enhanced"
# MASKS_PATH="$DATA/LES-AV/test/enhanced_masks"
# TEST_NAME="LES-AV"

# ---- HRF ----------------------------------------------------------
# IMAGES_PATH="$DATA/HRF_AVLabel_191219/test_karlsson_w1024/enhanced"
# MASKS_PATH="$DATA/HRF_AVLabel_191219/test_karlsson_w1024/enhanced_masks"
# TEST_NAME="HRF"

# ---- Run ----------------------------------------------------------
echo "Weights:  $WEIGHTS"
echo "Images:   $IMAGES_PATH"
echo "Masks:    $MASKS_PATH"
echo "Test:     $TEST_NAME"
echo "Save:     $SAVE_ROOT/$SAVE_PATH"

python3 test.py \
    --weights "$WEIGHTS" \
    --images_path "$IMAGES_PATH" \
    --masks_path "$MASKS_PATH" \
    --test_name "$TEST_NAME" \
    --save_root "$SAVE_ROOT" \
    --save_path "$SAVE_PATH"
