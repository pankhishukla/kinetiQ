"""
training_pipeline/config.py
============================
Single source of truth for all training hyperparameters.

WHY a dedicated config file?
    Scattering hyperparameters through multiple files is the #1 cause of
    "I changed it in one place but not the other" bugs.  A single config
    makes experiments reproducible: version-control this file and you have
    a full record of every setting used in every run.

HOW to use:
    from training_pipeline.config import Config
    cfg = Config()
    print(cfg.BATCH_SIZE)   # or just cfg.BATCH_SIZE anywhere
"""

from pathlib import Path

class Config:
    # -----------------------------------------------------------------------
    # PATHS
    # -----------------------------------------------------------------------
    BASE_DIR    = Path(__file__).resolve().parent.parent
    DATASET_DIR = BASE_DIR / "Dataset"
    TRAIN_JSON  = DATASET_DIR / "train" / "_annotations.coco.json"
    TEST_JSON   = DATASET_DIR / "test"  / "_annotations.coco.json"
    MODELS_DIR  = BASE_DIR / "models"
    MODELS_DIR.mkdir(exist_ok=True)   # create if missing

    # Output filenames
    ANOMALY_MODEL_PATH    = MODELS_DIR / "anomaly_model.pkl"
    CLASSIFIER_MODEL_PATH = MODELS_DIR / "classifier_model.pkl"
    FEATURES_CACHE_PATH   = MODELS_DIR / "features_cache.pkl"

    # -----------------------------------------------------------------------
    # DATA LOADING
    # -----------------------------------------------------------------------
    # WHY 0.5 confidence threshold?
    #   Below this the keypoint localisation is noisy enough to corrupt angles.
    KEYPOINT_CONF_THRESHOLD = 0.5

    # Train/val split ratio (fraction used for validation)
    VAL_SPLIT = 0.2

    # Reproducibility seed — set once here, used everywhere
    RANDOM_SEED = 42

    # -----------------------------------------------------------------------
    # FEATURE EXTRACTION
    # -----------------------------------------------------------------------
    # We compute joint angles as features.  The angle names below define
    # which joints to monitor.  Changing this list changes the feature vector
    # dimension — remember to retrain after any change.
    ANGLE_FEATURES = [
        "right_knee", "left_knee",
        "right_hip",  "left_hip",
        "right_elbow","left_elbow",
        "right_shoulder", "left_shoulder",
    ]

    # WHY normalise?  Angles are already scale-invariant, but normalising to
    # zero mean / unit variance makes distance-based algorithms (like k-NN or
    # Isolation Forest with Euclidean kernels) more stable.
    NORMALISE_FEATURES = True

    # -----------------------------------------------------------------------
    # ANOMALY DETECTION MODEL  (Isolation Forest)
    # -----------------------------------------------------------------------
    # n_estimators: more trees = more stable scores, but slower training.
    # 100 is a reasonable default for datasets of < 5,000 samples.
    ANOMALY_N_ESTIMATORS = 100

    # contamination: expected fraction of outlier poses in the training data.
    # 0.1 = assume 10% of training images show bad/unusual form.
    ANOMALY_CONTAMINATION = 0.10

    # -----------------------------------------------------------------------
    # CLASSIFIER MODEL  (Random Forest for exercise + phase recognition)
    # -----------------------------------------------------------------------
    CLASSIFIER_N_ESTIMATORS = 200
    CLASSIFIER_MAX_DEPTH    = None   # None = grow trees fully; int = limit depth
    CLASSIFIER_MIN_SAMPLES  = 2      # min samples required to split a node

    # -----------------------------------------------------------------------
    # TRAINING LOOP  (placeholders — not used until actual training)
    # -----------------------------------------------------------------------
    # Even for scikit-learn models these are useful as documentation of the
    # training configuration that produced a given model file.
    BATCH_SIZE  = 32    # number of samples per gradient step (for NN models)
    NUM_EPOCHS  = 50    # max training epochs (for NN models)
    LEARNING_RATE = 1e-3

    # -----------------------------------------------------------------------
    # EVALUATION
    # -----------------------------------------------------------------------
    # Anomaly threshold: scores below this are flagged as "bad form".
    # This is tuned after training using the validation set.
    ANOMALY_SCORE_THRESHOLD = -0.1

    # -----------------------------------------------------------------------
    # LOGGING
    # -----------------------------------------------------------------------
    VERBOSE = True   # print progress during training
