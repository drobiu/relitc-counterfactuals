seed=42
MIN_MASK_FRAC=0.2
MAX_MASK_FRAC=0.55
LEARNING_RATE=5e-5
TRAIN_BATCH_SIZE=16
EVAL_BATCH_SIZE=16
ACCUMULATION_STEPS=2
TRAIN_EPOCHS=10
WARMUP_STEPS=0
LR_SCHEDUDLER='linear'
PATIENCE=4
LOGGING_STRATEGY='steps'
LOGGING_STEPS=10
SAVE_STRATEGY='steps'
SAVE_TOTAL_LIMIT=4
EVAL_STRATEGY='steps'
LOAD_BEST_MODEL_AT_END=true