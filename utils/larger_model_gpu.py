import os
from typing import Literal

TRAINING_BATCH_SIZE_ENV_VARIABLE_NAME = "GRE_LARGE_MODEL_TRAIN_BATCH_SIZE"
DEFAULT_TRAINING_BATCH_SIZE = 16

INFERENCE_BATCH_SIZE_ENV_VARIABLE_NAME = "GRE_LARGE_MODEL_INFERENCE_BATCH_SIZE"
DEFAULT_INFERENCE_BATCH_SIZE = 128

NUM_WORKERS_ENV_VARIABLE_NAME = "GRE_LARGE_MODEL_NUM_WORKERS"
DEFAULT_NUM_WORKERS = 8

TORCH_MATMUL_PRECISION_ENV_VARIABLE_NAME = "GRE_TORCH_MATMUL_PRECISION"
DEFAULT_TORCH_MATMUL_PRECISION: Literal['high', 'medium'] = "high"

FINE_TUNE_EPOCHS_ENV_VARIABLE_NAME = "GRE_FINE_TUNE_EPOCHS"
DEFAULT_FINE_TUNE_EPOCHS = 8


def resolve_training_batch_size():
    """Gets training batch size from env or default"""
    if TRAINING_BATCH_SIZE_ENV_VARIABLE_NAME in os.environ.keys():
        env_training_batch_size = int(os.environ[TRAINING_BATCH_SIZE_ENV_VARIABLE_NAME])
        return env_training_batch_size
    else:
        return DEFAULT_TRAINING_BATCH_SIZE


def resolve_inference_batch_size():
    """Gets inference batch size from env or default"""
    if INFERENCE_BATCH_SIZE_ENV_VARIABLE_NAME in os.environ.keys():
        env_inference_batch_size = int(os.environ[INFERENCE_BATCH_SIZE_ENV_VARIABLE_NAME])
        return env_inference_batch_size
    else:
        return DEFAULT_INFERENCE_BATCH_SIZE


def resolve_num_workers():
    """Gets number of workers from env or default"""
    if NUM_WORKERS_ENV_VARIABLE_NAME in os.environ.keys():
        env_num_workers = int(os.environ[NUM_WORKERS_ENV_VARIABLE_NAME])
        return env_num_workers
    else:
        return DEFAULT_NUM_WORKERS


def resolve_torch_matmul_precision():
    """Gets torch matmul precision from env or default"""
    if TORCH_MATMUL_PRECISION_ENV_VARIABLE_NAME in os.environ.keys():
        env_torch_precision = os.environ[TORCH_MATMUL_PRECISION_ENV_VARIABLE_NAME]
        return env_torch_precision
    else:
        return DEFAULT_TORCH_MATMUL_PRECISION


def resolve_fine_tune_epochs():
    """Gets number of epochs for fine tuning from env or default"""
    if FINE_TUNE_EPOCHS_ENV_VARIABLE_NAME in os.environ.keys():
        env_fine_tune_epochs = int(os.environ[FINE_TUNE_EPOCHS_ENV_VARIABLE_NAME])
        return env_fine_tune_epochs
    else:
        return DEFAULT_FINE_TUNE_EPOCHS
