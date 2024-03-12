import os
from typing import Type, Any

import spacy
from loguru import logger

SPACY_MODE_ENV_VARIABLE_NAME = "GRE_SPACY_MODE"
SPACY_MODE_ALLOWED_VALUES = {"cpu", "gpu", "gpu_except_stylometrix"}


def autoconfigure_spacy_mode(caller_type: Type[Any]):
    """
    This sets the correct Spacy execution mode based on GRE_SPACY_MODE env variable
    """
    if SPACY_MODE_ENV_VARIABLE_NAME in os.environ.keys():
        spacy_mode_env = os.environ[SPACY_MODE_ENV_VARIABLE_NAME]
        if spacy_mode_env not in SPACY_MODE_ALLOWED_VALUES:
            logger.warning(f"Environment variable {SPACY_MODE_ENV_VARIABLE_NAME} "
                           f"has invalid value '{spacy_mode_env}' - defaulting to 'cpu'")
            spacy.require_cpu()
        else:
            logger.info(f"Using Spacy mode: {spacy_mode_env}")
            if spacy_mode_env == "gpu":
                spacy.require_gpu()
            elif spacy_mode_env == "gpu_except_stylometrix":
                if caller_type.__name__ == "StyloMetrix":
                    spacy.require_cpu()
                else:
                    spacy.require_cpu()
            elif spacy_mode_env == "cpu":
                spacy.require_gpu()
    else:
        logger.warning(f"Environment variable {SPACY_MODE_ENV_VARIABLE_NAME} not set - defaulting to 'cpu'")
        spacy.require_cpu()
