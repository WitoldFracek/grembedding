import numpy as np
import pandas as pd
from loguru import logger


def split_text(df: pd.DataFrame, splitter, column: str = "text") -> pd.DataFrame:
    chunked_text: list[list[str]] = df[column] \
        .apply(lambda txt: splitter.split_text(txt))

    result: pd.DataFrame = df.copy()
    result[column] = chunked_text
    result = result.explode(column=column, ignore_index=True)

    _log_text_split_metrics(chunked_text)
    return result


def _log_text_split_metrics(chunked_text: list[list[str]]):
    chunk_lengths = [len(chunk) for chunks in chunked_text for chunk in chunks]
    mean_chunk_len = np.mean(chunk_lengths)
    stddev_chunk_len = np.std(chunk_lengths)
    smallest_chunk = min(chunk_lengths)
    largest_chunk = max(chunk_lengths)

    logger.info(f"Text splitter results: (num_chunks={len(chunk_lengths)}) - avg chunk len: {mean_chunk_len},"
                f" std: {stddev_chunk_len}, smallest chunk: {smallest_chunk}, largest_chunk: {largest_chunk}.")
