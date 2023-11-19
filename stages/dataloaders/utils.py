from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split


def make_split(df: pd.DataFrame,
               stratify: bool = True,
               subset: Optional[float] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Makes a simple sklearn based train/test split
    Args:
        df: DataFrame to split
        stratify: Whether to stratify the split or not. If `True`, the split will be made on the `label` column
        subset: Percentage of the dataset to keep (if `None`, the whole dataset will be used)
    Returns:
        Tuple containing the train and test DataFrames
    """

    stratify_option: Optional[pd.Series] = df["label"] if stratify else None

    # Reduce dataset size if subset percentage is provided
    if subset is not None:
        data, _ = train_test_split(df, stratify=stratify_option, train_size=subset)
    else:
        data = df

    df_train, df_test = train_test_split(data, stratify=stratify_option)

    return df_train, df_test
