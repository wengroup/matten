from typing import Optional, Tuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split


def train_test_split_dataframe(
    df: pd.DataFrame,
    test_size: Union[int, float],
    stratify: Optional[str] = None,
    random_state: int = 35,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a pandas dataframe into a train dataframe and a test dataframe.

    Args:
        df: dataframe to split
        test_size: If float, should be between 0.0 and 1.0 and represent the proportion
            of the dataset to include in the test split. If int, represents the
            absolute number of test samples.
        stratify: name of the column used as label to do stratified split. This is
            mainly used for classification dataset. If `None`, do not perform stratified
            split.
        random_state: random seed to shuffle the data.

    Returns:
        train: the train split
        test: the test split
    """

    index = list(range(len(df)))

    if stratify is not None:
        stratify = df[stratify].tolist()

    train_index, test_index = train_test_split(
        index, test_size=test_size, random_state=random_state, stratify=stratify
    )

    train = df.iloc[train_index]
    test = df.iloc[test_index]

    return train, test


def train_val_test_split_dataframe(
    df: pd.DataFrame,
    val_size: Union[int, float],
    test_size: Union[int, float],
    stratify: Optional[str] = None,
    random_state: int = 35,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    """
    Split a pandas dataframe into a train dataframe and a test dataframe.

    Args:
        df: dataframe to split
        val_size : If float, should be between 0.0 and 1.0 and represent the proportion
            of the dataset to include in the val split. If int, represents the
            absolute number of val samples.
        test_size : If float, should be between 0.0 and 1.0 and represent the proportion
            of the dataset to include in the test split. If int, represents the
            absolute number of test samples.
        stratify: name of the column used as label to do stratified split. This is
            mainly used for classification dataset. If `None`, do not perform stratified
            split.
        random_state: random seed to shuffle the data.

    Returns:
        train: the train split
        val: the val split
        test: the test split
    """

    if isinstance(val_size, float) and isinstance(test_size, float):
        assert val_size + test_size < 1
        # convert to int for train/val split below
        n = len(df)
        val_size = int(n * val_size)
        test_size = int(n * test_size)
    elif isinstance(val_size, int) and isinstance(test_size, int):
        assert val_size + test_size < len(df)
    else:
        raise ValueError(
            "`val_size` and `test_size` should be either `int` or `float`. "
            f"Got `{type(val_size)}` and `{type(test_size)}, respectively."
        )

    dev, test = train_test_split_dataframe(
        df, test_size=test_size, random_state=random_state, stratify=stratify
    )

    train, val = train_test_split_dataframe(
        dev, test_size=val_size, random_state=random_state, stratify=stratify
    )

    return train, val, test
