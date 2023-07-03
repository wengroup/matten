import pandas as pd

from matten.data.split import train_test_split_dataframe


def create_df():

    df = pd.DataFrame(
        {
            "x": list(range(5)),
            "y": [f"a{i%2}" for i in range(5)],
        }
    )

    return df


def test_train_test_split_dataframe():
    df = create_df()

    ref_train = pd.DataFrame({"x": [0, 3, 1], "y": ["a0", "a1", "a1"]})
    ref_test = pd.DataFrame({"x": [2, 4], "y": ["a0", "a0"]})

    train, test = train_test_split_dataframe(df, test_size=2, random_state=35)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    assert train.equals(ref_train)
    assert test.equals(ref_test)

    train, test = train_test_split_dataframe(df, test_size=0.4, random_state=35)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    assert train.equals(ref_train)
    assert test.equals(ref_test)

    # stratify
    ref_train2 = pd.DataFrame({"x": [3, 0, 4], "y": ["a1", "a0", "a0"]})
    ref_test2 = pd.DataFrame({"x": [2, 1], "y": ["a0", "a1"]})
    train, test = train_test_split_dataframe(
        df, test_size=2, random_state=35, stratify="y"
    )
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    assert train.equals(ref_train2)
    assert test.equals(ref_test2)

    train, test = train_test_split_dataframe(
        df, test_size=0.4, random_state=35, stratify="y"
    )
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    assert train.equals(ref_train2)
    assert test.equals(ref_test2)
