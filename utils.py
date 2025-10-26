# utils.py
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset

def DataUtils(dataset_name, test_size=0.2, val_size=0.2, seed=42):
    """Loads dataset from Hugging Face and splits into train, val, test."""

    dataset = load_dataset(dataset_name)
    # Use the first available split if multiple exist
    ds = dataset[next(iter(dataset.keys()))]

    # Convert to pandas for easy splitting
    df = ds.to_pandas()

    # Split into train and temp (val+test)
    train_df, temp_df = train_test_split(df, test_size=(val_size + test_size), random_state=seed, shuffle=True)
    val_rel_size = val_size / (val_size + test_size)

    # Split temp into val and test
    val_df, test_df = train_test_split(temp_df, test_size=(1 - val_rel_size), random_state=seed, shuffle=True)

    # Convert back to Hugging Face datasets
    from datasets import Dataset
    return {
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(val_df),
        "test": Dataset.from_pandas(test_df)
    }


def evaluate_model(y_true, y_pred):
    """Compute standard classification metrics."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro F1-Score": f1_score(y_true, y_pred, average="macro"),
        "Macro Precision": precision_score(y_true, y_pred, average="macro"),
        "Macro Recall": recall_score(y_true, y_pred, average="macro"),
    }

def results_table(results_dict):
    """Convert dict of model results to a clean DataFrame."""
    df = pd.DataFrame(results_dict).T.reset_index().rename(columns={"index": "Algorithm"})
    return df
