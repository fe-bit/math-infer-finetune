import pandas as pd
import os
from pathlib import Path
import numpy as np
import json
from typing import Literal

dir = Path(__file__).parent.parent.parent.absolute()

blacklist_id_path = dir / "blacklist.txt"
blacklisted_ids = set(blacklist_id_path.open().read().splitlines())
nested_error_lookup = json.load(open(dir / "error_lookup.json"))


def get_error_lookup(example):
    for high_level, mid_levels in nested_error_lookup.items():
        for mid_level, errors in mid_levels.items():
            if example in errors:
                return {"high_level": high_level, "mid_level": mid_level}
            elif example == mid_level:
                return {"high_level": high_level, "mid_level": mid_level}
    
    return {"high_level": example, "mid_level": example}

def check_df(df):
    for idx, row in df.iterrows():
        if row["is_correct"] is False and pd.isna(row["Error Class"]):
            raise ValueError(f"Row {idx} is marked as incorrect but has no Error Class: {row}")

def read_svamp_df(fpath, method):
    try:
        df_r = pd.read_excel(fpath, sheet_name="Error-Analysis", engine="openpyxl")
        df_r["Dataset"] = "SVAMP"
        df_r["Method"] = method
        df_r["Model"] = Path(fpath).parent.name + "/" + Path(fpath).stem
        df_r = df_r[["Dataset", "ID", "Method", "Model", "question", "Question-only", "target_answer", "response", "is_correct", "input_tokens", "output_tokens", "total_tokens", "reasoning", "Error Class", "Type"]]
        return df_r
    except Exception as e:
        print(f"Error processing {fpath}: {e}")
        return None

def get_svamp_df(method: Literal["Direct Prompting", "ReAct", "ReWOO"], include_misclassified: bool=False) -> pd.DataFrame:
    dfs = []
    
    for root, dirs, files in os.walk("."):
        for file in files:
            if not file.endswith(".xlsx"):
                continue
            print(file)
            fpath = os.path.join(root, file)
            df_r = read_svamp_df(fpath, method)
            if df_r is not None:
                dfs.append(df_r)   

    df_dp = pd.concat(dfs)
    def map_is_correct(value):
        if isinstance(value, bool):
            return value

        if value == 'TRUE':
            return True
        elif value == 'FALSE':
            return False
        else:
            return np.nan

    df_dp['is_correct'] = df_dp['is_correct'].map(map_is_correct)

    ids_to_be_dropped = blacklisted_ids # df_dp[df_dp['is_correct'].isna()]['ID'].unique()

    # drop rows of ids_to_be_dropped
    df_dp = df_dp[~df_dp['ID'].isin(ids_to_be_dropped)]
    check_df(df_dp)
    df_dp["Error Type"] = df_dp["Error Class"].apply(lambda x: get_error_lookup(x.strip().lower())["high_level"] if isinstance(x, str) else x)
    df_dp["Error Class"] = df_dp["Error Class"].apply(lambda x: get_error_lookup(x.strip().lower())["mid_level"] if isinstance(x, str) else x)
    if not include_misclassified:
        # Replace Misclassified with None
        df_dp["Error Type"] = df_dp["Error Type"].replace("Misclassified", None)
        df_dp["Error Class"] = df_dp["Error Class"].replace("Misclassified", None)
    return df_dp