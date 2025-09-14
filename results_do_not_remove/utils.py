import pandas as pd
import os
from pathlib import Path
import numpy as np
import json

blacklist_id_path = Path(__file__).parent / "blacklist.txt"
blacklisted_ids = set(blacklist_id_path.open().read().splitlines())
nested_error_lookup = json.load(open("error_lookup.json"))


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
        df_r["Model"] = Path(fpath).stem
        df_r = df_r[["Dataset", "ID", "Method", "Model", "question", "Question-only", "target_answer", "response", "is_correct", "input_tokens", "output_tokens", "total_tokens", "reasoning", "Error Class", "Type"]]
        return df_r
    except Exception as e:
        print(f"Error processing {fpath}: {e}")
        return None

def get_svamp_df(include_misclassified: bool=False):
    dfs = []
    print("DP")
    for root, dirs, files in os.walk("dp/SVAMP"):
        for file in files:
            if not file.endswith(".xlsx"):
                continue
            print(file)
            fpath = os.path.join(root, file)
            df_r = read_svamp_df(fpath, "Direktes Prompting")
            if df_r is not None:
                dfs.append(df_r)

    print("DP-Reflection")
    for root, dirs, files in os.walk("dp_reflex/SVAMP"):
        for file in files:
            if not file.endswith(".xlsx"):
                continue
            print(file)
            fpath = os.path.join(root, file)
            df_r = read_svamp_df(fpath, "DP-Reflection")
            if df_r is not None:
                dfs.append(df_r)


    print("ReAct")
    for root, dirs, files in os.walk("react/SVAMP"):
        for file in files:
            if not file.endswith(".xlsx"):
                continue
            print(file)
            fpath = os.path.join(root, file)
            df_r = read_svamp_df(fpath, "ReAct")
            if df_r is not None:
                dfs.append(df_r)
    print("ReWOO")
    for root, dirs, files in os.walk("rewoo_v2/SVAMP"):
        for file in files:
            if not file.endswith(".xlsx"):
                continue
            print(file)
            fpath = os.path.join(root, file)
            df_r = read_svamp_df(fpath, "ReWOO")
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
    # check_df(df_dp)
    df_dp["Error Type"] = df_dp["Error Class"].apply(lambda x: get_error_lookup(x.strip().lower())["high_level"] if isinstance(x, str) else x)
    df_dp["Error Class"] = df_dp["Error Class"].apply(lambda x: get_error_lookup(x.strip().lower())["mid_level"] if isinstance(x, str) else x)
    if not include_misclassified:
        # Replace Misclassified with None
        df_dp["Error Type"] = df_dp["Error Type"].replace("Misclassified", None)
        df_dp["Error Class"] = df_dp["Error Class"].replace("Misclassified", None)
    return df_dp


def get_gsm8k(include_misclassified: bool=False):
    dfs = []
    print("GSM8K")
    for root, dirs, files in os.walk("dp/GSM8K"):
        for file in files:
            if not file.endswith(".xlsx"):
                continue
            print(file)
            fpath = os.path.join(root, file)
            df_r = pd.read_excel(fpath, sheet_name="Error-Analysis", engine="openpyxl")
            df_r["Dataset"] = "GSM8K"
            df_r["Method"] = "Direktes Prompting"
            df_r["Model"] = Path(fpath).stem.replace("_", ":")
            df_r = df_r[["Dataset", "Method", "Model", "question", "target_answer", "response", "is_correct", "input_tokens", "output_tokens", "total_tokens", "reasoning", "Error Class"]]
            dfs.append(df_r)
    print("DP-Reflection")
    for root, dirs, files in os.walk("dp_reflex/GSM8K"):
        for file in files:
            if not file.endswith(".xlsx"):
                continue
            print(file)
            fpath = os.path.join(root, file)
            df_r = pd.read_excel(fpath, sheet_name="Error-Analysis", engine="openpyxl")
            df_r["Dataset"] = "GSM8K"
            df_r["Method"] = "DP-Reflection"
            df_r["Model"] = Path(fpath).stem.replace("_", ":")
            df_r = df_r[["Dataset", "Method", "Model", "question", "target_answer", "response", "is_correct", "input_tokens", "output_tokens", "total_tokens", "reasoning", "Error Class"]]
            dfs.append(df_r)
    print("ReAct")
    for root, dirs, files in os.walk("react/GSM8K"):
        for file in files:
            if not file.endswith(".xlsx"):
                continue
            print(file)
            fpath = os.path.join(root, file)
            df_r = pd.read_excel(fpath, sheet_name="Error-Analysis", engine="openpyxl")
            df_r["Dataset"] = "GSM8K"
            df_r["Method"] = "ReAct"
            df_r["Model"] = Path(fpath).stem.replace("_", ":")
            df_r = df_r[["Dataset", "Method", "Model", "question", "target_answer", "response", "is_correct", "input_tokens", "output_tokens", "total_tokens", "reasoning", "Error Class"]]
            dfs.append(df_r)
    print("ReWOO")
    for root, dirs, files in os.walk("rewoo_v2/GSM8K"):
        for file in files:
            if not file.endswith(".xlsx"):
                continue
            print(file)
            fpath = os.path.join(root, file)
            try:
                df_r = pd.read_excel(fpath, sheet_name="Error-Analysis", engine="openpyxl")
                df_r["Dataset"] = "GSM8K"
                df_r["Method"] = "ReWOO"
                df_r["Model"] = Path(fpath).stem.replace("_", ":")
                df_r = df_r[["Dataset", "Method", "Model", "question", "target_answer", "response", "is_correct", "input_tokens", "output_tokens", "total_tokens", "reasoning", "Error Class"]]
                dfs.append(df_r)
            except Exception as e:
                print(f"Error processing {fpath}: {e}")

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
    # check_df(df_dp)
    df_dp["Error Type"] = df_dp["Error Class"].apply(lambda x: get_error_lookup(x.strip().lower())["high_level"] if isinstance(x, str) else x)
    df_dp["Error Class"] = df_dp["Error Class"].apply(lambda x: get_error_lookup(x.strip().lower())["mid_level"] if isinstance(x, str) else x)
    if not include_misclassified:
        # Replace Misclassified with None
        df_dp["Error Type"] = df_dp["Error Type"].replace("Misclassified", None)
        df_dp["Error Class"] = df_dp["Error Class"].replace("Misclassified", None)
    return df_dp


def get_all(include_misclassified: bool=False):
    df = pd.concat([get_gsm8k(include_misclassified=include_misclassified), get_svamp_df(include_misclassified=include_misclassified)])
    return df[["Dataset", "Method", "Model", "question", "target_answer", "is_correct", "Error Class", "Error Type", "reasoning", "input_tokens", "output_tokens", "total_tokens"]]