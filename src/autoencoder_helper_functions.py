from pickle import load, dump
import pandas as pd


def min_max_scaler(val: float, min_val: float, max_val: float) -> float:
    out = (val - min_val) / (max_val - min_val)
    if (pd.isna(out)) & (val >= 1):
        out = 1
    elif (pd.isna(out)) & (val < 1) & (val > 0):
        out = val
    elif (pd.isna(out)) & (val <= 0):
        out = 0
    return out


def scale_dataset(
    df2: pd.DataFrame, experiment_type: str, scale_location: str, rescale: bool = False
) -> pd.DataFrame:
    # rescale rest of columns using minmax scaler (except in [])
    x_columns = [col for col in df2.columns.tolist() if col not in []]

    min_max_scaling_dict = {}
    df2_minmax = df2.copy()

    if rescale:
        for col in x_columns:
            min_val = df2[col].min()
            max_val = df2[col].max()

            if max_val == 0:
                max_val = 1

            min_max_scaling_dict[col] = {"min": min_val, "max": max_val}
            df2_minmax[col] = df2_minmax[col].apply(
                min_max_scaler, args=[min_val, max_val]
            )

        dump(
            min_max_scaling_dict,
            open(f"{scale_location}{experiment_type}_min_max_scaling_dict.pkl", "wb"),
        )
        print(f"Saved new scaling dict")
    else:
        # apply previous scaling
        min_max_scaling_dict = load(
            open(f"{scale_location}{experiment_type}_min_max_scaling_dict.pkl", "rb")
        )

        for col in x_columns:
            min_val = min_max_scaling_dict[col]["min"]
            max_val = min_max_scaling_dict[col]["max"]
            df2_minmax[col] = df2_minmax[col].apply(
                min_max_scaler, args=[min_val, max_val]
            )
    return df2_minmax
