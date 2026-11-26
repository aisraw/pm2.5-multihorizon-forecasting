import numpy as np
import pandas as pd

DEG = {"N":0, "NNE":22.5, "NE":45, "ENE":67.5, "E":90, "ESE":112.5, "SE":135,
       "SSE":157.5, "S":180, "SSW":202.5, "SW":225, "WSW":247.5, "W":270,
       "WNW":292.5, "NW":315, "NNW":337.5}


def load_clean(path):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df[["year", "month", "day", "hour"]])

    # RH from temp + dew point (Magnus formula)
    T, Td = df["TEMP"], df["DEWP"]
    df["RH"] = 100 * np.exp(17.625 * Td / (243.04 + Td)) / np.exp(17.625 * T / (243.04 + T))

    # wind direction: text -> degree -> sin/cos
    rad = np.deg2rad(df["wd"].map(DEG))
    df["wd_sin"], df["wd_cos"] = np.sin(rad), np.cos(rad)

    df = df.rename(columns={"PM2.5": "pm25", "TEMP": "temp", "PRES": "pressure",
                            "RAIN": "rain", "WSPM": "wind"})
    keep = ["timestamp", "station", "pm25", "temp", "RH", "rain", "wind",
            "pressure", "wd_sin", "wd_cos"]
    return df[keep].sort_values(["station", "timestamp"]).reset_index(drop=True)


def split(df):
    # 2026 = hold-out test
    train = df[df.timestamp < "2025-01-01"]
    val = df[(df.timestamp >= "2025-01-01") & (df.timestamp < "2026-01-01")]
    test = df[df.timestamp >= "2026-01-01"]
    return train, val, test