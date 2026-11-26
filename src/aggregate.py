import glob
import pandas as pd


def main():
    files = sorted(glob.glob("results/seed*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    g = df.groupby(["h", "model"], sort=False)
    out = g.agg(n_seeds=("seed", "nunique"),
                MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
                RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
                R2_mean=("R2", "mean"), R2_std=("R2", "std")).reset_index()
    out = out.round(3)
    out.to_csv("results/summary.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()