"""
Plots the 4 training-dynamics curves from the ES training run's W&B history:
  1. train/reward: min/mean/max vs. iteration, one plot
  2. train/reward/std vs. iteration
  3. train/response-length: min/mean/max vs. iteration, one plot
  4. train/response-length/std vs. iteration

Usage (from a saved CSV, e.g. results/iter50-1.5b/training_curves.csv):
    python plot_training_curves.py --csv results/iter50-1.5b/training_curves.csv --out-dir results/iter50-1.5b

Or fetch fresh from W&B:
    python plot_training_curves.py --wandb-run chunhinma00-personal/es-finetuning/it2de910 --out-dir results/iter50-1.5b
"""
import argparse
import os

import pandas as pd


def plot_minmeanmax(df, metric_prefix, ylabel, title, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = df["global_step"]
    fig, ax = plt.subplots(figsize=(3.2, 2.5))
    ax.plot(x, df[f"{metric_prefix}/max"], label="max", color="#e74c3c", linewidth=1.4)
    ax.plot(x, df[f"{metric_prefix}/mean"], label="mean", color="#2c3e50", linewidth=2.0)
    ax.plot(x, df[f"{metric_prefix}/min"], label="min", color="#3498db", linewidth=1.4)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.4)
    ax.legend(loc="best", frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_std(df, metric, ylabel, title, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = df["global_step"]
    fig, ax = plt.subplots(figsize=(3.2, 2.5))
    ax.plot(x, df[metric], color="#8e44ad", linewidth=1.8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="Pre-saved W&B history CSV")
    ap.add_argument("--wandb-run", default=None, help="entity/project/run_id, fetches fresh if --csv not given")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
    elif args.wandb_run:
        import wandb
        api = wandb.Api()
        run = api.run(args.wandb_run)
        keys = ["global_step", "train/reward/mean", "train/reward/min", "train/reward/max", "train/reward/std",
                "train/response-length/mean", "train/response-length/min", "train/response-length/max", "train/response-length/std"]
        df = run.history(keys=keys, pandas=True)
    else:
        raise SystemExit("Need --csv or --wandb-run")

    df = df.sort_values("global_step")
    os.makedirs(args.out_dir, exist_ok=True)

    plot_minmeanmax(df, "train/reward", "Reward", "Training reward (min/mean/max)",
                     os.path.join(args.out_dir, "train_reward_minmeanmax.png"))
    plot_std(df, "train/reward/std", "Reward std", "Training reward std",
              os.path.join(args.out_dir, "train_reward_std.png"))
    plot_minmeanmax(df, "train/response-length", "Response length (tokens)", "Training response length (min/mean/max)",
                     os.path.join(args.out_dir, "train_response_length_minmeanmax.png"))
    plot_std(df, "train/response-length/std", "Response length std", "Training response length std",
              os.path.join(args.out_dir, "train_response_length_std.png"))

    print(f"Wrote 4 plots to {args.out_dir}")


if __name__ == "__main__":
    main()
