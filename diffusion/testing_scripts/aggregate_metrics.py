#!/usr/bin/env python3
"""
汇总多份 test_metrics.json，计算均值与标准差。
用法（在 diffusion 目录下）:
  python testing_scripts/aggregate_metrics.py --log_dir ./results_eval/.../logs/..._clean --splits 0 1 2 3 4
"""
import argparse
import json
import os
import numpy as np


def load_metrics(log_dir: str, splits: list) -> list:
    metrics_list = []
    for s in splits:
        path = os.path.join(log_dir, f"split_{s}", "test_metrics.json")
        if not os.path.isfile(path):
            print(f"Warning: not found {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            metrics_list.append(json.load(f))
    return metrics_list


def aggregate_scalar(values: list) -> tuple:
    a = np.array(values, dtype=float)
    return float(np.mean(a)), float(np.std(a))


def aggregate_list_of_floats(list_of_lists: list) -> tuple:
    arr = np.array(list_of_lists)
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    return mean.tolist(), std.tolist()


def main():
    parser = argparse.ArgumentParser(description="Aggregate test_metrics.json from multiple splits")
    parser.add_argument("--log_dir", type=str, required=True,
                        help="Directory containing split_0, split_1, ...")
    parser.add_argument("--splits", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="Split indices to aggregate, e.g. 0 1 2 3 4")
    args = parser.parse_args()

    metrics_list = load_metrics(args.log_dir, args.splits)
    if not metrics_list:
        print("No test_metrics.json found. Exit.")
        return

    n = len(metrics_list)
    print(f"Loaded {n} runs from {args.log_dir} (splits {args.splits})")
    print()

    accs = [m["accuracy"] for m in metrics_list]
    eces = [m["ece"] for m in metrics_list]
    acc_mean, acc_std = aggregate_scalar(accs)
    ece_mean, ece_std = aggregate_scalar(eces)

    extra_scalar_keys = [
        "sensitivity", "specificity", "balanced_accuracy",
        "auc", "ap", "ar", "f1", "precision", "recall",
    ]
    extra_agg = {}
    for key in extra_scalar_keys:
        vals = [m[key] for m in metrics_list if key in m and m[key] is not None]
        vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        if vals:
            extra_agg[key] = aggregate_scalar(vals)

    print("========== 标量指标（均值 ± 标准差）==========")
    print(f"  accuracy:  {acc_mean:.4f} ± {acc_std:.4f}  (即 {acc_mean*100:.2f}% ± {acc_std*100:.2f}%)")
    print(f"  ece:       {ece_mean:.6f} ± {ece_std:.6f}")
    for key in (
        "sensitivity", "specificity", "balanced_accuracy",
        "auc", "ap", "ar", "f1", "precision", "recall",
    ):
        if key in extra_agg:
            mean_v, std_v = extra_agg[key]
            print(f"  {key:20s} {mean_v:.4f} ± {std_v:.4f}")
    print()

    piw_correct_list = [m["piw_correct"] for m in metrics_list]
    piw_incorrect_list = [m["piw_incorrect"] for m in metrics_list]
    var_correct_list = [m["variances_correct"] for m in metrics_list]
    var_incorrect_list = [m["variances_incorrect"] for m in metrics_list]

    piw_c_mean, piw_c_std = aggregate_list_of_floats(piw_correct_list)
    piw_i_mean, piw_i_std = aggregate_list_of_floats(piw_incorrect_list)
    var_c_mean, var_c_std = aggregate_list_of_floats(var_correct_list)
    var_i_mean, var_i_std = aggregate_list_of_floats(var_incorrect_list)

    print("========== 各类别 PIW / 方差（均值 ± 标准差）==========")
    print("  piw_correct    (mean):", [f"{x:.4f}" for x in piw_c_mean], "  (std):", [f"{x:.4f}" for x in piw_c_std])
    print("  piw_incorrect  (mean):", [f"{x:.4f}" for x in piw_i_mean], "  (std):", [f"{x:.4f}" for x in piw_i_std])
    print("  variances_correct   (mean):", [f"{x:.4f}" for x in var_c_mean], "  (std):", [f"{x:.4f}" for x in var_c_std])
    print("  variances_incorrect (mean):", [f"{x:.4f}" for x in var_i_mean], "  (std):", [f"{x:.4f}" for x in var_i_std])
    print()

    summary = {
        "n_runs": n,
        "splits": args.splits,
        "accuracy_mean": acc_mean,
        "accuracy_std": acc_std,
        "ece_mean": ece_mean,
        "ece_std": ece_std,
        "piw_correct_mean": piw_c_mean,
        "piw_correct_std": piw_c_std,
        "piw_incorrect_mean": piw_i_mean,
        "piw_incorrect_std": piw_i_std,
        "variances_correct_mean": var_c_mean,
        "variances_correct_std": var_c_std,
        "variances_incorrect_mean": var_i_mean,
        "variances_incorrect_std": var_i_std,
    }
    for key, (mean_v, std_v) in extra_agg.items():
        summary[f"{key}_mean"] = mean_v
        summary[f"{key}_std"] = std_v

    out_path = os.path.join(args.log_dir, "aggregated_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"汇总已写入: {out_path}")


if __name__ == "__main__":
    main()
