import argparse
import glob
import os
import shutil

from sklearn.model_selection import train_test_split


def copy_files(file_list, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    for src in file_list:
        shutil.copy2(src, os.path.join(dest_dir, os.path.basename(src)))


def main():
    parser = argparse.ArgumentParser(description="60/20/20 ImageFolder split (seed=42).")
    parser.add_argument("--root_dir", required=True, help="Folder with Normal/ and Tuberculosis/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root_dir = args.root_dir
    normal_files = glob.glob(os.path.join(root_dir, "Normal", "*.png"))
    tuberculosis_files = glob.glob(os.path.join(root_dir, "Tuberculosis", "*.png"))
    if not normal_files or not tuberculosis_files:
        raise SystemExit(f"expected {root_dir}/Normal/*.png and Tuberculosis/*.png")

    for split in ["training", "testing", "validation"]:
        for class_name in ["Normal", "Tuberculosis"]:
            os.makedirs(os.path.join(root_dir, split, class_name), exist_ok=True)

    normal_train, normal_temp = train_test_split(normal_files, test_size=0.4, random_state=args.seed)
    normal_val, normal_test = train_test_split(normal_temp, test_size=0.5, random_state=args.seed)
    tb_train, tb_temp = train_test_split(tuberculosis_files, test_size=0.4, random_state=args.seed)
    tb_val, tb_test = train_test_split(tb_temp, test_size=0.5, random_state=args.seed)

    copy_files(normal_train, os.path.join(root_dir, "training", "Normal"))
    copy_files(normal_val, os.path.join(root_dir, "validation", "Normal"))
    copy_files(normal_test, os.path.join(root_dir, "testing", "Normal"))
    copy_files(tb_train, os.path.join(root_dir, "training", "Tuberculosis"))
    copy_files(tb_val, os.path.join(root_dir, "validation", "Tuberculosis"))
    copy_files(tb_test, os.path.join(root_dir, "testing", "Tuberculosis"))
    print(
        "done",
        len(normal_train),
        len(normal_val),
        len(normal_test),
        len(tb_train),
        len(tb_val),
        len(tb_test),
    )


if __name__ == "__main__":
    main()
