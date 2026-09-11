import argparse

import yaml

from infer import infer
from train import train
from utils.common import set_seed


def main():
    ap = argparse.ArgumentParser(description="AE / VAE / DDPM 生成模型实验")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--mode", choices=["train", "infer"], help="覆盖配置文件中的 run.mode")
    ap.add_argument("--digits", type=str, help="CVAE 指定生成的数字, 逗号分隔, 如 3,7")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.mode:
        cfg["run"]["mode"] = args.mode
    if args.digits:
        cfg["infer"]["digits"] = [int(d) for d in args.digits.split(",")]

    set_seed(cfg["run"]["seed"])
    (train if cfg["run"]["mode"] == "train" else infer)(cfg)


if __name__ == "__main__":
    main()
