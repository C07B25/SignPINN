# coding: utf-8
import argparse
import os
from training import train, test
from prediction import calculate_and_display_metrics

def main():
    ap = argparse.ArgumentParser("Sign-IDD: Iconicity Disentangled Diffusion for Sign Language Production")
    ap.add_argument("mode", choices=["train", "test"], help="train a model or test")
    ap.add_argument("config_path", default="./Configs/Sign-IDD.yaml", type=str, help="path to YAML config file")
    ap.add_argument("--ckpt", type=str, help="path to model checkpoint")
    ap.add_argument("--gpu_id", type=str, default="0", help="gpu to run your job on")
    args = ap.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    if args.mode == "train":
        train(cfg_file=args.config_path, ckpt=args.ckpt)
    elif args.mode == "test":
        output_dir = test(cfg_file=args.config_path, ckpt=args.ckpt)
        calculate_and_display_metrics(output_dir)
    else:
        raise ValueError("Unknown mode")

if __name__ == "__main__":
    main()