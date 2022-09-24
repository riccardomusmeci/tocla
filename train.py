import argparse
from src.core import train

def parse_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--data-dir",
        required=True,
        help="path to data directory"
    )
    
    parser.add_argument(
        "--config",
        default="config/config.yml",
        help="path to YAML configuration file.",
    )
    
    parser.add_argument(
        "--output-dir",
        default=".",
        help="where to save checkpoints during training"
    )
    
    parser.add_argument(
        "--resume-from",
        default=None,
        help="path to checkpoint (ckpt file) to resume training from"
    )
    
    parser.add_argument(
        "--seed",
        default=42
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)