import argparse
from src.core import inference

def parse_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--data-dir",
        required=True,
        help="path to data directory"
    )
    
    parser.add_argument(
        "--model-dir",
        required=True,
        help="path to model dir from train script with config and checkpoint files"
    )
    
    parser.add_argument(
        "--config",
        default="test.yml",
        help="name of configuration file within model-dir",
    )
    
    parser.add_argument(
        "--ckpt",
        required=True,
        help="name of the ckpt file to load from model-dir/checkpoints dir"
    )
    
    parser.add_argument(
        "--batch-size",
        default=128,
        type=int,
        help="inference batch size"
    )
    
    parser.add_argument(
        "--split",
        default=False,
        type=lambda x: (str(x).lower() == 'true'),
        help="if dataset is split into classes' folders"
    )
    
    parser.add_argument(
        "--output",
        default="predictions.json",
        help="name of output file for predictions."
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    inference(args)