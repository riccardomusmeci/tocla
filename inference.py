import argparse
from src.core import inference

def parse_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--data-dir",
        default="/Users/riccardomusmeci/Developer/data/enel/spain/broken_insulator/val/normal",
        #required=True,
        help="path to data directory"
    )
    
    parser.add_argument(
        "--model-dir",
        default="/Users/riccardomusmeci/Developer/experiments/classifier-playground/spain/insulator_broken/2022-09-23-22-10-28",
        help="path to model dir from train script with config and checkpoint files"
    )
    
    parser.add_argument(
        "--config",
        default="test.yml",
        help="name of configuration file within model-dir",
    )
    
    parser.add_argument(
        "--ckpt",
        default="epoch=38-step=4095-val_loss=0.172-val_acc=0.931-val_f1=0.928-cal_err=0.01691.ckpt",
        help="name of the ckpt file to load from model-dir/checkpoints dir"
    )
    
    parser.add_argument(
        "--batch-size",
        default=128,
        type=int,
        help="inference batch size"
    )
    
    parser.add_argument(
        "--pseudolabel",
        default=False,
        type=lambda x: (str(x).lower() == 'true'),
        help="to perform pseudo-labeling analysis on split dataset"
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