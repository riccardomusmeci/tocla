import os
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from src.utils import Device
from src.io import load_config
from src.model import create_model
from src.model import create_model
from src.transform import Transform
from torch.utils.data import DataLoader
from src.dataset import InferenceDataset
from typing import Dict, List, Tuple, Union

def save_output(
    img_paths: List[str],
    predictions: List[int],
    targets: List[int],
    scores: List[float],
    output_fpath: str,
    class_map: Dict[int, Union[str, List[str]]]
):
    """saves output from model inference

    Args:
        img_paths (List[str]): paths of images
        predictions (List[int]): predictions from model
        targets (List[int]): targets from dataset
        scores (List[float]): scores from model
        output_fpath (str): where to save output
        class_map (Dict[int, Union[str, List[str]]], optional): class map
    """
    output = []
    for i, (p, t) in enumerate(zip(predictions, targets)):
        output.append({
                "file_path": img_paths[i],
                "ground_truth": class_map[t] if t!=-1 else t,
                "prediction": class_map[p],
                "score": scores[i]
            })  
    
    with open(output_fpath, "w") as f:
        json.dump(output, f, indent=4)
    print(f"> Saved predictions at {output_fpath}")

def compute_predictions(
    model: nn.Module,
    data_loader: DataLoader
) -> Tuple[List[int], List[int], List[float]]:
    """computes predictions on dataset

    Args:
        model (nn.Module): model
        data_loader (DataLoader): data loader

    Returns:
        Tuple[List[int], List[int], List[float]]: predictions, targets, scores 
    """
    device = Device()
    model.eval()
    predictions, targets, scores = [], [], []
    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        with torch.no_grad():
            x, target = batch
            x = x.to(device)
            logits = model(x)
            outs = torch.nn.functional.softmax(logits, dim=1)
            max_outs, preds = torch.max(outs.data, 1)
            predictions += preds.tolist()
            targets += target.tolist()
            scores += max_outs.tolist()
    return predictions, targets, scores

def inference(args):

    config = load_config(os.path.join(args.model_dir, args.config))
    device = Device()
    
    class_map = config["datamodule"]["class_map"]
    num_classes = len(config["datamodule"]["class_map"])    
    dataset = InferenceDataset(
        root_dir=args.data_dir,
        class_map=class_map if args.pseudolabel else None,
        transform=Transform(train=False, **config["transform"])
    )
    
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size
    )
    
    model = create_model(
        model_name=config["model"]["model_name"],
        pretrained=config["model"]["pretrained"],
        num_classes=num_classes,
        checkpoint_path=os.path.join(args.model_dir, "checkpoints", args.ckpt) if args.ckpt is not None else None
    )
    
    model = model.to(device)
    print(f"> Starting inference on dataset {args.data_dir} (device set to {device})")
    predictions, targets, scores = compute_predictions(
        model=model,
        data_loader=data_loader
    )
    
    # save output json
    save_output(
        img_paths=dataset.images,
        predictions=predictions,
        targets=targets,
        scores=scores,
        output_fpath=os.path.join(args.model_dir, args.output),
        class_map=class_map
    ) 
    
            

    
    
    
    
    