from src.dataset import TrainDataset
from torch.utils.data import DataLoader
from src.sampler import ImbalancedSampler
from typing import List, Dict, Union, Callable, Optional

def create_dataloader(
    root_dir: str,
    train: bool,
    class_map: Dict[int, Union[str, List[str]]],
    batch_size: int,
    max_samples_per_class: int = None,
    random_samples: bool = False,
    transform: Callable = None, 
    imbalanced: bool = False,
    shuffle: bool = True,
    num_workers: int = 5,
    pin_memory: bool = True,
    drop_last: bool = False,
    persistent_workers: bool = True,
) -> DataLoader:
    """Setup a dataloader for a dataset

    Args:
        root_dir (str): dataset root dir
        train (bool): if True train loader else test/val loader.
        class_map (Dict[int, Union[str, List[str]]]): class map
        batch_size (int): batch size
        max_samples_per_class (int, optional): max samples per class in the dataset. Defaults to None.
        random_samples (bool, optional): if random sampling if max samples per class is not None. Defaults to False.
        train_transform (Callable, optional): train data augmentation. Defaults to None.
        val_transform (Callable, optional): val data augmentation. Defaults to None.
        imbalanced (bool, optional): if dataset is imbalanced. Defaults to False.
        shuffle (bool, optional): whether to shuffle dataset. Defaults to True.
        num_workers (int, optional): num workers. Defaults to 5.
        pin_memory (bool, optional): data loader pin memory. Defaults to True.
        drop_last (bool, optional): drop last data loader. Defaults to False.
        persistent_workers (bool, optional): persistent workers data loader. Defaults to True.

    Returns:
        DataLoader: dataset dataloader
    """
    
    dataset = TrainDataset(
        root_dir=root_dir,
        train=train,
        class_map=class_map,
        max_samples_per_class=max_samples_per_class if train else None,
        random_samples=random_samples if train else None,
        transform=transform
    )
    
    if persistent_workers and num_workers==0:
        print(f"> [WARNING] persistent_workers is set to True and num_workers is 0 (must be >0). This is not right. Setting persistent_workers to False.")
        persistent_workers = False
        
    return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle if not imbalanced else False,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            sampler=ImbalancedSampler(dataset=dataset) if imbalanced else None
        )
    
    
    
    