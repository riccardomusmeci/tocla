import pytorch_lightning as pl
from src.dataset import TrainDataset
from torch.utils.data import DataLoader
from src.sampler import ImbalancedSampler
from typing import List, Dict, Union, Callable, Optional

class DataModule(pl.LightningDataModule):
    
    def __init__(
        self, 
        root_dir: str,
        class_map: Dict[int, Union[str, List[str]]],
        batch_size: int,
        max_samples_per_class: int = None,
        random_samples: bool = False,
        train_transform: Callable = None, 
        val_transform: Callable = None, 
        imbalanced: bool = False,
        shuffle: bool = True,
        num_workers: int = 5,
        pin_memory: bool = True,
        drop_last: bool = False,
        persistent_workers: bool = True
    ):
        super().__init__()
        self.root_dir = root_dir
        self.class_map = class_map
        self.batch_size = batch_size
        self.max_samples_per_class = max_samples_per_class
        self.random_samples = random_samples
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.imbalanced = imbalanced
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last 
        self.persistent_workers=persistent_workers
        
    def prepare_data(self) -> None:
        return super().prepare_data()
    
    def setup(self, stage: Optional[str] = None) -> None:
        
        if stage == "fit" or stage is None:
            
            self.train_dataset = TrainDataset(
                root_dir=self.root_dir,
                train=True,
                class_map=self.class_map,
                max_samples_per_class=self.max_samples_per_class,
                random_samples=self.random_samples,
                transform=self.train_transform
            )
            
            self.val_dataset = TrainDataset(
                root_dir=self.root_dir,
                train=False,
                class_map=self.class_map,
                max_samples_per_class=None,
                random_samples=False,
                transform=self.val_transform
            )
        
        if stage == "test":
            self.test_dataset = TrainDataset(
                root_dir=self.root_dir,
                train=False,
                class_map=self.class_map,
                max_samples_per_class=None,
                random_samples=False,
                transform=self.val_transform
            )
            
    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle if not self.imbalanced else False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            sampler=ImbalancedSampler(dataset=self.train_dataset) if self.imbalanced else None
        )
        
    def val_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )