import yaml
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
import os

class DatasetWrapper(Dataset):
    """
    Used to apply different transform types on differen subsets 
    of the initial dataset
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.subset)
    
class DataModule:
    def __init__(self, config_path: str):

        self.config = self._load_config(config_path)
        self.data_dir = self.config['dataset']['processed_data_path']
        self.batch_size = self.config['dataset']['batch_size']
        self.base_size = self.config['dataset']['base_size']
        self.img_size = self.config['dataset']['image_size']
        self.num_workers = self.config['dataset']['num_workers']
        self.val_split = self.config['dataset']['validation_split']

        self.mean = self.config['dataset']['mean']
        self.std = self.config['dataset']['std']

        self.train_dataset = None
        self.val_dataset = None
        self.classes = None
        
    def _load_config(self, config_path: str) -> dict:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f'File {config_path} not found')
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
        
    def _get_train_transforms(self):
        """
        Returns the transformation pipeline for training set with 
        random crop and rndom flip for training
        """
        return transforms.Compose([
            transforms.Resize(self.base_size),
            transforms.RandomCrop(self.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
    
    def _get_val_transforms(self):
        """
        Returns the transformation pipeline for validation set
        without Data Augmentation only center crop
        """
        return transforms.Compose([
            transforms.Resize(self.base_size),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
    
    def setup(self):
        """

        """
        full_dataset = datasets.ImageFolder(root=self.data_dir)
        self.classes = full_dataset.classes

        #calc train and val size
        val_size = int(len(full_dataset) * self.val_split)
        train_size = len(full_dataset) - val_size

        generator = torch.Generator().manual_seed(42) #use seed to take the same split every time
        #splitting the dataset
        train_subset, val_subset = random_split(
            full_dataset, [train_size, val_size], generator=generator
        )

        #applying the transforms
        self.train_dataset = DatasetWrapper(train_subset, transform=self._get_train_transforms())
        self.val_dataset = DatasetWrapper(val_subset, transform=self._get_val_transforms())

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the train dataloader
        """
        if self.train_dataset is None:
            raise ValueError('Method setup() must be called first')
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_val_dataloader(self) -> DataLoader:
        """
        Returns the validation dataloader
        """
        if self.val_dataset is None:
            raise ValueError('Method setup() must be called first')

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        ) 




        
        
    

