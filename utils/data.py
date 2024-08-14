import os
from typing import Dict

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

class PredictorDataset(Dataset):
    def __init__(self, data_path: str,):
        datas = np.load(data_path)
        self.inputs: np.ndarray = datas['x']
        self.targets: np.ndarray = datas['y']
        self.x: np.ndarray = datas
        # min = np.min(self.inputs.reshape(-1, 11), axis=0)
        # max = np.max(self.inputs.reshape(-1, 11), axis=0)



    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):
        x, y = self.inputs[idx], self.targets[idx]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __minmax__(self):
        min = np.min(self.inputs.reshape(-1, 11), axis=0)
        max = np.max(self.inputs.reshape(-1, 11), axis=0)
        return torch.tensor(min, dtype=torch.float32), torch.tensor(max, dtype=torch.float32)

def get_datasets(dataset: str) -> Dict[str, PredictorDataset]:
    return {key: PredictorDataset(os.path.join('data', dataset, f'{key}.npz')) for key in
            ['train', 'val', 'test']}


def get_dataloaders(datasets: Dict[str, Dataset],
                    batch_size: int,
                    num_workers: int = 0) -> Dict[str, DataLoader]:
    return {key: DataLoader(dataset=ds,
                            batch_size=batch_size,
                            shuffle=(key == 'train'),
                            num_workers=num_workers) for key, ds in datasets.items()}