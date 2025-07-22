import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from data.schemas import SeqBatch

class CustomDataset(Dataset):
    """
    Use this class to load custom preprocessed data.

    The class expects the following files in the `root` directory:
    - item_embeddings.npy : numpy array of shape (num_items, embedding_dim)
    - is_train.npy : boolean numpy array of shape (num_items)
    """
    def __init__(self, root, *args, **kwargs):
        super().__init__()
        self.root = root
        self.item_embeddings = np.load(os.path.join(root, "item_embeddings.npy"))
        self.item_embeddings = torch.from_numpy(self.item_embeddings)
        is_train_path = os.path.join(root, "is_train.npy")
        if os.path.exists(is_train_path):
            self.item_is_train = np.load(is_train_path)
        else:
            self.item_is_train = np.ones(self.item_embeddings.shape[0], dtype=bool)
        self.text_dummy = np.zeros_like(self.item_embeddings[:, 0], dtype=bool)
    
    def __len__(self):
        return len(self.item_embeddings)
    
    def __getitem__(self, idx):
        item_ids = torch.tensor(idx).unsqueeze(0) if not isinstance(idx, torch.Tensor) else idx
        x = self.item_embeddings[idx]
        return SeqBatch(
            user_ids=-1 * torch.ones_like(item_ids.squeeze(0)),
            ids=item_ids,
            ids_fut=-1 * torch.ones_like(item_ids.squeeze(0)),
            x=x,
            x_fut=-1 * torch.ones_like(x),
            seq_mask=torch.ones_like(item_ids, dtype=bool)
        )
    
    @property
    def data(self):
        return {
            "item": {
                "x": self.item_embeddings,
                "is_train": self.item_is_train,
                "text": self.text_dummy,
            }
        }
    
    @property
    def processed_paths(self):
        return [self.root]
    
    def process(self, *args, **kwargs):
        # This class just loads preprocessed data
        pass
