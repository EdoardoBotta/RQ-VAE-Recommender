import torch

from torch.utils.data import Dataset
from torch_geometric.datasets import MovieLens1M

PROCESSED_MOVIE_LENS_SUFFIX = "/processed/data.pt"

class MovieLensMovieData(Dataset):
    def __init__(
        self,
        root: str,
        *args,
        **kwargs
    ) -> None:
        
        raw_movie_lens = MovieLens1M(root=root, *args, **kwargs)
        raw_movie_lens.process()

        data = torch.load(root + PROCESSED_MOVIE_LENS_SUFFIX)
        self.movie_data = data[0]["movie"]["x"]
    
    def __len__(self):
        return self.movie_data.shape[0]
    
    def __getitem__(self, idx):
        return self.movie_data[idx, :]

