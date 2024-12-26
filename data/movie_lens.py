import os
import torch

from data.ml1m import RawMovieLens1M
from data.ml32m import RawMovieLens32M
from data.schemas import SeqBatch
from enum import Enum
from torch.utils.data import Dataset

PROCESSED_MOVIE_LENS_SUFFIX = "/processed/data.pt"


class MovieLensSize(Enum):
    _1M = 1
    _32M = 2


MOVIE_LENS_SIZE_TO_RAW_DATASET = {
    MovieLensSize._1M: RawMovieLens1M,
    MovieLensSize._32M: RawMovieLens32M
}


class MovieLensMovieData(Dataset):
    def __init__(
        self,
        root: str,
        *args,
        force_process: bool = False,
        dataset_size: MovieLensSize = MovieLensSize._1M,
        **kwargs
    ) -> None:
        
        processed_data_path = root + PROCESSED_MOVIE_LENS_SUFFIX
        raw_dataset_class = MOVIE_LENS_SIZE_TO_RAW_DATASET[dataset_size]

        raw_movie_lens = raw_dataset_class(root=root, *args, **kwargs)
        if not os.path.exists(processed_data_path) or force_process:
            raw_movie_lens.process(max_seq_len=200)

        data = torch.load(root + PROCESSED_MOVIE_LENS_SUFFIX)
        self.movie_data = data[0]["movie"]["x"]

    def __len__(self):
        return self.movie_data.shape[0]

    def __getitem__(self, idx):
        movie_ids = torch.tensor(idx).unsqueeze(0) if not isinstance(idx, torch.Tensor) else idx
        x = self.movie_data[idx, :768]
        return SeqBatch(
            user_ids=-1 * torch.ones_like(movie_ids.squeeze(0)),
            ids=movie_ids,
            ids_fut=-1 * torch.ones_like(movie_ids.squeeze(0)),
            x=x,
            x_fut=-1 * torch.ones_like(movie_ids.squeeze(0)),
            seq_mask=torch.ones_like(movie_ids, dtype=bool)
        )


class MovieLensSeqData(Dataset):
    def __init__(
        self,
        root: str,
        *args,
        is_train: bool = True,
        force_process: bool = False,
        dataset_size: MovieLensSize = MovieLensSize._1M,
        **kwargs
    ) -> None:

        processed_data_path = root + PROCESSED_MOVIE_LENS_SUFFIX
        raw_dataset_class = MOVIE_LENS_SIZE_TO_RAW_DATASET[dataset_size]

        raw_movie_lens = raw_dataset_class(root=root, *args, **kwargs)
        if not os.path.exists(processed_data_path) or force_process:
            raw_movie_lens.process(max_seq_len=200)
  
        data = torch.load(root + PROCESSED_MOVIE_LENS_SUFFIX)

        split = "train" if is_train else "eval"
        self.sequence_data = data[0][("user", "rated", "movie")]["history"][split]
        self.movie_data = data[0]["movie"]["x"]
        self.split = split
    
    @property
    def max_seq_len(self):
        return self.sequence_data["movieId"].shape[-1]

    def __len__(self):
        return self.sequence_data["userId"].shape[0]
  
    def __getitem__(self, idx):
        user_ids = self.sequence_data["userId"][idx]
        movie_ids = self.sequence_data["movieId"][idx]
        movie_ids_fut = self.sequence_data["movieId_fut"][idx]
        
        assert (movie_ids >= -1).all(), "Invalid movie id found"
        x = self.movie_data[movie_ids, :768]
        x[movie_ids == -1] = -1

        x_fut = self.movie_data[movie_ids_fut, :768]
        x_fut[movie_ids_fut == -1] = -1

        return SeqBatch(
            user_ids=user_ids,
            ids=movie_ids,
            ids_fut=movie_ids_fut,
            x=x,
            x_fut=x_fut,
            seq_mask=(movie_ids >= 0)
        )


if __name__ == "__main__":
    dataset = MovieLensMovieData("dataset/ml-32m", dataset_size=MovieLensSize._32M, force_process=True)
    dataset[0]
    import pdb; pdb.set_trace()
