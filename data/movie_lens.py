import torch

from data.ml1m import RawMovieLens1M
from data.schemas import SeqBatch
from torch.utils.data import Dataset

PROCESSED_MOVIE_LENS_SUFFIX = "/processed/data.pt"


class MovieLensMovieData(Dataset):
    def __init__(
        self,
        root: str,
        *args,
        **kwargs
    ) -> None:

        raw_movie_lens = RawMovieLens1M(root=root, *args, **kwargs)
        raw_movie_lens.process()

        data = torch.load(root + PROCESSED_MOVIE_LENS_SUFFIX)
        self.movie_data = data[0]["movie"]["x"]

    def __len__(self):
        return self.movie_data.shape[0]

    def __getitem__(self, idx):
        movie_ids = torch.tensor(idx).unsqueeze(0) if not isinstance(idx, torch.Tensor) else idx
        x = self.movie_data[idx, :]
        return SeqBatch(
            user_ids=-1 * torch.ones_like(movie_ids.squeeze(0)),
            ids=movie_ids,
            x=x,
            seq_mask=torch.ones_like(movie_ids, dtype=bool)
        )


class MovieLensSeqData(Dataset):
    def __init__(
            self,
            root: str,
            *args,
            **kwargs
    ) -> None:

        raw_movie_lens = RawMovieLens1M(root=root, *args, **kwargs)
        raw_movie_lens.process()
  
        data = torch.load(root + PROCESSED_MOVIE_LENS_SUFFIX)
        self.sequence_data = data[0][("user", "rated", "movie")]["history"]
        self.movie_data = data[0]["movie"]["x"]

    def __len__(self):
        return self.sequence_data.shape[0]
  
    def __getitem__(self, idx):
        user_ids = self.sequence_data[idx, 0]
        movie_ids = self.sequence_data[idx, 1:]
        assert (movie_ids >= -1).all(), "Invalid movie id found"
        x = self.movie_data[movie_ids, :]
        x[movie_ids == -1] = -1

        return SeqBatch(
            user_ids=user_ids,
            ids=movie_ids,
            x=self.movie_data[movie_ids, :],
            seq_mask=(movie_ids >= 0)
        )


if __name__ == "__main__":
    dataset = MovieLensSeqData("dataset/ml-1m")
    dataset[0]
    import pdb; pdb.set_trace()
