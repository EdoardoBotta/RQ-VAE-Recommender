import os
import os.path as osp
import pandas as pd
import torch

from data.preprocessing import MovieLensPreprocessingMixin
from torch_geometric.data import HeteroData
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import download_url
from torch_geometric.data import extract_zip
from torch_geometric.io import fs
from typing import Callable, List, Optional


class MovieLens32M(InMemoryDataset):
    url = 'https://files.grouplens.org/datasets/movielens/ml-32m.zip'

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0], data_cls=HeteroData)
    
    @property
    def raw_file_names(self) -> List[str]:
        return ['links.csv', 'movies.csv', 'ratings.csv', 'tags.csv']
    
    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    @property
    def has_process(self) -> bool:
        return not os.path.exists(self.processed_paths[0])
    
    def download(self) -> None:
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.remove(path)
        folder = osp.join(self.root, 'ml-32m')
        fs.rm(self.raw_dir)
        os.rename(folder, self.raw_dir)

    def process():
        pass


class RawMovieLens32M(MovieLens32M, MovieLensPreprocessingMixin):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 force_reload=False) -> None:
        super(RawMovieLens32M, self).__init__(
            root, transform, pre_transform, force_reload
        )

    def _load_ratings(self):
        return pd.read_csv(self.raw_paths[2])
    
    def process(self, max_seq_len=None) -> None:
        data = HeteroData()
        ratings_df = self._load_ratings()

        # TODO: Extract actor name tag from tag dataset
        # TODO: Maybe use links to extract more item features
        # Process movie data:
        full_df = pd.read_csv(self.raw_paths[1], index_col='movieId')
        df = full_df # self._remove_low_occurrence(ratings_df, full_df, "movieId") # TODO: Maybe remove to retain more data for RQVAE training.
        
        movie_mapping = {idx: i for i, idx in enumerate(df.index)}

        genres = self._process_genres(df["genres"].str.get_dummies('|').values, one_hot=True)
        genres = torch.from_numpy(genres).to(torch.float)

        titles_text = df["title"].apply(lambda s: s.split("(")[0].strip()).tolist()
        titles_emb = self._encode_text_feature(titles_text)

        x = torch.cat([titles_emb, genres], axis=1)

        data['movie'].x = x
        # Process user data:
        full_df = pd.DataFrame({"userId": ratings_df["userId"].unique()})
        df = self._remove_low_occurrence(ratings_df, full_df, "userId")
        user_mapping = {idx: i for i, idx in enumerate(df["userId"])}
        self.int_user_data = df

        # Process rating data:
        df = self._remove_low_occurrence(
            ratings_df,
            ratings_df,
            ["userId", "movieId"]
        )
        src = [user_mapping[idx] for idx in df['userId']]
        dst = [movie_mapping[idx] for idx in df['movieId']]
        edge_index = torch.tensor([src, dst])
        data['user', 'rates', 'movie'].edge_index = edge_index

        rating = torch.from_numpy(df['rating'].values).to(torch.long)
        data['user', 'rates', 'movie'].rating = rating

        time = torch.from_numpy(df['timestamp'].values)
        data['user', 'rates', 'movie'].time = time

        data['movie', 'rated_by', 'user'].edge_index = edge_index.flip([0])
        data['movie', 'rated_by', 'user'].rating = rating
        data['movie', 'rated_by', 'user'].time = time

        df["movieId"] = df["movieId"].apply(lambda x: movie_mapping[x])

        df["rating"] = (2*df["rating"]).astype(int)
        data["user", "rated", "movie"].history = self._generate_user_history(
            df,
            features=["movieId", "rating"],
            window_size=max_seq_len if max_seq_len is not None else 200,
            stride=180
        )

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
