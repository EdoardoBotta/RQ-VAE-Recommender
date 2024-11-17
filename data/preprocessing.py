import pandas as pd
import numpy as np
import torch
from typing import List


class MovieLensPreprocessingMixin:
    @staticmethod
    def _process_genres(genres, one_hot=True):
        if one_hot:
            return genres

        max_genres = genres.sum(axis=1).max()
        idx_list = []
        for i in range(genres.shape[0]):
            idxs = np.where(genres[i, :] == 1)[0] + 1
            missing = max_genres - len(idxs)
            if missing > 0:
                idxs = np.array(list(idxs) + missing * [0])
            idx_list.append(idxs)
        out = np.stack(idx_list)
        return out

    @staticmethod
    def _remove_low_occurrence(source_df, target_df, index_col):
        if isinstance(index_col, str):
            index_col = [index_col]
        out = target_df.copy()
        for col in index_col:
            count = source_df.groupby(col).agg(ratingCnt=("rating", "count"))
            high_occ = count[count["ratingCnt"] >= 5]
            out = out.merge(high_occ, on=col).drop(columns=["ratingCnt"])
        return out
    
    @staticmethod
    def _rolling_window(group, features, window_size=200, stride=1):
        assert group["userId"].nunique() == 1, "Found data for too many users"

        if len(group) < window_size:
            window_size = len(group)
            stride = 1
        n_windows = (len(group)+1-window_size)//stride
        feats = group[features].to_numpy().T
        windows = np.lib.stride_tricks.as_strided(
            feats,
            shape=(len(features), n_windows, window_size),
            strides=(feats.strides[0], 8*stride, 8*1)
        )
        feat_seqs = np.split(windows, len(features), axis=0)
        rolling_df = pd.DataFrame({
            name: pd.Series(
                np.split(feat_seqs[i].squeeze(0), n_windows, 0)
            ).map(torch.tensor) for i, name in enumerate(features)
        })
        return rolling_df

    @staticmethod
    def _generate_user_history(ratings_df, rolling: bool = False, window_size: int = 200, stride: int = 1):
        if rolling:
            grouped_by_user = (ratings_df
                .sort_values(by=['userId', 'timestamp'])
                .groupby("userId")
                .apply(lambda g: MovieLensPreprocessingMixin._rolling_window(
                    g, ["movieId"], window_size=window_size, stride=stride)
                )
                .reset_index()
                .apply(
                    lambda x: (
                        torch.cat([torch.tensor(x["userId"]).unsqueeze(0).unsqueeze(0), x["movieId"]], axis=1).T
                    ),
                    axis=1      
                )
            )
        else:
            grouped_by_user = (ratings_df
                .sort_values(by=['userId', 'timestamp'])
                .groupby("userId")
                .agg(list)["movieId"]
                .reset_index()
                .apply(
                    lambda x: (
                        torch.tensor([x["userId"]] + x["movieId"]).unsqueeze(-1)
                    ),
                    axis=1
                )
            )

        padded_history = torch.nn.utils.rnn.pad_sequence(
            grouped_by_user.to_list(),
            batch_first=True,
            padding_value=-1
        ).squeeze(-1)

        return padded_history
