import numpy as np
import pandas as pd
import polars as pl
import torch
from einops import rearrange
from sentence_transformers import SentenceTransformer
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
    def _encode_text_feature(text_feat, model=None):
        if model is None:
            model = SentenceTransformer('sentence-transformers/sentence-t5-base')
        embeddings = model.encode(sentences=text_feat, show_progress_bar=True, convert_to_tensor=True).cpu()
        return embeddings
    
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
    def _generate_user_history(
        ratings_df,
        features: List[str] = ["movieId", "rating"],
        window_size: int = 200,
        stride: int = 1
    ) -> torch.Tensor:
        
        if isinstance(ratings_df, pd.DataFrame):
            ratings_df = pl.from_pandas(ratings_df)

        grouped_by_user = (ratings_df
            .sort("userId", "timestamp")
            .group_by_dynamic(
                index_column=pl.int_range(pl.len()),
                every=f"{stride}i",
                period=f"{window_size}i",
                by="userId")
            .agg(
                *(pl.col(feat) for feat in features),
                seq_len=pl.col(features[0]).len()
            )
        )
        
        max_seq_len = grouped_by_user.select(pl.col("seq_len").max()).item()
        padded_history = (grouped_by_user
            .with_columns(pad_len=max_seq_len-pl.col("seq_len"))
            .select(
                pl.col("userId"),
                *(pl.col(feat).list.concat(
                    pl.lit(-1, dtype=pl.Int64).repeat_by(pl.col("pad_len"))
                ).list.to_array(max_seq_len) for feat in features)
            )
        )

        out = {
            feat: torch.from_numpy(
                rearrange(
                    padded_history.select(feat).to_numpy().squeeze().tolist(), "b d -> b d"
                )
            ) for feat in features
        }
        out["userId"] = torch.from_numpy(padded_history.select("userId").to_numpy())
        
        return out
