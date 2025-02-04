import numpy as np
import pandas as pd
import polars as pl
import torch
from data.schemas import FUT_SUFFIX
from einops import rearrange
from sentence_transformers import SentenceTransformer
from typing import List


class PreprocessingMixin:
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
            model = SentenceTransformer('sentence-transformers/sentence-t5-xl')
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
    def _ordered_train_test_split(df, on, train_split=0.8):
        threshold = df.select(pl.quantile(on, train_split)).item()
        return df.with_columns(is_train=pl.col(on) <= threshold)
    
    @staticmethod
    def _df_to_tensor_dict(df, features):
        out = {
            feat: torch.from_numpy(
                rearrange(
                    df.select(feat).to_numpy().squeeze().tolist(), "b d -> b d"
                )
            ) if df.select(pl.col(feat).list.len().max() == pl.col(feat).list.len().min()).item()
            else df.get_column("itemId").to_list()
            for feat in features
        }
        fut_out = {
            feat + FUT_SUFFIX: torch.from_numpy(
                df.select(feat + FUT_SUFFIX).to_numpy()
            ) for feat in features
        }
        out.update(fut_out)
        out["userId"] = torch.from_numpy(df.select("userId").to_numpy())
        return out


    @staticmethod
    def _generate_user_history(
        ratings_df,
        features: List[str] = ["movieId", "rating"],
        window_size: int = 200,
        stride: int = 1,
        train_split: float = 0.8,
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
                seq_len=pl.col(features[0]).len(),
                max_timestamp=pl.max("timestamp")
            )
        )
        
        max_seq_len = grouped_by_user.select(pl.col("seq_len").max()).item()
        split_grouped_by_user = PreprocessingMixin._ordered_train_test_split(grouped_by_user, "max_timestamp", 0.8)
        padded_history = (split_grouped_by_user
            .with_columns(pad_len=max_seq_len-pl.col("seq_len"))
            .filter(pl.col("is_train").or_(pl.col("seq_len") > 1))
            .select(
                pl.col("userId"),
                pl.col("max_timestamp"),
                pl.col("is_train"),
                *(pl.when(pl.col("is_train"))
                    .then(
                        pl.col(feat).list.concat(
                            pl.lit(-1, dtype=pl.Int64).repeat_by(pl.col("pad_len"))
                        ).list.to_array(max_seq_len)
                    ).otherwise(
                        pl.col(feat).list.slice(0, pl.col("seq_len")-1).list.concat(
                            pl.lit(-1, dtype=pl.Int64).repeat_by(pl.col("pad_len")+1)
                        ).list.to_array(max_seq_len)
                    )
                    for feat in features
                ),
                *(pl.when(pl.col("is_train"))
                    .then(
                        pl.lit(-1, dtype=pl.Int64)
                    )
                    .otherwise(
                        pl.col(feat).list.get(-1)
                    ).alias(feat + FUT_SUFFIX)
                    for feat in features
                )
            )
        )
        
        out = {}
        out["train"] = PreprocessingMixin._df_to_tensor_dict(
            padded_history.filter(pl.col("is_train")),
            features
        )
        out["eval"] = PreprocessingMixin._df_to_tensor_dict(
            padded_history.filter(pl.col("is_train").not_()),
            features
        )
        
        return out

