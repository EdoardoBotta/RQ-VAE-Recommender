import numpy as np
import torch


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
    def _generate_user_history(ratings_df):
        grouped_by_user = (
            ratings_df.sort_values(by=['userId', 'timestamp'])
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
