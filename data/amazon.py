import gzip
import json
import os
import os.path as osp
import pandas as pd
import polars as pl

from collections import defaultdict
from data.preprocessing import PreprocessingMixin
from torch_geometric.data import download_google_url
from torch_geometric.data import extract_zip
from torch_geometric.data import HeteroData
from torch_geometric.data import InMemoryDataset
from torch_geometric.io import fs
from typing import Callable
from typing import List
from typing import Optional


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


class AmazonReviews(InMemoryDataset, PreprocessingMixin):
    gdrive_id = "1qGxgmx7G_WB7JE4Cn_bEcZ_o_NAJLE3G"
    gdrive_filename = "P5_data.zip"

    def __init__(
        self,
        root: str,
        split: str,  # 'beauty', 'sports', 'toys'
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.split = split
        super(AmazonReviews, self).__init__(
            root, transform, pre_transform, force_reload
        )
        self.load(self.processed_paths[0], data_cls=HeteroData)
    
    @property
    def raw_file_names(self) -> List[str]:
        return [self.split]
    
    @property
    def processed_file_names(self) -> str:
        return f'data_{self.split}.pt'
    
    def download(self) -> None:
        path = download_google_url(self.gdrive_id, self.root, self.gdrive_filename)
        extract_zip(path, self.root)
        os.remove(path)
        folder = osp.join(self.root, 'data')
        fs.rm(self.raw_dir)
        os.rename(folder, self.raw_dir)
    
    def _remap_ids(self, x):
        return x - 1

    def read_sequences_as_polars(self, max_seq_len=20):
        splits = ["train", "eval", "test"]
        sequences = {sp: defaultdict(list) for sp in splits}
        user_ids = []
        with open(os.path.join(self.raw_dir, self.split, "sequential_data.txt"), "r") as f:
            for line in f:
                parsed_line = list(map(int, line.strip().split()))
                user_ids.append(parsed_line[0])
                items = [self._remap_ids(id) for id in parsed_line[1:]]
                if len(items) > max_seq_len + 2:
                    items = items[-(max_seq_len + 2):]
                train_items = items[:-2]
                sequences["train"]["itemId"].append(train_items + [-1] * (max_seq_len - len(train_items)))
                sequences["train"]["itemId_fut"].append(-1)
                eval_items = items[:-2] + [-1]
                if len(eval_items) > max_seq_len:
                    eval_items = eval_items[-(max_seq_len):]
                sequences["eval"]["itemId"].append(eval_items + [-1] * (max_seq_len - len(eval_items)))
                sequences["eval"]["itemId_fut"].append(items[-2])
                test_items = items[:-1] + [-1]
                if len(test_items) > max_seq_len:
                    test_items = test_items[-(max_seq_len):]
                sequences["test"]["itemId"].append(test_items + [-1] * (max_seq_len - len(test_items)))
                sequences["test"]["itemId_fut"].append(items[-1])
        for sp in splits:
            sequences[sp]["userId"] = user_ids
            sequences[sp] = pl.from_dict(sequences[sp])
        return sequences
    
    def process(self, max_seq_len=20) -> None:
        data = HeteroData()

        with open(os.path.join(self.raw_dir, self.split, "datamaps.json"), 'r') as f:
            data_maps = json.load(f)
        
        asin2id = pd.DataFrame([{"asin": k, "id": self._remap_ids(int(v))} for k, v in data_maps["item2id"].items()])
        item_data = (
            pd.DataFrame([
                meta for meta in
                parse(path=os.path.join(self.raw_dir, self.split, "meta.json.gz"))
            ])
            .merge(asin2id, on="asin")
            .sort_values(by="id")
            .fillna({"brand": ""})
        )
        sentences = item_data.apply(
            lambda row:
                "Title: " +
                str(row["title"]) + "; " +
                "Description: " +
                str(row["description"]) + "; " +
                "Price: " +
                str(row["price"]) + "; " +
                "Categories: " +
                str(row["categories"][0]) + "; " +
                "Brand: " +
                str(row["brand"]) + "; ",
            axis=1
        )
        
        item_emb = self._encode_text_feature(sentences)
        data['item'].x = item_emb
        
        sequences = self.read_sequences_as_polars(max_seq_len=max_seq_len)
        data["user", "rated", "item"].history = {
            k: self._df_to_tensor_dict(v, ["itemId"])
            for k, v in sequences.items()
        }

        self.save([data], self.processed_paths[0])
        



#if __name__ == "__main__":
#    AmazonReviews("dataset/amazon")
