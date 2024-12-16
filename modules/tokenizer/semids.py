import torch

from data.movie_lens import MovieLensMovieData
from data.movie_lens import MovieLensSeqData
from data.schemas import SeqBatch
from einops import rearrange
from einops import repeat
from einops import pack
from modules.rqvae import RqVae
from typing import NamedTuple
from typing import List
from typing import Optional
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class TokenizedSeqBatch(NamedTuple):
    user_ids: torch.Tensor
    sem_ids: torch.Tensor
    seq_mask: torch.Tensor
    

class SemanticIdTokenizer(nn.Module):
    """
        Tokenizes a batch of sequences of item features into a batch of sequences of semantic ids.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dims: List[int],
                 codebook_size: int,
                 n_layers: int = 3,
                 n_cat_feats: int = 18,
                 commitment_weight: float = 0.25,
                 rqvae_weights_path: Optional[str] = None) -> None:
        super().__init__()

        self.rq_vae = RqVae(
            input_dim=input_dim,
            embed_dim=output_dim,
            hidden_dims=hidden_dims,
            codebook_size=codebook_size,
            codebook_kmeans_init=False,
            n_layers=n_layers,
            n_cat_features=n_cat_feats,
            commitment_weight=commitment_weight
        )
        
        if rqvae_weights_path is not None:
            self.rq_vae.load_pretrained(rqvae_weights_path)

        self.rq_vae.eval()

        self.codebook_size = codebook_size
        self.n_layers = n_layers
        self.n_ids = None
        self.cached_ids = None
    
    def _get_hits(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        return (rearrange(key, "b d -> 1 b d") == rearrange(query, "b d -> b 1 d")).all(axis=-1)
    
    @torch.no_grad
    def precompute_corpus_ids(self, movie_dataset: Dataset) -> torch.Tensor:
        cached_ids = None
        dedup_dim = []
        for batch in DataLoader(movie_dataset, batch_size=128, shuffle=False):
            batch_ids = self.forward(batch).sem_ids
            # Detect in-batch duplicates
            is_hit = self._get_hits(batch_ids, batch_ids)
            hits = torch.triu(is_hit, diagonal=1).sum(axis=-1)
            assert hits.min() >= 0
            if cached_ids is None:
                cached_ids = batch_ids.clone()
            else:
                # Detect batch-cache duplicates
                is_hit = self._get_hits(batch_ids, cached_ids)
                hits += torch.triu(is_hit, diagonal=1).sum(axis=-1)
                cached_ids = pack([cached_ids, batch_ids], "* d")[0]
            dedup_dim.append(hits)
        # Concatenate new column to deduplicate ids
        dedup_dim_tensor = pack(dedup_dim, "*")[0]
        self.cached_ids = pack([cached_ids, dedup_dim_tensor], "b *")[0]
        self.n_ids = self.cached_ids.max()+1
        return self.cached_ids
    
    @torch.no_grad
    def forward(self, batch: SeqBatch) -> TokenizedSeqBatch:
        if self.cached_ids is None or batch.ids.max() >= self.cached_ids.shape[0]:
            B, N = batch.ids.shape
            sem_ids = self.rq_vae.get_semantic_ids(batch.x).sem_ids
            D = sem_ids.shape[-1]
        else:
            B, N = batch.ids.shape
            _, D = self.cached_ids.shape
            sem_ids = rearrange(self.cached_ids[batch.ids.flatten(), :], "(b n) d -> b (n d)", n=N)
        seq_mask = repeat(batch.seq_mask, "b d -> b (rep d)", rep=D)
        return TokenizedSeqBatch(
            user_ids=batch.user_ids,
            sem_ids=sem_ids,
            seq_mask=seq_mask
        )

if __name__ == "__main__":
    dataset = MovieLensMovieData("dataset/ml-1m-movie")
    tokenizer = SemanticIdTokenizer(18, 32, [32], 32)
    tokenizer.precompute_corpus_ids(dataset)
    
    seq_data = MovieLensSeqData("dataset/ml-1m")
    batch = seq_data[:10]
    tokenized = tokenizer(batch)
    import pdb; pdb.set_trace()
