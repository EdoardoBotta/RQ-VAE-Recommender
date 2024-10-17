import torch
from modules.embedding.id_embedder import SemIdEmbedder
from modules.embedding.id_embedder import UserIdEmbedder
from modules.tokenizer.semids import TokenizedSeqBatch
from modules.transformer.model import TransformerDecoder
from typing import NamedTuple
from torch import nn
from torch.nn import functional as F


class ModelOutput(NamedTuple):
    loss: torch.Tensor


class DecoderRetrievalModel(nn.Module):
    def __init__(self,
                 embedding_dim,
                 d_out,
                 dropout,
                 num_heads,
                 n_layers,
                 num_embeddings,
                 max_pos=2048) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.sem_id_embedder = SemIdEmbedder(num_embeddings, embedding_dim)
        self.user_id_embedder = UserIdEmbedder(2000, embedding_dim)
        
        self.wpe = nn.Embedding(num_embeddings=max_pos, embedding_dim=embedding_dim)
        self.wte = nn.Embedding(num_embeddings=2, embedding_dim=embedding_dim)

        self.decoder = TransformerDecoder(
            d_in=embedding_dim,
            d_out=d_out,
            dropout=dropout,
            num_heads=num_heads,
            n_layers=n_layers,
            do_cross_attn=False
        )

        self.out_proj = nn.Linear(d_out, num_embeddings)
    
    def forward(self, batch: TokenizedSeqBatch) -> torch.Tensor:
        # TODO: Handle paddings in tokenization
        import pdb; pdb.set_trace()
        B, N = batch.seq_mask
        user_emb = self.user_id_embedder(batch.user_ids)
        sem_ids_emb = self.sem_id_embedder(batch.sem_ids)
        
        user_id_seq_mask = torch.zeros((B, 1), dtype=torch.bool)
        seq_mask = torch.cat([user_id_seq_mask, batch.seq_mask], axis=1)

        input_embedding = torch.cat([user_emb, sem_ids_emb], axis=1)
        transformer_output = self.decoder(input_embedding)

        if self.training:
            out = F.softmax(transformer_output, dim=-1)[seq_mask, :][:-1, :, :].reshape((-1, self.num_embeddings))
            target = batch.sem_ids[seq_mask, :][1:, :, :].flatten()
            loss = F.cross_entropy(out, target)

        return ModelOutput(loss=loss)
