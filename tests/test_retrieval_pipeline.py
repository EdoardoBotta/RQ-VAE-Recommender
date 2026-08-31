import torch
import pytest

from data.schemas import SeqBatch
from data.schemas import TokenizedSeqBatch
from modules.model import EncoderDecoderRetrievalModel
from modules.tokenizer.semids import SemanticIdTokenizer


def test_cached_tokenizer_maps_histories_and_masks_padding():
    tokenizer = SemanticIdTokenizer(
        input_dim=4,
        output_dim=2,
        hidden_dims=[4],
        codebook_size=3,
        n_layers=2,
        n_cat_feats=0,
    )
    tokenizer.cached_ids = torch.tensor(
        [
            [0, 1, 0],
            [1, 2, 0],
            [0, 1, 1],
        ]
    )
    batch = SeqBatch(
        user_ids=torch.tensor([[10], [11]]),
        ids=torch.tensor([[0, 1], [2, -1]]),
        ids_fut=torch.tensor([[2], [1]]),
        x=torch.zeros(2, 2, 4),
        x_fut=torch.zeros(2, 1, 4),
        seq_mask=torch.tensor([[True, True], [True, False]]),
    )

    output = tokenizer(batch)

    assert output.sem_ids.shape == (2, 6)
    assert output.sem_ids_fut.shape == (2, 3)
    assert output.seq_mask.shape == (2, 6)
    assert torch.equal(output.sem_ids[0], torch.tensor([0, 1, 0, 1, 2, 0]))
    assert torch.equal(output.sem_ids[1, 3:], torch.tensor([-1, -1, -1]))
    assert torch.equal(output.sem_ids_fut, torch.tensor([[0, 1, 1], [1, 2, 0]]))
    assert tokenizer.training


def test_cached_tokenizer_rejects_ids_outside_the_corpus():
    tokenizer = SemanticIdTokenizer(
        input_dim=4,
        output_dim=2,
        hidden_dims=[4],
        codebook_size=3,
        n_layers=2,
        n_cat_feats=0,
    )
    tokenizer.cached_ids = torch.tensor([[0, 1, 0], [1, 2, 0]])
    batch = SeqBatch(
        user_ids=torch.tensor([[10]]),
        ids=torch.tensor([[0, 2]]),
        ids_fut=torch.tensor([[1]]),
        x=torch.zeros(1, 2, 4),
        x_fut=torch.zeros(1, 1, 4),
        seq_mask=torch.ones(1, 2, dtype=torch.bool),
    )

    with pytest.raises(IndexError, match="outside the precomputed corpus-ID cache"):
        tokenizer(batch)


def test_retrieval_model_trains_and_generates_catalog_ids():
    torch.manual_seed(13)
    corpus_ids = torch.tensor(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]
    )
    model = EncoderDecoderRetrievalModel(
        codebooks=corpus_ids,
        num_hierarchies=2,
        num_embeddings_per_hierarchy=2,
        t5_d_model=8,
        t5_num_heads=2,
        t5_d_ff=16,
        t5_num_layers=1,
        top_k_for_generation=2,
        should_add_sep_token=False,
    )
    batch = TokenizedSeqBatch(
        user_ids=torch.tensor([[0], [1]]),
        sem_ids=torch.tensor(
            [
                [0, 0, 0, 1, 1, 0],
                [1, 0, 0, 0, 1, 0],
            ]
        ),
        sem_ids_fut=torch.tensor([[1, 0, 0], [0, 1, 0]]),
        seq_mask=torch.ones(2, 6, dtype=torch.bool),
        token_type_ids=torch.tensor([[0, 1, 2, 0, 1, 2]]).repeat(2, 1),
        token_type_ids_fut=torch.tensor([[0, 1, 2]]).repeat(2, 1),
    )

    training_output = model(batch)
    training_output.loss.backward()

    assert training_output.loss.ndim == 0
    assert torch.isfinite(training_output.loss)
    assert training_output.loss_d.shape == (2,)
    assert model.item_sid_embedding_table.weight.grad is not None

    model.eval()
    generated = model.generate_next_sem_id(batch)
    assert generated.sem_ids.shape == (2, 2, 2)
    assert generated.log_probas.shape == (2, 2)
    assert torch.isfinite(generated.log_probas).all()
    for candidate in generated.sem_ids.reshape(-1, 2):
        assert (corpus_ids == candidate).all(dim=1).any()
