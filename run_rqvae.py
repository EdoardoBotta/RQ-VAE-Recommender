import os
import gin
import torch
import numpy as np
from modules.rqvae import RqVae
from modules.quantize import QuantizeForwardMode
from modules.tokenizer.semids import SemanticIdTokenizer
from modules.utils import parse_config
from data.custom import CustomDataset
from train_rqvae import train  # Import the train function to make its parameters available to gin

@gin.configurable
def generate_semantic_ids(
    pretrained_model_path: str,
    output_path: str,
    batch_size: int = 64,
):
    dataset = CustomDataset(root=gin.query_parameter('train.dataset_folder'))
    
    tokenizer = SemanticIdTokenizer(
        input_dim=gin.query_parameter('train.vae_input_dim'),
        output_dim=gin.query_parameter('train.vae_embed_dim'),
        hidden_dims=gin.query_parameter('train.vae_hidden_dims'),
        codebook_size=gin.query_parameter('train.vae_codebook_size'),
        n_layers=gin.query_parameter('train.vae_n_layers'),
        n_cat_feats=gin.query_parameter('train.vae_n_cat_feats'),
        rqvae_weights_path=pretrained_model_path,
        rqvae_codebook_normalize=gin.query_parameter('train.vae_codebook_normalize'),
        rqvae_sim_vq=gin.query_parameter('train.vae_sim_vq')
    )
    
    device = tokenizer.rq_vae.device
    dataset.item_embeddings = dataset.item_embeddings.to(device)
    
    semantic_ids = tokenizer.precompute_corpus_ids(dataset)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, semantic_ids.cpu().numpy())
    print(f"Saved semantic IDs to {output_path}")

if __name__ == "__main__":
    parse_config()
    generate_semantic_ids() 