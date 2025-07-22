# RQ-VAE Recommender
This is a PyTorch implementation of a generative retrieval model using semantic IDs based on RQ-VAE from "Recommender Systems with Generative Retrieval". 
The model has two stages:
1. Items in the corpus are mapped to a tuple of semantic IDs by training an RQ-VAE (figure below).
2. Sequences of semantic IDs are tokenized by using a frozen RQ-VAE and a transformer-based is trained on sequences of semantic IDs to generate the next ids in the sequence.
![image](https://github.com/EdoardoBotta/RQ-VAE/assets/64335373/199b38ac-a282-4ba1-bd89-3291617e6aa5)

### Currently supports
* **Datasets:** Amazon Reviews (Beauty, Sports, Toys), MovieLens 1M, MovieLens 32M
* RQ-VAE Pytorch model implementation + KMeans initialization + RQ-VAE Training script.
* Decoder-only retrieval model + Training code with semantic id user sequences from randomly initialized or pretrained RQ-VAE.

### 🤗 Usage on Hugging Face 
RQ-VAE trained model checkpoints are available on Hugging Face 🤗: 
* [**RQ-VAE Amazon Beauty**](https://huggingface.co/edobotta/rqvae-amazon-beauty) checkpoint.

### Installing
Clone the repository and run `pip install -r requirements.txt`. 

No manual dataset download is required.

### Executing
RQ_VAE tokenizer model and the retrieval model are trained separately, using two separate training scripts. 
#### Custom configs
Configs are handled using `gin-config`. 

The `train` functions defined under `train_rqvae.py` and `train_decoder.py` are decorated with `@gin.configurable`, which allows all their arguments to be specified with `.gin` files. These include most parameters one may want to experiment with (e.g. dataset, model sizes, output paths, training length). 

Sample configs for the `train.py` functions are provided under `configs/`. Configs are applied by passing the path to the desired config file as argument to the training command. 
#### Sample usage
To train both models on the **Amazon Reviews** dataset, run the following commands:
* **RQ-VAE tokenizer model training:** Trains the RQ-VAE tokenizer on the item corpus. Executed via `python train_rqvae.py configs/rqvae_amazon.gin`
* **Retrieval model training:** Trains retrieval model using a frozen RQ-VAE: `python train_decoder.py configs/decoder_amazon.gin`

To train both models on the **MovieLens 32M** dataset, run the following commands:
* **RQ-VAE tokenizer model training:** Trains the RQ-VAE tokenizer on the item corpus. Executed via `python train_rqvae.py configs/rqvae_ml32m.gin`
* **Retrieval model training:** Trains retrieval model using a frozen RQ-VAE: `python train_decoder.py configs/decoder_ml32m.gin`

#### Semantic IDs for custom data
If you want to train an RQ-VAE model to produce semantic ids from a pretrained content model embeddings you can use a predefined `custom.gin` config. It utilizes a `CustomDataset` class that loads raw data without any preprocessing.

`CustomDataset` expects two files in a folder you provide in `train.dataset_folder`:
- `item_embeddings.npy`: a numpy array of shape `[num_items, dimension_dim]` containing item embeddings from a content model. Item position corresponds to item id.
- `is_train.npy`: an optional boolean numpy array indicating whether an item belongs to a train split (or eval). If the file does not exist then your config should have `train.do_eval=False`.

By default you would do the following:
1. Put `item_embeddings.npy` into `dataset/custom`
2. Train the model with `python train_rqvae.py configs/custom.gin`
3. Calculate semantic ids with `python run_rqvae.py configs/custom.gin`

The resulting file contains `train.vae_n_layers + 1` id for every item in dataset as in the original paper, so that the combination of semantic ids is unique for every item.

### Next steps
* Comparison encoder-decoder model vs. decoder-only model.

### References
* [Recommender Systems with Generative Retrieval](https://arxiv.org/pdf/2305.05065) by Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan H. Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Q. Tran, Jonah Samost, Maciej Kula, Ed H. Chi, Maheswaran Sathiamoorthy
* [Categorical Reparametrization with Gumbel-Softmax](https://openreview.net/pdf?id=rkE3y85ee) by Eric Jang, Shixiang Gu, Ben Poole
* [Restructuring Vector Quantization with the Rotation Trick](https://arxiv.org/abs/2410.06424) by Christopher Fifty, Ronald G. Junkins, Dennis Duan, Aniketh Iger, Jerry W. Liu, Ehsan Amid, Sebastian Thrun, Christopher Ré
* [vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch) by lucidrains
* [deep-vector-quantization](https://github.com/karpathy/deep-vector-quantization) by karpathy
  
