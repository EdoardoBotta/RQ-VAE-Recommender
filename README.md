# RQ-VAE Recommender
This is a PyTorch implementation of a generative retrieval model using semantic IDs based on RQ-VAE from "Recommender Systems with Generative Retrieval". 
The model has two stages:
1. Items in the corpus are mapped to a tuple of semantic IDs by training an RQ-VAE (figure below).
2. Sequences of semantic IDs are tokenized by using a frozen RQ-VAE and a transformer-based is trained on sequences of semantic IDs to generate the next ids in the sequence.
![image](https://github.com/EdoardoBotta/RQ-VAE/assets/64335373/199b38ac-a282-4ba1-bd89-3291617e6aa5).

### Currently supports
* **Datasets:** MovieLens 1M
* RQ-VAE Pytorch model implementation + KMeans initialization + RQ-VAE Training script.
* Decoder-only retrieval model + Training code with semantic id user sequences from randomly initialized RQ-VAE.

### Executing
RQ_VAE tokenizer model and the retrieval model are trained separately, using two separate training scripts.
* **RQ-VAE tokenizer model training:** Trains the RQ-VAE tokenizer on the item corpus. Executed via `python train_rqvae.py`
* **Retrieval model training:** Trains retrieval model using a frozen RQ-VAE: `python train_decoder.py`

### Next steps
* ML1M timestamp-based train/test split.
* Comparison encoder-decoder model vs. decoder-only model.
* Eval loops.

### References
* [Recommender Systems with Generative Retrieval](https://arxiv.org/pdf/2305.05065) by Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan H. Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Q. Tran, Jonah Samost, Maciej Kula, Ed H. Chi, Maheswaran Sathiamoorthy
* [Categorical Reparametrization with Gumbel-Softmax](https://openreview.net/pdf?id=rkE3y85ee) by Eric Jang, Shixiang Gu, Ben Poole
  
