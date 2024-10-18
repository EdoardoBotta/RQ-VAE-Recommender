# RQ-VAE Recommender
This is a PyTorch implementation of a generative retrieval model based on RQ-VAE from "Recommender Systems with Generative Retrieval". 
![image](https://github.com/EdoardoBotta/RQ-VAE/assets/64335373/199b38ac-a282-4ba1-bd89-3291617e6aa5)
### Currently supports
* RQ-VAE Pytorch model implementation + KMeans initialization + RQ-VAE Training on MovieLens 1M.

### Executing
RQ_VAE tokenizer model and the retrieval model are trained separately, using two separate training scripts.
* **RQ-VAE tokenizer model training:** Trains the RQ-VAE tokenizer on the item corpus. Executed via `python train_rqvae.py`
* **Retrieval model training:** Trains retrieval model using a frozen RQ-`python train_decoder.py` (Currently unstable)

### Next steps
* Retrieval model + Training code with semantic id user sequences.
* Comparison encoder-decoder model vs. decoder-only model.
* Properly package repository.

### References
* [Recommender Systems with Generative Retrieval](https://arxiv.org/pdf/2305.05065) by Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan H. Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Q. Tran, Jonah Samost, Maciej Kula, Ed H. Chi, Maheswaran Sathiamoorthy
  
