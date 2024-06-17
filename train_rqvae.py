from accelerate import Accelerator

from data.movie_lens import MovieLensMovieData
from modules.rqvae import RqVae
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader

def train(
    epochs=20,
    batch_size = 64,
    learning_rate = 0.001, 
    weight_decay = 0.01,
    max_grad_norm = 1,
    dataset_folder="data/ml-1m",
    split_batches=True,
    amp=False,
    mixed_precision_type="fp16"
):
    accelerator = Accelerator(
        split_batches=split_batches,
        mixed_precision=mixed_precision_type if amp else 'no'
    )

    device = accelerator.device

    dataset = MovieLensMovieData(root=dataset_folder)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataloder = accelerator.prepare(dataloader)

    optimizer = AdamW(
        lr = learning_rate,
        weight_decay = weight_decay
    )

    scheduler = LinearLR(optimizer)
    
    model = RqVae(
        input_dim = 18,
        embed_dim = 64,
        hidden_dim = 32,
        codebook_size = 32,
        n_layers = 3
    )

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    for epoch in range(epochs):
        model.train()
        for batch in dataloader:
            total_loss = 0

            optimizer.zero_grad()
            data = batch.to(device)
            batch_loss = model(data)

            with accelerator.autocast():
                loss = model(data)
                loss = loss
                total_loss += loss.item()
            
            accelerator.backward(loss)

            accelerator.wait_for_everyone()
            accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
        
            optimizer.step()
            scheduler.step()

            accelerator.wait_for_everyone()


    


    