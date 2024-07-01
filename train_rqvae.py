import gin

from accelerate import Accelerator
from data.movie_lens import MovieLensMovieData
from distributions.gumbel import TemperatureScheduler
from modules.rqvae import RqVae
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from tqdm import tqdm

def cycle(dataloader):
    while True:
        for data in dataloader:
            yield data

@gin.configurable
def train(
    iterations=500000,
    batch_size = 64,
    learning_rate = 0.001, 
    weight_decay = 0.01,
    max_grad_norm = 1,
    dataset_folder="dataset/ml-1m",
    split_batches=True,
    amp=False,
    mixed_precision_type="fp16",
    gradient_accumulate_every=1
):
    accelerator = Accelerator(
        split_batches=split_batches,
        mixed_precision=mixed_precision_type if amp else 'no'
    )

    device = accelerator.device

    dataset = MovieLensMovieData(root=dataset_folder)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataloader = cycle(dataloader)
    dataloader = accelerator.prepare(dataloader)
    
    model = RqVae(
        input_dim = 18,
        embed_dim = 64,
        hidden_dim = 32,
        codebook_size = 32,
        n_layers = 3
    )

    optimizer = AdamW(
        params=model.parameters(),
        lr = learning_rate,
        weight_decay = weight_decay
    )

    scheduler = LinearLR(optimizer)

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    temp_scheduler = TemperatureScheduler(
        t0=1,
        min_t=0.1,
        anneal_rate=0.00003,
        step_size=6000
    )

    for iter in tqdm(range(iterations)):
        model.train()
        total_loss = 0
        t = temp_scheduler.get_t(iter)

        optimizer.zero_grad()
        for _ in range(gradient_accumulate_every):
            data = next(dataloader).to(device)

            with accelerator.autocast():
                loss = model(data, gumbel_t=t)
                loss = loss / gradient_accumulate_every
                total_loss += loss.item()

            accelerator.backward(loss)

        accelerator.wait_for_everyone()
        accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
    
        optimizer.step()
        scheduler.step()

        accelerator.wait_for_everyone()

if __name__ == "__main__":
    train()


    


    