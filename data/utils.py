from data.schemas import SeqBatch


def cycle(dataloader):
    while True:
        for data in dataloader:
            yield data


def next_batch(dataloader, device):
    batch = next(dataloader)
    return SeqBatch(
        user_ids=batch.user_ids.to(device),
        ids=batch.ids.to(device),
        x=batch.x.to(device),
        seq_mask=batch.seq_mask.to(device)
    )
