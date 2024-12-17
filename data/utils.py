from data.schemas import SeqBatch


def cycle(dataloader):
    while True:
        for data in dataloader:
            yield data


def batch_to(batch, device):
    return SeqBatch(*[v.to(device) for _,v in batch._asdict().items()])


def next_batch(dataloader, device):
    batch = next(dataloader)
    return batch_to(batch, device)
