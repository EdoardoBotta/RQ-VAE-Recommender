from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class InverseSquareRootScheduler(LRScheduler):
    def __init__(self, optimizer: Optimizer, warmup_steps: int, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        super(InverseSquareRootScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        step = self.last_epoch + 1
        if step <= self.warmup_steps:
            return self.base_lrs
        scale_factor = (self.warmup_steps ** 0.5) / (step ** 0.5)
        return [base_lr * scale_factor for base_lr in self.base_lrs]
