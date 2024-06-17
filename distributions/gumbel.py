import torch

def sample_gumbel(shape, device, eps=1e-20):
  """Sample from Gumbel(0, 1)"""
  U = torch.rand(shape, device=device)
  return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature, device):
  """ Draw a sample from the Gumbel-Softmax distribution"""
  y = logits + sample_gumbel(logits.shape, device)
  return torch.nn.functional.softmax( y / temperature, dim=-1)