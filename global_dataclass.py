import torch
from dataclasses import dataclass

@dataclass
class G:
    device = torch.device("cpu")
    capacity = 20 #Ampere hours
    num_features = 3 # current, voltage, soc
    lstm_nodes = 256
    window_time = 64 #seconds
    window_size = 16
    slicing = window_time // window_size
    batch_size = 16
    epochs = 128 # should use a power of `T_mult` if you're using cosine annealing, because the cycles restart on a power of `T_mult`
    learning_rate = 0.0035
    weight_decay = 0 # Do not implement weight decay alongside batch normalization