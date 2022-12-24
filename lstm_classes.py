import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.modules.activation import Sigmoid

class BatterySet(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
    
        self.logits = torch.from_numpy(x.squeeze()).to(device)
        self.labels = torch.from_numpy(y).to(device)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (self.logits[idx], self.labels[idx])

def log_cosh_loss(y_pred: torch.tensor, y_ground: torch.tensor) -> torch.tensor:
    x = y_pred - y_ground
    return x + torch.nn.functional.softplus(-2. * x) - np.log(2.0)


class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                y_pred: torch.Tensor,
                y_true: torch.Tensor) -> torch.Tensor:
        return log_cosh_loss(y_pred, y_true)


class LSTMNetwork(nn.Module):
    
    def __init__(self):
        super(LSTMNetwork, self).__init__()

        self.lstm = nn.LSTM(G.num_features, G.lstm_nodes, 1, batch_first = True)
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.uniform_(param, a=0.001, b=0.09)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param, nonlinearity = "relu") # PyTorch equivalent to He Normalization
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

        self.linear_stack = nn.Sequential(
            nn.Linear(G.lstm_nodes, G.lstm_nodes), # shape == (G.batch_size, G.lstm_nodes)
            nn.BatchNorm1d(G.lstm_nodes, momentum = 0.92),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(G.lstm_nodes, G.lstm_nodes), # shape == (G.batch_size, G.lstm_nodes)
            nn.BatchNorm1d(G.lstm_nodes, momentum = 0.92),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(G.lstm_nodes, 1), # shape == (G.batch_size, 1)
            Sigmoid()
        )
        for i in range(0, len(self.linear_stack), 2):
            for name, param in self.linear_stack[i].named_parameters():
                if 'bias' in name:
                    nn.init.uniform_(param,
                                     a=(0.1 / (i/2 + 1)),
                                     b=(0.9/ (i/2 + 1))
                                    )
                elif 'weight' in name:
                    nn.init.kaiming_normal_(param, nonlinearity = "relu")
    
    def forward(self, x): # assert(x.shape == (G.batch_size, G.window_size, G.num_features))
        #lstm
        x_out, (h_n_lstm, c_n)  = self.lstm(x) # assert(h_n_lstm.shape == (1, G.batch_size, G.lstm_nodes))
        #Dense Layers
        # send the final lstm layer's hidden state values to the Dense Layers
        out = self.linear_stack(h_n_lstm.squeeze())
        return out # (G.batch_size, 1)