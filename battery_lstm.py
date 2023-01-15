import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.modules.activation import Sigmoid

import numpy as np

from global_dataclass import G


class BatterySet(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
    
        self.logits = torch.from_numpy(x.squeeze()).to(G.device)
        self.labels = torch.from_numpy(y).to(G.device)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (self.logits[idx], self.labels[idx])

class LSTMNetwork(nn.Module):
    
    def __init__(self):
        super().__init__()

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

def train_loop(dataloader, model, loss_fn, optimizer, scheduler, epoch):
    size = len(dataloader)
    train_loss, perc_error = 0.0, 0.0
    model.train()
    for batch, (x,y) in enumerate(dataloader):
        try:
            optimizer.zero_grad() #resets the gradient graph
            
            #forward
            predict = model(x)
            loss = loss_fn(predict, y).mean(0) # assert(loss.shape == (1))

            #backward
            loss.backward()

            with torch.no_grad():
                train_loss += loss.item()

            optimizer.step()
            ##### For OneCycleLR:
            scheduler.step()
            ##### For CosineAnnealingWarmRestarts:
            # scheduler.step(epoch + (batch+1) // size)

            if loss.isnan():
                print("loss was NaN")
                break

            if batch % (size // 3) == 0:
                print(f"batch mean loss: {loss.item():>7f}  [{batch:4d}/{size:4d}]")

            with torch.no_grad(): #used to check bias and variance by comparing with test set
                perc_error += torch.mean(torch.abs(predict - y) / (y+ 1e-2) * 100, (0,1))
        except KeyboardInterrupt:
            break
        
    train_loss /= size
    perc_error /= size
    print(f"Train Error: \nAverage Accuracy: {100 - perc_error}%, Avg Loss: {train_loss:>8f}\n")
    return train_loss, 100.0 - perc_error

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader)
    test_loss, perc_error = 0.0, 0.0
    model.eval()
    with torch.no_grad(): #doesnt update parameters (we are testing not training)
        for counter, (x,y) in enumerate(dataloader):
            try:
                predict = model(x).reshape(y.shape)
                test_loss += loss_fn(predict, y).mean(0).item()
                perc_error += torch.mean(torch.abs(predict - y) / (y+ 1e-2) * 100, (0,1))
            
                counter += 1
                if counter % (size // 2) == 0:
                    print(f"{counter} / {size} tested")
            except KeyboardInterrupt:
                break
    test_loss /= size
    perc_error /= size
    print(f"Test Error: \nAverage Accuracy: {100 - perc_error}%, Avg Loss: {test_loss:>8f}\n")
    return test_loss, 100.0 - perc_error