# SOC Estimation Using PyTorch and Pandas

This repo contains a pytorch LSTM network that will be used to estimate State of Charge for a LFP Li-ion (LFP chemistry) cell from real time voltage, current and time data

The LSTM input will be voltage, current, and previous SOC points in a batch of windowed data (of shape ```(G.batch_size, G.window_size, G.num_features)```) <br>
The voltage, current and soc data will be from time: $$t - \text{windowsize} - 1 \rightarrow t - 1$$

The output should be the SOC prediction at time $t$ for each batch, the output shape should be ```(G.batch_size, 1)```
