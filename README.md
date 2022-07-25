# SOC Estimation Using PyTorch and Pandas

This repo contains a pytorch network that will be used to estimate State of Charge for a LFP Li-ion (LFP chemistry) cell from real time voltage, current and time data

The LSTM input will be voltage, current, delta time and previous SOC points in a window of data (of size ```G.window_size```) <br>
The voltage, current and soc data will be from time: $$t - \text{windowsize} - 1 \rightarrow t - 1$$

The output should be the SOC prediction at time $t$
