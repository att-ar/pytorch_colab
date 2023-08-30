# SOC Estimation Using PyTorch and Pandas 98% accuracy on Dev set and entire dataset

**Dataset used is confidential test data and cannot be shared**

However, the repo [att-ar/ecm_battery_simulation](https://github.com/att-ar/ecm_battery_simulation) outputs data in the same structure as the dataset that was used:

|time|current|voltage|soc|
|---|---|---|---|
|0.0|0.0|3.65|95|
|1.0|8.0|3.50|94.99|
|...|...|...|...|

This repo contains a pytorch LSTM network that will be used to estimate State of Charge for a LFP Li-ion (LFP chemistry) cell from real time voltage, current and time data


The LSTM input will be voltage, current, and previous SOC points in a batch of windowed data of shape: <br>```(G.batch_size, G.window_size, G.num_features)```

The voltage, current and soc data will be from time: $$t - \text{windowsize} - 1 \rightarrow t - 1$$<br>
The output should be the SOC prediction at time $t$ for each batch, the output shape should be ```(G.batch_size, 1)```

98% accurate:

<img src="https://github.com/att-ar/pytorch_colab/blob/main/dataset_predict.png">
