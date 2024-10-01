# ICML 2024 ID #259

This demo code performs GWNET on the PEMS08-16 dataset and subsequently testing it on the shifted dataset PEMS08-17. To train GWNET, execute the following Python command in the terminal:

```bash
python experiments/gwnet/main.py --device cuda:0 --dataset PEMS08 --years 2016 --model_name gwnet
```

To evaluate the shifted performance, use the following command:

```bash
python experiments/gwnet/main.py --device cuda:0 --dataset PEMS08 --years 2016 --model_name gwnet --mode test --target 2017
```

# Acknowledgement
Our code is developed based on LargeST, an easy-to-use and powerful open-source ST-GNNs training framework. 