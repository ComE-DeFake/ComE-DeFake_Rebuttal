# WWW Rebuttal of ComE-DeFake 

## 1. Early Detection Results (The Newly Added Experiment)
[![Image text](https://github.com/ComE-DeFake/ComE-DeFake_Rebuttal/tree/main/figuresrecovery_early_acc.pdf)](https://github.com/ComE-DeFake/ComE-DeFake_Rebuttal/blob/main/figures/recovery_early_acc.pdf)

## 2. The Updated Table 2 (The Newly Added Baselines)

## 3. Dataset Introduction

## 4. t-SNE of original and ours

## 5. Statistical Significance

## 6. Running Time

## 7. Comparison of Structured Methods


## 8. PyTorch Implementation

### 8.1. Dependencies
* python = 3.8.0
* pytorch = 1.12.0
* numpy = 1.24.3
* scipy = 1.10.1
* scikit-learn = 1.3.0
* PyYAML = 6.0.1

### 8.2. Training Configuration
The hyperparameters needed in training can be set via the configuration file: `./configs/config.yaml`.

### 8.3. Run Training
First of all, we suggest looking through the available command line arguments defined in `./utility/globals.py`. 

To run the code on CPU

```
python train.py --data_prefix <dataset_path> --dataset <dataset_name>  --train_config ./configs/config.yaml --gpu -1
```

To run the code on GPU

```
python train.py --data_prefix <dataset_path> --dataset <dataset_name>  --train_config ./configs/config.yaml --gpu -1
```


  
