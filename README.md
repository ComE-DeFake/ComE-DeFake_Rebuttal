# WWW Rebuttal of ComE-DeFake 

## 1. Early Detection Results (The Newly Added Experiment)
### 1.1 Results on dataset ReCOV
<div style="display:inline-block">
  <img src="/figures/recovery_early_acc.png" alt="image1" width="300">
  <img src="/figures/recovery_early_users.png" alt="image2" width="300">
</div>

## 2. The Updated Table 2 (The Newly Added Baselines)
![Table2](https://github.com/ComE-DeFake/ComE-DeFake_Rebuttal/blob/main/figures/table2.png)

We compared our proposed method with the three latest methods: BERTweet [1], Llama 2 [2], and HG-SL [3]. 

[1] Dat Quoc Nguyen, Thanh Vu, and Anh Tuan Nguyen. 2020. BERTweet: A pre-trained language model for English Tweets. In EMNLP, 9–14. 

[2] Touvron, Hugo, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288. 

[3] Ling Sun, Yuan Rao, Yuqian Lan, Bingcan Xia, and Yangyang Li. 2023. HG-SL: Jointly Learning of Global and Local User Spreading Behavior for Fake News Early Detection. In AAAI. 5248–5256. 

## 3. Dataset Introduction

## 4. Comparison of t-SNE visualization
![tSNE](https://github.com/ComE-DeFake/ComE-DeFake_Rebuttal/blob/main/figures/tsne_all.png)


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

