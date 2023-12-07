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


| ReCOV | ComE-DeFake | ComE-DeFake |  ComE-DeFake | ComE-DeFake | ComE-DeFake | ComE-DeFake | ComE-DeFake |  ComE-DeFake | ComE-DeFake | ComE-DeFake |
| --- | ----------- | ----------- | ----------- |----------- |----------- |----------- |----------- |----------- |----------- |----------- |
|         | __TextCNN__     | __HAN__         |      __BERT__   |   __TextGCN__   |   __HyperGAT__  |   __DualEmo__   | __HGFND__       |    __BERTweet__ | __Llama__     |   __HG-SL__     |
| __ACC__  | 1.06e-05 | 3.65e-06 | 9.07e-05 | 5.12e-05 | 4.79e-07 | 0.0007   | 0.0066 | 2.61e-06 | 0.0097 | 0.0067 |
| __Pre__  | 1.79e-05 | 1.58e-06 | 0.0016   | 0.0023   | 1.38e-07 | 0.0005   | 0.0021 | 4.58e-06 | 0.0029 | 0.0067 |
| __Rec__  | 2.94e-05 | 0.0012   | 0.0039   | 0.0002   | 3.82e-07 | 0.0007   | 0.0067 | 9.07e-05 | 0.0218 | 0.0056 |
| __F1__   | 5.16e-05 | 1.26e-05 | 5.14e-05 | 9.00e-04 | 1.77e-07 | 5.00e-04 | 0.0023 | 6.37e-05 | 0.0099 | 0.0058 |


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

