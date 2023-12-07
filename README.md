# WWW Rebuttal of ComE-DeFake 

## 1. Early Detection Results (The Newly Added Experiment)
### 1.1 Results on dataset ReCOV
<div style="display:inline-block">
  <img src="/figures/recovery_early_acc.png" alt="image1" width="320">
  <img src="/figures/recovery_early_users.png" alt="image2" width="300">
</div>

### 1.1 Results on dataset MM-C
<div style="display:inline-block">
  <img src="/figures/mmcovid_early_acc.png" alt="image1" width="320">
  <img src="/figures/mmcovid_early_users.png" alt="image2" width="300">
</div>

## 2. The Updated Table 2 (The Newly Added Baselines)
![Table2](https://github.com/ComE-DeFake/ComE-DeFake_Rebuttal/blob/main/figures/table2.png)

We compared our proposed method with the three latest methods: BERTweet [1], Llama 2 [2], and HG-SL [3]. 

[1] Dat Quoc Nguyen, Thanh Vu, and Anh Tuan Nguyen. 2020. BERTweet: A pre-trained language model for English Tweets. In EMNLP, 9–14. 

[2] Touvron, Hugo, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288. 

[3] Ling Sun, Yuan Rao, Yuqian Lan, Bingcan Xia, and Yangyang Li. 2023. HG-SL: Jointly Learning of Global and Local User Spreading Behavior for Fake News Early Detection. In AAAI. 5248–5256. 

## 3. Dataset Introduction

### 3.1 User Attributes
We use 6 user-relevant features crawled by Twitter API to be user attributes of our model: 

* `'followers_count'`: The number of followers.  
* `'friends_count'`: The number of users this account is following (AKA their “followings”). 
* `'listed_count'`: The number of public lists that this user is a member of. 
* `'verified'`: When true, indicates that the user has a verified account.
* `'statuses_count'`: The number of Tweets (including retweets) issued by the user.  
* `'favourites_count'`: The number of Tweets this user has liked in the account’s lifetime. 


### 3.2 Format of Crawled Row Data
For each piece of source news, we crawl its relevant tweets through Twitter API. The format of a crawled tweet can be read as a dictionary format in Python. The list of Keys of this dictionary is shown below:

``
['created_at', 'id', 'id_str', 'text', 'truncated', 'entities', 'source', 'in_reply_to_status_id', 'in_reply_to_status_id_str', 'in_reply_to_user_id', 'in_reply_to_user_id_str', 'in_reply_to_screen_name', 'user', 'geo', 'coordinates', 'place', 'contributors', 'is_quote_status', 'retweet_count', 'favorite_count', 'favorited', 'retweeted', 'possibly_sensitive', 'lang']
``

The value of `'user'` in the above list is again a dictionary that contains, but not only, the features we described in __3.1__.


## 4. Comparison of t-SNE visualization
![tSNE](https://github.com/ComE-DeFake/ComE-DeFake_Rebuttal/blob/main/figures/tsne_all.png)


## 5. Statistical Significance - P Value


| ReCOV | ComE-DeFake | ComE-DeFake |  ComE-DeFake | ComE-DeFake | ComE-DeFake | ComE-DeFake | ComE-DeFake |  ComE-DeFake | ComE-DeFake | ComE-DeFake |
| --- | ----------- | ----------- | ----------- |----------- |----------- |----------- |----------- |----------- |----------- |----------- |
|         | __TextCNN__     | __HAN__         |      __BERT__   |   __TextGCN__   |   __HyperGAT__  |   __DualEmo__   | __HGFND__       |    __BERTweet__ | __Llama__     |   __HG-SL__     |
| __ACC__  | 1.06e-05 | 3.65e-06 | 9.07e-05 | 5.12e-05 | 4.79e-07 | 0.0007   | 0.0066 | 2.61e-06 | 0.0097 | 0.0067 |
| __Pre__  | 1.79e-05 | 1.58e-06 | 0.0016   | 0.0023   | 1.38e-07 | 0.0005   | 0.0021 | 4.58e-06 | 0.0029 | 0.0067 |
| __Rec__  | 2.94e-05 | 0.0012   | 0.0039   | 0.0002   | 3.82e-07 | 0.0007   | 0.0067 | 9.07e-05 | 0.0218 | 0.0056 |
| __F1__   | 5.16e-05 | 1.26e-05 | 5.14e-05 | 9.00e-04 | 1.77e-07 | 5.00e-04 | 0.0023 | 6.37e-05 | 0.0099 | 0.0058 |


## 6. Running Time

|  Datasets  | TextCNN  | HAN  |  BERT |  TextGCN  |  HyperGAT |  DualEmo  | HGFND  |  BERTweet| Llama |  HG-SL | ComE-DeFake |
| ---------- | -------- | -----| ----- |---------- |---------- |---------- |------- |--------- |------ |------- |------------ |
|            |  (C)     | (C)  | (G)   |   (G)     |    (G)    |    (G)    |   (G)  |   (G)    |   (G) | (G)    |   (G)       |
| __ReCOV__  |   33.4   | 136.1| 4.6   |  0.03     |   22.8    |   0.39    |  16.1  |   6.8    | 20.9  |   1.4  |   0.27      |
| __MM-C__   |   67     |290.6 |  26.9 |  0.09     |   --      |    0.29   |  10.2  |   30.5   | 106.4 |  1     |   0.19      |

## 7. Comparison of Graph-based Methods

![Structured-based Methds](https://github.com/ComE-DeFake/ComE-DeFake_Rebuttal/blob/main/figures/methods_comparison.png)

[1].	Jing Ma, Wei Gao, and Kam-Fai Wong. 2018. Rumor Detection on Twitter with Tree-structured Recursive Neural Networks. In ACL. 1980–1989. 

[2].	Tian Bian, Xi Xiao, Tingyang Xu, Peilin Zhao, Wenbing Huang, Yu Rong, and Junzhou Huang. 2020. Rumor detection on social media with bi-directional graph convolutional networks. In AAAI. 549–556. 

[3].	Nir Rosenfeld, Aron Szanto, and David C. Parkes. 2020. A Kernel of Truth: Determining Rumor Veracity on Twitter by Diffusion Pattern Alone. In WWW. 1018–1028. 

[4].	Van-Hoang Nguyen, Kazunari Sugiyama, Preslav Nakov, and Min-Yen Kan. 2020. FANG: Leveraging Social Context for Fake News Detection Using Graph Representation. In CIKM. 1165–1174. 

[5].	Yingtong Dou, Kai Shu, Congying Xia, Philip S Yu, and Lichao Sun. 2021. User preference-aware fake news detection. In SIGIR. 2051–2055. 

[6].	Yuxiang Ren and Jiawei Zhang. 2021. Fake news detection on news-oriented heterogeneous information networks through hierarchical graph attention. In IJCNN. 1–8. 

[7].	Ruichao Yang, Jing Ma, Hongzhan Lin, and Wei Gao. 2022. A Weakly Supervised Propagation Model for Rumor Verification and Stance Detection with Multiple Instance Learning. In SIGIR. 1761–1772. 

[8].	Yiqiao Jin, Xiting Wang, Ruichao Yang, Yizhou Sun, Wei Wang, Hao Liao, and Xing Xie. 2022. Towards fine-grained reasoning for fake news detection. In AAAI. 5746–5754.

[9].	Ruichao Yang, Xiting Wang, Yiqiao Jin, Chaozhuo Li, Jianxun Lian, and Xing Xie. 2022. Reinforcement Subgraph Reasoning for Fake News Detection. In KDD. 2253–2262. 

[10].	Erxue Min, Yu Rong, Yatao Bian, Tingyang Xu, Peilin Zhao, Junzhou Huang, and Sophia Ananiadou. 2022. Divide-and-Conquer: Post-User Interaction Network for Fake News Detection on Social Media. In WWW. 1148–1158. 

[11].	Ujun Jeong, Kaize Ding, Lu Cheng, Ruocheng Guo, Kai Shu, and Huan Liu. 2022. Nothing stands alone: Relational fake news detection with hypergraph neural networks. In Big Data. 596–605. 

[12].	Sun, X., Yin, H., Liu, B., Meng, Q., Cao, J., Zhou, A. and Chen, H.. 2022. Structure learning via meta-hyperedge for dynamic rumor detection. IEEE Transactions on Knowledge and Data Engineering. 

[13].	Nikhil Mehta, Maria Leonor Pacheco, and Dan Goldwasser. 2022. Tackling Fake News Detection by Continually Improving Social Context Representations using Graph Neural Networks. In ACL. 1363–1380.

[14].	Xing Su, Jian Yang, Jia Wu, and Yuchen Zhang. 2023. Mining User-Aware Multi-Relations for Fake News Detection in Large Scale Online Social Networks. In WSDM. 51–59. 

[15].	Ling Sun, Yuan Rao, Yuqian Lan, Bingcan Xia, and Yangyang Li. 2023. HG-SL: Jointly Learning of Global and Local User Spreading Behavior for Fake News Early Detection. In AAAI. 5248–5256. 


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

