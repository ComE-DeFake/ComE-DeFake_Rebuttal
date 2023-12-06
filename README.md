# ComE-DeFake
An PyTorch implement of ComE-DeFake in paper:

Debunking Fake News in Online Social Networks Without Text: A New Perspective from Community-Driven Hypergraph Learning

## Dependencies

* python = 3.8.0
* pytorch = 1.12.0
* numpy = 1.24.3
* scipy = 1.10.1
* scikit-learn = 1.3.0
* PyYAML = 6.0.1


## Datasets

We give an example dataset "toy" in /data/ directory to show the input formats of datasets. The toy dataset is just used to show the input format, it's not suitable for experiments.
```
* hypergraph.npy: incidence matrix, where nodes represent users and news represent hyperedges. 
* node_feats.npy: credibiliti-related user attributes. 
* edge_labels.json: labels of news.
* edge_feats.npy: news attributes, which are only used for ComE-DeFakeT, if you consider the textual information from news text. 
```
The datasets can be downloaded through: [Politifact, Gossipcop](https://github.com/KaiDMML/FakeNewsNet), [ReCOVery](https://github.com/apurvamulay/ReCOVery), and [MM-COVID](https://github.com/bigheiniu/MM-COVID). 


## Training Configuration

The hyperparameters needed in training can be set via the configuration file: `./configs/config.yaml`.


## Run Training

First of all, we suggest looking through the available command line arguments defined in `./utility/globals.py`. 

To run the code on CPU

```
python train.py --data_prefix <dataset_path> --dataset <dataset_name>  --train_config ./configs/config.yaml --gpu -1
```


To run the code on GPU

```
python train.py --data_prefix <dataset_path> --dataset <dataset_name>  --train_config ./configs/config.yaml --gpu -1
```

For example, to run dataset 'toy' on CPU:
```
python train.py --data_prefix ./data/ --dataset toy  --train_config ./configs/config.yaml --gpu -1
```


  
