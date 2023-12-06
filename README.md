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


  
