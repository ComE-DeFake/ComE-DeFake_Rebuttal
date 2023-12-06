import os,sys,time,datetime
import argparse
import subprocess

timestamp = time.time()
timestamp = datetime.datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d %H-%M-%S')

parser = argparse.ArgumentParser(description="argument for ComE-DeFake training")
parser.add_argument("--data_prefix",default="./data/",type=str,help="prefix identifying data")
parser.add_argument("--dataset",default="toy",type=str,help="path to the configuration of training (*.yml)")
parser.add_argument("--train_config",default="./configs/config.yaml",type=str,help="path to the configuration of training (*.yml)")
parser.add_argument("--gpus",default="-1",type=str,help="number of GPUs")
parser.add_argument("--gamma1",default="0.01",type=float,help="parameter of loss2")
parser.add_argument("--gamma2",default="10",type=int,help="parameter of loss_kl")
parser.add_argument('--update_interval', default=5, type=int)  # [1,3,5]
args_global = parser.parse_args()
