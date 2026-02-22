import time
import urllib.request
import zipfile
import os
import pandas as pd
import numpy as np
import sys

# Ultimate monkeypatch to defeat RecBole's configurator crash on numpy 2.0
np.float_ = np.float64
np.int_ = np.int64
np.bool_ = bool
sys.modules['numpy'].float = np.float64
sys.modules['numpy'].float_ = np.float64
sys.modules['numpy'].int = np.int64
sys.modules['numpy'].int_ = np.int64
sys.modules['numpy'].bool = bool
sys.modules['numpy'].bool_ = bool

# We must mock it inside recbole before it crashes
import recbole.config.configurator
def silent_settings(self):
    pass
recbole.config.configurator.Config.compatibility_settings = silent_settings

import rusket
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import BPR as RecBoleBPR
from recbole.model.general_recommender import EASE as RecBoleEASE
from recbole.model.general_recommender import ItemKNN as RecBoleItemKNN
from recbole.trainer import Trainer

os.makedirs("data/ml-100k", exist_ok=True)
if not os.path.exists("data/ml-100k/ml-100k.inter"):
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    urllib.request.urlretrieve(url, "data/ml-100k.zip")
    with zipfile.ZipFile("data/ml-100k.zip", 'r') as zip_ref:
        zip_ref.extractall("data")
    
    df = pd.read_csv("data/ml-100k/u.data", sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    df.rename(columns={'user_id': 'user_id:token', 'item_id': 'item_id:token', 'rating': 'rating:float', 'timestamp': 'timestamp:float'}, inplace=True)
    df.to_csv("data/ml-100k/ml-100k.inter", sep='\t', index=False)

import logging
logging.getLogger("recbole").setLevel(logging.CRITICAL)


def run_recbole_benchmark(model_name: str, config_dict: dict):
    config = Config(model=model_name, dataset='ml-100k', config_dict=config_dict, config_file_list=[])
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    if model_name == "BPR":
        model = RecBoleBPR(config, train_data.dataset).to(config['device'])
    elif model_name == "ItemKNN":
        model = RecBoleItemKNN(config, train_data.dataset).to(config['device'])
    elif model_name == "EASE":
        model = RecBoleEASE(config, train_data.dataset).to(config['device'])
    else:
        raise ValueError()

    trainer = Trainer(config, model)
    
    t0 = time.perf_counter()
    trainer.fit(train_data, show_progress=False)
    t1 = time.perf_counter()
    return t1 - t0


def run_rusket_benchmark():
    df = pd.read_csv("data/ml-100k/u.data", sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    
    print("\n" + "="*50)
    print("SPEED BENCHMARK: Rusket vs RecBole (ml-100k)")
    print("="*50)

    print("\n[ItemKNN]")
    rec_time = run_recbole_benchmark("ItemKNN", {'epochs': 1, 'state': 'INFO', 'data_path': 'data/'})
    t0 = time.perf_counter()
    rusket.ItemKNN.from_transactions(df, "user_id", "item_id", k=100)
    rus_time = time.perf_counter() - t0
    
    print(f"RecBole Fit Time: {rec_time:.4f}s")
    print(f"Rusket Fit Time:  {rus_time:.4f}s")
    print(f"Speedup: {rec_time / rus_time:.1f}x faster")

    print("\n[BPR (10 epochs)]")
    rec_time = run_recbole_benchmark("BPR", {'epochs': 10, 'state': 'INFO', 'data_path': 'data/'})
    t0 = time.perf_counter()
    rusket.BPR.from_transactions(df, "user_id", "item_id", iterations=10)
    rus_time = time.perf_counter() - t0
    
    print(f"RecBole Fit Time: {rec_time:.4f}s")
    print(f"Rusket Fit Time:  {rus_time:.4f}s")
    print(f"Speedup: {rec_time / rus_time:.1f}x faster")

    print("\n[EASE]")
    rec_time = run_recbole_benchmark("EASE", {'epochs': 1, 'state': 'INFO', 'data_path': 'data/'})
    t0 = time.perf_counter()
    rusket.EASE.from_transactions(df, "user_id", "item_id")
    rus_time = time.perf_counter() - t0
    
    print(f"RecBole Fit Time: {rec_time:.4f}s")
    print(f"Rusket Fit Time:  {rus_time:.4f}s")
    print(f"Speedup: {rec_time / rus_time:.1f}x faster")


if __name__ == "__main__":
    run_rusket_benchmark()
