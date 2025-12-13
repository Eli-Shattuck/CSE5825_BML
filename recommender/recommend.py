import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

from libreco.data import DataInfo, DatasetPure
from libreco.algorithms import SVD
from libreco.evaluation import evaluate

from datetime import datetime
def recommend(model_path='../checkpoints/checkpoint_final_2025-12-08 14:13:05.920304', user_puzzle_path='../data/german11_user_puzzles.npy'):
    data_info = DataInfo.load(model_path, model_name="puzzle_recommender")
    model = SVD.load(
        path=model_path, model_name="puzzle_recommender", data_info=data_info, manual=True
    )

    return model.recommend_user(
        user=382, # german11
        n_rec=10,
        cold_start='average',
        filter_consumed=True,
        random_rec=False
    )
    

if __name__ == '__main__':
    print(recommend())