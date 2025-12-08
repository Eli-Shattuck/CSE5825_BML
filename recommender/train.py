import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

from libreco.data import DatasetPure
from libreco.algorithms import SVD
from libreco.evaluation import evaluate

from datetime import datetime

dataset_df = pd.read_csv('../data/interactions.csv', index_col=0)

user_counts = dataset_df['user'].value_counts()

# remove users with only 1 interaction
dataset_df = dataset_df[dataset_df['user'].map(dataset_df['user'].value_counts()) >= 2]

train_df, eval_df = train_test_split(dataset_df, test_size=0.1, random_state=42, stratify=dataset_df['user'])

# remove items that only appear in eval set
print(len(eval_df), end=' -> ')
valid_items = set(train_df["item"].unique())
eval_df = eval_df[ eval_df["item"].isin(valid_items) ]
print(len(eval_df))

train_users = set(train_df['user'])
train_items = set(train_df['item'])

val_users = set(eval_df['user']) - train_users
val_items = set(eval_df['item']) - train_items

print("Users only in val:", len(val_users))
print("Items only in val:", len(val_items))

mn,mx = 0, 10

scaler = MinMaxScaler(feature_range=(mn, mx))
train_df['label'] = scaler.fit_transform(train_df[['label']])
eval_df['label']  = scaler.transform(eval_df[['label']])

train_data, data_info = DatasetPure.build_trainset(train_data=train_df)
eval_data = DatasetPure.build_evalset(eval_data=eval_df)

model = SVD(
    task='rating',
    data_info=data_info,
    embed_size=32,
    lr=0.1,
    lr_decay=True,
    reg=0.001,
    n_epochs=100,
    seed=42,
    lower_upper_bound=(mn, mx)
)

now = str(datetime.now())
print(f'starting training at {now}')

model.fit(
    train_data,
    neg_sampling=False,
    shuffle=True,
    eval_data=eval_data,
    metrics=["rmse", "mae"],
    verbose=2
)

metrics = evaluate(
    model,
    eval_data,
    neg_sampling=False,
    metrics=["rmse", "mae"],
    k=10,
    eval_batch_size=2048
)

print(f'FINAL | eval rmse: {metrics["rmse"]} eval mae: {metrics["mae"]}')

model.save(
    path=f"../checkpoints/checkpoint_final_{now}", 
    model_name="puzzle_recommender", 
    manual=True,
    inference_only=True
)