import os
import yaml
import pandas as pd
import numpy as np
import argparse
from pkgutil import get_data
from get_data import get_data, read_param
from load_data import load_save_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.linear_model import ElasticNet
import joblib
import json
import mlflow
from urllib.parse import urlparse


def eval_metrics(actual, pred):


def tarin_and_eveluate(config_path):
    config=read_param(config_path)
    train_data_path=config["split_data"]["train_path"]
    test_data_path=config["split_data"]["test_path"]
    raw_data_path=config["load_data"]["clean_data"]
    split_data=config["split_data"]["test_size"]
    random_state=config["base"]["random_state"]
    df=pd.read_csv(raw_data_path, sep=",")

    model_dir=config["model_path"]
    alpha=config["estimators"]["ElasticNet"]["params"]["alpha"]
    l1_ration=config["estimators"]["ElasticNet"]["params"]["l1_ratio"]

    target=config["base"]["target_col"]
    train=pd.read_csv("train_data_path")
    test=pd.read_csv("test_data_path")
    
    train_y=train[target]
    test_y=test[target]

    train_x=train.drop(target, axis=1)
    test_x=test.drop(target, axis=1)

    lr=ElasticNet(alpha=alpha,l1_ration=l1_ration,random_state=random_state)
    lr.fit(train_x,train_y)
    predicted_qualities= lr.predict(test_x)
    
    (rmse,mae,r2)=eval_metrics(test_y,predicted_qualities)
    print("Elasticnet model (alpha=%f, l1_ratio=%f) : " % (alpha,l1_ration))

    scores_files=config["reports"]["score"]
    params_files=config["reports"]["params"]

    with open(scores_files,"w") as f:
        scores={
            "rmse":rmse,
            "mae":mae,
            "r2":r2
        }
        json.dump(scores,f)

    with open(params_files,"w") as f:
        params={
            "alpha":alpha,
            "l1_ration":l1_ration
        }
        json.dump(params,f)

model_path=config("model_path")
joblib.dump(lr, model_path)        

if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--config",default="params.yml")
    parsed_args=args.parse_args()
    tarin_and_eveluate(config_path=parsed_args.config)
