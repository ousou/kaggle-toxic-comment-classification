import pandas as pd
import numpy as np

DATA_FOLDER = "data"
seed = 9

def read_data(train_data_perc=0.8):
    train_data_file = DATA_FOLDER + "/" + "train.csv"

    all_data = pd.read_csv(train_data_file)
    X = all_data[["id", "comment_text"]]
    y = all_data[["toxic", "severe_toxic", "obscene",
                        "threat", "insult", "identity_hate"]]

    if train_data_perc == 1:
        return X, y
    np.random.seed(seed)
    msk = np.random.rand(len(X)) < train_data_perc
    X_train = X[msk].reset_index(drop=True)
    X_test = X[~msk].reset_index(drop=True)
    y_train = y[msk].reset_index(drop=True)
    y_test = y[~msk].reset_index(drop=True)

    return X_train, y_train, X_test, y_test

def read_test_data():
    test_data_file = DATA_FOLDER + "/" + "test.csv"

    all_data = pd.read_csv(test_data_file)
    X = all_data[["id", "comment_text"]]

    return X