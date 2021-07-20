import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# No srcip, dstip, sport, dport, labels are "label"

exclude_cols = [
    "id",  # Have Pandas manage IDs
    "attack_cat",  # Only using binary labels for now
    "rate",  # Not documented?
]

catcols = [
    "proto",
    "service",
    "state",
    "swin",
    "dwin",
    "is_ftp_login",
    "is_sm_ips_ports",
]

numcols = [
    "dur",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "sttl",
    "dttl",
    "sload",
    "dload",
    "sloss",
    "dloss",
    "sinpkt",
    "dinpkt",
    "sjit",
    "djit",
    "tcprtt",
    "synack",
    "ackdat",
    "smean",
    "dmean",
    "stcpb",
    "dtcpb",
    "trans_depth",
    "response_body_len",
    "ct_state_ttl",
    "ct_ftp_cmd",
    "ct_flw_http_mthd",
]

ratecols = [
    "ct_srv_src",
    "ct_dst_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "ct_src_ltm",
    "ct_srv_dst",
]


def fit_encoders(dataset):
    ohe = OneHotEncoder(handle_unknown='ignore')
    ohe.fit(dataset[catcols].values)
    scaler = StandardScaler()
    scaler.fit(dataset[numcols].values)
    return ohe, scaler


def transform(dataset, ohe, scaler):
    a1 = scaler.transform(dataset[numcols])
    a2 = ohe.transform(dataset[catcols]).toarray()
    a3 = dataset[ratecols].values / 100  # Ratios are in "count per 100" format
    labels = (dataset["label"].values == 1).astype(int)
    return np.append(np.append(a1, a2, axis=1), a3, axis=1), labels


def read_files():
    # Naming error in the downloaded folder: According to the website, the train set is supposed to have
    # 175,341 records, but the file 'UNSW_NB15_training-set.csv' provided for download has 82,332 records.
    train_dataset = pd.read_csv('data/unsw-nb15/UNSW_NB15_testing-set.csv')
    train_dataset.drop(exclude_cols, axis=1, inplace=True)
    test_dataset = pd.read_csv('data/unsw-nb15/UNSW_NB15_training-set.csv')
    test_dataset.drop(exclude_cols, axis=1, inplace=True)
    return train_dataset, test_dataset


def get_coherent_split():
    train_dataset, test_dataset = read_files()
    logging.debug(train_dataset.head())
    ohe, scaler = fit_encoders(train_dataset)
    x_train, y_train = transform(train_dataset, ohe, scaler)
    x_test, y_test = transform(test_dataset, ohe, scaler)
    return x_train, y_train, x_test, y_test


def get_random_split():
    train_dataset_coherent, test_dataset_coherent = read_files()

    train_dataset_anomaly_count = len(train_dataset_coherent[train_dataset_coherent['label'] == 1])
    train_dataset_benign_count = len(train_dataset_coherent) - train_dataset_anomaly_count
    logging.debug("UNSW train set anomaly count : " + str(train_dataset_anomaly_count) +
                  ", benign count : " + str(train_dataset_benign_count))

    test_dataset_anomaly_count = len(test_dataset_coherent[test_dataset_coherent['label'] == 1])
    test_dataset_benign_count = len(test_dataset_coherent) - test_dataset_anomaly_count
    logging.debug("UNSW test set anomaly count : " + str(test_dataset_anomaly_count) +
                  ", benign count : " + str(test_dataset_benign_count))

    while True:  # Try to generate splits until valid (should not be necessary with OHE handle_unknown='ignore')
        try:
            # Combine datasets, build new index
            combined_dataset = pd.concat([train_dataset_coherent, test_dataset_coherent], ignore_index=True)
            combined_dataset_benign = combined_dataset[combined_dataset['label'] == 0]
            combined_dataset_anomaly = combined_dataset[combined_dataset['label'] == 1]

            # Randomly sample 82332 test samples, rest are train samples (should amount to 175341)
            # Keep anomaly/benign ratio from original split
            test_dataset_benign = combined_dataset_benign.sample(n=test_dataset_benign_count)
            train_dataset_benign = combined_dataset_benign.drop(test_dataset_benign.index)
            test_dataset_anomaly = combined_dataset_anomaly.sample(n=test_dataset_anomaly_count)
            train_dataset_anomaly = combined_dataset_anomaly.drop(test_dataset_anomaly.index)
            train_dataset = pd.concat([train_dataset_benign, train_dataset_anomaly])
            test_dataset = pd.concat([test_dataset_benign, test_dataset_anomaly])
            logging.debug("(Shape UNSW-NB15 Random Split) Train : " + str(train_dataset.shape) +
                          ", Test : " + str(test_dataset.shape))
            logging.debug(train_dataset.head())
            ohe, scaler = fit_encoders(train_dataset)
            x_train, y_train = transform(train_dataset, ohe, scaler)
            x_test, y_test = transform(test_dataset, ohe, scaler)
            return x_train, y_train, x_test, y_test
        except ValueError:
            logging.info("Discarding split, retrying...")
