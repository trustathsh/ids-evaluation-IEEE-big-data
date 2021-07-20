import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.io import arff


catcols = ['protocol_type', 'service', 'flag']

numcols = ['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
           'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
           'num_shells', 'num_access_files',  # 'num_outbound_cmds',
           'count', 'srv_count', 'dst_host_count', 'dst_host_srv_count',
           'land', 'logged_in', 'is_hot_login', 'is_guest_login']

ratecols = ['serror_rate',
            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_srv_error_rate', 'dst_host_rerror_rate']

exclude_cols = ['connection_number', 'start_time', 'orig_port', 'resp_port', 'orig_ip', 'resp_ip', 'num_outbound_cmds']


def fit_encoders(dataset):
    ohe = OneHotEncoder(handle_unknown='ignore')
    ohe.fit(dataset[catcols].values)
    scaler = StandardScaler()
    scaler.fit(dataset[numcols].values)
    return ohe, scaler


def transform(dataset, ohe, scaler):
    a1 = scaler.transform(dataset[numcols])
    a2 = ohe.transform(dataset[catcols]).toarray()
    a3 = dataset[ratecols].values
    labels = (dataset["class"].values != b'normal').astype(int)
    return np.append(np.append(a1, a2, axis=1), a3, axis=1), labels


def read_files():  # Sorts samples by timestamp, first 72% are train set, last 28% are test set
    data = arff.loadarff('data/gure-kdd-6-percent/gureKddcup6percent.arff')
    data = pd.DataFrame(data[0])
    data.sort_values(by='start_time', inplace=True)
    data.drop(exclude_cols, axis=1, inplace=True)
    num_samples = len(data)
    num_test_samples = int(num_samples * 0.28)
    num_train_samples = num_samples - num_test_samples
    train_dataset = data.head(num_train_samples)
    test_dataset = data.tail(num_test_samples)
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

    train_dataset_anomaly_count = len(train_dataset_coherent[train_dataset_coherent['class'] != b'normal'])
    train_dataset_benign_count = len(train_dataset_coherent) - train_dataset_anomaly_count
    logging.debug("Gure KDD train set anomaly count : " + str(train_dataset_anomaly_count) +
                  ", benign count : " + str(train_dataset_benign_count))

    test_dataset_anomaly_count = len(test_dataset_coherent[test_dataset_coherent['class'] != b'normal'])
    test_dataset_benign_count = len(test_dataset_coherent) - test_dataset_anomaly_count
    logging.debug("Gure KDD test set anomaly count : " + str(test_dataset_anomaly_count) +
                  ", benign count : " + str(test_dataset_benign_count))

    while True:  # Try to generate splits until valid (should not be necessary with OHE handle_unknown='ignore')
        try:
            # Combine datasets, build new index
            combined_dataset = pd.concat([train_dataset_coherent, test_dataset_coherent], ignore_index=True)
            combined_dataset_benign = combined_dataset[combined_dataset['class'] == b'normal']
            combined_dataset_anomaly = combined_dataset[combined_dataset['class'] != b'normal']

            # Keep sample count and anomaly/benign ratio from original split
            test_dataset_benign = combined_dataset_benign.sample(n=test_dataset_benign_count)
            train_dataset_benign = combined_dataset_benign.drop(test_dataset_benign.index)
            test_dataset_anomaly = combined_dataset_anomaly.sample(n=test_dataset_anomaly_count)
            train_dataset_anomaly = combined_dataset_anomaly.drop(test_dataset_anomaly.index)
            train_dataset = pd.concat([train_dataset_benign, train_dataset_anomaly])
            test_dataset = pd.concat([test_dataset_benign, test_dataset_anomaly])
            logging.debug("(Shape Gure KDD Random Split) Train : " + str(train_dataset.shape) +
                          ", Test : " + str(test_dataset.shape))
            logging.debug("(Shape Gure KDD Coherent Split) Train : " + str(train_dataset_coherent.shape) +
                          ", Test : " + str(test_dataset_coherent.shape))
            logging.debug(train_dataset.head())
            ohe, scaler = fit_encoders(train_dataset)
            x_train, y_train = transform(train_dataset, ohe, scaler)
            x_test, y_test = transform(test_dataset, ohe, scaler)
            return x_train, y_train, x_test, y_test
        except ValueError:
            logging.info("Discarding split, retrying...")
