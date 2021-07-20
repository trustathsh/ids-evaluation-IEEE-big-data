import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


files_by_day = {
    'mon': ['data/cicids2017/Monday-WorkingHours.pcap_ISCX.csv'],
    'tue': ['data/cicids2017/Tuesday-WorkingHours.pcap_ISCX.csv'],
    'wen': ['data/cicids2017/Wednesday-workingHours.pcap_ISCX.csv'],
    'thu': ['data/cicids2017/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
            'data/cicids2017/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv'],
    'fri': ['data/cicids2017/Friday-WorkingHours-Morning.pcap_ISCX.csv',
            'data/cicids2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
            'data/cicids2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv'],
}

# Configuration for generating the split for this dataset
train_files = [files_by_day['fri'][1]]
train_do_clean = True  # Turn any infinite entries into NaN, drop all rows with NaN entries
train_drop_percentage = 0.7  # Drop percentage of rows (after clean)
train_drop_benign_only = False  # If true, only benign traffic rows will be considered for dropping

test_files = [files_by_day['fri'][2]]
test_do_clean = True
test_drop_percentage = 0.8
test_drop_benign_only = True

random_seed = 1337  # Set fixed random seed to make subsampling repeatable


catcols = [' Destination Port',  'Fwd PSH Flags', ' Bwd PSH Flags', ' Fwd URG Flags', ' Bwd URG Flags']

ratecols = [' Down/Up Ratio']

numcols = [' Flow Duration', ' Total Fwd Packets', ' Total Backward Packets',
           'Total Length of Fwd Packets', ' Total Length of Bwd Packets', ' Fwd Packet Length Max',
           ' Fwd Packet Length Min', ' Fwd Packet Length Mean', ' Fwd Packet Length Std', 'Bwd Packet Length Max',
           ' Bwd Packet Length Min', ' Bwd Packet Length Mean', ' Bwd Packet Length Std', 'Flow Bytes/s',
           ' Flow Packets/s', ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min', 'Fwd IAT Total',
           ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max', ' Fwd IAT Min', 'Bwd IAT Total', ' Bwd IAT Mean',
           ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min', ' Fwd Header Length', ' Bwd Header Length', 'Fwd Packets/s',
           ' Bwd Packets/s', ' Min Packet Length', ' Max Packet Length', ' Packet Length Mean', ' Packet Length Std',
           ' Packet Length Variance', 'FIN Flag Count', ' SYN Flag Count', ' RST Flag Count', ' PSH Flag Count',
           ' ACK Flag Count', ' URG Flag Count', ' CWE Flag Count', ' ECE Flag Count',
           ' Average Packet Size', ' Avg Fwd Segment Size', ' Avg Bwd Segment Size', ' Fwd Header Length.1',
           'Fwd Avg Bytes/Bulk', ' Fwd Avg Packets/Bulk', ' Fwd Avg Bulk Rate', ' Bwd Avg Bytes/Bulk',
           ' Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', ' Subflow Fwd Bytes',
           ' Subflow Bwd Packets', ' Subflow Bwd Bytes', 'Init_Win_bytes_forward', ' Init_Win_bytes_backward',
           ' act_data_pkt_fwd', ' min_seg_size_forward', 'Active Mean', ' Active Std', ' Active Max', ' Active Min',
           'Idle Mean', ' Idle Std', ' Idle Max', ' Idle Min']


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
    labels = (dataset[' Label'].values != 'BENIGN').astype(int)
    return np.append(np.append(a1, a2, axis=1), a3, axis=1), labels


def read_files():
    np.random.seed(random_seed)

    if len(train_files) > 1:
        _train_list = []
        for filename in train_files:
            df = pd.read_csv(filename, index_col=None)
            _train_list.append(df)
        train_dataset = pd.concat(_train_list, axis=0, ignore_index=True)
    else:
        train_dataset = pd.read_csv(train_files[0])

    if train_do_clean:
        train_dataset = train_dataset.replace([np.inf, -np.inf], np.nan)
        train_dataset = train_dataset.dropna()
    benign_count = len(train_dataset[train_dataset[' Label'] == 'BENIGN'])
    logging.debug('Train dataset ...')
    logging.debug('Sample count: '+str(len(train_dataset)))
    logging.debug('Benign count: '+str(benign_count))
    logging.debug('Benign ratio: '+str(benign_count / len(train_dataset)))

    if train_drop_percentage > 0:
        if train_drop_benign_only:
            drop_indices = np.random.choice(train_dataset[train_dataset[' Label'] == 'BENIGN'].index,
                                            int(len(train_dataset) * train_drop_percentage),
                                            replace=False)
        else:
            drop_indices = np.random.choice(train_dataset.index,
                                            int(len(train_dataset) * train_drop_percentage),
                                            replace=False)
        train_dataset = train_dataset.drop(drop_indices)

        benign_count = len(train_dataset[train_dataset[' Label'] == 'BENIGN'])
        logging.debug('After dropping ...')
        logging.debug('Sample count: '+str(len(train_dataset)))
        logging.debug('Benign count: '+str(benign_count))
        logging.debug('Benign ratio: '+str(benign_count / len(train_dataset)))

    if len(test_files) > 1:
        _test_list = []
        for filename in test_files:
            df = pd.read_csv(filename, index_col=None)
            _test_list.append(df)
        test_dataset = pd.concat(_test_list, axis=0, ignore_index=True)
    else:
        test_dataset = pd.read_csv(test_files[0])

    if test_do_clean:
        test_dataset = test_dataset.replace([np.inf, -np.inf], np.nan)
        test_dataset = test_dataset.dropna()
    benign_count = len(test_dataset[test_dataset[' Label'] == 'BENIGN'])
    logging.debug('Test dataset ...')
    logging.debug('Sample count: '+str(len(test_dataset)))
    logging.debug('Benign count: '+str(benign_count))
    logging.debug('Benign ratio: '+str(benign_count / len(test_dataset)))

    if test_drop_percentage > 0:
        if test_drop_benign_only:
            drop_indices = np.random.choice(test_dataset[test_dataset[' Label'] == 'BENIGN'].index,
                                            int(benign_count * test_drop_percentage),
                                            replace=False)
        else:
            drop_indices = np.random.choice(test_dataset.index,
                                            int(benign_count * test_drop_percentage),
                                            replace=False)
        test_dataset = test_dataset.drop(drop_indices)

        benign_count = len(test_dataset[test_dataset[' Label'] == 'BENIGN'])
        logging.debug('After dropping ...')
        logging.debug('Sample count: '+str(len(test_dataset)))
        logging.debug('Benign count: '+str(benign_count))
        logging.debug('Benign ratio: '+str(benign_count / len(test_dataset)))

    np.random.seed(None)  # Set random random seed

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

    train_dataset_anomaly_count = len(train_dataset_coherent[train_dataset_coherent[' Label'] != 'BENIGN'])
    train_dataset_benign_count = len(train_dataset_coherent) - train_dataset_anomaly_count
    logging.debug("CICIDS2017 train set anomaly count : " + str(train_dataset_anomaly_count) +
                  ", benign count : " + str(train_dataset_benign_count))

    test_dataset_anomaly_count = len(test_dataset_coherent[test_dataset_coherent[' Label'] != 'BENIGN'])
    test_dataset_benign_count = len(test_dataset_coherent) - test_dataset_anomaly_count
    logging.debug("CICIDS2017 test set anomaly count : " + str(test_dataset_anomaly_count) +
                  ", benign count : " + str(test_dataset_benign_count))

    while True:  # Try to generate splits until valid (should not be necessary with OHE handle_unknown='ignore')
        try:
            # Combine datasets, build new index
            combined_dataset = pd.concat([train_dataset_coherent, test_dataset_coherent], ignore_index=True)
            combined_dataset_benign = combined_dataset[combined_dataset[' Label'] == 'BENIGN']
            combined_dataset_anomaly = combined_dataset[combined_dataset[' Label'] != 'BENIGN']

            # Keep sample count and anomaly/benign ratio from original split
            test_dataset_benign = combined_dataset_benign.sample(n=test_dataset_benign_count)
            train_dataset_benign = combined_dataset_benign.drop(test_dataset_benign.index)
            test_dataset_anomaly = combined_dataset_anomaly.sample(n=test_dataset_anomaly_count)
            train_dataset_anomaly = combined_dataset_anomaly.drop(test_dataset_anomaly.index)
            train_dataset = pd.concat([train_dataset_benign, train_dataset_anomaly])
            test_dataset = pd.concat([test_dataset_benign, test_dataset_anomaly])
            logging.debug("(Shape CICIDS2017 Random Split) Train : " + str(train_dataset.shape) +
                          ", Test : " + str(test_dataset.shape))
            logging.debug("(Shape CICIDS2017 Coherent Split) Train : " + str(train_dataset_coherent.shape) +
                          ", Test : " + str(test_dataset_coherent.shape))
            logging.debug(train_dataset.head())
            ohe, scaler = fit_encoders(train_dataset)
            x_train, y_train = transform(train_dataset, ohe, scaler)
            x_test, y_test = transform(test_dataset, ohe, scaler)
            return x_train, y_train, x_test, y_test
        except ValueError:
            logging.info("Discarding split, retrying...")
