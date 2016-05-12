from utils import *
from sklearn.cross_validation import StratifiedKFold
from random import seed, shuffle


def read_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            data.append(line)
    seed(6)
    shuffle(data)
    return data


def extract_labels(data):
    labels = []
    for d in data:
        tokens = d.split(' ')
        labels.append(LABEL[tokens[0]])
    return labels


def train_test_split(labels, data):
    train, test = [], []
    n_folds = 5     # for 80-20 split
    skf = StratifiedKFold(y=labels, n_folds=n_folds, random_state=6)
    for train_idx, test_idx in skf:
        for idx in train_idx:
            train.append(data[idx])
        for idx in test_idx:
            test.append(data[idx])
        return train, test


def write_file(filename, data):
    with open(filename, 'w') as f:
        for line in data:
            f.write(line)


def main():
    infile = "./WordLists/training_pairs.txt"
    train_file = "./WordLists/train.txt"
    test_file = "./WordLists/test.txt"

    data = read_data(infile)
    labels = extract_labels(data)
    train, test = train_test_split(labels, data)
    write_file(train_file, train)
    write_file(test_file, test)

if __name__ == '__main__':
    main()
