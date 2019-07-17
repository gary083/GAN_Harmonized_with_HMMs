import os, sys
import numpy as np
import _pickle as cPickle

def nor(train, test):
    train_all = np.concatenate(train, axis = 0)
    mean = np.mean(train_all, axis = 0)
    std = np.var(train_all, axis = 0) ** 0.5
    # print(mean.shape)
    # print(std.shape)
    for i, x in enumerate(train):
        x = (x - mean) / std
        train[i] = x

    for i, x in enumerate(test):
        x = (x - mean) / std
        test[i] = x

    return train, test, mean, std

def nor_save(train_path, test_path, train_nor_path, test_nor_path, meta_path, task):
    # print('normalizing', task)
    train = cPickle.load(open(train_path, 'rb'))
    test = cPickle.load(open(test_path, 'rb'))
    meta = cPickle.load(open(meta_path, 'rb'))

    train_nor, test_nor, mean, std = nor(train, test)
    meta[task] = {'mean':mean, 'std':std}

    cPickle.dump(train_nor, open(train_nor_path, 'wb'))
    cPickle.dump(test_nor, open(test_nor_path, 'wb'))
    cPickle.dump(meta, open(meta_path, 'wb'))

if __name__ == '__main__':
    root = sys.argv[1]
    all_task = ['mfcc', 'fbank', 'spec']
    for task in all_task:
        meta_path = os.path.join(root, 'timit-train-meta.pkl')
        train_path = os.path.join(root, 'timit-train-'+task+'.pkl')
        test_path = os.path.join(root, 'timit-test-'+task+'.pkl')
        train_nor_path = os.path.join(root, 'timit-train-'+task+'-nor.pkl')
        test_nor_path = os.path.join(root, 'timit-test-'+task+'-nor.pkl')
        nor_save(train_path, test_path, train_nor_path, test_nor_path, meta_path, task)
