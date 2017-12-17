# -*- coding: utf-8 -*-
from __future__ import print_function
from lang_tagger_crnn import LanguageTaggerCRNN
from data_prepare import load_data_from_one_dir
import numpy as np
import argparse

def proc_args():
    parser = argparse.ArgumentParser(description='command line for run_train')
    parser.add_argument('-b', '--batch_size', default=1, type=int, help='batch_size')
    parser.add_argument('-w', '--weights', default='model.weights.h5', type=str, help='load weights filename')
    parser.add_argument('-te', '--test_en_path', required=True, type=str, help='english test file directory')
    parser.add_argument('-tc', '--test_cn_path', required=True, type=str, help='chinese test file directory')
    args = parser.parse_args()
    return args

def count_acc(ground_truth, preds, name='test'):
    #print(ground_truth)
    pp = np.asarray([[1,0] if p[0]>p[1] else [0,1] for p in preds])
    #print(pp)

    #print(ground_truth.shape,preds.shape)
    errs = np.sum(np.abs(pp-ground_truth))
    print('samples:',ground_truth.shape[0], 'accury:', 1- errs/ground_truth.shape[0])



if __name__ == '__main__':

    args = proc_args()

    model = LanguageTaggerCRNN(None)
    model.summary()

    if args.weights !='':
        model.load_weights(args.weights)

    test_ex, test_ey = load_data_from_one_dir(args.test_en_path, tag=[1,0], max_file=200)
    test_ex = test_ex.transpose((0,2,3,1))

    test_cx, test_cy = load_data_from_one_dir(args.test_cn_path, tag=[0, 1], max_file=200)
    test_cx = test_cx.transpose((0, 2, 3, 1))

    pred_ey = model.predict(x=test_ex, batch_size=args.batch_size)
    count_acc(test_ey, pred_ey, 'english test')

    pred_cy = model.predict(x=test_cx, batch_size=args.batch_size)
    count_acc(test_cy, pred_cy, 'chinese test')


