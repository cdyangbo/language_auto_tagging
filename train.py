# -*- coding: utf-8 -*-
from lang_tagger_crnn import LanguageTaggerCRNN
from data_prepare import  load_h5_dataset
import argparse

def proc_args():
    parser = argparse.ArgumentParser(description='command line for run_train')
    parser.add_argument('-b', '--batch_size', default=32, type=int, help='batch_size')
    parser.add_argument('-e', '--epochs', default=10, type=int, help='epochs')
    parser.add_argument('-i', '--initial_epoch', default=0, type=int, help='initial_epoch')
    parser.add_argument('-r', '--learning_rate', default=1e-4, type=float, help='learning_rate')
    parser.add_argument('-w', '--weights', default='', type=str, help='save and load weights filename')
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = proc_args()

    model = LanguageTaggerCRNN(None)
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    if args.weights !='':
        model.load_weights(args.weights)

    train_x , train_y, dev_x,dev_y = load_h5_dataset('ch_en_speech_dataset_5000.h5')
    model.fit(x=train_x, y=train_y, batch_size=args.batch_size, epochs=args.epochs)

    print(model.evaluate(x=dev_x,y=dev_y))

    model.save_weights(args.weights if args.weights !='' else 'model.weights.h5')
