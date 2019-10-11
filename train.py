from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.utils import Sequence
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, TensorBoard, ModelCheckpoint
from keras import backend as K
from keras.optimizers import Adam

import numpy as np
import resnet
import tensorflow as tf
import os
import pandas as pd


class XTensorBoard(TensorBoard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)


class DataGenerator(Sequence):
    def __init__(self, record_file, sample_num, batch_size=32, dim=(32, 512, 512),
                 n_channels=1, n_classes=10, shuffle=True, mean=None, std=None, per=False):
        self.dim = dim
        self.batch_size = batch_size
        self.record_file = record_file
        self.sample_num = sample_num
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.mean = mean
        self.std = std
        # self.std[std == 0] = 1
        train_file_queue = tf.train.string_input_producer([record_file], shuffle=True)
        reader = tf.TFRecordReader()
        _, serialized_ex = reader.read(train_file_queue)
        features = tf.parse_single_example(serialized_ex, features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenSequenceFeature([], tf.float32, True),
        })
        image = features['img_raw']
        image = tf.reshape(image, [img_rows, img_cols, img_channels])
        if per:
            image = tf.image.per_image_standardization(image)
        label = tf.cast(features['label'], tf.int32)
        label = tf.reshape(label, [1])
        example_queue = tf.RandomShuffleQueue(
            capacity=8 * batch_size,
            min_after_dequeue=4 * batch_size,
            dtypes=[tf.float32, tf.int32],
            shapes=[[img_rows, img_cols, img_channels], [1]])
        num_threads = 3
        example_enqueue_op = example_queue.enqueue([image, label])
        tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(
            example_queue, [example_enqueue_op] * num_threads))
        self.images, labels = example_queue.dequeue_many(batch_size)
        labels = tf.reshape(labels, [batch_size, 1])
        indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
        self.labels = tf.sparse_to_dense(
            tf.concat(values=[indices, labels], axis=1),
            [batch_size, nb_classes], 1.0, 0.0)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        tf.train.start_queue_runners(self.sess)

    def __len__(self):
        return int(np.floor(self.sample_num / self.batch_size))

    def __getitem__(self, index):
        X, y = self.__data_generation()
        return X, y

    def __data_generation(self):
        image, label = self.sess.run([self.images, self.labels])
        if self.mean is not None and self.std is not None:
            for i in range(image.shape[3]):
                image[:, :, :, i] = (image[:, :, :, i] - self.mean) / self.std
        # image = tf.transpose(image, [0, 3, 1, 2])
        return image, label


batch_size = 32  # batch size
nb_classes = 10  # class number
nb_epoch = 1540  # epoch number
data_augmentation = False  # use data augmentation, False only for now
img_rows, img_cols = 512, 512  # input image size
img_channels = 1  # input image channels
fold_num = 10  # fold number

if __name__ == '__main__':
    train_num = []
    test_num = []
    lr_reducer = ReduceLROnPlateau(factor=0.1, cooldown=490, patience=10, min_lr=0.00001, min_delta=1e-5)
    early_stopper = EarlyStopping(min_delta=1e-6, patience=1000)

    for f in range(fold_num):
        tmp = pd.read_csv('./data/{}fold/index_train_{}.csv'.format(fold_num, f), header=None)  # train data index save path
        test_num.append(tmp.shape[0])
        tmp = pd.read_csv('./data/{}fold/index_test_{}.csv'.format(fold_num, f), header=None)  # test data index save path
        train_num.append(tmp.shape[0])
    del tmp

    for f in range(fold_num):
        if not os.path.exists('./{}fold/split_{}'.format(fold_num, f)):  # prepare folders
            os.mkdir('./{}fold/split_{}'.format(fold_num, f))
            os.mkdir('./{}fold/split_{}/log'.format(fold_num, f))
        csv_logger = CSVLogger('./{}fold/split_{}/resnet18_bio_classified.csv'.format(fold_num, f))  # train log save path
        test_file = './data/{}fold/bio_data_train_{}.tfrecords'.format(fold_num, f)  # test data file path
        train_file = './data/{}fold/bio_data_test_{}.tfrecords'.format(fold_num, f)  # train data file path
        train_generator = DataGenerator(train_file, train_num[f], per=False)
        test_generator = DataGenerator(test_file, test_num[f], per=False)
        model = resnet.ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), nb_classes)
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=0.01),
                      metrics=['categorical_accuracy'])

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(train_generator.sess, coord)

        tb = XTensorBoard(log_dir='./{}fold/split_{}/log'.format(fold_num, f),
                          histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=False)  # tensorboard log save path
        ck = ModelCheckpoint('./{}fold/split_{}'.format(fold_num, f)+'/model_{epoch:02d}-{val_loss:.2f}.hdf5', period=120)  # checkpoint save path


        if not data_augmentation:
            print('Not using data augmentation.')
            model.fit_generator(generator=train_generator,
                                validation_data=test_generator,
                                callbacks=[lr_reducer, ck, early_stopper, csv_logger, tb],
                                epochs=nb_epoch)

            model.save('./{}fold/split_{}/bio_classified.h5'.format(fold_num, f))  # model save path
        else:
            pass
