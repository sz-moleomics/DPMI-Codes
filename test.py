from __future__ import print_function
from keras.models import load_model
from keras.utils import Sequence

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class DataGenerator(Sequence):
    def __init__(self, record_file, sample_num, batch_size=32, dim=(32, 512, 512), n_channels=1, n_classes=10, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.record_file = record_file
        self.sample_num = sample_num
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        train_file_queue = tf.train.string_input_producer([record_file], shuffle=shuffle)
        reader = tf.TFRecordReader()
        _, serialized_ex = reader.read(train_file_queue)
        features = tf.parse_single_example(serialized_ex, features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenSequenceFeature([], tf.float32, True),
        })
        image = features['img_raw']
        image = tf.reshape(image, [img_rows, img_cols, img_channels])
        label = tf.cast(features['label'], tf.int32)
        label = tf.reshape(label, [1])
        example_queue = tf.FIFOQueue(
            capacity=8 * batch_size,
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
        return image, label

    def get_batch(self):
        image, label = self.sess.run([self.images, self.labels])
        return image, label


if __name__ == '__main__':
    test_num = [193, 191, 188, 188, 187, 186, 185, 184, 184, 184]  # number of test samples for each class
    img_rows, img_cols, img_channels = 512, 512, 1  # image size
    nb_classes = 10  # number of class

    loss = np.zeros([nb_classes, nb_classes])
    p_error = loss*1
    num = loss*1
    k = 0
    statistic = np.zeros([np.sum(test_num), 12])
    for f in range(10):
        model_file = './model/{}.h5'.format(f)  # model file path
        test_file = './data/bio_data_test_{}.tfrecords'.format(f)  # test data path

        test_generator = DataGenerator(test_file, test_num[f], batch_size=1, dim=(1, 512, 512), shuffle=False)
        model = load_model(model_file)

        for i in range(test_num[f]):
            img, ground_truth = test_generator.get_batch()
            ground_truth = ground_truth[0]
            label = np.argmax(ground_truth)
            predicted = model.predict(img)[0]
            statistic[k, 0] = label
            statistic[k, 1] = np.argmax(predicted)
            statistic[k, 2::] = predicted
            loss[label, :] += -ground_truth*np.log2(predicted+1e-100)-(1-ground_truth)*np.log2(1-predicted+1e-100)
            p_error[label, :] = np.abs(predicted - ground_truth)
            num[label, np.argmax(predicted)] += 1
            if label != np.argmax(predicted):
                plt.imsave('./wrong/split{}_{}.png'.format(f, i), img[0, :, :, 0], cmap='jet')
            k += 1

    print('Probability Error:')
    print(p_error/test_num)
    print('Sum Loss:')
    print(np.sum(loss))
    print('Confusion:')
    print(num)
    np.savetxt('./result.csv', statistic, delimiter=',')
