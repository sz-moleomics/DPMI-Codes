import tensorflow as tf
from keras.models import load_model
import numpy as np
import shap
import matplotlib.pyplot as plt
from shap.plots import colors
from scipy.ndimage import gaussian_filter
from shap.plots import colors

img_rows, img_cols, img_channels = 512, 512, 1  # image size
nb_classes = 10  # number of classes
f = 10  # fold number
k = 1  # fold index
test_num = np.array([193, 191, 188, 188, 187, 186, 185, 184, 184, 184])  # number of test samples for each class
total = 1870  # total number of samples
np.set_printoptions(formatter={'float': '{:0.2f}'.format})



class DataGenerator():
    def __init__(self, record_file, sample_num, batch_size=1, dim=(1, 512, 512), n_channels=1, n_classes=10, shuffle=False):
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
        # image = tf.transpose(image, [0, 3, 1, 2])
        return image, label

    def get_batch(self):
        image, label = self.sess.run([self.images, self.labels])
        # image = tf.transpose(image, [0, 3, 1, 2])
        return image, label


def get_single(train_file, n_sample, n_class=10, n_ch=1, dim=(512, 512)):
    generator = DataGenerator(train_file, n_sample, 1, (1, dim[0], dim[1]), n_ch, n_class)
    img, lab = generator.get_batch()
    yield img, [np.argmax(lab, axis=1)]


def get_sample(train_file, n_sample, num_per_class=10, n_class=10, n_ch=1, dim=(512, 512)):
    counter = np.zeros(n_class)
    bkg = np.array([])
    lab = []
    generator = DataGenerator(train_file, n_sample, 1, (1, dim[0], dim[1]), n_ch, n_class)
    for i in range(n_sample):
        images, labels = generator.get_batch()
        labels = np.argmax(labels, axis=1)
        for j in range(1):
            if counter[labels[j]] < num_per_class:
                counter[labels[j]] += 1
                bkg = np.vstack(
                    [bkg, images[j, :, :, :].reshape([1, 512, 512, 1])]) if bkg.size else images[j, :, :, :]\
                    .reshape([1, 512, 512, 1])
                lab.append(labels[j])
    return bkg, np.array(lab)


def explain(train_data, test_data, model):
    e = shap.DeepExplainer(model, train_data)
    return np.array(e.shap_values(test_data))


if __name__ == '__main__':
    train_num = total - test_num
    div = 10
    importance = np.zeros([1, img_rows, img_cols])
    bkg, lab = get_sample('./data/{}fold/bio_data_train_{}.tfrecords'.format(f, k),  # train data path
                          train_num[k], 10)
    tar, _ = get_single('./data/{}fold/bio_data_test_{}.tfrecords'.format(f, k),  # test data path
                        test_num[k], 2).__next__()
    model = load_model('./model/{}.h5'.format(k))  # model path

    for i in range(div):
        print('Starting part {}/{}'.format(i, 10))
        importance += np.mean(explain(bkg[i*10:(i+1)*10, :, :, :], tar, model)[0, :, :, :, :], axis=3)
    for j in range(1):
        # save path for shap ditribution of the first test sample
        np.savetxt('./importance_test_1_{}_c1_noabs.csv'.format(j), importance[j, :, :], delimiter=',')
