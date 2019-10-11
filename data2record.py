import os
import random
import sys
import tensorflow as tf
import csv
import math
import xlrd
import numpy as np
from scipy import stats
import scipy.ndimage as ndi
import scipy.misc as sc
import weighted_gaussian_kde as wg
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


def xls2img_sample(file, weighted=False, res=[100, 100], limit=[3000, 16]):
    workbook = xlrd.open_workbook(file)

    booksheet = workbook.sheet_by_index(0)

    N = booksheet.nrows - 1

    coord = np.zeros([N, 2])
    I = np.zeros(N)

    for i in range(N):
        coord[i, 0] = booksheet.cell_value(i + 1, 0)
        coord[i, 1] = booksheet.cell_value(i + 1, 1)
        I[i] = booksheet.cell_value(i + 1, 2)

    if weighted:
        image, _ = np.histogramdd(coord, bins=tuple(res), range=((0, limit[0]), (0, limit[1])), weights=np.log2(I))
    else:
        image, _ = np.histogramdd(coord, bins=tuple(res), range=((0, limit[0]), (0, limit[1])))
    return image


def xls2img_gaussian(file, weighted=False, res=[100, 100], limit=[3000, 16], kernel=False, sigma=None):
    if not sigma:
        sigma = (res[0] / 100, res[1] / 100)
    if kernel:
        workbook = xlrd.open_workbook(file)
        booksheet = workbook.sheet_by_index(0)
        N = booksheet.nrows - 1

        coord = np.zeros([2, N])
        I = np.zeros(N)

        for i in range(N):
            coord[0, i] = booksheet.cell_value(i + 1, 0)
            coord[1, i] = booksheet.cell_value(i + 1, 1)
            I[i] = booksheet.cell_value(i + 1, 2)

        if weighted:
            kernel = wg.weighted_gaussian_kde(coord, weights=np.power(I, 1 / 3))
        else:
            kernel = stats.gaussian_kde(coord)

        global pos
        image = np.reshape(kernel(pos), [res, res])
    else:
        sampled = xls2img_sample(file, weighted=weighted, res=res, limit=limit)
        image = ndi.gaussian_filter(sampled, sigma)
    return image


def convert_and_split_xls(root, out_path, test_percent=0.2, mode='sampled', res=[3000, 96], limit=[3000, 16],
                          weighted=True):
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    index_file_train = open(os.path.join(out_path, "index_train.csv"), "a", newline="")
    index_file_test = open(os.path.join(out_path, "index_test.csv"), "a", newline="")
    train_csv_writer = csv.writer(index_file_train, quoting=csv.QUOTE_NONNUMERIC)
    test_csv_writer = csv.writer(index_file_test, quoting=csv.QUOTE_NONNUMERIC)
    train_data_writer = tf.python_io.TFRecordWriter(os.path.join(out_path, "bio_data_train.tfrecords"))
    test_data_writer = tf.python_io.TFRecordWriter(os.path.join(out_path, "bio_data_test.tfrecords"))
    i = 0
    j = 0

    for _, classes, _ in os.walk(root):
        for index, name in enumerate(classes):
            class_path = os.path.join(root, name)
            N = len(os.listdir(class_path))
            random.seed(0)
            test_index = random.sample(range(N), math.floor(N * test_percent))
            for in_class_index, xls_name in enumerate(os.listdir(class_path)):
                if mode == "gaussian_kernel":
                    data = xls2img_gaussian(os.path.join(class_path, xls_name), res=res, limit=limit, weighted=weighted,
                                            kernel=True)
                elif mode == "gaussian_sampled":
                    data = xls2img_gaussian(os.path.join(class_path, xls_name), res=res, limit=limit, weighted=weighted,
                                            kernel=False)
                elif mode == "sampled":
                    data = xls2img_sample(os.path.join(class_path, xls_name), res=res, limit=limit, weighted=weighted)
                else:
                    print("Unexpected mode. Choose from gaussian_kernel, gaussian_sampled and sampled.")
                    sys.exit()
                fname, ext = os.path.splitext(xls_name)
                img_path = os.path.join(class_path, xls_name)
                if not os.path.exists(os.path.join(out_path, name)):
                    os.mkdir(os.path.join(out_path, name))
                data = data.reshape([data.size])
                example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    "img_raw": tf.train.Feature(float_list=tf.train.FloatList(value=data))
                }))
                if in_class_index not in test_index:
                    train_data_writer.write(example.SerializeToString())
                    train_csv_writer.writerow([i, img_path])
                    i += 1
                else:
                    test_data_writer.write(example.SerializeToString())
                    test_csv_writer.writerow([j, img_path])
                    j += 1
            print("Finish files in " + class_path)
        index_file_test.close()
        index_file_train.close()
        break


def convert_and_corss_split_xls(root, out_path, fold=10, mode='sampled', res=[3000, 96], limit=[3000, 16],
                                weighted=True):
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    for f in range(fold):
        index_file_train = open(os.path.join(out_path, "index_train_{}.csv".format(f)), "a", newline="")
        index_file_test = open(os.path.join(out_path, "index_test_{}.csv".format(f)), "a", newline="")
        train_csv_writer = csv.writer(index_file_train, quoting=csv.QUOTE_NONNUMERIC)
        test_csv_writer = csv.writer(index_file_test, quoting=csv.QUOTE_NONNUMERIC)
        train_data_writer = tf.python_io.TFRecordWriter(os.path.join(out_path, "bio_data_train_{}.tfrecords".format(f)))
        test_data_writer = tf.python_io.TFRecordWriter(os.path.join(out_path, "bio_data_test_{}.tfrecords".format(f)))
        i = 0
        j = 0

        for _, classes, _ in os.walk(root):
            for index, name in enumerate(classes):
                class_path = os.path.join(root, name)
                spliter = KFold(fold, random_state=100, shuffle=True)
                namelist = os.listdir(class_path)
                f_iter = 0
                for _, test_split in spliter.split(namelist):
                    if f_iter == f:
                        test_index = test_split
                        break
                    f_iter += 1
                for in_class_index, xls_name in enumerate(os.listdir(class_path)):
                    if mode == "gaussian_kernel":
                        data = xls2img_gaussian(os.path.join(class_path, xls_name), res=res, limit=limit,
                                                weighted=weighted, kernel=True)
                    elif mode == "gaussian_sampled":
                        data = xls2img_gaussian(os.path.join(class_path, xls_name), res=res, limit=limit,
                                                weighted=weighted, kernel=False)
                    elif mode == "sampled":
                        data = xls2img_sample(os.path.join(class_path, xls_name), res=res, limit=limit,
                                              weighted=weighted)
                    else:
                        print("Unexpected mode. Choose from gaussian_kernel, gaussian_sampled and sampled.")
                        sys.exit()
                    fname, ext = os.path.splitext(xls_name)
                    img_path = os.path.join(class_path, xls_name)
                    if not os.path.exists(os.path.join(out_path, name)):
                        os.mkdir(os.path.join(out_path, name))
                    plt.imsave(os.path.join(os.path.join(out_path, name), fname + ".png"), data, cmap='jet')
                    data = data.reshape([data.size])
                    example = tf.train.Example(features=tf.train.Features(feature={
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                        "img_raw": tf.train.Feature(float_list=tf.train.FloatList(value=data))
                    }))
                    if in_class_index not in test_index:
                        train_data_writer.write(example.SerializeToString())
                        train_csv_writer.writerow([i, img_path])
                        i += 1
                    else:
                        test_data_writer.write(example.SerializeToString())
                        test_csv_writer.writerow([j, img_path])
                        j += 1
                print("Finish files in " + class_path)
            train_data_writer.close()
            test_data_writer.close()
            index_file_test.close()
            index_file_train.close()
            break


if __name__ == '__main__':
    # your xls root folder (samples of the same class should be in the same folder) and output folder
    convert_and_corss_split_xls("./CNN_0517", "./data/10fold",
                                res=[512, 512], limit=[3000, 30], mode="gaussian_sampled", fold=10)
