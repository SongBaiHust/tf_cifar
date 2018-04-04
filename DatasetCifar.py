import numpy as np
import pickle
import os
import math


class DatasetCifar:
    def __init__(self, cifar_folder='/data3/xchen/workspace/datasets/cifar-10-batches-py', one_hot=True,  class_number=10, batch_size=64):
        self.cifar_folder = cifar_folder
        self.one_hot = one_hot
        self.class_number = class_number #10 for cifar-10, 100 for cifar 100
        self.batch_size = batch_size
        self.train_batch_index = 0
        self.test_batch_index = 0
        self.test_image_label = None
        self.train_image_label = None
    def unpickle(self, filename):
        with open(filename, 'rb') as f:
            dict = pickle.load(f, encoding='bytes')
        return dict

    def decode_cifar(self, input_data):
        images = list();
        labels = list();
        if self.one_hot:
            for im, lb in input_data:
                images.append(np.reshape(np.reshape(im, [3, 1024]).T ,[32, 32, 3]))
                onehot = np.zeros(self.class_number).astype(np.float32)
                onehot[int(lb)] = 1
                labels.append(onehot)
        else:
            for im, lb in input_data:
                images.append(np.reshape(np.reshape(im, [3, 1024]).T ,[32, 32, 3]))
                labels.append(int(lb))
        return images, labels

    def read_train(self):
        assert (self.class_number == 10 or self.class_number == 100), 'class_number must be 10 or 100'
        images = list();
        labels = list();
        if self.class_number == 10:
            for i in range(5):
                filename = os.path.join(self.cifar_folder, 'data_batch_%d' % (i + 1))
                dict = self.unpickle(filename)
                raw_data_label = zip(dict[b'data'], dict[b'labels'])
#                np.random.shuffle(raw_data_label)
                ims, lbs = self.decode_cifar(raw_data_label)
                images.extend(ims)
                labels.extend(lbs)
        elif self.class_number == 100:
            filename = os.path.join(self.cifar_folder, 'train')
            dict = self.unpickle(filename)
            raw_data_label = zip(dict[b'data'], dict[b'fine_labels'])
#            np.random.shuffle(raw_data_label)
            ims, lbs = self.decode_cifar(raw_data_label)
            images.extend(ims)
            labels.extend(lbs)
        else:
            print('Not cifar-10 or cifar-100.')
            return None
        return images, labels
    def read_test(self):
        assert (self.class_number == 10 or self.class_number == 100), 'class_number must be 10 or 100'
        if self.class_number == 10:
            filename = os.path.join(self.cifar_folder, 'test_batch')
            dict = self.unpickle(filename)
            raw_data_label = zip(dict[b'data'],  dict[b'labels'])
            images, labels = self.decode_cifar(raw_data_label)
        elif self.class_number == 100:
            filename = os.path.join(self.cifar_folder, 'test')
            dict = self.unpickle(filename)
            raw_data_label = zip(dict[b'data'], dictp[b'fine_labels'])
            images, labels = self.decode_cifar(raw_data_label)
        else:
            print('Not cifar-10 or cifar-100.')
            return None
        return images, labels
    def next_train_batch(self):
        if self.train_image_label is None:
            images, labels = self.read_train()
            self.train_image_label = zip(images, labels)
        num_batches = math.floor(50000/self.batch_size)
        if self.train_batch_index < num_batches:
            next_images ,next_labels = \
                self.train_image_label[self.train_batch_index*self.batch_size:(self.train_batch_index + 1)*self.batch_size]
            self.train_batch_index = self.train_batch_index + 1
        else:
            self.train_batch_index = 0
            next_images ,next_labels = \
                self.train_image_label[self.train_batch_index*self.batch_size:(self.train_batch_index + 1)*self.batch_size]
            self.train_batch_index = self.train_batch_index + 1
        return next_images, next_labels

    def next_test_batch(self):
        if self.test_image_label is None:
            images, labels = self.read_test()
            self.test_image_label = zip(images, labels)
        num_batches = math.floor(10000/self.batch_size)
        if self.test_batch_index < num_batches:
            next_images, next_labels = \
                self.test_image_label[self.test_batch_index*self.batch_size:(self.test_batch_index + 1)*self.batch_size]
            self.test_batch_index = self.test_batch_index + 1
        else:
            print('One epoch has been processed')
            raise OutOfRangeError('Max batch number reached')
        return next_images, next_labels
