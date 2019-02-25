# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 09:16:52 2018
func：网络训练,以及对应的模型保存
@author: kuangyongjian
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from network import Network

CKPT_DIR = '/home/dl/song/CNN-MNIST/model_mnist'
batch_size = 100


class Train(object):

    def __init__(self):
        self.net = Network()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.data = input_data.read_data_sets('MNIST_data', one_hot=True)

    def train(self):
        # 一个循环需要训练多少个batch
        n_batch = self.data.train.num_examples // batch_size

        saver = tf.train.Saver(max_to_keep=1)  # 只保留最近一次的模型
        # 训练模型
        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())  # 初始化变量
            for i in range(1001):
                batch_xs, batch_ys = self.data.train.next_batch(batch_size)
                sess.run(self.net.train_step, feed_dict={self.net.x: batch_xs, self.net.y: batch_ys, self.net.keep_prob: 0.5})
                if i % 100 == 0:
                    saver.save(sess, CKPT_DIR + '/model', global_step=i)
                    test_acc = sess.run(self.net.accuracy,
                                        feed_dict={self.net.x: self.data.test.images, self.net.y: self.data.test.labels, self.net.keep_prob: 1.0})
                    print("当前训练到{}batch,准确率为{}".format(i, test_acc))


if __name__ == '__main__':
    model = Train()
    model.train()