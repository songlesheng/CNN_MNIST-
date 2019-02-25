"""
简单的CNN网络实现 mnist jpg的测试，使用测试好的模型
2019.02.25
by song
stay hungry stay foolish
"""
import tensorflow as tf
import numpy as np
from PIL import Image
from network import Network

CKPT_DIR = '/home/dl/song/CNN-MNIST/model_mnist'


class Predict(object):

    def __init__(self):
        # 清除默认图的堆栈，并设置全局图为默认图
        # 若不进行清楚则在第二次加载的时候报错，因为相当于重新加载了两次
        tf.reset_default_graph()
        self.net = Network()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # 加载模型到sess中
        self.restore()
        print('load susess')

    def restore(self):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(CKPT_DIR)
        print(ckpt.model_checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise FileNotFoundError('未保存模型')

    def predict(self, image_path):
        # 读取图片并灰度化
        img = Image.open(image_path).convert('L')
        x = np.reshape(img, [1, 784])
        y = self.sess.run(self.net.prediction, feed_dict={self.net.x: x, self.net.keep_prob: 1.0})

        print(image_path)
        print(' Predict digit', np.argmax(y[0]))  # 提取出可能性最大的值


if __name__ == '__main__':
    model = Predict()
    model.predict('/home/dl/song/CNN-MNIST/full_network/data_jpg/4.jpg')  # 单张图片路径
