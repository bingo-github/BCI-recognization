# coding: utf-8
import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split

''' 加载数据 '''
data_path = './BCI Data/BCI_data.mat'
ori_data = sio.loadmat(data_path)

channel_num = 3
cnt, pos, y = ori_data['data'], ori_data['pos'][0], ori_data['label'][0]

cnt = cnt.T


''' 按照通道进行切分 '''
data_split = []
for i in range(len(pos)):
    data_per_ex = []
    for j in range(len(cnt)):
        data_per_ex.append(cnt[j][pos[i]:pos[i] + 800])
    data_split.append(data_per_ex)
data_split = np.array(data_split)
# print('shape of data_split is', data_split.shape)
# # shape of data_split is (200, 59, 800)


''' 滤波处理 '''
from scipy.fftpack import fft
import scipy.signal as signal

# b是针对100Hz采样频率下的8~30Hz带通滤波
b = np.array(
    [-0.0002, 0.0007, -0.0006, -0.0020, -0.0005, -0.0004, -0.0028, 0.0000, 0.0041, 0.0009, 0.0017, 0.0091, 0.0034,
     -0.0056, 0.0017, -0.0017, -0.0207, -0.0118, 0.0038, - 0.0138, - 0.0064, 0.0358, 0.0240, 0.0009, 0.0467, 0.0389,
     - 0.0526, - 0.0353, 0.0035, - 0.1647, - 0.2256, 0.1488, 0.4393, 0.1488, -0.2256, - 0.1647, 0.0035, - 0.0353,
     - 0.0526, 0.0389, 0.0467, 0.0009, 0.0240, 0.0358, - 0.0064, - 0.0138, 0.0038, - 0.0118, - 0.0207, - 0.0017, 0.0017,
     -0.0056, 0.0034, 0.0091, 0.0017, 0.0009, 0.0041, 0.0000, -0.0028, -0.0004, -0.0005, -0.0020, -0.0006, 0.0007,
     -0.0002])
data_split = signal.lfilter(b=b, a=[1], x=data_split)

''' 对信号按时间截取 '''
sample_freq = 100
time_head = 4
time_tail = 7
data_split = data_split[:, :, time_head * sample_freq:time_tail * sample_freq]
# print('shape of data_split_win is', data_split.shape)
# # shape of data_split_win is (200, 59, 300)


''' 整体进行fft变换，变到频域下进行分析处理 '''
data_split = abs(fft(data_split))
# print('shape of data_split is', data_split.shape)
# # shape of data_split is (200, 59, 300)
data_split_win = data_split[:, :, :int(data_split.shape[2] / 2)]
print('shape of data_split_win is', data_split_win.shape)

''' 频域下的数据，提取5~35Hz的信号 '''
data_win_freq_split = data_split_win[:, :, 15:105]
print(data_win_freq_split.shape)

''' 处理y为one hot '''
y_one_hot = []
for i in range(len(y)):
    if y[i] == 1:
        y_one_hot.append([0, 1])
    elif y[i] == -1:
        y_one_hot.append([1, 0])
y_one_hot = np.array(y_one_hot)
# print(len(y_one_hot))
# print(y_one_hot)


''' 分割为训练集和测试集 '''
data_win_freq_split_ss = data_win_freq_split / np.max(data_win_freq_split)
# print(data_win_freq_split_ss.shape)
# print(np.max(data_win_freq_split_ss))
train_x, test_x, train_y, test_y = train_test_split(data_win_freq_split_ss, y_one_hot, test_size=0.2, random_state=1)
# print('shape of train_x is', train_x.shape, '; shape of test_x is', test_x.shape)
# print('shape of train_y is', train_y.shape, '; shape of test_y is', test_y.shape)


''' 数据标准化 '''
from sklearn.preprocessing import StandardScaler

train_x = np.float32(np.reshape(train_x, [len(train_x), -1]))
test_x = np.float32(np.reshape(test_x, [len(test_x), -1]))
# print('shape of train_x is', train_x.shape, '; shape of test_x is', test_x.shape)
# # shape of train_x is (160, 3540) ; shape of test_x is (40, 3540)
ss = StandardScaler()
train_x = ss.fit_transform(train_x)
test_x = ss.transform(test_x)

''' 扩充数据集 '''
for _ in range(2):
    for i in range(len(train_x)):
        tmp = np.random.normal(loc=0, scale=0.005, size=train_x.shape[1])
        tmp_append = train_x[i] + tmp
        train_x = np.row_stack((train_x, tmp_append))  # 添加x数据
        train_y = np.concatenate((train_y, [train_y[i]]), axis=0)
print(train_x.shape, train_y.shape)
train_x_reshape = train_x.reshape(-1, channel_num, 90, 1)
test_x_reshape = test_x.reshape(-1, channel_num, 90, 1)

''' 建立模型 '''
import tensorflow as tf

X = tf.placeholder(tf.float32, [None, channel_num, 90, 1])
Y = tf.placeholder(tf.float32, [None, 2])
p_keep_conv = tf.placeholder(tf.float32)
p_keep_hidden = tf.placeholder(tf.float32)


def init_weights(shape):
    """
    :description: 权重w初始化
    :param shape: w的大小
    :return: 权重w
    """
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01))


w = init_weights([channel_num, 1, 1, 8])
w2 = init_weights([1, 9, 8, 10])
w3 = init_weights([10 * 17, 30])
w_o = init_weights([30, 2])

''' 建立CNN模型各层 '''
# 第一组卷积层和池化层，最后dropout
l1 = tf.nn.relu(tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='VALID'))
l1 = tf.nn.dropout(l1, p_keep_conv)
# 第二组卷积层和池化层，最后dropout
l2 = tf.nn.relu(tf.nn.conv2d(l1, w2, strides=[1, 1, 5, 1], padding='VALID'))
shape_l2m = tf.shape(l2)
l2 = tf.reshape(l2, [-1, w3.get_shape().as_list()[0]])
l2 = tf.nn.dropout(l2, p_keep_conv)
# 全连接层，最后dropout
l3 = tf.nn.relu(tf.matmul(l2, w3))
l3 = tf.nn.dropout(l3, p_keep_hidden)
# 输出层
py_x = tf.matmul(l3, w_o)

''' 模型优化 '''
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)

''' 模型训练与评估 '''
# 训练参数
batch_size = 20
test_size = 10
train_epoches = 10
# 评估
correct_pred = tf.equal(tf.argmax(py_x, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for epoch in range(800):
        training_batch = zip(range(0, len(train_x_reshape), batch_size),
                             range(batch_size, len(train_x_reshape) + 1, batch_size))
        iterate_num = 0
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: train_x_reshape[start:end], Y: train_y[start:end], p_keep_conv: 0.8,
                                          p_keep_hidden: 0.5})
            # sess.run(train_op, feed_dict={X: train_x_reshape[start:end], p_keep_conv:0.8, p_keep_hidden:0.5})
            iterate_num += 1
            if iterate_num % 3 == 0:
                train_loss, train_acc = sess.run([cost, accuracy],
                                                 feed_dict={X: train_x_reshape[start:end], Y: train_y[start:end],
                                                            p_keep_conv: 0.8, p_keep_hidden: 0.5})
                print('epoch:', epoch, 'iterate:', iterate_num, 'train loss:', train_loss, '; train accuracy:',
                      train_acc)
        test_indices = np.arange(len(test_x_reshape))
        test_indices = test_indices[0:test_size]
        test_loss, test_acc = sess.run([cost, accuracy], feed_dict={X: test_x_reshape[:], Y: test_y[:],
                                                                    p_keep_conv: 1.0, p_keep_hidden: 1.0})
        print('epoch', epoch, 'test loss:', test_loss, '; test accuracy:', test_acc)
