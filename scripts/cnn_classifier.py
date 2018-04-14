import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data  # 读取数据集，本地没有缓存就下载一个


# 读取MNIST数据, 读取标量作为标签，而不one-hot形式，
# 因为tf.estimator.DNNClassifier不支持one-hot标签
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)


def cnn_model_fn(features, labels, mode):
    input_layer = features.get("input")  # 从字典中取到这次训练的图片，我们会拿到两张图，所以input_layer的形状是[2, 784]
    input_layer = tf.reshape(input_layer, [-1, 28, 28, 1])

    # 第一层：一层卷积接一层最大池化
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5, 5], padding="same")
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # 第二层：一层卷积接一层最大池化
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same")
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    pool2_flat = tf.reshape(pool2, shape=[-1, 7 * 7 * 64])  # 将多维张量展开成向量，方便全连接
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)  # 全连接1
    logits = tf.layers.dense(inputs=dense, units=10)  # 全连接2

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),  # 概率最大的就是预测出的分类
        "probabilities": tf.nn.softmax(logits)  # 返回各个数字的概率
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    labels = tf.one_hot(indices=labels, depth=10)  # 将标量的标签转成one-hot形式, 不想在这里转换可以在上面读取的时候就设置one_hot=True
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)  # 交叉熵，下面都不是这次的重点
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        train_op = optimizer.minimize(loss, global_step)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(labels=tf.argmax(labels, 1),
                                        predictions=predictions['classes'],
                                        name='accuracy')
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops=eval_metric_ops
    )


cnn_mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="./cnn_mnist")


train_data = mnist.train.images  # 训练集
train_labels = mnist.train.labels
eval_data = mnist.test.images  # 验证集，用来验证模型的好坏
eval_labels = mnist.test.labels


#  使用estimator需要定义输入函数，这里 estimator 自带了一个常用的
train_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"input": train_data},                    # 输入的图片
                y=train_labels.astype(np.int32),            # 对应的标签，原始类型是uint8，需要转成tensorflow能处理的int32
                batch_size=8,                               # 每次传入八张图片，同时传入多张图片做mini batch训练结果更好
                num_epochs=None,                            # 将整个数据集循环几次，None表示无限循环
                shuffle=True                                # 是否打乱顺序
            )

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                    x={"input": eval_data},
                    y=eval_labels.astype(np.int32),
                    num_epochs=1,                            # 将整个数据集循环一次就结束生成器
                    shuffle=False
                )  # 同上


# cnn_mnist_classifier.train(input_fn=train_input_fn, steps=1000)  # 训练1000步就可以达到很好的效果, 几秒钟就可以训练好了
# eval_results = cnn_mnist_classifier.evaluate(input_fn=eval_input_fn)
# print(eval_results)


for i in range(10):
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"input": np.expand_dims(eval_data[i], 0)},
        batch_size=1,
        shuffle=False
    )
    predictions = cnn_mnist_classifier.predict(predict_input_fn)
    for prediction in predictions:
        plt.title('prediction is {label}'.format(label=prediction.get("classes")))
        plt.imshow(eval_data[i].reshape([28, 28]), cmap='gray')
        plt.show()
