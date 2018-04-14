import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data  # 读取数据集，本地没有缓存就下载一个


# 读取MNIST数据, 读取标量作为标签，而不one-hot形式，
# 因为tf.estimator.DNNClassifier不支持one-hot标签
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

dnn_mnist_classifier = tf.estimator.DNNClassifier(  # 就是这么简单
    hidden_units=[16, 16],  # 隐藏层，也就是中间层的神经元个数，这里就是两层，每层16个
    feature_columns=[tf.feature_column.numeric_column("input", shape=[784])],  # 定义输入的类别和形状，数字类型，形状是784
    model_dir="./dnn_mnist",  # 训练好的模型会保持在这里，方便下次继续训练或者使用
    n_classes=10  # 类别个数
)

train_data = mnist.train.images  # 训练集
train_labels = mnist.train.labels
eval_data = mnist.test.images  # 验证集，用来验证模型的好坏
eval_labels = mnist.test.labels

#  使用estimator需要定义输入函数，这里 estimator 自带了一个常用的
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"input": train_data},  # 输入的图片
    y=train_labels.astype(np.int32),  # 对应的标签，原始类型是uint8，需要转成tensorflow能处理的int32
    batch_size=8,  # 每次传入八张图片，同时传入多张图片做mini batch训练结果更好
    num_epochs=None,  # 将整个数据集循环几次，None表示无限循环
    shuffle=True  # 是否打乱顺序
)

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"input": eval_data},
    y=eval_labels.astype(np.int32),
    num_epochs=1,  # 将整个数据集循环一次就结束生成器
    shuffle=False
)  # 同上

dnn_mnist_classifier.train(input_fn=train_input_fn, steps=10000)  # 训练10000步就可以达到很好的效果, 几秒钟就可以训练好了
eval_results = dnn_mnist_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)


for i in range(10):
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"input": np.expand_dims(eval_data[i], 0)},
        batch_size=1,
        shuffle=False
    )
    predictions = dnn_mnist_classifier.predict(predict_input_fn)
    for prediction in predictions:
        plt.title('prediction is {label}'.format(label=prediction.get("classes")))
        plt.imshow(eval_data[i].reshape([28, 28]), cmap='gray')
        plt.show()
