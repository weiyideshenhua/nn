# python: 3.5.2
# encoding: utf-8

import numpy as np


def load_data(fname):

    """
    载入数据。
    """
    with open(fname, 'r') as f:
        data = []
        line = f.readline()
        for line in f:
            line = line.strip().split()
            x1 = float(line[0])
            x2 = float(line[1])
            t = int(line[2])
            data.append([x1, x2, t])
        return np.array(data)


def eval_acc(label, pred):
    """
    计算准确率。
    """
    return np.sum(label == pred) / len(pred)


class SVM():
    """
    SVM模型。
    $$
E_{\text{SVM}} = \sum_{n=1}^{N} \left[1 - y_nt_n \right] + \lambda \left\| \mathbf{w} \right\|^2
$$
其中$$
y_{n} = \mathbf{w}^{T} x_{n} + b
$$，$t_n$为类别标签。
    """

    def __init__(self, lambda_reg=0.1, learning_rate=0.01):
        # 请补全此处代码
        self.lambda_reg = lambda_reg
        self.learning_rate = learning_rate
        self.W = None
        self.b = None
        pass


    def hinge_loss(self, X, y):
        """
        计算SVM的铰链损失。
        """
        margins = 1 - y * (np.dot(X, self.W) + self.b)
        loss = np.sum(np.maximum(0, margins)) + 0.5 * self.lambda_reg * np.dot(self.W, self.W)
        return loss
    
    def compute_gradients(self, X, y):
        """
        计算铰链损失的梯度。
        """
        margins = 1 - y * (np.dot(X, self.W) + self.b)
        error = (margins > 0).astype(float)
        grad_W = -np.dot(X.T, y * error) / len(X) + self.lambda_reg * self.W
        grad_b = -np.sum(y * error) / len(X)
        return grad_W, grad_b
    


    def train(self, X, y, epochs=100):
        """
        训练模型。
        使用梯度下降法训练SVM。
        """
        self.W = np.random.randn(X.shape[1])
        self.b = np.random.randn()

        for epoch in range(epochs):
            loss = self.hinge_loss(X, y)
            grad_W, grad_b = self.compute_gradients(X, y)

            self.W -= self.learning_rate * grad_W
            self.b -= self.learning_rate * grad_b

            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

        # 请补全此处代码


    def predict(self, x):
        """
        预测标签。
        """
        return np.sign(np.dot(X, self.W) + self.b)
        # 请补全此处代码


if __name__ == '__main__':
    # 载入数据，实际实用时将x替换为具体名称
    train_file = 'data/train_linear.txt'
    test_file = 'data/test_linear.txt'
    data_train = load_data(train_file)  # 数据格式[x1, x2, t]
    data_test = load_data(test_file)

    # 使用训练集训练SVM模型
    svm = SVM()  # 初始化模型
    svm.train(data_train)  # 训练模型

    # 使用SVM模型预测标签
    x_train = data_train[:, :2]  # feature [x1, x2]
    t_train = data_train[:, 2]  # 真实标签
    t_train_pred = svm.predict(x_train)  # 预测标签
    x_test = data_test[:, :2]
    t_test = data_test[:, 2]
    t_test_pred = svm.predict(x_test)

    # 评估结果，计算准确率
    # 评估结果，计算准确率
    # 评估结果，计算准确率
    acc_train = eval_acc(t_train, t_train_pred)
    acc_test = eval_acc(t_test, t_test_pred)
    print("train accuracy: {:.1f}%".format(acc_train * 100))
    print("test accuracy: {:.1f}%".format(acc_test * 100))
