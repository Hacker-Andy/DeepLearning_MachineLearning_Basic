from torch.utils import data
import torch
import torch.nn as nn
import numpy as np

def getData(n_data):
    """
    通过torch随机生成数据，如果习惯torch可以将相应的部分转化成numpy来实现
        实现思路
        1. 对于正类样本：先在0-9中随机生成4个整数（n_data行），再生成n_data*1的纯1矩阵。
            将两者合并再水平打乱即可得到含1的样本。对应label为1
        2. 对于负类样本：先在0-9中随机生成5个整数（n_data行），再对其中小于2的数全部替换成0。
            即可得到不含1的样本。对应label为0
    :param n_data:  每类样本的个数
    :return:        共生成n_data * 2份样本。
    """
    X_neg = torch.randint(0, 9, [n_data, 5])
    zero = torch.zeros_like(X_neg)
    X_neg = torch.where(X_neg<2, zero, X_neg)
    y_neg = torch.zeros([n_data, 1])
    X_pos = torch.randint(0, 9, [n_data-1, 4])
    X_one = torch.ones([n_data-1, 1])
    X_pos = torch.cat([X_pos, X_one], dim=1)
    X_pos_ = torch.tensor([[1, 2, 3, 5, 4]])
    for i in range(n_data-1):
        idx_ = torch.randperm(5).reshape(1, 5)
        X_tmp = X_pos[i, idx_]
        X_pos_ = torch.cat([X_pos_, X_tmp], dim=0)
    X_pos = X_pos_
    y_pos = torch.ones([n_data, 1])
    X = torch.cat([X_pos, X_neg], dim=0)
    y = torch.cat([y_pos, y_neg], dim=0)
    return data.TensorDataset(X, y)

def init_weights(m):
    """
    通过net对象自动调用进行权重初始化
    :param m:
    :return:    对w,b初始化权重
    """
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

class Accumulator:  #@save
    """在`n`个变量上累加。"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def accuracy(y_hat, y):
    """
    :param y_hat:   预测的类别
    :param y:       真实标签
    :return:        预测正确的样本数量
    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())
def evaluate_accuracy(net, data_iter):
    """
    :param net:         定义的网络对象
    :param data_iter:   数据迭代器：测试集
    :return:            返回测试集上的准确率
    """
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    for X, y in data_iter:
        y = y.reshape(1, y.shape[0])
        metric.add(accuracy(net(X).argmax(dim=1), y), y.numel())
    return metric[0] / metric[1]

def train_epoch(net, train_iter, loss, updater, last_epoch):  #
    """
    :param net:         定义的网络对象
    :param train_iter:  训练数据迭代器
    :param loss:        每轮数据损失值
    :param updater:     定义训练优化器
    :param last_epoch:  用于做最后输出梯度的flag
    :return:            返回训练损失和训练准确率
    """
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)

    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        y = y.squeeze(1).long()
        l = loss(y_hat, y)

        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.backward()
            # print(last_epoch)
            if last_epoch < 3:                                  #最后输出当前的梯度
                for name, parms in net.named_parameters():
                    print('-->parms.grad:\t', parms.grad)

            updater.step()
            # print("Training::\taccuracy(y_hat, y)\n", accuracy(y_hat, y))
            metric.add(float(l) * len(y), accuracy(y_hat, y),
                       y.size().numel())
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练准确率
    return metric[0] / metric[2], metric[1] / metric[2]

def training(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    for epoch in range(num_epochs):
        last_epoch = num_epochs-epoch
        train_metrics = train_epoch(net, train_iter, loss, updater, last_epoch)
        test_acc = evaluate_accuracy(net, test_iter)
        if epoch%20 == 0: # 每二十个输出一次
            print("Epoch::{}\ttest_acc::{}\t".format(epoch, test_acc))
    train_loss, train_acc = train_metrics
    print(train_loss, train_acc)

def MLP():
    return nn.Sequential(nn.Flatten(),nn.Linear(5, 70),
                         nn.ReLU(),nn.Linear(70, 2))

if __name__ == '__main__':
    # init params
    batch_size = 64
    num_epochs = 600
    lr = 0.03
    # 构建数据集
    Data = getData(1000)
    train, val = torch.utils.data.random_split(Data, [1792, 208])
    train_iter = torch.utils.data.DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True
    )
    val_iter = torch.utils.data.DataLoader(
        val,
        batch_size=batch_size,
        shuffle=True
    )

    # 定义损失函数
    loss = nn.CrossEntropyLoss()

    # 定义网络
    net = MLP()

    # 定义优化算法
    trainer = torch.optim.SGD(net.parameters(), lr=lr)

    # 参数初始化
    net.apply(init_weights)

    # 训练
    training(net, train_iter, val_iter, loss, num_epochs, trainer)

    # 测试
    test = np.array([[1, 2, 3, 4, 5],   # 1
                     [2, 3, 4, 5, 6],   # 0
                     [2, 4, 5, 1, 3],   # 1
                     [3, 4, 5, 9, 2],   # 0
                     [1, 1, 2, 3, 4],   # 1
                     [2, 3, 4, 2, 0]])  # 0
    # test = test.reshape(1, 4, 5)
    test = test.astype(np.float32)
    test = torch.from_numpy(test)
    y_pred = net(test)
    print("Pred::\n", y_pred.argmax(dim=1))