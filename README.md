# 1. 作业内容描述
## 1.1 背景
- 数据集大小150
- 该数据有4个属性，分别如下
	- Sepal.Length：花萼长度(cm)
	-  Sepal.Width：花萼宽度单位(cm) 
	- Petal.Length：花瓣长度(cm)
	-  Petal.Width：花瓣宽度(cm)
	-  category：类别（Iris Setosa\Iris Versicolour\Iris Virginica)
## 1.2 要求
在不调用机器学习库的情况下，使用神经网络模型来预测一个花所属的种类。
     



# 2. 作业已完成部分和未完成部分
该作业已经全部完成，没有未完成的部分。全部代码我已经放在GitHub上和colab上了，可以点击下面的链接进行跳转。

|  <img src="https://gitee.com/the-blade-is-in-the-scabbard/typora_photo_repo/raw/master/img/202401192255505.png" alt="github_icon" style="zoom:50%;" /> | <img src="https://gitee.com/the-blade-is-in-the-scabbard/typora_photo_repo/raw/master/img/202401192255555.png" alt="colab_icon" style="zoom:50%;" /> |
-------- | -----
[GitHub For DNN](https://github.com/hiddenSharp429/DNN-Python)  | [Colab For DNN](https://colab.research.google.com/drive/1gbArpy5Oy0RGYPrAepD5FcR81HeDmYXz?usp=sharing) 

# 3. 作业运行结果截图
最后得出使用深度神经网络的模型预测的准确率为$95.555\%$

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/56c01e4f0bf547ea8dfd26adb694dd67.png)

# 4. 核心代码和步骤
## 4.1 第一步将数据集读入
### 4.1.1 原始的数据集 data.txt 部分截图：
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/f38a56b5c911452f93861c26cd9898d0.png =300x300)
稍微进行改动一下（添加了属性列并将格式转换为.csv）
### 4.1.2 修改后的数据集 data.csv 部分截图
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/e188dae79e0f455697f11b5f202d776c.png =300x300)
### 4.1.3 将 data.csv 读入并且将其存入标识符 df 中，定义数据集的筛选条件 expr_1;expr_2;expr_3
- expr_1: 用于赛选 Category 属性列为 iris-setosa 的类 sql 语句
- expr_2: 用于赛选 Category 属性列为 Iris-versicolor 的类 sql 语句
- expr_3: 用于赛选 Category 属性列为 Iris-virginica 的类 sql 语句
用上面定义的筛选条件筛选出数据集中三个类别的数据，并分别存入对应的标识符中。
- Iris_setosa_dataframe：所属类别为 Iris-setosa 的全部数据
- Iris_versicolor_dataframe：所属类别为 Iris-versicolor 的全部数据
- Iris_virginica_dataframe：所属类别为 Iris-virginica 的全部数据

### 4.1.4 代码部分
<span style="font-size: large;">**In[1]:**</span>
```python
import pandas as pd

df = pd.read_csv("data.csv") # 读取全部列表数据

expr_1 = "Category == 'Iris-setosa'" # 用于赛选Category属性列为iris-setosa的类sql语句
expr_2 = "Category == 'Iris-versicolor'" # 用于赛选Category属性列为Iris-versicolor的类sql语句
expr_3 = "Category == 'Iris-virginica'" # 用于赛选Category属性列为Iris-virginica的类sql语句

Iris_setosa_dataframe = df.query(expr_1)
Iris_versicolor_dataframe = df.query(expr_2)
Iris_virginica_dataframe = df.query(expr_3)

print("Iris_setosa_dataframe is:\n",Iris_setosa_dataframe)
print("Iris_versicolor_dataframe is\n",Iris_versicolor_dataframe)
print("Iris_virginica_dataframe is\n",Iris_virginica_dataframe)
```

<br>

<span style="font-size: large;">**out[1]:**</span>

```
Iris_setosa_dataframe is:
Sepal.Length Sepal.Width Petal.Length Petal.Width Category
0 5.1 3.5 1.4 0.2 Iris-setosa
1 4.9 3.0 1.4 0.2 Iris-setosa
2 4.7 3.2 1.3 0.2 Iris-setosa
3 4.6 3.1 1.5 0.2 Iris-setosa
4 5.0 3.6 1.4 0.2 Iris-setosa
5 5.4 3.9 1.7 0.4 Iris-setosa
6 4.6 3.4 1.4 0.3 Iris-setosa
7 5.0 3.4 1.5 0.2 Iris-setosa
8 4.4 2.9 1.4 0.2 Iris-setosa
9 4.9 3.1 1.5 0.1 Iris-setosa
10 5.4 3.7 1.5 0.2 Iris-setosa
11 4.8 3.4 1.6 0.2 Iris-setosa
12 4.8 3.0 1.4 0.1 Iris-setosa
13 4.3 3.0 1.1 0.1 Iris-setosa
14 5.8 4.0 1.2 0.2 Iris-setosa
15 5.7 4.4 1.5 0.4 Iris-setosa
16 5.4 3.9 1.3 0.4 Iris-setosa
17 5.1 3.5 1.4 0.3 Iris-setosa
18 5.7 3.8 1.7 0.3 Iris-setosa
19 5.1 3.8 1.5 0.3 Iris-setosa
20 5.4 3.4 1.7 0.2 Iris-setosa
21 5.1 3.7 1.5 0.4 Iris-setosa
22 4.6 3.6 1.0 0.2 Iris-setosa
23 5.1 3.3 1.7 0.5 Iris-setosa
24 4.8 3.4 1.9 0.2 Iris-setosa
25 5.0 3.0 1.6 0.2 Iris-setosa
26 5.0 3.4 1.6 0.4 Iris-setosa
27 5.2 3.5 1.5 0.2 Iris-setosa
28 5.2 3.4 1.4 0.2 Iris-setosa
29 4.7 3.2 1.6 0.2 Iris-setosa
30 4.8 3.1 1.6 0.2 Iris-setosa
31 5.4 3.4 1.5 0.4 Iris-setosa
32 5.2 4.1 1.5 0.1 Iris-setosa
33 5.5 4.2 1.4 0.2 Iris-setosa
34 4.9 3.1 1.5 0.1 Iris-setosa
35 5.0 3.2 1.2 0.2 Iris-setosa
36 5.5 3.5 1.3 0.2 Iris-setosa
37 4.9 3.1 1.5 0.1 Iris-setosa
38 4.4 3.0 1.3 0.2 Iris-setosa
39 5.1 3.4 1.5 0.2 Iris-setosa
40 5.0 3.5 1.3 0.3 Iris-setosa
41 4.5 2.3 1.3 0.3 Iris-setosa
42 4.4 3.2 1.3 0.2 Iris-setosa
43 5.0 3.5 1.6 0.6 Iris-setosa
44 5.1 3.8 1.9 0.4 Iris-setosa
45 4.8 3.0 1.4 0.3 Iris-setosa
46 5.1 3.8 1.6 0.2 Iris-setosa
47 4.6 3.2 1.4 0.2 Iris-setosa
48 5.3 3.7 1.5 0.2 Iris-setosa
49 5.0 3.3 1.4 0.2 Iris-setosa
Iris_versicolor_dataframe is
Sepal.Length Sepal.Width Petal.Length Petal.Width Category
50 7.0 3.2 4.7 1.4 Iris-versicolor
51 6.4 3.2 4.5 1.5 Iris-versicolor
52 6.9 3.1 4.9 1.5 Iris-versicolor
53 5.5 2.3 4.0 1.3 Iris-versicolor
54 6.5 2.8 4.6 1.5 Iris-versicolor
55 5.7 2.8 4.5 1.3 Iris-versicolor
56 6.3 3.3 4.7 1.6 Iris-versicolor
57 4.9 2.4 3.3 1.0 Iris-versicolor
58 6.6 2.9 4.6 1.3 Iris-versicolor
59 5.2 2.7 3.9 1.4 Iris-versicolor
60 5.0 2.0 3.5 1.0 Iris-versicolor
61 5.9 3.0 4.2 1.5 Iris-versicolor
62 6.0 2.2 4.0 1.0 Iris-versicolor
63 6.1 2.9 4.7 1.4 Iris-versicolor
64 5.6 2.9 3.6 1.3 Iris-versicolor
65 6.7 3.1 4.4 1.4 Iris-versicolor
66 5.6 3.0 4.5 1.5 Iris-versicolor
67 5.8 2.7 4.1 1.0 Iris-versicolor
68 6.2 2.2 4.5 1.5 Iris-versicolor
69 5.6 2.5 3.9 1.1 Iris-versicolor
70 5.9 3.2 4.8 1.8 Iris-versicolor
71 6.1 2.8 4.0 1.3 Iris-versicolor
72 6.3 2.5 4.9 1.5 Iris-versicolor
73 6.1 2.8 4.7 1.2 Iris-versicolor
74 6.4 2.9 4.3 1.3 Iris-versicolor
75 6.6 3.0 4.4 1.4 Iris-versicolor
76 6.8 2.8 4.8 1.4 Iris-versicolor
77 6.7 3.0 5.0 1.7 Iris-versicolor
78 6.0 2.9 4.5 1.5 Iris-versicolor
79 5.7 2.6 3.5 1.0 Iris-versicolor
80 5.5 2.4 3.8 1.1 Iris-versicolor
81 5.5 2.4 3.7 1.0 Iris-versicolor
82 5.8 2.7 3.9 1.2 Iris-versicolor
83 6.0 2.7 5.1 1.6 Iris-versicolor
84 5.4 3.0 4.5 1.5 Iris-versicolor
85 6.0 3.4 4.5 1.6 Iris-versicolor
86 6.7 3.1 4.7 1.5 Iris-versicolor
87 6.3 2.3 4.4 1.3 Iris-versicolor
88 5.6 3.0 4.1 1.3 Iris-versicolor
89 5.5 2.5 4.0 1.3 Iris-versicolor
90 5.5 2.6 4.4 1.2 Iris-versicolor
91 6.1 3.0 4.6 1.4 Iris-versicolor
92 5.8 2.6 4.0 1.2 Iris-versicolor
93 5.0 2.3 3.3 1.0 Iris-versicolor
94 5.6 2.7 4.2 1.3 Iris-versicolor
95 5.7 3.0 4.2 1.2 Iris-versicolor
96 5.7 2.9 4.2 1.3 Iris-versicolor
97 6.2 2.9 4.3 1.3 Iris-versicolor
98 5.1 2.5 3.0 1.1 Iris-versicolor
99 5.7 2.8 4.1 1.3 Iris-versicolor
Iris_virginica_dataframe is
Sepal.Length Sepal.Width Petal.Length Petal.Width Category
100 6.3 3.3 6.0 2.5 Iris-virginica
101 5.8 2.7 5.1 1.9 Iris-virginica
102 7.1 3.0 5.9 2.1 Iris-virginica
103 6.3 2.9 5.6 1.8 Iris-virginica
104 6.5 3.0 5.8 2.2 Iris-virginica
105 7.6 3.0 6.6 2.1 Iris-virginica
106 4.9 2.5 4.5 1.7 Iris-virginica
107 7.3 2.9 6.3 1.8 Iris-virginica
108 6.7 2.5 5.8 1.8 Iris-virginica
109 7.2 3.6 6.1 2.5 Iris-virginica
110 6.5 3.2 5.1 2.0 Iris-virginica
111 6.4 2.7 5.3 1.9 Iris-virginica
112 6.8 3.0 5.5 2.1 Iris-virginica
113 5.7 2.5 5.0 2.0 Iris-virginica
114 5.8 2.8 5.1 2.4 Iris-virginica
115 6.4 3.2 5.3 2.3 Iris-virginica
116 6.5 3.0 5.5 1.8 Iris-virginica
117 7.7 3.8 6.7 2.2 Iris-virginica
118 7.7 2.6 6.9 2.3 Iris-virginica
119 6.0 2.2 5.0 1.5 Iris-virginica
120 6.9 3.2 5.7 2.3 Iris-virginica
121 5.6 2.8 4.9 2.0 Iris-virginica
122 7.7 2.8 6.7 2.0 Iris-virginica
123 6.3 2.7 4.9 1.8 Iris-virginica
124 6.7 3.3 5.7 2.1 Iris-virginica
125 7.2 3.2 6.0 1.8 Iris-virginica
126 6.2 2.8 4.8 1.8 Iris-virginica
127 6.1 3.0 4.9 1.8 Iris-virginica
128 6.4 2.8 5.6 2.1 Iris-virginica
129 7.2 3.0 5.8 1.6 Iris-virginica
130 7.4 2.8 6.1 1.9 Iris-virginica
131 7.9 3.8 6.4 2.0 Iris-virginica
132 6.4 2.8 5.6 2.2 Iris-virginica
133 6.3 2.8 5.1 1.5 Iris-virginica
134 6.1 2.6 5.6 1.4 Iris-virginica
135 7.7 3.0 6.1 2.3 Iris-virginica
136 6.3 3.4 5.6 2.4 Iris-virginica
137 6.4 3.1 5.5 1.8 Iris-virginica
138 6.0 3.0 4.8 1.8 Iris-virginica
139 6.9 3.1 5.4 2.1 Iris-virginica
140 6.7 3.1 5.6 2.4 Iris-virginica
141 6.9 3.1 5.1 2.3 Iris-virginica
142 5.8 2.7 5.1 1.9 Iris-virginica
143 6.8 3.2 5.9 2.3 Iris-virginica
144 6.7 3.3 5.7 2.5 Iris-virginica
145 6.7 3.0 5.2 2.3 Iris-virginica
146 6.3 2.5 5.0 1.9 Iris-virginica
147 6.5 3.0 5.2 2.0 Iris-virginica
148 6.2 3.4 5.4 2.3 Iris-virginica
149 5.9 3.0 5.1 1.8 Iris-virginica
```
## 4.2 第二步数据集按照测试集和训练集分类
<br>
人为规定训练集占比 0.7，数据集为 0.3 下面将定义一个名为 `get_train_and_test_dataframe`
的函数，并返回训练集和测试集的 DataFrame。
<br>
<br>

<span style="font-size: large;">**In[2]:**</span>
```python
total_record, attribute_rows = df.shape # 获取总记录条数和其属性列

train_data_rate = 0.7 # 训练集占数据集的比例，即70%
test_data_rate = 1 - train_data_rate # 测试集与训练集为互补集

def get_train_and_test_dataframe(df1, df2, df3, train_data_rate):
    train_df = pd.DataFrame() # 创建一个空的dataframe
    test_df = pd.DataFrame() # 创建一个空的dataframe
    df_array = [df1, df2, df3] # 将各个df子集存入一个列表用于变量

    for i in range(3):
        item_df_record_num, _ = df_array[i].shape # 获取每个df子集的记录总条数
        item_df_train_record_num = int(item_df_record_num* train_data_rate) # 计算每个df子集的训练数据总记录条数

        # 随机从df子集中抽取数量为 itemDf_trainRecordNum 的记录作为训练集
        train_records = df_array[i].sample(item_df_train_record_num)

        # 子集中除去被选出为测试集的其余记录作为测试集
        test_records = df_array[i][~df_array[i].index.isin(train_records.index)]

        # 将每个子集中的训练集添加到trainDf中
        train_df = pd.concat([train_df, train_records])

        # 将每个子集中的测试集添加到testDf中
        test_df = pd.concat([test_df, test_records])

    return train_df, test_df
```

<br>
<br>

<span style="font-size: large;">**In[3]:**</span>
```python

train_data, test_data = get_train_and_test_dataframe(Iris_setosa_dataframe, Iris_versicolor_dataframe, Iris_virginica_dataframe, train_data_rate)
```


<br>


## 4.3 第三步定义深度神经网络和其所需的函数法
### 4.3.1 简介
- 该深度神经网络具有两个隐藏层，这两个隐藏层的大小分别为 5 和 4
- 除此之外输入层有四个特征值，输出层有三个输出类别，
- 大致过程为：用训练集来训练各层的权重 `w` 和偏移常量 `b`，单次训练的大小是整个训练集的大小即 105*5, 训练完成后使用测试集的单个测试记录来进行模型评估。

整个该深度神经网络的形状如下图所示。
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/b6f29ff584654d51ba5634ca25b8dbbf.png)

### 4.3.2 作用
能够通过输入层输入的四个特征值，经过已经训练好的参数进行一层一层的传播，从而在输出
层输出一个所属各个类别概率的列表。

### 4.3.3 接口说明
需要传入下面几个参数:
- X_train: 训练集的特征集
- Y_train: 训练集的结果集（所属类别的哑变量）
- hidden_sizes: 隐藏层的层数以及其对应大小
- num_epochs: 训练次数
- learning_rate：学习效率
### 4.3.4 返回参数说明
1. trained_parameters：已经训练好的参数
2. loss_history：损失函数的变化历史

### 4.3.5 代码部分
<span style="font-size: large;">**In[4]:**</span>
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 定义激活函数sigomid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 定义softmax函数
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# 初始化参数
def initialize_parameters(input_size, hidden_sizes, output_size):
    sizes = [input_size] + hidden_sizes + [output_size] # 使用列表拼接，将输入层的大小，隐藏层大小列表，输出层大小全部放入一个列表里
    parameters = {} # 定义一个空的参数字典

    for i in range(1, len(sizes)): # 从1循环到sizes的大小减一
        parameters['W' + str(i)] = np.random.randn(sizes[i-1], sizes[i]) * 0.01 # 随机生成每层的权重矩阵
        parameters['b' + str(i)] = np.zeros((1, sizes[i])) # 初始化每层的偏移矩阵

    return parameters

# 前向传播
def forward_propagation(X, parameters):
    cache = {'A0': X}
    # print("init cache:", cache)

    for i in range(1, len(parameters)//2):
        cache['Z' + str(i)] = np.dot(cache['A' + str(i-1)], parameters['W' + str(i)]) + parameters['b' + str(i)]
        cache['A' + str(i)] = sigmoid(cache['Z' + str(i)])

    # 输出层不使用sigmoid激活函数
    cache['A' + str(len(parameters)//2)] = np.dot(cache['A' + str(len(parameters)//2 - 1)], parameters['W' + str(len(parameters)//2)]) + parameters['b' + str(len(parameters)//2)]
    cache['O' + str(len(parameters)//2)] = softmax(cache['A' + str(len(parameters)//2)])

    # print('final cache:', cache)
    return cache

# 计算交叉熵损失
def compute_loss(Y, Y_hat):
    m = Y.shape[0] # 获取Y的样本数
    epsilon = 1e-8  # 微小常数，用于数值稳定性，防止出现0的情况
    loss = -1/m * np.sum(Y * np.log(Y_hat + epsilon)) # 将总损失除以样本数m，得到平均损失。负号表示最小化损失。
    return loss

# 反向传播
def backward_propagation(X, Y, parameters, cache):
    m = X.shape[0]  # 获取样本数

    grads = {}  # 初始化梯度字典

    # 计算输出层的梯度
    dZ_last = cache['O' + str(len(parameters)//2)] - Y  # 计算输出层激活值的梯度
    grads['dW' + str(len(parameters)//2)] = 1/m * np.dot(cache['A' + str(len(parameters)//2 - 1)].T, dZ_last)  # 计算输出层权重的梯度
    grads['db' + str(len(parameters)//2)] = 1/m * np.sum(dZ_last, axis=0, keepdims=True)  # 计算输出层偏差的梯度

    dZ = dZ_last  # 初始化激活值梯度

    # 循环计算隐藏层的梯度
    for i in range(len(parameters)//2, 1, -1):
        dA = np.dot(dZ, parameters['W' + str(i)].T)  # 计算上一层激活值的梯度
        dZ = dA * cache['A' + str(i-1)] * (1 - cache['A' + str(i-1)])  # 计算当前层激活值的梯度
        grads['dW' + str(i-1)] = 1/m * np.dot(cache['A' + str(i-2)].T, dZ)  # 计算当前层权重的梯度
        grads['db' + str(i-1)] = 1/m * np.sum(dZ, axis=0, keepdims=True)  # 计算当前层偏差的梯度

    return grads  # 返回计算得到的梯度字典



# 更新参数
def update_parameters(parameters, grads, learning_rate):
    for i in range(1, len(parameters)//2 + 1):
        parameters['W' + str(i)] -= learning_rate * grads['dW' + str(i)]
        parameters['b' + str(i)] -= learning_rate * grads['db' + str(i)]

    return parameters

# 模型训练
def train_neural_network(X, Y, hidden_sizes, num_epochs, learning_rate=0.01):
    input_size = X.shape[1] # 训练集的特征数量（有多少个属性列）
    output_size = Y.shape[1] # 期望输出（预测）的所属类型的个数（有多少个类型）

    parameters = initialize_parameters(input_size, hidden_sizes, output_size) # 初始化权重w和偏移量b

    loss_history = [] # 定义空列表来记录损失函数的状态

    for epoch in range(num_epochs):
        # 前向传播
        cache = forward_propagation(X, parameters)
        Y_hat = cache['O' + str(len(parameters)//2)]

        # 计算损失
        loss = compute_loss(Y, Y_hat)
        loss_history.append(loss)

        # 反向传播
        grads = backward_propagation(X, Y, parameters, cache)

        # 更新参数
        parameters = update_parameters(parameters, grads, learning_rate)

        # 打印损失
        if epoch % 2000 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')


    return parameters, loss_history
```


## 4.4 第四步定义验证和打印函数
### 4.4.1 简介
`verify`函数是每次循环测试记录时需要调用的，可以用来打印结果和验证是否预测正确。
### 4.4.2 入口参数说明
- index：测试记录在数据集中的索引
- p_catagory：该测试记录经过贝叶斯分类后返回的结果（属于各类别的概率）
- real_category：该测试记录真实所属类别
- record_num：已经遍历测试记录的数量
- correct_num：已经遍历测试记录并且预测结果为正确的数量
- correct_rate：该模型的正确率

### 4.4.3 返回参数说明
- correct_num：同上
- correct_rate：同上

### 4.4.4 代码部分
<span style="font-size: large;">**In[5]:**</span>
```python
def verify(index, p_catagory, real_category, record_num, correct_num, correct_rate):

    print("测试结果已出，该测试记录所属类别的概率为\n",p_category) # 打印该记录所对应类别的概率
    max_probability = max(p_category.values())  # 获取最大的概率值

    for key, key_value in p_category.items(): # 寻找概率最大的类别
        if key_value == max_probability: ## 找到概率最大的类别
            print(f"第{index}记录的预测最可能的所属类别为:{key}")
            print(f"第{index}记录的真实属性为:{real_category}")
            if key == real_category: ## 查看预测的类别和真实的类别是否一样
                correct_num = correct_num + 1 # 若一样则correct_num++
            print("-------------------------")

    correct_rate = correct_num / record_num # 计算新的正确率
    return correct_num, correct_rate
```


## 4.5 第五步将训练集数据放入深度神经网络中训练
<span style="font-size: large;">**In[6]:**</span>
```python
# 提取训练集特征和标签
X_train = train_data.iloc[:, :-1].values
Y_train = pd.get_dummies(train_data['Category']).values

# 训练神经网络
trained_parameters, loss_history = train_neural_network(X_train, Y_train, hidden_sizes=[5, 4], num_epochs=100000, learning_rate=0.09)

# 展示loss的折线图
plt.plot(loss_history)
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
```
<br>
<br>

<span style="font-size: large;">**out[6]:**</span>
```
Epoch 0, Loss: 1.0986374636813678
Epoch 2000, Loss: 1.0986098115977763
Epoch 4000, Loss: 1.0985974991821654
Epoch 6000, Loss: 1.0979638994470924
Epoch 8000, Loss: 0.3051250142896731
Epoch 10000, Loss: 0.09047674524698732
Epoch 12000, Loss: 0.06427539824439313
Epoch 14000, Loss: 0.05738894582331716
Epoch 16000, Loss: 0.05422622522965926
Epoch 18000, Loss: 0.052323171513319895
Epoch 20000, Loss: 0.050989527225192836
Epoch 22000, Loss: 0.04995500647173487
Epoch 24000, Loss: 0.04908545311867662
Epoch 26000, Loss: 0.048302169206777316
Epoch 28000, Loss: 0.04755683502388777
Epoch 30000, Loss: 0.04682381035782021
Epoch 32000, Loss: 0.04609479360679712
Epoch 34000, Loss: 0.045371734234237376
Epoch 36000, Loss: 0.04466000953822288
Epoch 38000, Loss: 0.043963956316713244
Epoch 40000, Loss: 0.04328524335321981
Epoch 42000, Loss: 0.042623351179843484
Epoch 44000, Loss: 0.041976754101067736
Epoch 46000, Loss: 0.04134382847704943
Epoch 48000, Loss: 0.040723316848883334
Epoch 50000, Loss: 0.040114532246993004
Epoch 52000, Loss: 0.03951746513256648
Epoch 54000, Loss: 0.03893284700966053
Epoch 56000, Loss: 0.038362146671127625
Epoch 58000, Loss: 0.0378074459549261
Epoch 60000, Loss: 0.037271164594935534
Epoch 62000, Loss: 0.03675567096214186
Epoch 64000, Loss: 0.036262888491245114
Epoch 66000, Loss: 0.0357940258522333
Epoch 68000, Loss: 0.03534949830432913
Epoch 70000, Loss: 0.03492901608330322
Epoch 72000, Loss: 0.034531761988498534
Epoch 74000, Loss: 0.034156583734556774
Epoch 76000, Loss: 0.033802158226160436
Epoch 78000, Loss: 0.03346711379140615
Epoch 80000, Loss: 0.03315011215846706
Epoch 82000, Loss: 0.03284989755621375
Epoch 84000, Loss: 0.03256532097901557
Epoch 86000, Loss: 0.03229534661051307
Epoch 88000, Loss: 0.03219234060425748
Epoch 90000, Loss: 0.03201415844578465
Epoch 92000, Loss: 0.03182184539911233
Epoch 94000, Loss: 0.03163239691494281
Epoch 96000, Loss: 0.031446408554093044
Epoch 98000, Loss: 0.031264326953139694
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/31b7c22059c045f6af220e13e1cf6069.png)


## 4.6 第六步将测试集数据放入神经网络中验证是否正确
<span style="font-size: large;">**In[7]:**</span>
```python
# 提取测试集特征和标签
X_test = test_data.iloc[:, :-1].values
Y_test = pd.get_dummies(test_data['Category']).values

# 初始化参数
correct_rate = 0
correct_num = 0
record_num = 0
class_labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

for index, record in test_data.iterrows():
    record_num = record_num + 1 # 所遍历的测试记录条数+1

    Category = record['Category'] # 获取当前测试记录的真实所属类别
    print(f'记录:{index}, 所属类别:{Category}')

    record_x = [record['Sepal.Length'], record['Sepal.Width'], record['Petal.Length'], record['Petal.Width']] # 将当前测试记录的所有特征值合并成一个列表
    print(record_x)

    record_x_cache = forward_propagation(record_x, trained_parameters) # 调用深度神经网络进行测试，返回一个字典
    result_y = record_x_cache['O' + str(len(trained_parameters)//2)][0] # 用result_y来接收输出层的数据，具体形式是一个概率数组
    p_category = {label: value for label, value in zip(class_labels, result_y)} # 将概率数组拓广到字典，即原本的概率列表变成一个概率字典（键值对）
    print(p_category)

    correct_num, correct_rate = verify(index, p_category, Category, record_num, correct_num, correct_rate) # 更新正确条数和正确率

print(f'该模型的预测准确率为:{correct_rate}')

```


<br>
<br>

<span style="font-size: large;">**out[7]:**</span>
```
记录:9, 所属类别:Iris-setosa
[4.9, 3.1, 1.5, 0.1]
{'Iris-setosa': 0.9994073024809125, 'Iris-versicolor': 0.0005926974101059076, 'Iris-virginica': 1.089814284856423e-10}
测试结果已出，该测试记录所属类别的概率为
 {'Iris-setosa': 0.9994073024809125, 'Iris-versicolor': 0.0005926974101059076, 'Iris-virginica': 1.089814284856423e-10}
第9记录的预测最可能的所属类别为:Iris-setosa
第9记录的真实属性为:Iris-setosa
-------------------------
记录:10, 所属类别:Iris-setosa
[5.4, 3.7, 1.5, 0.2]
{'Iris-setosa': 0.9994194567509896, 'Iris-versicolor': 0.0005805431416654768, 'Iris-virginica': 1.0734484569031509e-10}
测试结果已出，该测试记录所属类别的概率为
 {'Iris-setosa': 0.9994194567509896, 'Iris-versicolor': 0.0005805431416654768, 'Iris-virginica': 1.0734484569031509e-10}
第10记录的预测最可能的所属类别为:Iris-setosa
第10记录的真实属性为:Iris-setosa
-------------------------
记录:13, 所属类别:Iris-setosa
[4.3, 3.0, 1.1, 0.1]
{'Iris-setosa': 0.9994344552564192, 'Iris-versicolor': 0.0005655446415884805, 'Iris-virginica': 1.0199248245814307e-10}
测试结果已出，该测试记录所属类别的概率为
 {'Iris-setosa': 0.9994344552564192, 'Iris-versicolor': 0.0005655446415884805, 'Iris-virginica': 1.0199248245814307e-10}
第13记录的预测最可能的所属类别为:Iris-setosa
第13记录的真实属性为:Iris-setosa
-------------------------
记录:14, 所属类别:Iris-setosa
[5.8, 4.0, 1.2, 0.2]
{'Iris-setosa': 0.9993839003436563, 'Iris-versicolor': 0.0006160995360430794, 'Iris-virginica': 1.203005937453023e-10}
测试结果已出，该测试记录所属类别的概率为
 {'Iris-setosa': 0.9993839003436563, 'Iris-versicolor': 0.0006160995360430794, 'Iris-virginica': 1.203005937453023e-10}
第14记录的预测最可能的所属类别为:Iris-setosa
第14记录的真实属性为:Iris-setosa
-------------------------
记录:15, 所属类别:Iris-setosa
[5.7, 4.4, 1.5, 0.4]
{'Iris-setosa': 0.9993971683516336, 'Iris-versicolor': 0.0006028315327860123, 'Iris-virginica': 1.1558054531504385e-10}
测试结果已出，该测试记录所属类别的概率为
 {'Iris-setosa': 0.9993971683516336, 'Iris-versicolor': 0.0006028315327860123, 'Iris-virginica': 1.1558054531504385e-10}
第15记录的预测最可能的所属类别为:Iris-setosa
第15记录的真实属性为:Iris-setosa
-------------------------
记录:16, 所属类别:Iris-setosa
[5.4, 3.9, 1.3, 0.4]
{'Iris-setosa': 0.9993562667737085, 'Iris-versicolor': 0.0006437330976151712, 'Iris-virginica': 1.28676261460329e-10}
测试结果已出，该测试记录所属类别的概率为
 {'Iris-setosa': 0.9993562667737085, 'Iris-versicolor': 0.0006437330976151712, 'Iris-virginica': 1.28676261460329e-10}
第16记录的预测最可能的所属类别为:Iris-setosa
第16记录的真实属性为:Iris-setosa
-------------------------
记录:23, 所属类别:Iris-setosa
[5.1, 3.3, 1.7, 0.5]
{'Iris-setosa': 0.9992156354832021, 'Iris-versicolor': 0.000784364351532988, 'Iris-virginica': 1.652649064805943e-10}
测试结果已出，该测试记录所属类别的概率为
 {'Iris-setosa': 0.9992156354832021, 'Iris-versicolor': 0.000784364351532988, 'Iris-virginica': 1.652649064805943e-10}
第23记录的预测最可能的所属类别为:Iris-setosa
第23记录的真实属性为:Iris-setosa
-------------------------
记录:26, 所属类别:Iris-setosa
[5.0, 3.4, 1.6, 0.4]
{'Iris-setosa': 0.9993422506434927, 'Iris-versicolor': 0.0006577492281176229, 'Iris-virginica': 1.2838960352069869e-10}
测试结果已出，该测试记录所属类别的概率为
 {'Iris-setosa': 0.9993422506434927, 'Iris-versicolor': 0.0006577492281176229, 'Iris-virginica': 1.2838960352069869e-10}
第26记录的预测最可能的所属类别为:Iris-setosa
第26记录的真实属性为:Iris-setosa
-------------------------
记录:28, 所属类别:Iris-setosa
[5.2, 3.4, 1.4, 0.2]
{'Iris-setosa': 0.9994022919498919, 'Iris-versicolor': 0.0005977079379210146, 'Iris-virginica': 1.1218683099221791e-10}
测试结果已出，该测试记录所属类别的概率为
 {'Iris-setosa': 0.9994022919498919, 'Iris-versicolor': 0.0005977079379210146, 'Iris-virginica': 1.1218683099221791e-10}
第28记录的预测最可能的所属类别为:Iris-setosa
第28记录的真实属性为:Iris-setosa
-------------------------
记录:37, 所属类别:Iris-setosa
[4.9, 3.1, 1.5, 0.1]
{'Iris-setosa': 0.9994073024809125, 'Iris-versicolor': 0.0005926974101059076, 'Iris-virginica': 1.089814284856423e-10}
测试结果已出，该测试记录所属类别的概率为
 {'Iris-setosa': 0.9994073024809125, 'Iris-versicolor': 0.0005926974101059076, 'Iris-virginica': 1.089814284856423e-10}
第37记录的预测最可能的所属类别为:Iris-setosa
第37记录的真实属性为:Iris-setosa
-------------------------
记录:41, 所属类别:Iris-setosa
[4.5, 2.3, 1.3, 0.3]
{'Iris-setosa': 0.9990698970091187, 'Iris-versicolor': 0.0009301027868999484, 'Iris-virginica': 2.03981361107607e-10}
测试结果已出，该测试记录所属类别的概率为
 {'Iris-setosa': 0.9990698970091187, 'Iris-versicolor': 0.0009301027868999484, 'Iris-virginica': 2.03981361107607e-10}
第41记录的预测最可能的所属类别为:Iris-setosa
第41记录的真实属性为:Iris-setosa
-------------------------
记录:43, 所属类别:Iris-setosa
[5.0, 3.5, 1.6, 0.6]
{'Iris-setosa': 0.9992318991352795, 'Iris-versicolor': 0.0007681007014263888, 'Iris-virginica': 1.6329400930324945e-10}
测试结果已出，该测试记录所属类别的概率为
 {'Iris-setosa': 0.9992318991352795, 'Iris-versicolor': 0.0007681007014263888, 'Iris-virginica': 1.6329400930324945e-10}
第43记录的预测最可能的所属类别为:Iris-setosa
第43记录的真实属性为:Iris-setosa
-------------------------
记录:44, 所属类别:Iris-setosa
[5.1, 3.8, 1.9, 0.4]
{'Iris-setosa': 0.9993864980408808, 'Iris-versicolor': 0.0006135018444908369, 'Iris-virginica': 1.146282328612296e-10}
测试结果已出，该测试记录所属类别的概率为
 {'Iris-setosa': 0.9993864980408808, 'Iris-versicolor': 0.0006135018444908369, 'Iris-virginica': 1.146282328612296e-10}
第44记录的预测最可能的所属类别为:Iris-setosa
第44记录的真实属性为:Iris-setosa
-------------------------
记录:45, 所属类别:Iris-setosa
[4.8, 3.0, 1.4, 0.3]
{'Iris-setosa': 0.9993337339145498, 'Iris-versicolor': 0.0006662659548606306, 'Iris-virginica': 1.3058942609729524e-10}
测试结果已出，该测试记录所属类别的概率为
 {'Iris-setosa': 0.9993337339145498, 'Iris-versicolor': 0.0006662659548606306, 'Iris-virginica': 1.3058942609729524e-10}
第45记录的预测最可能的所属类别为:Iris-setosa
第45记录的真实属性为:Iris-setosa
-------------------------
记录:46, 所属类别:Iris-setosa
[5.1, 3.8, 1.6, 0.2]
{'Iris-setosa': 0.999441154585264, 'Iris-versicolor': 0.0005588453143751629, 'Iris-virginica': 1.0036090050646012e-10}
测试结果已出，该测试记录所属类别的概率为
 {'Iris-setosa': 0.999441154585264, 'Iris-versicolor': 0.0005588453143751629, 'Iris-virginica': 1.0036090050646012e-10}
第46记录的预测最可能的所属类别为:Iris-setosa
第46记录的真实属性为:Iris-setosa
-------------------------
记录:52, 所属类别:Iris-versicolor
[6.9, 3.1, 4.9, 1.5]
{'Iris-setosa': 0.00034673300859923385, 'Iris-versicolor': 0.9996312297167484, 'Iris-virginica': 2.203727465222196e-05}
测试结果已出，该测试记录所属类别的概率为
 {'Iris-setosa': 0.00034673300859923385, 'Iris-versicolor': 0.9996312297167484, 'Iris-virginica': 2.203727465222196e-05}
第52记录的预测最可能的所属类别为:Iris-versicolor
第52记录的真实属性为:Iris-versicolor
-------------------------
记录:57, 所属类别:Iris-versicolor
[4.9, 2.4, 3.3, 1.0]
{'Iris-setosa': 0.0011225847462562426, 'Iris-versicolor': 0.998874153393992, 'Iris-virginica': 3.261859751878275e-06}
测试结果已出，该测试记录所属类别的概率为
 {'Iris-setosa': 0.0011225847462562426, 'Iris-versicolor': 0.998874153393992, 'Iris-virginica': 3.261859751878275e-06}
第57记录的预测最可能的所属类别为:Iris-versicolor
第57记录的真实属性为:Iris-versicolor
-------------------------
记录:60, 所属类别:Iris-versicolor
[5.0, 2.0, 3.5, 1.0]
{'Iris-setosa': 0.000699627908179903, 'Iris-versicolor': 0.9992955250491328, 'Iris-virginica': 4.8470426874968965e-06}
测试结果已出，该测试记录所属类别的概率为
 {'Iris-setosa': 0.000699627908179903, 'Iris-versicolor': 0.9992955250491328, 'Iris-virginica': 4.8470426874968965e-06}
第60记录的预测最可能的所属类别为:Iris-versicolor
第60记录的真实属性为:Iris-versicolor
-------------------------
记录:67, 所属类别:Iris-versicolor
[5.8, 2.7, 4.1, 1.0]
{'Iris-setosa': 0.0006969336614828119, 'Iris-versicolor': 0.9992982666461563, 'Iris-virginica': 4.799692360888943e-06}
测试结果已出，该测试记录所属类别的概率为
 {'Iris-setosa': 0.0006969336614828119, 'Iris-versicolor': 0.9992982666461563, 'Iris-virginica': 4.799692360888943e-06}
第67记录的预测最可能的所属类别为:Iris-versicolor
第67记录的真实属性为:Iris-versicolor
-------------------------
记录:70, 所属类别:Iris-versicolor
[5.9, 3.2, 4.8, 1.8]
{'Iris-setosa': 8.092532228554832e-07, 'Iris-versicolor': 0.04124258002202099, 'Iris-virginica': 0.9587566107247563}
测试结果已出，该测试记录所属类别的概率为
 {'Iris-setosa': 8.092532228554832e-07, 'Iris-versicolor': 0.04124258002202099, 'Iris-virginica': 0.9587566107247563}
第70记录的预测最可能的所属类别为:Iris-virginica
第70记录的真实属性为:Iris-versicolor
-------------------------
记录:75, 所属类别:Iris-versicolor
[6.6, 3.0, 4.4, 1.4]
{'Iris-setosa': 0.0004904963785951448, 'Iris-versicolor': 0.9995006634330806, 'Iris-virginica': 8.840188324211333e-06}
测试结果已出，该测试记录所属类别的概率为
 {'Iris-setosa': 0.0004904963785951448, 'Iris-versicolor': 0.9995006634330806, 'Iris-virginica': 8.840188324211333e-06}
第75记录的预测最可能的所属类别为:Iris-versicolor
第75记录的真实属性为:Iris-versicolor
-------------------------
记录:76, 所属类别:Iris-versicolor
[6.8, 2.8, 4.8, 1.4]
{'Iris-setosa': 0.0003802333354495639, 'Iris-versicolor': 0.9996035606243237, 'Iris-virginica': 1.6206040226774196e-05}
测试结果已出，该测试记录所属类别的概率为
 {'Iris-setosa': 0.0003802333354495639, 'Iris-versicolor': 0.9996035606243237, 'Iris-virginica': 1.6206040226774196e-05}
第76记录的预测最可能的所属类别为:Iris-versicolor
第76记录的真实属性为:Iris-versicolor
-------------------------
记录:78, 所属类别:Iris-versicolor
[6.0, 2.9, 4.5, 1.5]
{'Iris-setosa': 0.00021263981498677786, 'Iris-versicolor': 0.9996074531085687, 'Iris-virginica': 0.000179907076444516}
测试结果已出，该测试记录所属类别的概率为
 {'Iris-setosa': 0.00021263981498677786, 'Iris-versicolor': 0.9996074531085687, 'Iris-virginica': 0.000179907076444516}
第78记录的预测最可能的所属类别为:Iris-versicolor
第78记录的真实属性为:Iris-versicolor
-------------------------
记录:79, 所属类别:Iris-versicolor
[5.7, 2.6, 3.5, 1.0]
{'Iris-setosa': 0.0014117930910004294, 'Iris-versicolor': 0.9985851980770214, 'Iris-virginica': 3.0088319782903333e-06}
测试结果已出，该测试记录所属类别的概率为
 {'Iris-setosa': 0.0014117930910004294, 'Iris-versicolor': 0.9985851980770214, 'Iris-virginica': 3.0088319782903333e-06}
第79记录的预测最可能的所属类别为:Iris-versicolor
第79记录的真实属性为:Iris-versicolor
-------------------------
记录:81, 所属类别:Iris-versicolor
[5.5, 2.4, 3.7, 1.0]
{'Iris-setosa': 0.0008627167806437697, 'Iris-versicolor': 0.9991333416086959, 'Iris-virginica': 3.941610660265416e-06}
测试结果已出，该测试记录所属类别的概率为
 {'Iris-setosa': 0.0008627167806437697, 'Iris-versicolor': 0.9991333416086959, 'Iris-virginica': 3.941610660265416e-06}
第81记录的预测最可能的所属类别为:Iris-versicolor
第81记录的真实属性为:Iris-versicolor
-------------------------
记录:84, 所属类别:Iris-versicolor
[5.4, 3.0, 4.5, 1.5]
{'Iris-setosa': 4.914358190106048e-05, 'Iris-versicolor': 0.881793576222092, 'Iris-virginica': 0.11815728019600699}
测试结果已出，该测试记录所属类别的概率为
 {'Iris-setosa': 4.914358190106048e-05, 'Iris-versicolor': 0.881793576222092, 'Iris-virginica': 0.11815728019600699}
第84记录的预测最可能的所属类别为:Iris-versicolor
第84记录的真实属性为:Iris-versicolor
-------------------------
记录:89, 所属类别:Iris-versicolor
[5.5, 2.5, 4.0, 1.3]
{'Iris-setosa': 0.00043234470441415025, 'Iris-versicolor': 0.999555742853914, 'Iris-virginica': 1.1912441671869986e-05}
测试结果已出，该测试记录所属类别的概率为
 {'Iris-setosa': 0.00043234470441415025, 'Iris-versicolor': 0.999555742853914, 'Iris-virginica': 1.1912441671869986e-05}
第89记录的预测最可能的所属类别为:Iris-versicolor
第89记录的真实属性为:Iris-versicolor
-------------------------
记录:94, 所属类别:Iris-versicolor
[5.6, 2.7, 4.2, 1.3]
{'Iris-setosa': 0.00033893326798138754, 'Iris-versicolor': 0.9996339971676897, 'Iris-virginica': 2.7069564329036698e-05}
测试结果已出，该测试记录所属类别的概率为
 {'Iris-setosa': 0.00033893326798138754, 'Iris-versicolor': 0.9996339971676897, 'Iris-virginica': 2.7069564329036698e-05}
第94记录的预测最可能的所属类别为:Iris-versicolor
第94记录的真实属性为:Iris-versicolor
-------------------------
记录:95, 所属类别:Iris-versicolor
[5.7, 3.0, 4.2, 1.2]
{'Iris-setosa': 0.00047520993314104494, 'Iris-versicolor': 0.9995150352793134, 'Iris-virginica': 9.754787545568637e-06}
测试结果已出，该测试记录所属类别的概率为
 {'Iris-setosa': 0.00047520993314104494, 'Iris-versicolor': 0.9995150352793134, 'Iris-virginica': 9.754787545568637e-06}
第95记录的预测最可能的所属类别为:Iris-versicolor
第95记录的真实属性为:Iris-versicolor
-------------------------
记录:99, 所属类别:Iris-versicolor
[5.7, 2.8, 4.1, 1.3]
{'Iris-setosa': 0.00048434746193381145, 'Iris-versicolor': 0.9995065789295492, 'Iris-virginica': 9.073608517006889e-06}
测试结果已出，该测试记录所属类别的概率为
 {'Iris-setosa': 0.00048434746193381145, 'Iris-versicolor': 0.9995065789295492, 'Iris-virginica': 9.073608517006889e-06}
第99记录的预测最可能的所属类别为:Iris-versicolor
第99记录的真实属性为:Iris-versicolor
-------------------------
记录:101, 所属类别:Iris-virginica
[5.8, 2.7, 5.1, 1.9]
{'Iris-setosa': 6.230993565240641e-08, 'Iris-versicolor': 0.004904991357197354, 'Iris-virginica': 0.995094946332867}
测试结果已出，该测试记录所属类别的概率为
 {'Iris-setosa': 6.230993565240641e-08, 'Iris-versicolor': 0.004904991357197354, 'Iris-virginica': 0.995094946332867}
第101记录的预测最可能的所属类别为:Iris-virginica
第101记录的真实属性为:Iris-virginica
-------------------------
记录:104, 所属类别:Iris-virginica
[6.5, 3.0, 5.8, 2.2]
{'Iris-setosa': 3.1285347946103443e-09, 'Iris-versicolor': 0.0004093612664117814, 'Iris-virginica': 0.9995906356050535}
测试结果已出，该测试记录所属类别的概率为
 {'Iris-setosa': 3.1285347946103443e-09, 'Iris-versicolor': 0.0004093612664117814, 'Iris-virginica': 0.9995906356050535}
第104记录的预测最可能的所属类别为:Iris-virginica
第104记录的真实属性为:Iris-virginica
-------------------------
记录:105, 所属类别:Iris-virginica
[7.6, 3.0, 6.6, 2.1]
{'Iris-setosa': 4.518643270279261e-09, 'Iris-versicolor': 0.0005552351498570809, 'Iris-virginica': 0.9994447603314996}
测试结果已出，该测试记录所属类别的概率为
 {'Iris-setosa': 4.518643270279261e-09, 'Iris-versicolor': 0.0005552351498570809, 'Iris-virginica': 0.9994447603314996}
第105记录的预测最可能的所属类别为:Iris-virginica
第105记录的真实属性为:Iris-virginica
-------------------------
记录:108, 所属类别:Iris-virginica
[6.7, 2.5, 5.8, 1.8]
{'Iris-setosa': 8.268626469165646e-08, 'Iris-versicolor': 0.006206423895269242, 'Iris-virginica': 0.993793493418466}
测试结果已出，该测试记录所属类别的概率为
 {'Iris-setosa': 8.268626469165646e-08, 'Iris-versicolor': 0.006206423895269242, 'Iris-virginica': 0.993793493418466}
第108记录的预测最可能的所属类别为:Iris-virginica
第108记录的真实属性为:Iris-virginica
-------------------------
记录:111, 所属类别:Iris-virginica
[6.4, 2.7, 5.3, 1.9]
{'Iris-setosa': 9.976341957164077e-09, 'Iris-versicolor': 0.0010721415234597504, 'Iris-virginica': 0.9989278485001982}
测试结果已出，该测试记录所属类别的概率为
 {'Iris-setosa': 9.976341957164077e-09, 'Iris-versicolor': 0.0010721415234597504, 'Iris-virginica': 0.9989278485001982}
第111记录的预测最可能的所属类别为:Iris-virginica
第111记录的真实属性为:Iris-virginica
-------------------------
记录:117, 所属类别:Iris-virginica
[7.7, 3.8, 6.7, 2.2]
{'Iris-setosa': 1.3832457944150949e-08, 'Iris-versicolor': 0.0014048982490742054, 'Iris-virginica': 0.9985950879184678}
测试结果已出，该测试记录所属类别的概率为
 {'Iris-setosa': 1.3832457944150949e-08, 'Iris-versicolor': 0.0014048982490742054, 'Iris-virginica': 0.9985950879184678}
第117记录的预测最可能的所属类别为:Iris-virginica
第117记录的真实属性为:Iris-virginica
-------------------------
记录:122, 所属类别:Iris-virginica
[7.7, 2.8, 6.7, 2.0]
{'Iris-setosa': 1.0114546252578506e-08, 'Iris-versicolor': 0.001083446341577254, 'Iris-virginica': 0.9989165435438766}
测试结果已出，该测试记录所属类别的概率为
 {'Iris-setosa': 1.0114546252578506e-08, 'Iris-versicolor': 0.001083446341577254, 'Iris-virginica': 0.9989165435438766}
第122记录的预测最可能的所属类别为:Iris-virginica
第122记录的真实属性为:Iris-virginica
-------------------------
记录:126, 所属类别:Iris-virginica
[6.2, 2.8, 4.8, 1.8]
{'Iris-setosa': 1.211053511352366e-06, 'Iris-versicolor': 0.05826189317047926, 'Iris-virginica': 0.9417368957760094}
测试结果已出，该测试记录所属类别的概率为
 {'Iris-setosa': 1.211053511352366e-06, 'Iris-versicolor': 0.05826189317047926, 'Iris-virginica': 0.9417368957760094}
第126记录的预测最可能的所属类别为:Iris-virginica
第126记录的真实属性为:Iris-virginica
-------------------------
记录:128, 所属类别:Iris-virginica
[6.4, 2.8, 5.6, 2.1]
{'Iris-setosa': 3.5398422692944902e-09, 'Iris-versicolor': 0.0004535013010799169, 'Iris-virginica': 0.9995464951590779}
测试结果已出，该测试记录所属类别的概率为
 {'Iris-setosa': 3.5398422692944902e-09, 'Iris-versicolor': 0.0004535013010799169, 'Iris-virginica': 0.9995464951590779}
第128记录的预测最可能的所属类别为:Iris-virginica
第128记录的真实属性为:Iris-virginica
-------------------------
记录:129, 所属类别:Iris-virginica
[7.2, 3.0, 5.8, 1.6]
{'Iris-setosa': 2.9681892387650685e-06, 'Iris-versicolor': 0.12000192448513033, 'Iris-virginica': 0.8799951073256309}
测试结果已出，该测试记录所属类别的概率为
 {'Iris-setosa': 2.9681892387650685e-06, 'Iris-versicolor': 0.12000192448513033, 'Iris-virginica': 0.8799951073256309}
第129记录的预测最可能的所属类别为:Iris-virginica
第129记录的真实属性为:Iris-virginica
-------------------------
记录:134, 所属类别:Iris-virginica
[6.1, 2.6, 5.6, 1.4]
{'Iris-setosa': 5.6707124370751285e-05, 'Iris-versicolor': 0.9231303912341395, 'Iris-virginica': 0.0768129016414897}
测试结果已出，该测试记录所属类别的概率为
 {'Iris-setosa': 5.6707124370751285e-05, 'Iris-versicolor': 0.9231303912341395, 'Iris-virginica': 0.0768129016414897}
第134记录的预测最可能的所属类别为:Iris-versicolor
第134记录的真实属性为:Iris-virginica
-------------------------
记录:135, 所属类别:Iris-virginica
[7.7, 3.0, 6.1, 2.3]
{'Iris-setosa': 4.027193249103518e-10, 'Iris-versicolor': 7.550066584932601e-05, 'Iris-virginica': 0.9999244989314313}
测试结果已出，该测试记录所属类别的概率为
 {'Iris-setosa': 4.027193249103518e-10, 'Iris-versicolor': 7.550066584932601e-05, 'Iris-virginica': 0.9999244989314313}
第135记录的预测最可能的所属类别为:Iris-virginica
第135记录的真实属性为:Iris-virginica
-------------------------
记录:137, 所属类别:Iris-virginica
[6.4, 3.1, 5.5, 1.8]
{'Iris-setosa': 8.799431580711186e-07, 'Iris-versicolor': 0.04410462329103846, 'Iris-virginica': 0.9558944967658034}
测试结果已出，该测试记录所属类别的概率为
 {'Iris-setosa': 8.799431580711186e-07, 'Iris-versicolor': 0.04410462329103846, 'Iris-virginica': 0.9558944967658034}
第137记录的预测最可能的所属类别为:Iris-virginica
第137记录的真实属性为:Iris-virginica
-------------------------
记录:142, 所属类别:Iris-virginica
[5.8, 2.7, 5.1, 1.9]
{'Iris-setosa': 6.230993565240641e-08, 'Iris-versicolor': 0.004904991357197354, 'Iris-virginica': 0.995094946332867}
测试结果已出，该测试记录所属类别的概率为
 {'Iris-setosa': 6.230993565240641e-08, 'Iris-versicolor': 0.004904991357197354, 'Iris-virginica': 0.995094946332867}
第142记录的预测最可能的所属类别为:Iris-virginica
第142记录的真实属性为:Iris-virginica
-------------------------
记录:144, 所属类别:Iris-virginica
[6.7, 3.3, 5.7, 2.5]
{'Iris-setosa': 3.840864653692801e-10, 'Iris-versicolor': 7.252557578190236e-05, 'Iris-virginica': 0.9999274740401316}
测试结果已出，该测试记录所属类别的概率为
 {'Iris-setosa': 3.840864653692801e-10, 'Iris-versicolor': 7.252557578190236e-05, 'Iris-virginica': 0.9999274740401316}
第144记录的预测最可能的所属类别为:Iris-virginica
第144记录的真实属性为:Iris-virginica
-------------------------
该模型的预测准确率为:0.9555555555555556
```

---
# 5. 结束语
如果有疑问欢迎大家留言讨论，你如果觉得这篇文章对你有帮助可以给我一个免费的赞吗？我们之间的交流是我最大的动力！
