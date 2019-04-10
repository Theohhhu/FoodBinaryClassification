# FoodBinaryClassification
few_shot_learning_in_binary_classicfication

## 问题描述
> 解决个别标签所属数据集较小的情况下 如何对该数据集不同状态进行分类的神经网络设计
## 问题分析
> 将小数据集可以作为分类依据的一个问题是，在没有设计标签的情况下，如何使神经网络上具有明显区分不同输入的能力，本工程使用的仅为二分类，可以扩充为多分类（没熟/熟了/烂了/霉了）
## 算法分析
> 改进自 Prototypical Networks for Few-shot Learning（Jake Snell et. al.）中的算法 以适应二分类问题 并加快效率


**Input**: 60 normal fruits (of different kinds, apple, pear, e.g.) denoted as a set D_1, and 60 rotten fruits (within the former fruits kinds) denoted as a set D_2. 

**Output**: an updated NN, denoted by a function f. the loss J ( to judge whether the training is enough)

**Begin**:

	1. For k in {1,2}: #actually, 1 means normal set X, 2 means rotten set Y 

	2. S_k -< Randomsample(D_k) # get the n ways support examples( including normal and rotten)

	3. Q_k-< Randomsample(D_k/S_k) # get the n ways query examples( including normal and rotten) but different from S_k
	4. c_k -< mean(f(x) for x in S_k) #average all samples in S_k to obtain a common feature vector denoted as c_k 
	5.	Q_distance_1 = distance.pow(c_1, Q_1)
		Q_distance_2 = distance.pow(c_2, Q_2)
	6. P_Q_1 = Q_distance_1/( Q_distance_1+ Q_distance_2)
	   P_Q_2 = Q_distance_2/( Q_distance_1+ Q_distance_2)
	7. J-<0
	8.	If k = 1:
		J-<J+( P_Q_1- P_Q_2)
		If k = 2:
		J-<J+( P_Q_2- P_Q_1)
	End for
	Opitimize J

**End**
>After enough episodes, the NN is supposed to detect whether the fruits is rotten. Then: A new kind of fruits comes, with one pair of images labeled with normal and rotten already, denoted as x1 and x2,respectively.
C1 = f(x1)
C2 = f(x2) # as the f is assumed to be trained to detect the rotten fruits, thus we believe it can separate c1 and c2 without extra training.

**Judge new fruits:**

```
Begin:

C = f(x)

If ||c-c1|| > ||c-c2||:

Output the fruits is normal

Else:

Output the fruits is rotten.

```

## Contribution
> although without sufficient training, out NN can quickly separate the new kind fruit via the few-shot learning efficiently.

> 与原始论文不同，在固定分类状态个数的问题中，我们的n ways 并不是起到标签作用，而是起到增加计算效率以及类似于momentum的效果，理论上来说，当n=训练集全部类 时，few-shot-learning的效率最高 梯度下降方向最具有代表性，但是受制于硬件，我们只采用n=1-10。相较于标准的few-shot-learning模型，我们的模型收敛于更为固定的分类点，同时拉大同类水果不同状态的特征。


##  程序设计
> 环境: 
win10 python 3.6 pytorch 0.4.1 cpu: intel 8770H gpu: nivida GTX1060

数据集: 

train:果蔬及各类食物120类（腐烂60类 成熟60类）

dev/test: 果蔬及各类食物30类（腐烂15类 成熟15类）

每种类型大于30张图片

超参数设置：见parser_new 

epoch:50 iteration:50

代码逻辑：

My_Dataset:取数据 打标签 resize(3,100,100) 继承自torch.utils.data.Dataset

My_Sampler:采样数据 继承自torch.utils.data.Sampler

My_DataLoader:传输数据

Protonet:神经网络 四层卷积+池化 最终fc 64维特征向量 继承自torch.nn.modules


My_Loss:自定义的损失函数 继承自torch.nn.modules

Model_val:利用测试集 验证神经网络性能


## 实验及性能结果
> 对于5组测试集类(5成熟 5腐败)，训练集上在验证集上取得的最好结果，在每组15个query 的情况下， 分类正确个数在（118-127）/150 平均正确率为80-85个点

