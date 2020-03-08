## cifar10 数据集识别. 
学习理解了 resnet 之后, 根据参考链接写出了一个 resnet 32 层网络, 对 cifar10 数据进行了分类识别. 

在 training.py 文件中训练模型, 因为我没有设置 validation 数据集, 最终的结果是在训练集上的表现达到了 100%. (在我的 CPU 8G 内存电脑上训练了 24 小时). 

在 demo.py 文件上, 用训练好的模型运行了一下测试集, 精确度只有 68% 左右. 

主要还是为了学习 resnet 的网络结构. 同时其中用函数封装网络的方式很不错, 方便复用. 

**参考链接:**
```text
https://blog.csdn.net/gzroy/article/details/82386540
https://www.jianshu.com/p/23c73b90657f
https://blog.csdn.net/u013841196/article/details/80713314
```

模型链接: 
```text
链接：https://pan.baidu.com/s/198sYkuJno1zbhWYHFsBEdg 
提取码：h1us
```

数据集 cifar-10 网上能搜到, 分 python, matlab, c 语言, 三种版本. 我用的是 python 版本. 在参考链接中, 它用 tensorflow 去加载 c 版本. 
数据集 python 版本: 
```text
链接：https://pan.baidu.com/s/1AlX7k4p8dtvEPCdqmvaiRg 
提取码：euk6
```
