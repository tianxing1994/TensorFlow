### 卷积神经网络作回归
通过卷积神经网络作回归任务, 检测图像中的单个目标. 
这里的数据集是通过代码生成的. 图片底为黑色, 有单个物体为任意白色矩形. 

其实就是普通的分类任务的网络, 在损失函数那里改成让它输出 4 个值, 
计算这 4 个值与目标 bounding box 4 个值之间的最小方差和. 

#### 参考链接:
https://www.kaggle.com/soumikrakshit/object-detection-single-rectangle

#### 说明:
因为当前目录原本是一个工程的根目录, 所以若要执行此工程, 须将当前目录移至工程根目录. 
如者在对应的要执行的文件顶部添加环境变量: 
```cython
# 添加环境变量.
import os
import sys
p = os.getcwd()
sys.path.append(p)
```

**load_data.py:** 包含一个 generate_training_set 函数, 用于生成训练数据. 
**modeling.py:** 这是神经网络模型的部分. 
**optimization.py:** 模型训练, 模型保存到: model/box_regression.ckpt. 
**demo.py:** 在 demo 文件中有一个对已经训练完成的模型效果的一个演示. 
