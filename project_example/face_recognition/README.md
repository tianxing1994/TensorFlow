### 深度学习作人脸识别

参考链接:   
https://blog.csdn.net/codes_first/article/details/79223524  
数据集下载地址:   
https://pan.baidu.com/s/1QlyTAOd-LDC7oiUtS5McnQ  

#### 说明
**load_data.py:** 用于加载图片数据的模块, 因为图片都存在文件夹里, 且如果一下子将 3 万张图片
加载到内存中, 电脑承受不了. 所以先将图片的路径及其对应的类别加载进来, 
形式如: [(image_path, classify), (image_path, classify), ..., (image_path, classify)]
包含元组的列表. 
当需要向模型中批次输入数据时, 再分批地读取当前需要的图片. 

**modeling.py:** 这是神经网络模型的部分, 因为在模型训练, 和重新加载已保存的 ckpt 检查点文件时
, 都需要用到这部分内容. 
原作者的输入数据是 (None, 128, 128, 3), 但是我的电脑好像内存不够, 于是就改成了 (None, 64, 64, 1), 
且卷积层也减少了一层. 

**optimization.py:** 这是模型训练的部分, 训练完的模型被存到 'model/faces.ckpt', 由于文件比较大, 
没有将训练完的模型上传到 GitHub. 模型训练比较慢. 它将整个数据循环输入 10 遍, 大概需要两个小时吧. 
训练的时候, 可能没什么反应. 在电脑的资源管理器看一下, python 占用了比较多的 CPU 资源, 内存资源. 
应该就是在训练了. 

**main.py:** 本来 modeling.py 加 optimization.py 就可以进行模型训练了, 但是我既然已经
将这两部分分开了, 它们应该算是同一个级别的模型. 如果直接在 optimization.py 中启动训练
过程, 感觉不太好. 就单独写了一个 main.py. 


**classification.py:** 分类部分. 我们把模型训练好之后, 保存到 'model/faces.ckpt' 文件. 此模型
加载已训练好的 ckpt 文件. 并支持输入一张图片来执行分类. 
