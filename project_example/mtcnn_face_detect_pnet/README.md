### MTCNN 人脸检测网络中的 PNet 网络

MTCNN 是由 PNet, RNet, ONet 三部分组成的. 我的参考链接中, 他是将三个网络整合到一起的, 这里我为了学习, 
将 PNet 单独拿出来. 一个一个地学习. 

**参考链接:**  
https://github.com/LeslieZhoa/tensorflow-MTCNN  
**生成数据:**   
https://github.com/LeslieZhoa/tensorflow-MTCNN/tree/master/preprocess  
**模型:**   
https://github.com/LeslieZhoa/tensorflow-MTCNN/blob/master/train/model.py

##### PNet 网络
我们的数据集中包含的是图片和其中人脸位置的 bounding box 标注.   
PNet 要求输入数据是 (None, 12, 12, 3).   
也就是说, 需要将原图像切割成 12*12 的大小 ROI 子图, 根据 bounding box 标注, 计算这个子图中是否包含人脸. 
做一个分类训练, 分类结果为: pos, neg, part. 三种, 即: 全部为人脸部分, 没有人脸部分, 部分为人脸部分. 


