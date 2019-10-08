### MTCNN 人脸检测网络中的 PNet 网络

MTCNN 是由 PNet, RNet, ONet 三部分组成的. 我的参考链接中, 他是将三个网络整合到一起的, 这里我为了学习, 
将 PNet 单独拿出来. 一个一个地学习. 

**参考链接:**  
```angular2html
https://github.com/LeslieZhoa/tensorflow-MTCNN 
```

**生成数据:**  
```angular2html
https://github.com/LeslieZhoa/tensorflow-MTCNN/tree/master/preprocess 
``` 
 
**模型:**   
```angular2html
https://github.com/LeslieZhoa/tensorflow-MTCNN/blob/master/train/model.py
```


##### 数据集下载地址
```angular2html
http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/ 
http://shuoyang1213.me/WIDERFACE/
```


数据集分 4 个 .zip 文件.
```angular2html
wider_face_split.zip
WIDER_test.zip
WIDER_train.zip
WIDER_val.zip
``` 

其中
```angular2html
WIDER_test.zip
WIDER_train.zip
WIDER_val.zip
```

三个压缩包中都存放着图片. 
```angular2html
wider_face_split.zip
```

中包含我们需要的 4 个 txt 文件: 
```angular2html
readme.txt
wider_face_train_bbx_gt.txt
wider_face_val_bbx_gt.txt
wider_face_test_filelist.txt
```


其将图片按 4, 1, 5 分为了训练集, 验证集和测试集. 
```angular2html
wider_face_train_bbx_gt.txt
wider_face_val_bbx_gt.txt
```

训练集和验证集都是按: 图片相对路径, 人脸个数, 人脸标注.   
其中人脸标注有 10 项数字, 前四个是人脸位置的 bounding box 的位置
(x, y, w, h). 后 6 项数字是 readme.txt 中对图片的其它描述. 这里我们不需要. 
```angular2html
wider_face_test_filelist.txt
```  
测试集中只有图片的路径, 没有标注. 

你可以用以下代码对图片标注进行测试观察: 
```python
"""
在 wider_face_train_bbx_gt.txt 文件中找到一个图片链接, 并复制其标注信息. 如: 
image_path = 'project_example/mtcnn_face_detect_pnet/dataset/WIDER_train/0--Parade/0_Parade_marchingband_1_849.jpg'
# 449 330 122 149 0 0 0 0 0 0
"""
import numpy as np
import cv2 as cv
  
  
def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return
  
  
image_path = 'project_example/mtcnn_face_detect_pnet/dataset/WIDER_train/0--Parade/0_Parade_marchingband_1_849.jpg'
  
image = cv.imread(image_path)
print(image.shape)
  
# 449 330 122 149 0 0 0 0 0 0
cv.cv2.rectangle(image, (449, 330), (449+122, 330+149), (0, 255, 0), 2)
  
show_image(image)
```


##### PNet 网络
我们的数据集中包含的是图片和其中人脸位置的 bounding box 标注.   
PNet 要求输入数据是 (None, 12, 12, 3).   
也就是说, 需要将原图像切割成 12*12 的大小 ROI 子图, 根据 bounding box 标注, 计算这个子图中是否包含人脸. 
做一个分类训练, 分类结果为: pos, neg, part. 三种, 即: 全部为人脸部分, 没有人脸部分, 部分为人脸部分. 
如果子图与 bounding box 重叠部分小于 0.3 为 neg, 小于等于 0.7 为 part, 大于0.7 为 pos. 

