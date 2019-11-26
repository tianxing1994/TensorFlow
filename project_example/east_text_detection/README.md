### EAST 文本检测模型
支持中文检测, 参考链接:
```angular2html
https://github.com/argman/EAST
``` 

原作者已经训练好了模型并分享了 ckpt 训练参数文件. 我需要将其在本地运行起来. 
运行 demo.py 文件以执行图像文本检测. 

1. 需要将该部分移动到工程根目录, 则可以运行. 
2. Shapely 库, 需要先下载 whl 文件. 
下载路径: 
```angular2html
https://www.lfd.uci.edu/~gohlke/pythonlibs/
```
再用 pip 安装, 下载好 whl 文件后, 将其存放到当前工程对应的环境 site-packages 文件夹下, 
如: 
```angular2html
C:\ProgramData\Anaconda3\envs\TensorFlow\Lib\site-packages
```
激活工程环境,  Terminal 终端切换路径至 whl 文件路径, 执行: 
```angular2html
pip install Shapely-1.6.4.post2-cp36-cp36m-win_amd64.whl
```
以完成安装. 
参考链接: https://www.cnblogs.com/wgy1/p/11137722.html
