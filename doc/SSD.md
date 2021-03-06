
## SSD
<https://arxiv.org/pdf/1512.02325.pdf>


参考链接:
<https://blog.csdn.net/zhangjunp3/article/details/80597312>



### 摘要

我们提出了一种使用单个深度神经网络检测图像中的对象的方法. 我们的实现称为 SSD, 其将边界框的输出空间离散化为一组默认框, 这些默认框具有不同的长宽比和每个要素图位置的比例. 在预测时, 网络会为每个默认框中的每个对象类别的存在生成分数, 并对该框进行调整以更好地匹配对象形状. 此外, 该网络将来自具有不同分辨率的多个特征图的预测进行组合, 以自然地处理各种大小的对象. 相对于需要对象提案的方法, SSD 简单, 因为它完全消除了建议生成和后续的像素或特征重采样阶段, 并将所有计算封装在一个网络中. 这使得 SSD 易于训练, 并且可以直接集成到需要检测组件的系统中. 在 PASCAL VOC, COCO 和 ILSVRC 数据集上的实验结果证实, SSD 与采用额外对象提案步骤的方法相比具有有竞争力的精确度, 并且速度更快, 而为训练和推理提供统一的框架. 对于 300×300 的输入, 在Nvidia Titan X上以59 FPS的速度进行 VOC2007 测试时 SSD 有 74.3% 的 mAP. 对于 512×512 输入, SSD 达到76.9％ 的 mAP 优于同类最新的 Faster R-CNN 模型. 与其他单步方法相比, 即使输入图像尺寸比较小, SSD 的精度也要好得多. 代码在: https://github.com/weiliu89/caffe/tree/ssd



### 1. 介绍
当前最新的对象检测系统是以下方法的变体: 假设边界框, 为每个框重新采样像素或特征, 并应用高质量的分类器. 自从选择性搜索以来, 该管道一直在检测基准上占主导地位, 尽管 PASCAL VOC, COCO 和 ILSVRC 检测的当前领先结果全部基于 Faster R-CNN, 尽管具有更深的特征, 例如[3]. 尽管这些方法准确无误, 但对于嵌入式系统而言, 它们的计算量太大, 即使高端硬件, 对于实时应用程序也太慢. 这些方法的检测速度通道以每秒每帧 (SPF) 来衡量, 即使是最快的高精度检测器 Faster R-CNN, 也只能以每秒 7 帧 (FPS) 的速度运行. 已经有很多尝试通过攻击检测管道的每个阶段来构建更快的检测器, 但是到目前为止, 速度的显著提高仅是以大大降低检测精度为代价的.

本文介绍了第一个基于深度网络的对象检测器, 该对象检测器不对边界框假设的像素或特征进行重采样, 并且与执行此操作的方法一样准确. 这样可以显着提高高精度检测的速度 (VOC2007 测试中 59 mPS 的 mAP为 74.3％, 而 Faster R-CNN 7 FPS 的 mAP 为 73.2％ 或 YOLO 45 FPS 的 mAP 为 63.4％). 速度的根本提高来自取消边界框建议和后续的像素或特征重采样阶段. 我们不是第一个这样做的人, 但是通过添加一系列改进, 我们比以前的尝试显着提高了准确性. 我们的改进包括使用小型卷积核来预测对象类别和边界框位置中的偏移量, 使用单独的预测器进行不同的宽高比检测, 以及将这些过滤器应用于网络后期的多特征图, 以便执行多尺度检测. 通过这些修改 (尤其是使用多层进行不同规模的预测), 我们可以使用相对较低的分辨率输入来实现高精度, 从而进一步提高检测速度.

尽管这些贡献似乎很小, 但我们注意到, 结果系统将 PASCAL VOC 的实时检测精度从 YOLO 的 63.4％mAP 提高到了 SSD 的 74.3％ mAP. 与最近关于残差网络的非常引人注目的工作相比, 这在检测精度上有较大的相对提高. 此外, 显着提高的高质量检测速度可以拓宽使用计算机视觉的范围.

我们将我们的贡献总结如下:

* 我们推出了SSD, 这是一种适用于多种类别的单步检测器, 其速度比以前的最新单步检测器 (YOLO) 更快, 并且精度更高, 实际上与执行区域提案的速度较慢的技术一样准确 (包括 Faster R-CNN).

* SSD 的核心是使用应用于特征图的小型卷积滤波器来预测一组固定的默认边界框的类别得分和框偏移.

* 为了获得较高的检测精度, 我们从不同比例的特征图生成不同比例的预测, 并按宽高比明确地分离预测.

  这些设计特征即使在低分辨率输入图像上也可以实现简单的端到端训练和高精度, 从而进一步提高了速度与精度之间的权衡.* 实验包括在 PASCAL VOC, COCO 和 ILSVRC 上评估的具有可变输入大小的模型的时序和精度分析, 并将其与一系列最新技术进行了比较.



### 2. 单步检测 (SSD)

本节介绍了我们建议的 SSD 检测框架和相关的训练方法. 之后, 呈现特定于数据集的模型详细信息和实验结果.



#### 2.1 模型

SSD 方法基于前馈卷积网络, 该网络会生成固定大小的边界框集合, 并为这些框中存在的对象类实例打分, 然后进行非最大抑制步骤以产生最终检测结果. 早期的网络层基于用于高质量图像分类的标准体系结构 (在任何分类层之前均被截断), 我们将其称为基础网络. 然后, 我们将辅助结构添加到网络, 以产生具有以下关键特征的检测结果:

**用于检测的多尺度特征图**: 我们将卷积特征层添加到截断的基础网络的末尾. 这些层的大小逐渐减小, 并可以预测多个尺度的检测. 每个特征层用于预测检测的卷积模型是不同的 (参见在单个比例尺特征图上运行的 Overfeat 和 YOLO).

**卷积预测器用于检测**: 每个添加的特征层 (或可选地, 来自基础网络的现有特征层) 可以使用一组卷积滤波器来生成一组固定的检测预测. 这些在图 2 中的 SSD 网络体系结构之上显示. 对于具有 p 个通道的大小为 m×n 的特征层, 用于预测参数的基本元素是 3×3×p 的小卷积核, 该核产生类别的得分或相对于默认框的形状偏移坐标. 在 m×n 的每一个位置, 它都会产生一个输出值. 边界框偏移输出值是相对于每个特征图上位置的默认框位置进行测量的 (参见YOLO 的体系结构, 此步骤使用中间完全连接层而不是卷积滤波器).

**默认框和宽高比**: 我们将一组默认边界框与每个要素图单元相关联, 以用于网络顶部的多个特征图. 默认框以卷积方式平铺突然方法征图, 因此每个框相对于其对应单元格的位置是固定的. 每个特征图单元, 我们预测其相对于该单元中默认框形状的偏移量, 以及每个类的分数, 这些分数指示哪些框中存在类实例. 具体来说, 对于给定位置的 k 个盒子, 我们计算 c 类得分和相对于原始默认盒子形状的 4 个偏移量. 这导致总共 (c+4)k 个滤波应用在特征图中的每个位置周围, 从而生成 m×n 的特征图有 $(c+4) \times k \times m \times n$ 个输出值. 有关默认框的说明, 参阅图 1. 我们的默认框与 Faster R-CNN 中使用的锚框相似, 但是我们将它们应用于多个不同分辨率的特征图. 在多个特征图中允许使用不同的默认盒子形状, 可以使我们有效地离散可能的输出盒子形状的空间.



#### 2.2 训练

训练 SSD 和训练使用区域提案的典型控测器之间的玉要区别在于, 需要将目标区域信息分配给探测器输出的固定集合中的特定输出. YOLO 中的训练以及 Faster R-CNN 和 MultiBox 的区域提案阶段也需要一些这种形式. 确定此分配后, 就可以对损失函数和反向传播进行端到端应用. 训练还涉及选择一组默认的检测框的尺度进行检测, 以及难例与数据增强策略.

**匹配策略**: 在训练期间, 我们需要确定哪些默认框对应于目标区域并相应地训练网络. 对于每个目标框, 我们从默认框中进行选择, 这些默认框随位置, 宽高比和比例而变化. 我们首先将每个目标框与具有最佳 jaccard 重叠度的默认框进行匹配 (像 MultiBox 中一样). 与 MultiBox 不同的是, 我们随后将默认框与 jaccard 重叠度高于阈值 (0.5) 的任何目标框进行匹配. 这简化了学习问题, 它允许网络为多个具有相当重叠度的默认框预测出高评分, 而不是要求它只对重叠度最大的框预测高分.

**训练目标**: SSD 训练目标是从 MultiBox 目标派生而来的, 但可以扩展为处理多个对象类别. 令: $x_{ij}^{p} = \{1, 0\}$ 为标识第 i 个默认框与第 j 个目标区域的第 p 个类别的匹配指示符. 在上面的匹配策略中, 我们有 $\sum_{i}{x_{ij}^{p}} \ge 1$. 总体的目标损失函数是定位损失 (loc) 和置信度损失 (conf) 的加权和:

$$\begin{aligned} L(x, c, l, g) = \frac{1}{N} (L_{conf}(x, c) + \alpha L_{loc}(x, l, g)) \end{aligned}$$

其中: N 是匹配的默认框的数量.
如果 N=0, 则设置损失为 0. 定位损失是平滑预测框 l 与真实框 g 参数之间的 L1 损失. 与 Faster R-CNN 类似, 我们回归到默认边界框 d 的中心 (cx, cy) 及其宽度 w 和高度 h 的偏移量.

$$\begin{aligned} L_{loc}(x, l, g) = \sum_{i \in Pos}^{N} \sum_{m \in \{cx, cy, w, h\}} {x_{ij}^{k}{smooth_{L1}(l_{i}^{m} - \hat{g}_{j}^{m})}} \end{aligned}$$

* 备注: 对于目标框 j, 计算所有与 j 有足够 IOU 的锚框之间定位的 "平滑 L1 损失" 之和.



$$\begin{aligned} \hat{g}_{j}^{cx} = (g_{j}^{cx} - d_{i}^{cx}) / d_{i}^{w} \end{aligned}$$
$$\begin{aligned} \hat{g}_{j}^{cy} = (g_{j}^{cy} - d_{i}^{cy}) / d_{i}^{h} \end{aligned}$$

* 备注: 预测的关于 $\hat{g}_{j}^{cx}$ 的偏移量, 应为真实 $g_{j}^{cx}$ 与锚点框 $d_{i}^{cx}$ 之差, 再除以锚点框的宽度 $d_{i}^{w}$. 看起来, 这样可以使所有大小不同的锚点框都预测出相似范围的值, 应该更加易于训练.

  

  $$\begin{aligned} \hat{g}_{j}^{w} = log(\frac{g_{j}^{w}}{d_{i}^{w}}) \end{aligned}$$
  $$\begin{aligned} \hat{g}_{j}^{h} = log(\frac{g_{j}^{h}}{d_{i}^{h}}) \end{aligned}$$

* 备注: 对宽度的预测值 $\hat{g}_{j}^{w}$ 为 真实框 $g_{j}^{w}$ 宽度与锚点框 $d_{i}^{cx}$ 宽长之比, $e^{\hat{g}_{j}^{w}} = \frac{g_{j}^{h}}{d_{i}^{h}}$容易理解, 但为什么要加一个 $log$ 对数呢. 网上说, 是为了确保长度的缩放值必须大于 0. 但是也注意到这种写法好像更容易从锚点框的尺寸放大, 而不是缩小 ($\hat{g}_{j}^{w}$ 大于 0 时, 只需要增加一点, 就可以将从 $e^{\hat{g}_{j}^{w}} $ 的值放得很大, 而需要减小很多, 才能将其缩小).

  

置信度损失是多个类别置信度 (c) 上的 softmax 损失.

$$\begin{aligned} L_{conf}(x, c) = -\sum_{i \in Pos}^{N}{x_{ij}^{p}{log(\hat{c}_{i}^{p})}} - \sum_{i \in Neg} {log(\hat{c}_{i}^{0})} \end{aligned}$$
$$\begin{aligned} \hat{c}_{i}^{p} = \frac{exp(c_{i}^{p})}{\sum_{p} exp(c_{i}^{p})} \end{aligned}$$

并通过交叉验证将权重项 $\alpha$ 设置为 1.

* 备注:

* 类别损失 Pos 部分, 计算交叉熵, 但最后只取有足够 IOU 的默认框部分.

* 将类别只看作正样本和负样本(背景). 先取出真实值为负样本的数据中被预测出负样本的概率较大的 k 个数据, 使得 使负样本与正样本的比例为 3:1. 然后只对这里面的负样本计算其交叉熵作为 Neg 部分.

* softmax. 考虑到在任何情况下, 不管一个样本属于某一类别的置信度得分有多么低, 但它仍然有可能属于那个类别, 所以当我们为其预测一个 $(- \infty, +\infty)$ 之间的 $p$ 值时, 先用 $e^{p}$ 将这个置信度值映射到 $(0, +\infty)$, 之后用其属于其它类别的置信度计算出它属于各个类别的概率. 如: $$\begin{aligned} \hat{c}_{i}^{p} = \frac{exp(c_{i}^{p})}{\sum_{p} exp(c_{i}^{p})} \end{aligned}$$.* 交叉熵: $H(p, q) = \sum_{x}{p(x) \cdot log(\frac{1}{q(x)})}$. $p(x)$ 表示真实分布, $q(x)$ 表示预测分布.

  

```python
# 我猜测:
# 对于特征图, 先根据图像标注的 ground true box,
# 生成与指定特征图相匹配的对每一个锚点框的分类(one_hot),
# 根据 ground_true_box 与每个锚点的位置和宽高计算 location 回归的目标值. 生成如下 n2 的数组.
# 计算 location 损失, 先通过 n2 确定目标的锚点框, 并从 n1, n2 索引这部份内容, 只对这些数据计算损失.

# 计算 classify 损失, 可能是如下:
# 类别损失的 Pos 部分, 先计算交叉熵, 但最后只取有足够 IOU 的默认框部分.
# 类别损失的 Neg 部分, 考虑到负样本(背景)远比正样本多, 先从 n1 将真实值为负样本的数据提取出,
# 再根据其对负样本的置信度评分, 将概率较大的 k 个样本选出, 使负样本与正样本的比例为 3:1.
# 然后只对这里面的负样本计算其交叉熵作为 Neg 部分.
import numpy as np

n1 = np.array([[0, 1, 7, 3, 2, 0, 1, 11, 22, 33, 44],
[7, 1, 2, 1, 0, 0, 3, 11, 22, 33, 44],
[7, 3, 2, 1, 1, 0, 1, 11, 22, 33, 44],
[3, 1, 2, 7, 1, 0, 2, 11, 22, 33, 44],
[1, 1, 2, 7, 3, 0, 7, 11, 22, 33, 44],
[2, 7, 2, 1, 0, 2, 3, 11, 22, 33, 44]])

n2 = np.array([[0, 0, 0, 0, 0, 0, 1, -1, -1, -1, -1],
[0, 0, 0, 0, 0, 0, 1, -1, -1, -1, -1],
[0, 0, 0, 0, 0, 0, 1, -1, -1, -1, -1],
[0, 1, 0, 0, 0, 0, 0, 12, 12, 12, 12],
[0, 0, 0, 1, 0, 0, 0, 23, 23, 23, 23],
[0, 0, 0, 0, 0, 0, 1, -1, -1, -1, -1]])
```



**选择默认框的比例和宽高比**: 为了处理不同的对象比例, 一些方法建议以不同的尺寸处理图像, 然后将结果合并. 但是, 通过利用单个网络中不同层的特征图进行预测, 我们可以模拟相同的效果, 同时还可以在所有对象比例尺上共享参数. 先前的工作已经表明, 使用较低层的特征图可以提高语义分割的质量, 因为较低层可以捕获输入对象的更多详细信息. 同样, [12] 显示, 添加从特征图中合并的全局上下文可以帮助平滑分割结果. 受到这些方法的启发, 我们同时使用上下特征图进行检测. 图 1 显示了框架中使用的两个示例性特征图 (8 × 8 和 4 × 4). 在实践中, 我们可以以较小的计算开销使用更多. 已知网络中来自不同级别的特征图具有不同的接受域大小. 幸运的是, 在 SSD 框架内, 默认框不需要与每个层的实际接收字段相对应. 我们设计默认框的拼贴, 以便特定的特征图学会对特定比例的对象做出响应. 假设我们要使用 m 个特征图进行预测. 每个特征图的默认框的比例计算如下:

$$\begin{aligned} s_{k} = s_{min} + \frac{s_{max} - s_{min}}{m - 1}(k-1), k \in [1, m] \end{aligned}$$

其中: $s_{k}$ 表示单边的长度比例, $s_{min}$ 是 0.2, $s_{max}$ 是 0.9, 意味着最底层的尺度是 0.2, 最上层的尺度是 0.9, 且介于两者之间的所有层都按规律间隔. 我们为默认框设置不同的宽高比, 并将其表示为 $a_{r} \in \{1, 2, 3, \frac{1}{2}, \frac{1}{3}\}$. 我们可以计算每个默认框的宽度 ($w_{k}^{a} = s_{k}\sqrt{a_{r}}$) 和高度 ($h_{k}^{a} = s_{k} / \sqrt{a_{r}}$). 对于宽高比为 1, 我们额外添加了一个默认框, 其尺度为 $s_{k}^{'} = \sqrt{s_{k}s_{k+1}}$ (备注: 为什么不取平均呢 ?), 这样, 特征图的每个位置有 6 个默认框. 每个默认框的中心为 $(\frac{i + 0.5}{|f_{k}|}, \frac{j + 0.5}{|f_{k}|})$, 其中 $|f_{k}|$ 是第 k 个方形特征图的大小, $i, j \in [0, |f_{k}|)$. 实际上, 还可以设计默认框的分布以最适合特定的数据集. 如何设计最佳平铺也是一个开放性的问题.

通过将来自许多特征图的所有位置的具有不同比例和长宽比的所有默认框预测组合在一起, 我们获得了一组多样化的预测, 涵盖了各种输入对象的大小和形状. 例如, 在图 1 中, 狗与 4×4 特征图中的默认框匹配, 但与 8×8 特征图中的任何默认框都不匹配. 这是因为这些盒子的比例不同并且与狗的盒子不匹配, 因此在训练过程中被视为背景.



* 备注:
* 在整个的目标检测过程中, 认为: 最小目标的面积除以原图面积应为 $s_{min}^{2}$ , 最大的目标则为 $s_{max}^{2}$. 则当我们有 m 个特征图时, 按梯度比例, 就可以计算每一层的比例关系 $s_{k}$. 如上面的公式.
* $s_{k}$ 乘以图像的宽或高, 就是此锚框对应于原图中的大小. 再结合每个默认框的中心坐标, 将其还原到原图中. 得到此默认框在原图中的 $(x, y, w, h)$, 这个结果可用来与 ground true box 的标注进行 IOU 计算. (特征图的每一个像素(坐标)都被看作是一个锚点).
* 我们预先就可以知道: 原图的宽高, 指定特征图的宽高, 因此可以计算指定特征图的标注矩阵 (但是, 特征图之后只是跟随了一个 3×3 卷积, 好像每一个宽高比的锚框在计算时并没有什么差别. 我原本的想像是每一个特征图会被应用不同宽高的卷积核).



**难例挖掘**: 匹配步骤之后, 大多数默认框都是负例, 尤其是当可能的默认框数量很大时. 这在正负样本的训练之间造成了严重的不平衡. 与其使用所有的负例, 我们对每个默认框使用最高的置信度损失对它们进行排序, 然后选择最上面的一些框, 以使**负样本与正样本之间的比率最大为 3: 1**. 我们发现这导致更快的优化和更稳定的训练.



**数据增强**:
为了使模型对各种输入对象的大小和形状更加健壮, 每个训练图像均通过以下选项之一随机采样:

* 使用整个原始输入图像.
* 图像截取采样, 以使与对象的最小 jaccard 重叠为 0.1、0.3、0.5、0.7 或 0.9.
* 随机图像截取采样. 每个图像截取采样的大小是原始图像大小的 [0.1, 1], 长宽比在 0.5 和 2 之间. 如果目标区域的中心在截取区域内部, 我们将保留它的重叠部分. 在上述采样步骤之后, 除了应用某些类似于[14]中所述的光度失真之外, 每个采样的截图都将大小调整为固定大小并以 0.5 的概率水平翻转.



### 3. 实验结果