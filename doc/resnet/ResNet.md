## ResNet (深度残差学习用于图像识别)

https://arxiv.org/pdf/1512.03385.pdf



### 摘要

越深的神经网络则越难训练. 我们提出了一种简化训练的残差学习框架, 其实际网络深度比先前的都深. 我们显式地将层重新设置为根据指定的输入层学习残差函数, 而不是学习未指定的函数. 我们提供了全面的经验证据, 表明这些残差网络更易于优化, 并且可以通过大大增加的深度来获得准确性. 在 ImageNet 数据集上, 我们评估深度最大为 152 层的残差网络 - 比 VGG 网络深 8 位, 但复杂性仍然较低. 这些残差网络整体上在 ImageNet 测试集上实现了 3.57% 的误差. 该结果在 ILSVRC 2015 分类任务中获得第一名. 我们还介绍了具有 100 和 1000 层的 CIFAR-10 分析. 

表示的深度对于许多视觉识别任务至关重要. 仅由于我们极深表示, 我们在 COCO 对象检测数据集上获得了 28% 的相对改进. 深度残差网络是我们提交 ILSVRC 和 COCO 2015 竞赛1 的基础, 在该竞赛中, 我们还获得了 ImageNet 检测, ImageNet 本地化, COCO 检测和 COCO 分割等任务的第一名. 



### 介绍

深度卷积神经网络导致了图像分类的一系列突破. 深度网络自然地以端到端的多层方式集成了低/中/高级特征和分类器, 并且特征的"级别"可以通过堆叠的层数(深度)来丰富. 最新证据揭示了网络深度至关重要, 在具有挑战性的 ImageNet 数据集上的领先结果都利用了 "非常深" 的模型, 深度为 16 到 30. 许多其他非平凡的视觉识别任务也从非常深入的模型中受益匪浅. 在深度意义的驱动下, 出现了一个问题: 学习更好的网络是否像堆叠更多的层一样容易 ? 回答这个问题的障碍是臭名昭著的梯度消失/爆炸, 从一开始就阻碍了收敛. 但是这个问题已经通初始归一化和中间归一化层在很大程度上解决了, 它使具有数十层的网络能够通过反向传播开始收敛用于随机梯度下降 (SGD). 

当更深层的网络能够开始聚合时, 就会出现降级问题: 随着网络深度的增加, 精度达到饱和 (这可能不足为奇), 然后迅速降级. 出乎意料的是, 这种降级不是由过拟合引起的, 将更多的层添加到适当深度的模型中会导致更高的训练误差, 并通过我们的实验进行了全面验证, 图 1 显示了一个典型示例. 

训练准确性的下降表明并非所有系统都同样容易优化. 让我们考虑一个较浅的体系结构, 以及一个较深的体系结构, 它在其上添加了更多层. 通过构建更深层的模型可以找到解决方案: 添加的层是身份映射, 其他层是从学习的浅层模型中复制的. 此构造解决方案的存在表明, 较深的模型不会比浅的模型产生更高的训练误差. 但实验表明, 我们现有的求解器无法找到比构造的解决方案好或更好的解决方案 (或在可行时间内无法找到). 

在此论文中, 我们通过深度残差学习框架解决此降级问题. 我们明确让这些层适合残差映射, 而不是希望每个堆叠的层都直接适合所需的基础映射. 形式上, 将所需的基础映射表示为 H(x), 我们让堆叠的非线性层拟合 F(x)=H(x)-x, 则原始的映射将变为 F(x)+x. 我们假设优化残差映射比优化原始未引用映射要容易. 极端地, 如果恒等映射是最佳的, 则将残差迭代到零比通过非线性层的堆叠拟合恒等映射要容易. 

F(x)+x 的公式可通过具有"快捷连接"的前馈神经网络实现. 快捷连接是跳过一层或多层的连接. 在我们的例子中, 快捷连接仅执行恒等映射, 并将其输出添加到堆叠的输出中 (图2). 恒等快捷连接即不会增加额外的参数, 也不会增加计算复杂性. 整个网络仍然可以通过 SGD 反向传播进行端到端训练, 并且可以使用通用库轻松实现, 而无需修改求解器. 我们在 ImageNet 上进行了全面的实验, 以显示退化问题并评估我们的方法. 我们证明: 1) 我们极深的残差网络易于优化, 但是当深度增加时, 对应的 "普通" 网络 (简单地堆叠层) 显示出更高的训练误差, 2) 我们的深层残差网络可以通过深度的增加而轻松地提高准确性, 从而产生比以前的网络更好的结果. 在 CIFAR-10 集上也显示了类似的现象, 这表明优化困难和我们方法的效果不仅限于特定的数据集. 我们在此数据集上展示了经过成功训练的 100 层以上的模型, 并探索了 1000 层以上的模型. 

在 ImageNet 分类数据集上, 我们通过极深的残差网络获得了出色的结果. 我们的 152 层残差网络是 ImageNet 上提出的最深的网络, 同时其复杂度仍低于 VGG 网络. 我们在 ImageNet 测试集上的有 3.57% 到 5 的错误率在 ILSVRC 2015 分类竞赛中获得第一名. 极深的表现形式在其他识别任务上也具有出色的泛化性能, 使我们在 ILSVRC 和 COCO 2015 竞赛中进一步赢得了第一名: ImageNet 检测, ImageNet 本地化, COCO 检测和 COCO 分割. 有力的证据表明, 残差学习原理是通用的, 我们希望它适用于其他视觉和非视觉问题. 



### 相关工作

**残差表示**. 在图像识别中, VLAD 是通过相对于字典的残差矢量进行编码的表示, Fisher Vector 可公式化为 VLAD 的概率版本. 它们都是用于图像检索和分类的有力的浅层表示. 对于矢量量化, 编码残差矢量比编码原始矢量更有效. 在低级视觉和计算机图形学中, 为了求解偏微分方程 (PDE), 广泛使用的 Multigrid 方法将系统重新构建为多个尺度的子问题, 其中每个子问题负责较粗和较细的剩余解. Multigrid 的替代方法是分层基础预处理, 它依赖于表示两个尺度之间残差矢量的变量. 已经显示, 这些求解器的收敛速度比不知道解决方案剩余性质的标准求解器快得多. 这些方法表明, 良好的重构或预处理可以简化优化过程. 

**快捷连接**. 导致快捷连接的实践和理论已经研究了很长时间. 训练多层感知器(MLP) 的早期实践是添加从网络输入连接到输出的线性层. 在 [44, 24] 中, 一些中间层直接连接到辅助分类器, 以解决消失/爆炸梯度. [39, 38, 31, 47] 的论文提出了通过快捷连接实现居中层响应, 梯度和传播误差居中的方法. 在 [44] 中, "起始" 层由一个快捷分支和一些更深的分支组成. 

我们工作的同时, "高速公路网络" 提供了具有选通函数的快捷连接. 与我们的不带参数的身份快捷方式相反, 这些门取决于数据关具有参数. 当封闭的快捷方式 "关闭" (接近零) 时, 公路网络中的图层表示非残留功能. 相反, 我们的公式总是学习残差函数. 我们的恒等快捷连接永远不会被关闭, 所有信息始终都会通过传递, 还有其他剩余功能需要学习. 另外, 高速公路网络还没有显示出深度极大增加 (例如超过 100 层) 的精度. 



### 深度残差学习

#### 残差学习























