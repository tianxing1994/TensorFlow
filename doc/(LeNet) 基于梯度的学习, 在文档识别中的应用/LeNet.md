## (LeNet) 基于梯度的学习, 在文档识别中的应用. 

Yann LeCun, Leon Bottou, Yoshua Bengio, and Patrick Haffner



http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf



### 摘要

使用反向传播算法训练的多层神经网络构成了成功的基于梯度学习技术的最佳示例. 给定适当的网络体系结构, 可以使用基于梯度的学习算法来合成复杂的决策面, 该决策面可以以最少的预处理对诸如手写字符之类的高维模式进行分类. 本文回顾了应用于手写字符识别的各种方法, 并将它们与标准的手写数字识别任务进行了比较. 卷积神经网络专门设计用于处理 2D 形状的可变性, 其性能优于所有其他技术. 

现实生活中的文档识别系统由多个模块组成, 包括字段提取, 分段, 识别和语言建模. 一种新的学习范例, 称为图形变换网络 (GTN), 允许使用基于梯度的方法对此类多模块系统进行分局训练, 以最大程度地降低总体性能指标. 

描述了两种用于在线手写识别的系统. 实验证明了全局训练的优势以及图形转换网络 (GTN) 的灵活性. 

还描述了用于读取银行支票的图形转换网络. 它使用卷积神经网络字符识别器与全局训练技术相结合, 以提供商务和个人支票的记录准确性. 它已商业化部署, 每天读取数百万张支票. 

字键字: Neural Networks, OCR, Document Recognition, Machine Learning, Gradient-Based Learning, Convolutional Neural Networks, Graph Transformer Networks, Finite State Transducers. 



命名: 

* GT Graph transformer. 
* GTN Graph transformer network. 
* HMM Hidden Markov model. 
* HOS Heuristic oversegmentation. 
* K-NN K-nearest neighbor. 
* NN Neural network. 
* OCR Optical character recognition. 
* PCA Principal component analysis. 
* RBF Radial basis function. 
* RS-SVM Reduced-set support vector method. 
* SDNN Space displacement neural network. 
* SVM Support vector method. 
* TDNN Time delay neural network. 
* V-SVM Virtual support vector method. 



作者与语音和图像处理服务研究实验室 AT＆T Labs {yann，leonb，yoshua，haffner {@ research.att.com 在一起. Yoshua Bengio 还是蒙特利尔大学 dInformatique et de Recherche Opérationelle 系的学生 (C.P. 6128 Succ). Center-Ville，2920 Chemin de la Tour, 蒙特利尔, 魁北克, 加拿大H3C 3J7. 



### 介绍

在过去的几年中, 机器学习技术, 尤其是应用于神经网络的机器学习技术, 在模式识别系统的设计中起着越来越重要的作用. 实际上, 可以说, 学习技术的可用性一直是模式识别应用 (连续语音识别和手写识别) 近来成功关键因素. 

本论文的主要信息是, 可以通过更多地依赖自动学习而不是手工设计的启发式方法来构建更好的模式识别系统. 机器学习计算机技术的最新进展使这成为可能. 通过使用字符识别作为案例研究, 我们表明, 可以通过精心设计的直接在像素图像上运行的学习机器来代替手工制作的特征提取. 通过使用文档理解作为案例研究, 我们表明, 通过手工集成单独设计的模块来构建识别系统的传统方式可以被称为 Graph Transformer Networks 的统一且原则明确的设计范例所代替, 该范例可以训练所有模块来优化全局性能标准. 

从模式识别的早期开始, 人们就知道自然数据 (无论是语音, 字形还是其他类型的模式) 的可变性和实用性使得几乎不可能完全手动建立准确的识别系统. 因此, 大多数模式识别系统都是使用自动学习技术和手工算法组合而成的. 识别单个模式的常用方法是将系统分为两个主要模块, 如图 1 所示. 第一个模块称为特征提取器, 对输入模式进行转换, 以使它们可以由低维向量或短符号串表示, 它们 (a) 可以轻松匹配或比较, 并且 (b) 相对而言是变体, 不改变其性质的输入模式的变换和变形. 特征提取器包含大多数先验知识, 并且非常特定于任务. 它也是大多数设计工作的重点, 因为它通常是完全手工制作的. 另一方面, 分类器通常是通用且可训练的. 这种方法的主要问题之一是识别精度在很大程序上取决于设计人员提出适当的特征集的能力. 事实证明, 这是一项艰巨的任务, 不幸的是, 每个新问题都必须重做. 大量的模式识别文献致力于描述和比较针对特定任务的不同特征集的相对优点. 

从历史上看, 对适当的特征提取器的需求是由于以下事实: 分类器使用的学习技术仅限于具有易于分离类的低维空间. 在过去十年中, 三个因素的结合改变了这一愿景. 首先, 具有快速算术算元的低成本机器的可用性允许更多地依靠蛮力 "数学" 方法, 而不是依靠算法改进. 其次, 大型数据库可解决诸如手写识别之类的具有广阔市场和广泛关注的问题, 这使设计人员能够更多地依赖真实数据, 而较少依赖手工特征提取来构建识别系统. 第三个也是非常重要的因素是强大的机器学习技术的可用性, 该技术可以处理高维输入, 并在喂入这些数据集时可以生成复杂的决策功能. 可以说, 语音和手写识别系统在准确性方面的最新进展在很大程序上可以归因于对学习技术和大量训练数据集的日益依赖. 为此, 很多现代商业 OCR 系统使用经过反向传播训练的某种形式的多层神经网络. 





in this study, we consider the tasks of handwritten character recognition (Sections I and II) and compare the performance of several learning techniques on a benchmark data set for handwritten digit recognition (Section III). 











