"""
查看 imagenet-vgg-verydeep-19.mat 文件的详细信息.

graph:
https://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.svg

data = scipy.io.loadmat(vgg_model_path)
data:
type: dict. {key: type(value)}
{
    __header__: <class 'bytes'>
    __version__: <class 'str'>
    __globals__: <class 'list'>
    # layers 中保存了每一层的参数. 总共 43 层.
    layers: <class 'numpy.ndarray'>
    classes: <class 'numpy.ndarray'>
    normalization: <class 'numpy.ndarray'>
}

data = {
    __header__: 'MATLAB 5.0 MAT-file Platform: posix, Created on: Sat Sep 19 12:27:40 2015',
    __version__: 1.0,
    __globals__: [],
    layers: {type: <class 'numpy.ndarray'>, shape: (1, 43)},
}

## 说明:
### layers:
conv 层 (0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34)
params, pad, operation_type, operation_name, stride = data['layers'][0][index][0][0]
kernal, bias = params[0]
print(kernal.shape, bias.shape)

relu 层 (1, 3, 6, 8, 11, 13, 15, 17, 20, 22, 24, 26, 29, 31, 33, 35, 38, 40)
operation_type, operation_name = data['layers'][0][index][0][0]
print(operation_type, operation_name)

pool 层 (4, 9, 18, 27, 36)
operation_name, _, _, operation_type, pool_type, _ = data['layers'][0][index][0][0]
print(operation_name, operation_type, pool_type)

fc 层 (37, 39, 41)
params, _, operation_type, operation_name, _ = data['layers'][0][index][0][0]
kernal, bias = params[0]
print(kernal.shape)
print(bias.shape)

prob 层 (42, )
operation_type, operation_name = data['layers'][0][index][0][0]
print(operation_type, operation_name)

### classes:
没有搞清楚, 每一种类别分别代表什么.
cls = data['classes'][0][0]
cls0, cls1 = cls
print(cls0)

### normalization:
normalization = data['normalization'][0][0]
mean, _, _, input_image_size, interpolate_type = normalization
我猜是: 将图像 resize 到 input_image_size 图像大小, 采用 interpolate_type 的插值方法, 之后减去 mean.
"""
import scipy.io


vgg_model_path = '../../../datasets/models/vgg/vgg_scipy/imagenet-vgg-verydeep-19.mat'
data = scipy.io.loadmat(vgg_model_path)

normalization = data['normalization'][0][0]
mean, _, _, input_image_size, Interpolate_type = normalization








