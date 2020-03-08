"""
在这里运行模型训练过程.
主要包括: load_data, modeling, optimization. 数据加载, 网络模型, 优化器.
"""
from project_example.face_recognition.load_data import DataLoader
from project_example.face_recognition.modeling import model
from project_example.face_recognition.optimization import Optimizer


if __name__ == '__main__':
    data_loader = DataLoader(data_path='dataset/faces', grayscale=True, test_size=0.2, one_hot=True)
    optimizer = Optimizer(data_loader=data_loader, model=model, n_epoch=10, batch_size=64, learning_rate=0.001)
    optimizer.run()
