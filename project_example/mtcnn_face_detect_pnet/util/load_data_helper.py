import re

import numpy as np


def is_path(string):
    """
    从 wider_face_train_bbx_gt.txt 中按行读取数据后, 用于判断当字符串是否是那个图片路径.
    :param string: 如: "0--Parade/0_Parade_marchingband_1_849.jpg"
    :return:
    """
    pattern = re.compile(".*(?:\.jpg)")
    match = re.search(pattern, string)
    if match is None:
        return False
    else:
        return True


def is_count(string):
    """
    从 wider_face_train_bbx_gt.txt 中按行读取数据后, 用于判断当字符串是否是那个图片中人脸的计数.
    :param string: 如: "32"
    :return:
    """
    pattern = re.compile("\d+\n")
    match = re.search(pattern, string)
    if match is None:
        return False
    else:
        return True


def is_bounding_box(string):
    """
    从 wider_face_train_bbx_gt.txt 中按行读取数据后, 用于判断当字符串是否是那个图片中人脸的 bounding box.
    :param string: 如: "570 126 13 16 2 0 0 0 0 0 "
    :return:
    """
    pattern = re.compile("(\d+ ){10}")
    match = re.search(pattern, string)
    if match is None:
        return False
    else:
        return True


def get_bounding_box(string):
    """
    人脸标注有 10 项数字, 前四个是人脸位置的 bounding box 的位置 (x, y, w, h).
    后 6 项数字是 readme.txt 中对图片的其它描述. 这里我们不需要.
    返回 bounding box 的四个数字的元组.
    :param string: string: 如: "570 126 13 16 2 0 0 0 0 0 "
    :return: 如: (570, 126, 13, 16)
    """
    pattern = re.compile("(\d+ \d+ \d+ \d+) (?:\d+ ){6}")
    match = re.search(pattern, string)
    box_string = match.group(1)
    result = list(map(lambda x: int(x), box_string.split(' ')))
    return result


def load_image_data(file_path):
    """
    将 txt 文件中的数据读取成一个一个的对象. 以便生成数据.
    :param file_path: 应该是 wider_face_train_bbx_gt.txt 文件.
    :return: 包含三元元组的列表. 如: [[image_path, face_count, bounding_box_list], ...,
    [image_path, face_count, bounding_box_list]]

    demo:
    file_path = "../dataset/wider_face_split/wider_face_train_bbx_gt.txt"
    training_data = load_training_data(file_path)
    while True:
        try:
            print(next(training_data))
        except StopIteration:
            break

    输出结果如:
    ['9--Press_Conference/9_Press_Conference_Press_Conference_9_417.jpg\n', 1, [(404, 162, 242, 298)]]
    ['9--Press_Conference/9_Press_Conference_Press_Conference_9_770.jpg\n', 1, [(207, 174, 570, 631)]]
    ['9--Press_Conference/9_Press_Conference_Press_Conference_9_864.jpg\n', 3, [(52, 138, 162, 202), (446, 206, 138, 202), (584, 162, 146, 200)]]
    ['9--Press_Conference/9_Press_Conference_Press_Conference_9_88.jpg\n', 1, [(523, 147, 241, 352)]]
    ['9--Press_Conference/9_Press_Conference_Press_Conference_9_591.jpg\n', 2, [(473, 131, 32, 50), (759, 267, 10, 14)]]
    ['9--Press_Conference/9_Press_Conference_Press_Conference_9_67.jpg\n', 1, [(504, 102, 190, 276)]]
    """
    image_obj = []
    with open(file_path, 'r', encoding='utf-8') as f:
        while True:
            training_data = f.readline()
            if training_data == '':
                break
            if is_path(training_data):
                if len(image_obj) != 0:
                    image_obj[-1] = np.array(image_obj[-1])
                    yield image_obj
                image_obj = [training_data.strip()]
                continue
            if is_count(training_data):
                image_obj.append(int(training_data))
                continue
            if is_bounding_box(training_data):
                # print(training_data)
                bounding_box = get_bounding_box(training_data)
                if len(image_obj) == 2:
                    image_obj.append([bounding_box])
                else:
                    image_obj[-1].append(bounding_box)


def bounding_box_iou(box, boxes):
    """
    求 box 与 boxes 的重合部分的面积比, 这处一个盒子对应多个标记盒子.
    之后取面积比最大的值返回.
    :param box: bounding box 坐标 (x, y, w, h), 形状为: (1, 4)
    :param boxes: 图像标记中的 bounding box 从标 (x, y, w, h), 形状为: (-1, 4)
    :return:
    """
    box_area = box[2] * box[3]
    area = boxes[:, 2] * boxes[:, 3]
    x = np.maximum(box[0], boxes[:, 0])
    y = np.maximum(box[1], boxes[:, 1])
    x1 = np.minimum(box[0] + box[2], boxes[:, 0] + boxes[:, 2])
    y1 = np.minimum(box[1] + box[3], boxes[:, 1] + boxes[:, 3])

    w = np.maximum(0, x1 - x)
    h = np.maximum(0, y1 - y)
    inter = w * h
    result = inter / (box_area + area - inter + 1e-10)
    result = np.max(result)
    return result


if __name__ == '__main__':
    box = np.array([[10, 10, 20, 20]])
    boxes = np.array([[12, 12, 30, 30], [9, 9, 40, 40]])
    result = bounding_box_iou(box, boxes)
    print(result)
