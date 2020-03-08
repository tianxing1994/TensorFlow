import numpy as np
from shapely.geometry import Polygon


def calc_iou(rect1, rect2):
    """
    计算由两个线性环界定的多边形的相交部分的面积与两个多边形的并集的面积的比值.
    :param rect1: ndarray, 形状为 (9,), 其中前 8 项用于表示四边形的左上, 右上, 右下, 左下四个角,
    分别表示 (x0, y0, x1, y1, x2, y2, x3, y3) 四个坐标. 最后一项为该四边形的权值.
    :param rect2: ndarray, 跟 rect1 一样的另一个四边形.
    :return: 两个多边形的交集比上其并集的值 iou.
    """
    rect1 = Polygon(shell=rect1[:8].reshape((4, 2)))
    rect2 = Polygon(shell=rect2[:8].reshape((4, 2)))
    inter = rect1.intersection(rect2).area
    union = rect1.area + rect2.area - inter
    if union == 0:
        result = 0
    else:
        result = inter/union
    return result


def merge_rect(rect1, rect2):
    """
    当两个矩形 IOU 值足够高, 即两个 rect 矩形重叠, 按 probs 权重合并两个矩形.
    这两个矩形在空间位置上相近, 都具有较高的概率指示为文本.
    这种情况, 很有可能是两个分离的文字, 我们需要将它们合成一个文本字符串.
    :param rect1: ndarray, 形状为 (9,), 其中前 8 项用于表示四边形的左上, 右上, 右下, 左下四个角,
    分别表示 (x0, y0, x1, y1, x2, y2, x3, y3) 四个坐标. 最后一项为该四边形的权值.
    :param rect2: rect2: ndarray, 跟 rect1 一样的另一个四边形.
    :return: ndarray, 将两个 rect1, rect2 合并成一个.
    """
    rect1[:8] = (rect1[8] * rect1[:8] + rect2[8] * rect2[:8])/(rect1[8] + rect2[8])
    rect1[8] = (rect1[8] + rect2[8])
    return rect1


def standard_nms(boxes, iou_threshold):
    """
    标准非极大值抑制.
    :param boxes: ndarray, 形状为: (m, 9) 表示 m 个四边形, 每一行的值分别表示: (x0, y0, x1, y1, x2, y2, x3, y3, probs)
    :param iou_threshold:
    :return:
    """
    # 按 probs 降序排序, 获取排序索引.
    order = np.argsort(boxes[:, 8])[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        iou = np.array([calc_iou(boxes[i], boxes[t]) for t in order[1:]])
        # iou > iou_threshold 的矩形都被去除.
        index = np.where(iou <= iou_threshold)[0]
        order = order[index+1]
    return boxes[keep]


def locality_non_max_suppression(boxes, merge_iou_threshold=0.1, nms_iou_threshold=0.3):
    """
    非极大值抑制.
    如果在空间位置上相近的两个矩形, 且都具有较高的概率指示为文本.
    在这种情况, 很有可能是两个文字(一个词语或句子),
    首先, 我们需要将它们合成一个文本串. 再用合并后的矩形框进行非极大值抑制.
    :param boxes: ndarray, 形状为: (m, 9) 表示 m 个四边形, 每一行的值分别表示: (x0, y0, x1, y1, x2, y2, x3, y3, probs)
    :param merge_iou_threshold: Rect 矩形合并时的 IOU 阈值.
    :param nms_iou_threshold: 非极大值抑制的 IOU 阈值.
    :return: 非极大值抑制后的 boxes.
    """
    pre_nms_boxes = []
    previous_box = None
    for box in boxes:
        if previous_box is not None and calc_iou(box, previous_box) > merge_iou_threshold:
            previous_box = merge_rect(box, previous_box)
        else:
            if previous_box is not None:
                pre_nms_boxes.append(previous_box)
            previous_box = box
    if previous_box is not None:
        pre_nms_boxes.append(previous_box)

    if len(pre_nms_boxes) == 0:
        return np.array([])
    return standard_nms(boxes=np.array(pre_nms_boxes), iou_threshold=nms_iou_threshold)
