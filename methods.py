import logging

import cv2

# import matplotlib
# matplotlib.use("tkagg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import convolve as conv2d
from tqdm import tqdm

logging.basicConfig(
    filename="exp.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
)


def canny(
    image,
    high_threshold: int = 200,
    low_threshold: int = 100,
    debugging=True,
):
    """
    给出numpy数组图像，和迟滞阈值法两个阈值
    """
    # 展示原图性质
    if debugging:
        show_img(image, "image")
    else:
        show_img(image, "image", analysis=False)
    # ----------------------------------------高斯滤波,默认5*5滤波
    # 输入image
    gaussian_kernel = np.array(
        [
            [2, 4, 5, 4, 2],
            [4, 9, 12, 9, 4],
            [5, 12, 15, 12, 5],
            [4, 9, 12, 9, 4],
            [2, 4, 5, 4, 2],
        ],
        dtype=np.float32,
    )  # 使用固定大小的kernel
    gaussian_kernel = gaussian_kernel / 159.0
    # TODO 改成生成不同大小的高斯kernel做滤波
    if debugging:
        print(gaussian_kernel)
    h, w = image.shape
    print(f"image shape:{image.shape}")

    # 应用高斯平滑
    image_smoothed = conv2d(
        input=image,
        weights=gaussian_kernel,
        mode="constant",
    )
    logging.info("高斯平滑完成")

    # show_img(image_smoothed, "image_smoothed")
    image_smoothed = np.float32(image_smoothed)
    if debugging:
        show_img(image_smoothed, "image_smoothed")

    # ----------------------------------------sobel梯度幅值
    # 现在有image_smmothed
    # 创建sobel算子
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)
    h, w = image.shape
    # 应用sobel算子，计算两个方向的梯度
    grad_x = conv2d(
        image_smoothed,
        sobel_x,
        mode="constant",
    )
    grad_y = conv2d(
        image_smoothed,
        sobel_y,
        mode="constant",
    )
    if debugging:
        show_img(grad_x, "grad_x")
        show_img(grad_y, "grad_y")
    grad = np.hypot(
        grad_x, grad_y
    )  # 计算对应元素的梯度均值，hypot相当于直角三角形求斜边
    logging.info("计算梯度完成")

    # print(grad.max())
    grad = grad / grad.max() * 255  # 整理到灰度区间内
    if debugging:
        show_img(grad, "grad")

    direction = np.arctan2(
        grad_y, grad_x
    )  # arctan2函数相比于arctan接受两个参数，可以忽略可能遇到的非常值，比如，分母为0，x=C

    direction[direction < 0] += (
        np.pi
    )  # 将角度统一到正区间，减少方向判断数量（不能用绝对值，虽然不影响边缘检测，但是影响霍夫圆识别

    if debugging:
        analyze_array(direction, "direction")
    logging.info("计算梯度方向完成")

    # ----------------------------------------非极大值抑制NMS
    # 现在有grad,direction
    # 四个方向的梯度求最大的极值（上下，左右，斜左，斜右）
    h, w = grad.shape
    grad_nms = np.zeros((h, w), dtype=np.int32)

    # 计算nms，只保留每个像素梯度方向的像素
    for x in tqdm(range(1, h - 1), desc="nms calculating"):
        for y in range(1, w - 1):
            # 临近像素p 设置为255是因为避免在没有进入分支的情况下被误分
            p1, p2 = 256, 256
            # 方向判断
            if (
                0 < direction[x, y] < np.pi / 8
                or np.pi * 7 / 8 < direction[x, y] < np.pi
            ):
                p1 = grad[x, y + 1]
                p2 = grad[x, y - 1]
            elif np.pi / 8 < direction[x, y] < np.pi * 3 / 8:
                p1 = grad[x - 1, y - 1]
                p2 = grad[x + 1, y + 1]
            elif np.pi * 3 / 8 < direction[x, y] < np.pi * 5 / 8:
                p1 = grad[x + 1, y]
                p2 = grad[x - 1, y]
            elif np.pi * 5 / 8 < direction[x, y] < np.pi * 7 / 8:
                p1 = grad[x + 1, y - 1]
                p2 = grad[x - 1, y + 1]

            # 如果同时大于周围像素则保留，否则为0
            if grad[x, y] >= max(p1, p2):
                grad_nms[x, y] = grad[x, y]
            else:
                grad_nms[x, y] = 0

    logging.info("计算NMS完成")
    grad_nms = grad_nms / grad_nms.max() * 255
    if debugging:
        show_img(grad_nms, "grad_nms")

    # ----------------------------------------迟滞阈值 门限法
    # 输入 grad_nms
    # high_thr = high_threshold
    # low_thr = low_threshold

    h, w = grad_nms.shape
    edge = np.zeros((h, w), dtype=np.int32)
    strong = np.where(grad_nms >= high_threshold)
    uncertain = np.where((low_threshold <= grad_nms) & (grad_nms < high_threshold))

    edge[strong[0], strong[1]] = 255
    edge[uncertain[0], uncertain[1]] = 100

    if debugging:
        show_img(edge, "edge")
    else:
        show_img(edge, "edge", analysis=False)

    # print(image.shape, edge.shape, direction.shape) #这行代表处理前后形状不改变

    return edge, direction


def hough_circle(
    image,
    high_threshold=200,
    low_threshold=100,
    min_r=40,
    max_r=200,
    min_voting=30,
    min_center_distance=10,
    debugging=True,
    vt_debugging=False,  # voting_threshold_debugging
    vtd_step=5,
):
    # 使用canny提取边缘，得到的是int32类型的图像数组最大255
    edge, direction = canny(
        image,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
        debugging=debugging,
    )

    # 定义参数空间x,y,r
    h, w = edge.shape
    vote_space = np.zeros((h, w, max_r), dtype=np.int32)  # 投票假设落于整数空间
    # FIXME 难以处理大图片，4096*3072*500就已经需要分配23G，霍夫投票空间实际上是一个相当稀疏的矩阵
    # 可以改进为int16

    # 投票,每个点投出一个圆锥的两条母线
    x, y = np.where(edge > 250)
    for i in tqdm(range(len(x)), desc="voting"):
        # 遍历所有的边缘点
        for r in range(min_r, max_r):
            # 遍历所有的半径，投票
            delta_x = r * np.sin(direction[x[i], y[i]])
            delta_y = r * np.cos(direction[x[i], y[i]])
            vx = int(np.around(x[i] + delta_x))
            vy = int(np.around(y[i] + delta_y))
            if 0 < vx < h and 0 < vy < w:
                vote_space[vx, vy, r] += 1
            vx = int(np.around(x[i] - delta_x))
            vy = int(np.around(y[i] - delta_y))
            if 0 < vx < h and 0 < vy < w:
                vote_space[vx, vy, r] += 1

    # 抑制圆心
    # analyze_array(vote_space[vote_space > 10], "vote space")  # 检查形状
    if debugging:  # 分析圆心分布,按照票数从低到高
        for i in range(2, 20, 3):
            analyze_array(vote_space[vote_space > i], "vote space")

    # 筛选出可能的候选圆心（调试方法，
    circles = []
    if vt_debugging:
        """
        因为发现筛选圆心的过程时间代价比较小，
        所以可以尝试使用不同的最小投票数量来画出不同的圆，找到最适合的圆
        """
        for v in range(5, np.max(vote_space), vtd_step):
            print(f"voting threshold:{v}")
            circles = _center_combine(vote_space, v, min_center_distance)
            _draw_circle(image, circles)
            cl = [c[2] for c in circles]  # 找到最大值
            print(f"Above circles: min_r{min(cl)},max_r{max(cl)}")
            print("-" * 100)
    else:
        circles = _center_combine(vote_space, min_voting, min_center_distance)
        _draw_circle(image, circles)

    # print(circles)  # 打印所有的圆坐标和半径
    return circles


def _center_combine(
    vote_space,
    min_voting,
    min_dist,
):
    # 使用最小投票数筛选圆心
    circles_to_be_selected = np.where(
        vote_space > min_voting
    )  # 返回x,y,r三个数组组成的list
    # print(circles_to_be_selected)
    circles_to_be_selected = np.stack(
        circles_to_be_selected, axis=1
    )  # 返回圆心[x,y,r]组成的列表
    # 判断圆心之间的距离，如果小于最小圆心距离就当作是一个圆
    # 相对比较笨的方法，双层循环
    selected_label = np.zeros(
        len(circles_to_be_selected), dtype=np.int8
    )  # 已经选中的圆标记为1
    circles = []  # 存放已经找到的圆
    for i, centeri in tqdm(enumerate(circles_to_be_selected), desc="filter circle"):
        # 对于每一个待选的圆，找到和其他圆一样的点
        # 如果选中的店已经被标记为圆心
        if selected_label[i] > 0:
            continue
        selected_label[i] += 1  # 圆心选中，标记
        # 圆心列表,同一个圆心
        same_center = []
        same_center.append(centeri)
        x, y, r = np.mean(same_center, axis=0)  # 列表套数组可以求,双列表也可以求
        for j, centerj in enumerate(circles_to_be_selected):
            if selected_label[j] > 0:
                continue
            # 判断距离是否小于阈值，小于则归为一个圆
            if (x - centerj[0]) ** 2 + (y - centerj[1]) ** 2 < min_dist**2:
                # 同一个圆
                same_center.append(centerj)
                x, y, r = np.mean(
                    same_center, axis=0
                )  # 取圆心列表所有点的平均值作为中心点
                # 标记
                selected_label[j] += 1

        circles.append([x, y, r])

    circles = np.uint32(np.around(circles))  # 整数圆心
    return circles


def _draw_circle(img, circles):
    img_c = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i in circles:
        cv2.circle(img_c, (i[1], i[0]), i[2], (0, 255, 0), 5)  # 绿色的圆
        cv2.circle(img_c, (i[1], i[0]), 2, (255, 0, 0), 5)  # 红色的圆心
    show_img(img_c, analysis=False)


def show_img(image, title="", analysis=True):
    """
    展示灰度图片,并可选的分析像素分布
    """
    plt.imshow(image)
    # plt.imshow(image, "gray")
    plt.axis("off")
    plt.title(title)
    plt.show()
    if analysis:
        analyze_array(image, title)  #


def analyze_array(array, title="", bins=50):
    """
    分析一个数组的数据分布，绘图的方式展现
    """
    try:
        if np.size(array) == 0 or array is None:
            print("analyze: 数组不存在或者大小为0")
            return
    except:  # noqa: E722
        print("analyze: 数组不存在或者大小为0")
        return

    print(array.dtype)
    print(array.shape)
    min_value = np.min(array)
    max_value = np.max(array)

    # 统计数据分布
    histogram, bins = np.histogram(array.flatten(), bins=bins)

    # 绘制柱形图
    plt.figure(figsize=(10, 5))
    plt.bar(
        bins[:-1], histogram, width=(bins[1] - bins[0]), align="edge", edgecolor="black"
    )
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {title}")

    plt.show()

    print(f"min{min_value},max{max_value}")
    print("-" * 100)


if __name__ == "__main__":
    # 针对图片的精调参数
    config = {
        "./img/coins3.jpg": {
            "high_threshold": 100,
            "low_threshold": 50,
            "min_r": 50,
            "max_r": 200,
            "min_voting": 15,
            "debugging": False,
        },
    }

    img_path = "./img/coins3.jpg"
    # test hough
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    circles = hough_circle(img, **config[img_path])
