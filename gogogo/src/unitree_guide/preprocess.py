#！/usr/bin/python

import cv2
from cv_bridge import CvBridge
import numpy as np
import rospy
from sensor_msgs.msg import Image
from skimage import morphology
import threading

class tupian:
    def __init__(self, topic_name = "/a112138/camera/color_raw"):
        self.offset = 0
        self.lock = threading.Lock()
        rospy.Subscriber(topic_name, Image, self.callback)

    def callback(self, msg):
        rospy.loginfo(f"Received image: {msg.width}x{msg.height}")
        # image = msg
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(gray, 100, 1, cv2.THRESH_BINARY_INV)

        # 中值滤波去噪
        denoised = cv2.medianBlur(binary, 5)

        # 形态学开运算（先腐蚀后膨胀）
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # 形态学闭运算（先膨胀后腐蚀）
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

        edges = cv2.Canny(closed, 50, 150)

        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 筛选最大连通区域
        max_contour = max(contours, key=cv2.contourArea)
        cv_image = cv_image.copy()
        cv2.drawContours(cv_image, [max_contour], -1, (0,255,0), 2)

        # 骨架化算法（需自定义或使用第三方库）
        skeleton0 = morphology.skeletonize(binary)
        skeleton = skeleton0.astype(np.uint8)*255
        # cv2.imshow('drawimg',skeleton)
        # cv2.waitKey(0)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, threshold=50,
            minLineLength=50, maxLineGap=10
        )
        
        # 计算偏离量
        height, width = cv_image.shape[:2]
        x_center = width // 2  

        # 提取骨架中的点
        points = cv2.findNonZero(skeleton)
        if points is not None:
            points = points[:, 0, :]  
            # 选择底部区域（例如最后100行）
            y_min = height - 100
            bottom_points = points[points[:, 1] >= y_min]

            if bottom_points.size > 0:
                # 计算底部区域x坐标的中位数（抗噪声）
                path_x = np.median(bottom_points[:, 0])
                offset = int(path_x - x_center)
                direction = "右" if offset > 0 else "左"
                print(f"偏离量：{abs(offset)}像素，方向：{direction}")
            else:
                # 若无底部点，尝试使用直线拟合所有点
                y = points[:, 1]
                x = points[:, 0]
                A = np.vstack([y, np.ones(len(y))]).T
                m, b = np.linalg.lstsq(A, x, rcond=None)[0]
                # 预测底部x坐标
                x_bottom = m * (height - 1) + b
                offset = int(x_bottom - x_center)
                direction = "右" if offset > 0 else "左"
                print(f"预测偏离量：{abs(offset)}像素，方向：{direction}")
        else:
            print("未检测到路径中心线")
        with self.lock:
            print(offset)
            self.offset = offset

    def getoffset(self):
        with self.lock:
            return self.offset


# if __name__=="__main__":
#     rospy.init_node('listener')
#     rospy.Subscriber("/a112138/camera/color_raw", Image, callback)
#     rospy.loginfo("Waiting for messages...")
#     rospy.spin()



def zhang_suen_thinning(binary_image):
    # 确保输入图像是二值图像
    binary_image = binary_image.copy()
    binary_image[binary_image != 0] = 1

    # 定义8邻域的偏移量
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                (0, -1),          (0, 1),
                (1, -1),  (1, 0), (1, 1)]

    # 迭代直到没有像素被删除
    while True:
        # 记录需要删除的像素
        to_remove = []

        # 第一步：标记需要删除的像素
        for i in range(1, binary_image.shape[0] - 1):
            for j in range(1, binary_image.shape[1] - 1):
                if binary_image[i, j] == 1:
                    # 计算P2到P9的值
                    P2, P3, P4, P5, P6, P7, P8, P9 = [binary_image[i + x, j + y] for x, y in neighbors]

                    # 计算A(P1)：从P2到P9的0到1的转换次数
                    A = sum((P2, P3, P4, P5, P6, P7, P8, P9, P2))
                    A = sum((1 if P2 == 0 and P3 == 1 else 0,
                            1 if P3 == 0 and P4 == 1 else 0,
                            1 if P4 == 0 and P5 == 1 else 0,
                            1 if P5 == 0 and P6 == 1 else 0,
                            1 if P6 == 0 and P7 == 1 else 0,
                            1 if P7 == 0 and P8 == 1 else 0,
                            1 if P8 == 0 and P9 == 1 else 0,
                            1 if P9 == 0 and P2 == 1 else 0))

                    # 计算B(P1)：P2到P9中1的个数
                    B = sum((P2, P3, P4, P5, P6, P7, P8, P9))

                    # 条件1：2 <= B(P1) <= 6
                    # 条件2：A(P1) == 1
                    # 条件3：P2 * P4 * P6 == 0
                    # 条件4：P4 * P6 * P8 == 0
                    if 2 <= B <= 6 and A == 1 and P2 * P4 * P6 == 0 and P4 * P6 * P8 == 0:
                        to_remove.append((i, j))

        # 如果没有像素需要删除，算法结束
        if not to_remove:
            break

        # 删除标记的像素
        for i, j in to_remove:
            binary_image[i, j] = 0

        # 第二步：标记需要删除的像素
        to_remove = []

        for i in range(1, binary_image.shape[0] - 1):
            for j in range(1, binary_image.shape[1] - 1):
                if binary_image[i, j] == 1:
                    # 计算P2到P9的值
                    P2, P3, P4, P5, P6, P7, P8, P9 = [binary_image[i + x, j + y] for x, y in neighbors]

                    # 计算A(P1)：从P2到P9的0到1的转换次数
                    A = sum((1 if P2 == 0 and P3 == 1 else 0,
                            1 if P3 == 0 and P4 == 1 else 0,
                            1 if P4 == 0 and P5 == 1 else 0,
                            1 if P5 == 0 and P6 == 1 else 0,
                            1 if P6 == 0 and P7 == 1 else 0,
                            1 if P7 == 0 and P8 == 1 else 0,
                            1 if P8 == 0 and P9 == 1 else 0,
                            1 if P9 == 0 and P2 == 1 else 0))

                    # 计算B(P1)：P2到P9中1的个数
                    B = sum((P2, P3, P4, P5, P6, P7, P8, P9))

                    # 条件1：2 <= B(P1) <= 6
                    # 条件2：A(P1) == 1
                    # 条件3：P2 * P4 * P8 == 0
                    # 条件4：P2 * P6 * P8 == 0
                    if 2 <= B <= 6 and A == 1 and P2 * P4 * P8 == 0 and P2 * P6 * P8 == 0:
                        to_remove.append((i, j))

        # 如果没有像素需要删除，算法结束
        if not to_remove:
            break

        # 删除标记的像素
        for i, j in to_remove:
            binary_image[i, j] = 0

    return binary_image