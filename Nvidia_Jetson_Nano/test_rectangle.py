#!/usr/bin/python
# -*- coding:utf-8 -*-
import cv2
import numpy as np
import serial as ser


# 计算透视变换参数矩阵
def cal_perspective_params(img, points):
    # 设置偏移点。如果设置为(0,0),表示透视结果只显示变换的部分（也就是画框的部分）
    offset_x = 0
    offset_y = 0
    # img_size = (img.shape[1], img.shape[0])  # 640*480
    # print(img_size)
    img_size = (500, 500)
    src = np.float32(points)
    # 透视变换的四个点
    dst = np.float32([[offset_x, offset_y], [img_size[0] - offset_x, offset_y],
                      [offset_x, img_size[1] - offset_y], [img_size[0] - offset_x, img_size[1] - offset_y]])
    # 透视矩阵
    M = cv2.getPerspectiveTransform(src, dst)
    # print(M)
    # 透视逆矩阵
    M_inverse = cv2.getPerspectiveTransform(dst, src)
    # print(M_inverse)
    return M, M_inverse


# 透视变换
def img_perspect_transform(img, M):
    # img_size = (img.shape[1], img.shape[0])
    # print(img_size)
    img_size = (500, 500)
    return cv2.warpPerspective(img, M, img_size)


# 计算透视变换参数矩阵
def cal_perspective_params1(img, points):
    # 设置偏移点。如果设置为(0,0),表示透视结果只显示变换的部分（也就是画框的部分）
    offset_x = 0
    offset_y = 0
    # img_size = (img.shape[1], img.shape[0])  # 640*480
    # print(img_size)
    img_size = (500, 500)
    src = np.float32(points)
    # 透视变换的四个点
    dst = np.float32([[offset_x, offset_y], [img_size[0] - offset_x, offset_y],
                      [offset_x, img_size[1] - offset_y], [img_size[0] - offset_x, img_size[1] - offset_y]])
    # 透视矩阵
    M = cv2.getPerspectiveTransform(src, dst)
    # print(M)
    # 透视逆矩阵
    M_inverse = cv2.getPerspectiveTransform(dst, src)
    # print(M_inverse)
    return M, M_inverse


# 透视变换
def img_perspect_transform1(img, M):
    # img_size = (img.shape[1], img.shape[0])
    # print(img_size)
    img_size = (500, 500)
    return cv2.warpPerspective(img, M, img_size)


def draw_line(img, p1, p2, p3, p4):
    points = [list(p1), list(p2), list(p3), list(p4)]
    # 画线
    # img = cv2.line(img, p1, p2, (0, 0, 255), 3)
    # img = cv2.line(img, p2, p4, (0, 0, 255), 3)
    # img = cv2.line(img, p4, p3, (0, 0, 255), 3)
    # img = cv2.line(img, p3, p1, (0, 0, 255), 3)
    return points, img


def counter(contours):
    if (len(contours)) > 0:
        # 找出最大坐标对应的索引
        area = []
        for i in range(len(contours)):
            area.append(cv2.contourArea(contours[i]))
        max_id = np.argmax(area)
        # print(area)
        # print(len(area))
        del area[max_id]
        second_max = np.argmax(area)
        # print(len(area))
        # x, y, w, h = cv2.boundingRect(contours[max_id])
        # cv2.drawContours(copy, [contours[second_max]], 0, (255, 0, 255), 2)  # 画最大轮廓
        # cv2.circle(copy, (x,y), 10, (0, 255, 0), 2)
        # rect = cv2.minAreaRect(contours[max_id])  # 最小外包矩形来包围目标边缘
        # points = np.int0(cv2.boxPoints(rect))
        cnt = contours[max_id]
        cnt1 = contours[second_max]
        epsilon = 0.1 * cv2.arcLength(cnt, True)
        epsilon1 = 0.1 * cv2.arcLength(cnt1, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        approx1 = cv2.approxPolyDP(cnt1, epsilon1, True)
    return approx, approx1


def calculate(points_):
    b = np.array(points_[:, 0])
    index = np.lexsort((b[:, 0], b[:, 1]))
    points = b[index]
    index1 = np.zeros(4, dtype=int)
    # print("init_point", points)
    if points[1][0] - points[0][0] < 0:
        index1[0] = index[1]
        index1[1] = index[0]
        index1[2] = index[3]
        index1[3] = index[2]
        # print(index, index1)
        points = b[index1]  # 按照排序结果重置数组元素的位置
    # print("last_point:", points)
    return points


def cal_point(point_last, point_last1):
    send_direction = []
    x1 = int((point_last[0][0] + point_last1[0][0]) / 2)
    y1 = int((point_last[0][1] + point_last1[0][1]) / 2)
    x2 = int((point_last[1][0] + point_last1[1][0]) / 2)
    y2 = int((point_last[1][1] + point_last1[1][1]) / 2)
    x3 = int((point_last[3][0] + point_last1[3][0]) / 2)
    y3 = int((point_last[3][1] + point_last1[3][1]) / 2)
    x4 = int((point_last[2][0] + point_last1[2][0]) / 2)
    y4 = int((point_last[2][1] + point_last1[2][1]) / 2)
    send_direction.append(int(x1 / 2))
    send_direction.append(int(y1 / 2))
    send_direction.append(int(x2 / 2))
    send_direction.append(int(y2 / 2))
    send_direction.append(int(x3 / 2))
    send_direction.append(int(y3 / 2))
    send_direction.append(int(x4 / 2))
    send_direction.append(int(y4 / 2))
    return send_direction


if __name__ == '__main__':
    Video = cv2.VideoCapture(0, cv2.CAP_V4L2)    # 打开摄像头设备
    ret, frame = Video.read()  # 读取摄像头画面
    frame = cv2.flip(frame, -1)  # 存在镜像，进行翻转
    ser = ser.Serial("/dev/ttyUSB0", 115200, timeout=1)   # 初始化串口
    send_flag = 0     # 是否发送坐标的标志位
    while 1:    # 保持循环
        send_follow = []       # 发送坐标数组
        b = ser.inWaiting()    # 串口接收到的数据
        ret, frame = Video.read()
        frame = cv2.flip(frame, -1)
        frame = frame[116:380, 200:475]   # 对摄像头画面进行剪裁
        cv2.imshow("frame", frame)         # 显示画面
        img_resize = cv2.resize(frame, (500, 500))  # 规范化画面
        points, img = draw_line(img_resize, (34, 39), (458, 36), (40, 475),(454, 472))  # 通过四个点透视变换图片
        M, M_inverse = cal_perspective_params(img, points)   # 得到透视变换的矩阵
        trasform_img = img_perspect_transform(img, M)       # 透视变换之后的图片
        copy = trasform_img.copy() # 图片进行复制
        frame = cv2.cvtColor(trasform_img, cv2.COLOR_BGR2HSV)  # 色彩空间的转化
        lower_green = np.array([0, 0, 224])           # 绿色激光笔的阈值
        upper_green = np.array([125, 255, 255])
        lower_red = np.array([139, 89, 84])           # 红色激光笔的阈值
        upper_red = np.array([180, 255, 255])
        lower_red_white = np.array([129, 15, 164])    # 红色激光笔在白板的阈值，以防丢点
        upper_red_white = np.array([180, 255, 255])
        lower_black = np.array([0, 0, 0])             # 黑色边框的阈值
        upper_black = np.array([255, 255, 80])
        thresh_black = cv2.inRange(trasform_img, lower_black, upper_black)  # 阈值化处理找黑色边框
        thresh_red = cv2.inRange(frame, lower_red, upper_red)   # 阈值化处理找红色激光笔
        thresh_green = cv2.inRange(frame, lower_green, upper_green) # 阈值化处理找绿色激光笔
        thresh_red_white = cv2.inRange(frame, lower_red_white, upper_red_white)
        contours_black, _ = cv2.findContours(thresh_black, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # 找到黑色边框的轮廓
        points, points1 = counter(contours_black)  # 自定义函数获得轮廓
        cv2.drawContours(copy, [points], 0, (255, 0, 255), 3)    # 画边框
        cv2.drawContours(copy, [points1], 0, (0, 0, 255), 3)
        if (len(contours_black)) > 0 and len(points) > 3 and len(points1) > 3:  # 如果轮廓的顶点大于3个，认为是矩形
            point_last = calculate(points)    # 自定义函数用于将边框四个顶点排序
            point_last1 = calculate(points1)
            send_goal = []
            send_goal = cal_point(point_last, point_last1)  # 该函数用来返回两个矩形之间的四个角点
        contours_red, _ = cv2.findContours(thresh_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   # 找到红色激光笔的轮廓
        contours_red_white, _ = cv2.findContours(thresh_red_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if (len(contours_red)) > 0:
            area = []
            for i in range(len(contours_red)):
                area.append(cv2.contourArea(contours_red[i]))
            max_id = np.argmax(area)      # 找到最大的轮廓
            x, y, w, h = cv2.boundingRect(contours_red[max_id])
            cx_red = x + w / 2
            cy_red = y + h / 2
            cv2.circle(copy, (int(x + w / 2), int(y + h / 2)), 3, (255, 0, 0), 2)   # 画出轮廓中心
        else:
            if (len(contours_red_white)) > 0:
                area = []
                for i in range(len(contours_red_white)):
                    area.append(cv2.contourArea(contours_red_white[i]))
                max_id = np.argmax(area)
                x, y, w, h = cv2.boundingRect(contours_red_white[max_id])
                cx_red = x + w / 2
                cy_red = y + h / 2
                cv2.circle(copy, (int(x + w / 2), int(y + h / 2)), 3, (255, 0, 0), 2)
        if (len(contours_red)) == 0 and len(contours_red_white) == 0:
            cx_green = 0
            cy_green = 0
        if cx_red != 0 and cy_red != 0:
            send_follow.append(int(cx_red/2))
            send_follow.append(int(cy_red/2))
        else:
            send_follow.append(254)
            send_follow.append(254)
        contours_green, _ = cv2.findContours(thresh_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 找到绿色激光笔的轮廓
        if (len(contours_green)) > 0:
            area = []
            for i in range(len(contours_green)):
                area.append(cv2.contourArea(contours_green[i]))
            max_id = np.argmax(area)    # 最大的轮廓
            x, y, w, h = cv2.boundingRect(contours_green[max_id])
            cx_green = x + w / 2
            cy_green = y + h / 2
            cv2.circle(copy, (int(x + w / 2), int(y + h / 2)), 3, (255, 0, 255), 2)
        if (len(contours_green)) == 0:
            cx_green = 0
            cy_green = 0
        if cx_green != 0 and cy_green != 0:
            send_follow.append(int(cx_green/2))
            send_follow.append(int(cy_green/2))
        else:
            send_follow.append(254)
            send_follow.append(254)
        if(b!=0):   # 如果串口接收到数据
            res = ser.read(b)   # 解读数据
            print(res.decode())
            if res == b'A':
                ser.write(send_goal)    # 发送矩形的四个顶点
                print(send_goal)
            if res == b'B':
                send_flag = 1          # 目标信号标志位等于1
            if res == b'C':
                send_flag = 0
        if send_flag == 1:
            send_follow.append(255)
            ser.write(send_follow)    # 发送激光笔的坐标
            print(send_follow)
        cv2.imshow("1", frame)
        cv2.imshow("2", thresh_green)
        cv2.imshow("4", thresh_red)
        cv2.imshow("3", copy)
        cv2.waitKey(30)
