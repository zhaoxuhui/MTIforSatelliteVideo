# coding=utf-8
import cv2
import numpy as np
import math


def calcVelocity(x1, x2, y1, y2, res, wT):
    dist = pow(pow(y1 - y2, 2) + pow(x1 - x2, 2), 0.5) * res
    v = dist / (wT / 1000.0) * 3.6
    return v


# ---------------必要参数---------------
# 待识别视频路径
video_path = 'E:\\object\\test_real.mp4'
# 卫星视频地表分辨率
resolution = 2
# 估计最快运动速度
velocity = 850
# ---------------必要参数---------------

# ---------------可选参数---------------
# 提取的模板是否为正方形
isSquare = True
# 是否自动根据速度信息计算阈值
isAutoDisThresh = True
# 是否为多模板
isMultiTemplate = True
# 是否采用均值对轨迹进行平滑
isSmooth = True
# 相邻轨迹点之间的距离阈值
dis_thresh = 10
# 多模板个数
templateNum = 8
# 初始待选窗口大小半径
range_d = 30
# 灰度阈值敏感度，越大灰度阈值越低
gray_factor = 0.2
# 识别框缩放因子，越大绘制的识别框越大
scale_factor = 1.5
# 模板缩放因子，越大模板图像越大
template_factor = 0.6
# 识别框颜色
color = (0, 0, 255)
# 输出路径
parent_path = video_path.replace(video_path.split("\\")[-1], '')
out_path = parent_path + "object.avi"
out_path2 = parent_path + "track.avi"
out_path3 = parent_path + "points.txt"
out_path4 = parent_path + "velocity.txt"
out_path5 = parent_path + "template.jpg"
# ---------------可选参数---------------

# 循环变量
count = 0

# 打开视频
cap = cv2.VideoCapture(video_path)
cap2 = cv2.VideoCapture(video_path)
# 获取视频图像大小
# video_h对应竖直方向，video_w对应水平方向
video_h = int(cap.get(4))
video_w = int(cap.get(3))
total = int(cap.get(7))

# 新建一张与视频等大的影像用于绘制轨迹
track = np.zeros((video_h, video_w, 3), np.uint8)

# tlp用于存放待选窗口的左上角点
tlp = []
# rbp用于存放待选窗口的右下角点
rbp = []
# bottom_right_points用于存放目标区域的右下角点
bottom_right_points = []
# center_points用于存放目标区域的中心点
center_points = []
# trackPoints用于存放目标区域的左上角点
trackPoints = []
# Vs用于存放目标各帧速度
Vs = []

# 根据视频信息计算每一帧的等待时间
if cap.get(5) != 0:
    waitTime = int(1000.0 / cap.get(5))
    fps = cap.get(5)

# 如果为真，则自动确定距离阈值
if isAutoDisThresh:
    # 计算物体帧间最大运动范围(像素)
    max_range = math.ceil((5.0 * velocity) / (18.0 * resolution * (fps - 1)))
    # 计算最大移动距离，作为阈值
    dis_thresh = math.ceil(pow(pow(max_range, 2) + pow(max_range, 2), 0.5))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(out_path, fourcc, fps, (video_w, video_h))
out2 = cv2.VideoWriter(out_path2, fourcc, fps, (video_w, video_h))

# 首先提取模板图像
if cap2.isOpened():
    # 读取前两帧
    ret, frame1 = cap2.read()
    ret, frame2 = cap2.read()
    # 相减做差
    sub = cv2.subtract(frame1, frame2)
    # 得到的结果灰度化
    gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
    # 判断作差后的结果是否全为0
    if gray.max() != 0:
        # 找到最大值位置
        loc = np.where(gray == gray.max())
        loc_x = loc[1][0]
        loc_y = loc[0][0]

        # 以loc为中心，range_d为距离向外拓展得到window
        win_tl_x = loc_x - range_d
        win_tl_y = loc_y - range_d
        win_rb_x = loc_x + range_d
        win_rb_y = loc_y + range_d

        # 一些越界的判断
        if win_tl_x < 0:
            win_tl_x = 0
        if win_tl_y < 0:
            win_tl_y = 0
        if win_rb_x > video_w:
            win_rb_x = video_w
        if win_rb_y > video_h:
            win_rb_y = video_h

        # 根据窗口坐标提取窗口内容
        win_ini = cv2.cvtColor(frame1[win_tl_y:win_rb_y, win_tl_x:win_rb_x, :], cv2.COLOR_BGR2GRAY)
        # 获取最大值位置对应的灰度值
        tem_img = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        # 由最大值对应灰度值计算合适的灰度阈值
        gray_thresh = tem_img[loc_y, loc_x] - gray_factor * tem_img[loc_y, loc_x]
        # 初始窗口二值化处理
        ret, thresh = cv2.threshold(win_ini, gray_thresh, 255, cv2.THRESH_BINARY)

        # 在初始窗口中寻找轮廓
        img2, contours, hi = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # 有可能找到多个轮廓，但认为包含点数最多的那个轮廓是要找的轮廓
        length = []
        for item in contours:
            length.append(item.shape[0])
        target_contour = contours[length.index(max(length))]
        # 获取目标轮廓的坐标信息
        x, y, w, h = cv2.boundingRect(target_contour)

        if isSquare:
            # 保证提取的模板为正方形
            tem_tl_x = win_tl_x + x
            tem_tl_y = win_tl_y + y
            tem_rb_x = win_tl_x + x + w
            tem_rb_y = win_tl_y + y + h

            center_x = (tem_tl_x + tem_rb_x) / 2
            center_y = (tem_tl_y + tem_rb_y) / 2

            delta = int(template_factor * max(w, h))

            real_tl_x = center_x - delta
            real_rb_x = center_x + delta
            real_tl_y = center_y - delta
            real_rb_y = center_y + delta
        else:
            # 不保证模板为正方形
            real_tl_x = win_tl_x + x
            real_tl_y = win_tl_y + y
            real_rb_x = win_tl_x + x + w
            real_rb_y = win_tl_y + y + h

        # 一些越界判断
        if real_tl_x < 0:
            real_tl_x = 0
        if real_tl_y < 0:
            real_tl_y = 0
        if real_rb_x > video_w:
            real_rb_x = video_w
        if real_rb_y > video_h:
            real_rb_y = video_h

        # 提取模板内容
        template = frame1[real_tl_y:real_rb_y, real_tl_x:real_rb_x, :]

        # 获取模板的宽高，h竖直方向，w水平方向
        h = template.shape[0]
        w = template.shape[1]
        d = max(w, h)

        # 是否是多模板匹配
        if isMultiTemplate:
            if templateNum == 16:
                M22_5 = cv2.getRotationMatrix2D((d / 2, d / 2), -22.5, 1)
                M45 = cv2.getRotationMatrix2D((d / 2, d / 2), -45, 1)
                M67_5 = cv2.getRotationMatrix2D((d / 2, d / 2), -67.5, 1)
                M90 = cv2.getRotationMatrix2D((d / 2, d / 2), -90, 1)
                M112_5 = cv2.getRotationMatrix2D((d / 2, d / 2), -112.5, 1)
                M135 = cv2.getRotationMatrix2D((d / 2, d / 2), -135, 1)
                M157_5 = cv2.getRotationMatrix2D((d / 2, d / 2), -157.5, 1)
                M180 = cv2.getRotationMatrix2D((d / 2, d / 2), -180, 1)
                M202_5 = cv2.getRotationMatrix2D((d / 2, d / 2), -202.5, 1)
                M225 = cv2.getRotationMatrix2D((d / 2, d / 2), -225, 1)
                M247_5 = cv2.getRotationMatrix2D((d / 2, d / 2), -247.5, 1)
                M270 = cv2.getRotationMatrix2D((d / 2, d / 2), -270, 1)
                M292_5 = cv2.getRotationMatrix2D((d / 2, d / 2), -292.5, 1)
                M315 = cv2.getRotationMatrix2D((d / 2, d / 2), -315, 1)
                M337_5 = cv2.getRotationMatrix2D((d / 2, d / 2), -337.5, 1)

                template22_5 = cv2.warpAffine(template, M22_5, (d, d))
                template45 = cv2.warpAffine(template, M45, (d, d))
                template67_5 = cv2.warpAffine(template, M67_5, (d, d))
                template90 = cv2.warpAffine(template, M90, (d, d))
                template112_5 = cv2.warpAffine(template, M112_5, (d, d))
                template135 = cv2.warpAffine(template, M135, (d, d))
                template157_5 = cv2.warpAffine(template, M157_5, (d, d))
                template180 = cv2.warpAffine(template, M180, (d, d))
                template202_5 = cv2.warpAffine(template, M202_5, (d, d))
                template225 = cv2.warpAffine(template, M225, (d, d))
                template247_5 = cv2.warpAffine(template, M247_5, (d, d))
                template270 = cv2.warpAffine(template, M270, (d, d))
                template292_5 = cv2.warpAffine(template, M292_5, (d, d))
                template315 = cv2.warpAffine(template, M315, (d, d))
                template337_5 = cv2.warpAffine(template, M337_5, (d, d))
            elif templateNum == 8:
                M45 = cv2.getRotationMatrix2D((d / 2, d / 2), -45, 1)
                M90 = cv2.getRotationMatrix2D((d / 2, d / 2), -90, 1)
                M135 = cv2.getRotationMatrix2D((d / 2, d / 2), -135, 1)
                M180 = cv2.getRotationMatrix2D((d / 2, d / 2), -180, 1)
                M225 = cv2.getRotationMatrix2D((d / 2, d / 2), -225, 1)
                M270 = cv2.getRotationMatrix2D((d / 2, d / 2), -270, 1)
                M315 = cv2.getRotationMatrix2D((d / 2, d / 2), -315, 1)

                template45 = cv2.warpAffine(template, M45, (d, d))
                template90 = cv2.warpAffine(template, M90, (d, d))
                template135 = cv2.warpAffine(template, M135, (d, d))
                template180 = cv2.warpAffine(template, M180, (d, d))
                template225 = cv2.warpAffine(template, M225, (d, d))
                template270 = cv2.warpAffine(template, M270, (d, d))
                template315 = cv2.warpAffine(template, M315, (d, d))
            elif templateNum == 4:
                M90 = cv2.getRotationMatrix2D((d / 2, d / 2), -90, 1)
                M180 = cv2.getRotationMatrix2D((d / 2, d / 2), -180, 1)
                M270 = cv2.getRotationMatrix2D((d / 2, d / 2), -270, 1)

                template90 = cv2.warpAffine(template, M90, (d, d))
                template180 = cv2.warpAffine(template, M180, (d, d))
                template270 = cv2.warpAffine(template, M270, (d, d))

        cv2.imshow("Template", template)
        cv2.imwrite(out_path5, template)

        offset = int(scale_factor * d)
        # 计算待选窗口左上角点坐标
        tlx = loc_x - d
        tly = loc_y - d
        # 判断是否越界，越界则设置为0
        if tlx < 0:
            tlx = 0
        if tly < 0:
            tly = 0
        range_tl = (tlx, tly)

        # 计算待选窗口右下角点坐标
        rbx = loc_x + w + d
        rby = loc_y + h + d
        # 判断是否越界，越界设置为视频长宽最大值
        if rbx > video_w:
            rbx = video_w
        if rby > video_h:
            rby = video_h
        range_rb = (rbx, rby)

        # 放入角点坐标列表
        tlp.append(range_tl)
        rbp.append(range_rb)
        cap2.release()

# 然后进行模板匹配
while cap.isOpened():
    # 读取每帧内容
    ret, frame = cap.read()
    # 判断帧内容是否为空，不为空继续
    if frame is None:
        break
    else:
        # 是否为多模板匹配模式
        if isMultiTemplate:
            if templateNum == 16:
                # 逐个模板进行匹配
                res = cv2.matchTemplate(frame[tlp[count][1]:rbp[count][1], tlp[count][0]:rbp[count][0], :], template,
                                        cv2.TM_CCOEFF_NORMED)
                res22_5 = cv2.matchTemplate(frame[tlp[count][1]:rbp[count][1], tlp[count][0]:rbp[count][0], :],
                                            template22_5,
                                            cv2.TM_CCOEFF_NORMED)
                res67_5 = cv2.matchTemplate(frame[tlp[count][1]:rbp[count][1], tlp[count][0]:rbp[count][0], :],
                                            template67_5,
                                            cv2.TM_CCOEFF_NORMED)
                res112_5 = cv2.matchTemplate(frame[tlp[count][1]:rbp[count][1], tlp[count][0]:rbp[count][0], :],
                                             template112_5,
                                             cv2.TM_CCOEFF_NORMED)
                res157_5 = cv2.matchTemplate(frame[tlp[count][1]:rbp[count][1], tlp[count][0]:rbp[count][0], :],
                                             template157_5,
                                             cv2.TM_CCOEFF_NORMED)
                res202_5 = cv2.matchTemplate(frame[tlp[count][1]:rbp[count][1], tlp[count][0]:rbp[count][0], :],
                                             template202_5,
                                             cv2.TM_CCOEFF_NORMED)
                res247_5 = cv2.matchTemplate(frame[tlp[count][1]:rbp[count][1], tlp[count][0]:rbp[count][0], :],
                                             template247_5,
                                             cv2.TM_CCOEFF_NORMED)
                res292_5 = cv2.matchTemplate(frame[tlp[count][1]:rbp[count][1], tlp[count][0]:rbp[count][0], :],
                                             template292_5,
                                             cv2.TM_CCOEFF_NORMED)
                res337_5 = cv2.matchTemplate(frame[tlp[count][1]:rbp[count][1], tlp[count][0]:rbp[count][0], :],
                                             template337_5,
                                             cv2.TM_CCOEFF_NORMED)

                res90 = cv2.matchTemplate(frame[tlp[count][1]:rbp[count][1], tlp[count][0]:rbp[count][0], :],
                                          template90,
                                          cv2.TM_CCOEFF_NORMED)
                res180 = cv2.matchTemplate(frame[tlp[count][1]:rbp[count][1], tlp[count][0]:rbp[count][0], :],
                                           template180,
                                           cv2.TM_CCOEFF_NORMED)
                res270 = cv2.matchTemplate(frame[tlp[count][1]:rbp[count][1], tlp[count][0]:rbp[count][0], :],
                                           template270,
                                           cv2.TM_CCOEFF_NORMED)

                res45 = cv2.matchTemplate(frame[tlp[count][1]:rbp[count][1], tlp[count][0]:rbp[count][0], :],
                                          template45,
                                          cv2.TM_CCOEFF_NORMED)
                res135 = cv2.matchTemplate(frame[tlp[count][1]:rbp[count][1], tlp[count][0]:rbp[count][0], :],
                                           template135,
                                           cv2.TM_CCOEFF_NORMED)
                res225 = cv2.matchTemplate(frame[tlp[count][1]:rbp[count][1], tlp[count][0]:rbp[count][0], :],
                                           template225,
                                           cv2.TM_CCOEFF_NORMED)
                res315 = cv2.matchTemplate(frame[tlp[count][1]:rbp[count][1], tlp[count][0]:rbp[count][0], :],
                                           template315,
                                           cv2.TM_CCOEFF_NORMED)

                # 获取各模板对应的最大值
                m22_5 = np.max(res22_5)
                m67_5 = np.max(res67_5)
                m112_5 = np.max(res112_5)
                m157_5 = np.max(res157_5)
                m202_5 = np.max(res202_5)
                m247_5 = np.max(res247_5)
                m292_5 = np.max(res292_5)
                m337_5 = np.max(res337_5)

                m45 = np.max(res45)
                m135 = np.max(res135)
                m225 = np.max(res225)
                m315 = np.max(res315)

                m0 = np.max(res)
                m90 = np.max(res90)
                m180 = np.max(res180)
                m270 = np.max(res270)

                # 寻找最佳匹配结果
                m = max(m0, m22_5, m45, m67_5, m90,
                        m112_5, m135, m157_5, m180,
                        m202_5, m225, m247_5, m270,
                        m292_5, m315, m337_5)

                # 获取最佳匹配结果对应的坐标信息
                if m == m0:
                    mIndex = 0
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                elif m == m90:
                    mIndex = 90
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res90)
                elif m == m180:
                    mIndex = 180
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res180)
                elif m == m270:
                    mIndex = 270
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res270)
                elif m == m45:
                    mIndex = 45
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res45)
                elif m == m135:
                    mIndex = 135
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res135)
                elif m == m225:
                    mIndex = 225
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res225)
                elif m == m315:
                    mIndex = 315
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res315)
                elif m == m22_5:
                    mIndex = 22.5
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res22_5)
                elif m == m67_5:
                    mIndex = 67.5
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res67_5)
                elif m == m112_5:
                    mIndex = 112.5
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res112_5)
                elif m == m157_5:
                    mIndex = 157.5
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res157_5)
                elif m == m202_5:
                    mIndex = 202.5
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res202_5)
                elif m == m247_5:
                    mIndex = 247.5
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res247_5)
                elif m == m292_5:
                    mIndex = 292.5
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res292_5)
                elif m == m337_5:
                    mIndex = 337.5
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res337_5)
            elif templateNum == 8:
                res = cv2.matchTemplate(frame[tlp[count][1]:rbp[count][1], tlp[count][0]:rbp[count][0], :], template,
                                        cv2.TM_CCOEFF_NORMED)
                res90 = cv2.matchTemplate(frame[tlp[count][1]:rbp[count][1], tlp[count][0]:rbp[count][0], :],
                                          template90,
                                          cv2.TM_CCOEFF_NORMED)
                res180 = cv2.matchTemplate(frame[tlp[count][1]:rbp[count][1], tlp[count][0]:rbp[count][0], :],
                                           template180,
                                           cv2.TM_CCOEFF_NORMED)
                res270 = cv2.matchTemplate(frame[tlp[count][1]:rbp[count][1], tlp[count][0]:rbp[count][0], :],
                                           template270,
                                           cv2.TM_CCOEFF_NORMED)

                res45 = cv2.matchTemplate(frame[tlp[count][1]:rbp[count][1], tlp[count][0]:rbp[count][0], :],
                                          template45,
                                          cv2.TM_CCOEFF_NORMED)
                res135 = cv2.matchTemplate(frame[tlp[count][1]:rbp[count][1], tlp[count][0]:rbp[count][0], :],
                                           template135,
                                           cv2.TM_CCOEFF_NORMED)
                res225 = cv2.matchTemplate(frame[tlp[count][1]:rbp[count][1], tlp[count][0]:rbp[count][0], :],
                                           template225,
                                           cv2.TM_CCOEFF_NORMED)
                res315 = cv2.matchTemplate(frame[tlp[count][1]:rbp[count][1], tlp[count][0]:rbp[count][0], :],
                                           template315,
                                           cv2.TM_CCOEFF_NORMED)

                m45 = np.max(res45)
                m135 = np.max(res135)
                m225 = np.max(res225)
                m315 = np.max(res315)

                m0 = np.max(res)
                m90 = np.max(res90)
                m180 = np.max(res180)
                m270 = np.max(res270)
                m = max(m0, m45, m90, m135, m180, m225, m270, m315)

                if m == m0:
                    mIndex = 0
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                elif m == m90:
                    mIndex = 90
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res90)
                elif m == m180:
                    mIndex = 180
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res180)
                elif m == m270:
                    mIndex = 270
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res270)
                elif m == m45:
                    mIndex = 45
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res45)
                elif m == m135:
                    mIndex = 135
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res135)
                elif m == m225:
                    mIndex = 225
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res225)
                elif m == m315:
                    mIndex = 315
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res315)
            elif templateNum == 4:
                res = cv2.matchTemplate(frame[tlp[count][1]:rbp[count][1], tlp[count][0]:rbp[count][0], :], template,
                                        cv2.TM_CCOEFF_NORMED)
                res90 = cv2.matchTemplate(frame[tlp[count][1]:rbp[count][1], tlp[count][0]:rbp[count][0], :],
                                          template90,
                                          cv2.TM_CCOEFF_NORMED)
                res180 = cv2.matchTemplate(frame[tlp[count][1]:rbp[count][1], tlp[count][0]:rbp[count][0], :],
                                           template180,
                                           cv2.TM_CCOEFF_NORMED)
                res270 = cv2.matchTemplate(frame[tlp[count][1]:rbp[count][1], tlp[count][0]:rbp[count][0], :],
                                           template270,
                                           cv2.TM_CCOEFF_NORMED)
                m0 = np.max(res)
                m90 = np.max(res90)
                m180 = np.max(res180)
                m270 = np.max(res270)
                m = max(m0, m90, m180, m270)

                if m == m0:
                    mIndex = 0
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                elif m == m90:
                    mIndex = 90
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res90)
                elif m == m180:
                    mIndex = 180
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res180)
                elif m == m270:
                    mIndex = 270
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res270)
        else:
            res = cv2.matchTemplate(frame[tlp[count][1]:rbp[count][1], tlp[count][0]:rbp[count][0], :], template,
                                    cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        window = frame[tlp[count][1]:rbp[count][1], tlp[count][0]:rbp[count][0], :]
        cv2.imshow("Window", window)

        # top_left坐标顺序(水平，竖直)(→，↓)
        top_left = (max_loc[0] + tlp[count][0], max_loc[1] + tlp[count][1])
        bottom_right = (top_left[0] + w, top_left[1] + h)
        center_point = ((top_left[0] + bottom_right[0]) / 2, (top_left[1] + bottom_right[1]) / 2)

        if trackPoints.__len__() == 0:
            # 计算待选窗口左上角点坐标
            tlx = top_left[0] - d
            tly = top_left[1] - d
            # 判断是否越界，越界则设置为0
            if tlx < 0:
                tlx = 0
            if tly < 0:
                tly = 0
            range_tl = (tlx, tly)

            # 计算待选窗口右下角点坐标
            rbx = top_left[0] + w + d
            rby = top_left[1] + h + d
            # 判断是否越界，越界设置为视频长宽最大值
            if rbx > video_w:
                rbx = video_w
            if rby > video_h:
                rby = video_h
            range_rb = (rbx, rby)

            # 将待选窗口左上角点坐标和右下角点坐标依次添加到列表中
            tlp.append(range_tl)
            rbp.append(range_rb)

            # 将目标区域的左上角点、中心点、右下角点坐标依次加入列表
            trackPoints.append(top_left)
            bottom_right_points.append(bottom_right)
            center_points.append(center_point)
            cv2.circle(track, center_point, 2, (0, 0, 255), -1)
        else:
            # 加入运动连续性约束，若相邻轨迹点距离相差大于阈值，则认为错误
            distance = abs(trackPoints[-1][0] - top_left[0]) + abs(trackPoints[-1][1] - top_left[1])
            if distance > dis_thresh:
                print '100%'
                break
            else:
                # 计算待选窗口左上角点坐标
                tlx = top_left[0] - d
                tly = top_left[1] - d
                # 判断是否越界，越界则设置为0
                if tlx < 0:
                    tlx = 0
                if tly < 0:
                    tly = 0
                range_tl = (tlx, tly)

                # 计算待选窗口右下角点坐标
                rbx = top_left[0] + w + d
                rby = top_left[1] + h + d
                # 判断是否越界，越界设置为视频长宽最大值
                if rbx > video_w:
                    rbx = video_w
                if rby > video_h:
                    rby = video_h
                range_rb = (rbx, rby)

                # 将待选窗口左上角点坐标和右下角点坐标依次添加到列表中
                tlp.append(range_tl)
                rbp.append(range_rb)

                # 将目标区域的左上角点、中心点、右下角点坐标依次加入列表
                trackPoints.append(top_left)
                bottom_right_points.append(bottom_right)

                # 判断是否采用均值平滑
                if isSmooth:
                    # 采用均值平滑，平滑轨迹
                    center_point = ((center_point[0] + center_points[-1][0]) / 2,
                                    (center_point[1] + center_points[-1][1]) / 2)
                center_points.append(center_point)

                # 绘制目标识别框
                cv2.rectangle(frame,
                              (center_point[0] - offset, center_point[1] - offset),
                              (center_point[0] + offset, center_point[1] + offset),
                              color, 2)
                # 绘制运动轨迹
                cv2.line(track, center_points[-2], center_points[-1], (255, 255, 255), 1)

                # 计算速度
                Vs.append(calcVelocity(center_points[-2][0],
                                       center_points[-1][0],
                                       center_points[-2][1],
                                       center_points[-1][1],
                                       resolution,
                                       waitTime))

        # 输出目标、轨迹视频
        out.write(frame)
        out2.write(track)
        count += 1
        print round((count * 1.0 / total) * 100, 2), '%'

        # 显示结果
        cv2.imshow("Tr", track)
        cv2.imshow("Fr", frame)

        # 退出控制
        k = cv2.waitKey(waitTime) & 0xFF
        if k == 27:
            break

# 打印轨迹坐标
print trackPoints

print '相邻帧距离阈值:', dis_thresh
print '灰度阈值:', gray_thresh
print '模板缩放因子:', template_factor
print '识别框缩放因子:', scale_factor

# 输出中心点轨迹
output = open(out_path3, 'w')
for item in center_points:
    output.write(item.__str__() + "\n")

# 输出各帧速度
output2 = open(out_path4, 'w')
for item in Vs:
    output2.write(item.__str__() + "\n")

# 释放对象
cap.release()
out.release()
out2.release()
output.close()
output2.close()
