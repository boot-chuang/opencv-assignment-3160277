import cv2
import numpy as np


net = cv2.dnn.readNet("frozen_east_text_detection.pb")

image = cv2.imread("english.jpg")

if image is None:
    print("Error: Image not found")
    exit()

image = cv2.imread('english.jpg')

(h, w) = image.shape[:2]

start_x = 0
start_y = 0

cropped_image = image[start_y:start_y + 512, start_x:start_x + 480]

cv2.imwrite('cropped_english.jpg', cropped_image)

"""cv2.imshow("Cropped Image", cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()"""

(h, w) = cropped_image.shape[:2]

blob = cv2.dnn.blobFromImage(cropped_image, 1.0, (w, h), (123.68, 116.779, 103.939), swapRB=True, crop=False)

net.setInput(blob)

scores, geometry = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])

(numRows, numCols) = scores.shape[2:4]
rects = []
confidences = []

for y in range(0, numRows):
    scoresData = scores[0, 0, y]
    x0 = geometry[0, 0, y]
    x1 = geometry[0, 1, y]
    x2 = geometry[0, 2, y]
    x3 = geometry[0, 3, y]
    anglesData = geometry[0, 4, y]

    # 遍历每一个框
    for x in range(0, numCols):
        score = scoresData[x]
        if score < 0.3:
            continue

        # 计算框的角度和坐标
        angle = anglesData[x]
        (offsetX, offsetY) = (x * 4.0, y * 4.0)
        angleRad = angle * np.pi / 180.0
        cos = np.cos(angleRad)
        sin = np.sin(angleRad)
        h = x0[x] + x2[x]
        w = x1[x] + x3[x]
        endX = int(offsetX + (cos * x1[x]) + (sin * x2[x]))
        endY = int(offsetY - (sin * x1[x]) + (cos * x2[x]))
        startX = int(endX - w)
        startY = int(endY - h)

        # 绘制边界框
        cv2.rectangle(cropped_image, (startX, startY), (endX, endY), (255, 0, 0), 1)

# 显示结果
cv2.imshow("Text Detection", cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()