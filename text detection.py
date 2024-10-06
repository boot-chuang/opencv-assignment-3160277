import cv2
import pytesseract
import numpy as np


def EAST_text_detection(image_path):
    net = cv2.dnn.readNet("frozen_east_text_detection.pb")

    image = cv2.imread(image_path)

    # 裁剪图像
    start_x = 0
    start_y = 0
    cropped_image = image[start_y:start_y + 512, start_x:start_x + 480]
    cv2.imwrite('cropped_english.jpg', cropped_image)

    # 用户选择预处理方式
    print("Choose preprocessing methods (select multiple by separating with commas, e.g., 1,2,3):")
    print("1. Noise Removal")
    print("2. Dilation")
    print("3. Erosion")
    print("4. Opening")
    choice = input("Enter your choices (1-4): ")

    choices = [int(x.strip()) for x in choice.split(',')]

    preprocessed_image = cropped_image.copy()

    for choice in choices:
        if choice == 1:
            preprocessed_image = remove_noise(preprocessed_image)
        elif choice == 2:
            preprocessed_image = dilate(preprocessed_image)
        elif choice == 3:
            preprocessed_image = erode(preprocessed_image)
        elif choice == 4:
            preprocessed_image = opening(preprocessed_image)

    (h, w) = preprocessed_image.shape[:2]

    blob = cv2.dnn.blobFromImage(preprocessed_image, 1.0, (w, h), (123.68, 116.779, 103.939), swapRB=True, crop=False)

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

        for x in range(0, numCols):
            score = scoresData[x]
            if score < 0.3:
                continue

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

            cv2.rectangle(preprocessed_image, (startX, startY), (endX, endY), (255, 0, 0), 1)

    cv2.imshow("Text Detection", preprocessed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def character_detection(img):
    # 用户选择预处理方式
    print("Choose preprocessing methods (select multiple by separating with commas, e.g., 1,2,3,4,5,6,7):")
    print("1. Noise Removal")
    print("2. Thresholding")
    print("3. Dilation")
    print("4. Erosion")
    print("5. Opening")
    print("6. Canny Edge Detection")
    print("7. Skew Correction")
    choice = input("Enter your choices (1-7): ")

    choices = [int(x.strip()) for x in choice.split(',')]

    preprocessed_image = img.copy()

    for choice in choices:
        if choice == 1:
            preprocessed_image = remove_noise(preprocessed_image)
        elif choice == 2:
            preprocessed_image = thresholding(preprocessed_image)
        elif choice == 3:
            preprocessed_image = dilate(preprocessed_image)
        elif choice == 4:
            preprocessed_image = erode(preprocessed_image)
        elif choice == 5:
            preprocessed_image = opening(preprocessed_image)
        elif choice == 6:
            preprocessed_image = canny(preprocessed_image)
        elif choice == 7:
            preprocessed_image = deskew(preprocessed_image)


    hImg,wImg=preprocessed_image .shape
    boxes = pytesseract.image_to_boxes(preprocessed_image )
    for b in boxes.splitlines():
        #print(b)
        b = b.split(' ')
        #print(b)
        x,y,w,h = int(b[1]),int(b[2]),int(b[3]),int(b[4])
        cv2.rectangle(preprocessed_image ,(x,hImg-y),(w,hImg-h),(0,255,0),1)
        # cv2.putText(img_copy ,b[0],(x+30,hImg-y),cv2.FONT_HERSHEY_SIMPLEX,1,(50,50,255),2)

    scale_percent = 50
    width = int(preprocessed_image .shape[1] * scale_percent / 100)
    height = int(preprocessed_image .shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_img_2 = cv2.resize(preprocessed_image , dim, interpolation=cv2.INTER_AREA)

    cv2.imshow('Original Image',resized_img_2 )
    cv2.waitKey(0)
    cv2.destroyAllWindows()




def word_detection(img):
    # 用户选择预处理方式
    print("Choose preprocessing methods (select multiple by separating with commas, e.g., 1,2,3,4,5,6,7):")
    print("1. Noise Removal")
    print("2. Thresholding")
    print("3. Dilation")
    print("4. Erosion")
    print("5. Opening")
    print("6. Canny Edge Detection")
    print("7. Skew Correction")
    choice = input("Enter your choices (1-7): ")

    choices = [int(x.strip()) for x in choice.split(',')]

    preprocessed_image = img.copy()

    for choice in choices:
        if choice == 1:
            preprocessed_image = remove_noise(preprocessed_image)
        elif choice == 2:
            preprocessed_image = thresholding(preprocessed_image)
        elif choice == 3:
            preprocessed_image = dilate(preprocessed_image)
        elif choice == 4:
            preprocessed_image = erode(preprocessed_image)
        elif choice == 5:
            preprocessed_image = opening(preprocessed_image)
        elif choice == 6:
            preprocessed_image = canny(preprocessed_image)
        elif choice == 7:
            preprocessed_image = deskew(preprocessed_image)
            hImg, wImg = preprocessed_image.shape

    recognized_text = []

    boxes = pytesseract.image_to_data(preprocessed_image)

    for x,b in enumerate(boxes.splitlines()):
        if x!=0:
            b = b.split()
            print(b)
            if len(b)==12:
                x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
                cv2.rectangle(preprocessed_image, (x, y), (w + x, h+y), (0, 255, 0), 2)
                cv2.putText(preprocessed_image, b[11], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)
                recognized_text.append(b[11])

    with open('recognized_text.txt', 'w') as file:
        for text in recognized_text:
            file.write(text + '\n')
    scale_percent = 50
    width = int(preprocessed_image.shape[1] * scale_percent / 100)
    height = int(preprocessed_image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_img = cv2.resize(preprocessed_image, dim, interpolation=cv2.INTER_AREA)

    cv2.imshow('Original Image',resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 预处理函数
def remove_noise(image):
    return cv2.medianBlur(image, 5)


def thresholding(image):
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def canny(image):
    return cv2.Canny(image, 100, 200)


def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def main():
        user_input=input("if you want use EAST to detect text,press 1\nif you want use tesseract to detect character,press 2\n"
                         "if you want use tesseract to detect words,press 3:")

        pytesseract.pytesseract.tesseract_cmd = 'D:\\Tesseract-OCR\\tesseract.exe'
        img = cv2.imread('report.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        img = gray
        if user_input=="1":
            image_path = 'english.jpg'
            EAST_text_detection(image_path)
        elif user_input=="2":
            character_detection(img)
        elif user_input=="3":
            word_detection(img)
main()


