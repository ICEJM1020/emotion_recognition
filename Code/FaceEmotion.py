import cv2 as cv
import tensorflow as tf
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class EmotionDetector:
    # -1为CPU运算，0为GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    rect_color = (0, 255, 0)
    Labels = np.array(['愤怒', '厌恶', '恐惧', '开心', '伤心', '惊讶', '平淡'])
    dataLength = 7

    # 地址设置
    haar_path = 'models/haarcascade_frontalface_alt2.xml'
    path_emotion_test = 'models/test.h5'
    path_emotion_final = 'models/emotion-final.h5'

    # 加载头像检测分类器xml
    faceCascade = cv.CascadeClassifier(haar_path)

    # 加载表情分类器
    model = load_model(path_emotion_final)

    def __init__(self):
        print("Face Emotion Init...")

    # 表情检测函数
    def predict_emotion(self, face_image_gray):
        #cv.imwrite("out/origin.jpg", face_image_gray)
        resized_img = cv.resize(face_image_gray, (48, 48), interpolation=cv.INTER_AREA)
        #cv.imwrite("out/resize.jpg", resized_img)
        image = resized_img.reshape(1, 48, 48, 1)
        return self.model.predict(image)

    # 定义检测人脸的函数
    def get_face(self, pic):
        ans = self.faceCascade.detectMultiScale(
        pic,
        scaleFactor=1.1,  # 缩小比例
        minNeighbors=5,  # 指定每个候选矩阵至少包含的邻近元素个数，也就是要被检测到多少次才能算
        minSize=(30, 30),  # 最小可能的对象的大小
        #maxSize=(150, 150)  # 最大可能的对象的大小
        )
        return ans

    # 画人脸框的函数
    def draw_face_rect(self, now_pic_ans, pic_origin):
        if len(now_pic_ans) > 0:            #大于0则检测到人脸
            for faceRect in now_pic_ans:  #单独框出每一张人脸
                x, y, w, h = faceRect  #5画图
                cv.rectangle(pic_origin, (x, y), (x + w, y + h), self.rect_color, 1)

    # 把人脸剪切下来
    def cut_face(self, now_pic_ans, pic_origin, type):
        if len(now_pic_ans) > 0:
            for detect_Face in now_pic_ans:
                x, y, w, h = detect_Face
                Face = pic_origin[y:y+h, x:x+w]
                Face = cv.cvtColor(Face, cv.COLOR_BGR2GRAY)
                Emotion_Data = self.predict_emotion(Face)
                radarPic = self.draw_emotion_pic(Emotion_Data[0])
                if type == 2:
                    return radarPic
                elif type == 3:
                    cv.imshow("radar map", radarPic)
                else:
                    radarPic = cv.resize(radarPic, (int(pic_origin.shape[1]*0.3), int(pic_origin.shape[0]*0.3)))
                    pic_origin[:radarPic.shape[0], :radarPic.shape[1]] = radarPic

    # 画雷达图
    def draw_emotion_pic(self, Emotion_Data):
        data = np.array(Emotion_Data)
        angles = np.linspace(0, 2*np.pi, self.dataLength, endpoint=False)
        data = np.concatenate((data, [data[0]]))
        angles = np.concatenate((angles, [angles[0]]))

        # 利用matplotlib画图
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, data, 'bo-', linewidth=2, markersize=3)
        ax.fill(angles, data, facecolor='r', alpha=0.35)
        ax.set_thetagrids(angles*180/np.pi, self.Labels, fontproperties="SimHei", fontsize=16)
        ax.set_rlim(0, 0.7)
        ax.grid(True)

        plt.rcParams['figure.dpi'] = 100

        # 用缓存区的方法把figure以jpg的形式存储在缓存中
        temp_buffer = BytesIO()
        plt.savefig(temp_buffer, format='jpg')
        temp_buffer.seek(0)
        pic_bytes = np.asarray(bytearray(temp_buffer.read()), dtype=np.uint8)
        data_Pic = cv.imdecode(pic_bytes, cv.IMREAD_COLOR)
        return data_Pic


    def do(self, pic, type, out_name=None):
        # 图像灰化
        gray = cv.cvtColor(pic, cv.COLOR_BGR2GRAY)

        get = self.get_face(gray)
        self.draw_face_rect(get, pic)
        radar = self.cut_face(get, pic, type)


        if type == 1:
            cv.imshow("camera", pic)
        elif type == 0:
            cv.imwrite(out_name, pic)
        elif type == 2:
            cv.imwrite(out_name, radar)
        elif type == 3:
            cv.imshow("camera", pic)
