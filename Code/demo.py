from FaceEmotion import EmotionDetector
import cv2 as cv
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(cv.__version__)
emotion = EmotionDetector()
# 主入口函数：EmotionDetector.do(pic, type, out_name=None)
# pic:输入的图片，cv::UMat格式（实际是numpy）
# type：4种模式:
#               type=0：在左上角显示雷达图，保存图片到out_name文件（需要填写out_name）
#               type=1：在左上角显示雷达图，显示输出窗口，不保存文件
#               type=2：单独输出雷达图，保存到out_name（需要填写out_name）
#               type=3：单独输出雷达图，显示图片在camera窗口，显示雷达图在radar map窗口
# 建议：使用摄像头+type3模式，可以更改，将输出结果的pic和radar返回，用于别的地方（返回到前端）


# 使用摄像头进行检测demo
# demo：detection using the camera
cap = cv.VideoCapture(0)
while (1):
    ret, img = cap.read()
    emotion.do(img, 3)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()  # 释放摄像头
cv.destroyAllWindows()  # 释放窗口资源

# 对图片进行检测demo
# demo：detection with the picture
# img = cv.imread("test/test_5.jpg")
# emotion.do(img, 2, "out_5.jpg")

