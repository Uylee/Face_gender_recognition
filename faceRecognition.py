from keras.models import load_model
import os
import face_recognition
import cv2
import numpy as np

# 这是一个在网络摄像头上对实时视频进行人脸识别的演示。




# 模型数据图片目录
path = "img/face_recognition"
# 性别模型
face_classifier = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

# 首先需要将模型数据图片的人脸读入
known_face_encoding = []
known_face_names = []
for fn in os.listdir(path):  #fn 表示的是文件名q
    print(path + "/" + fn)
    known_face_encoding.append(face_recognition.face_encodings(face_recognition.load_image_file(path + "/" + fn))[0])
    fn = fn[:(len(fn) - 4)]  #截取图片名（这里应该把images文件中的图片名命名为为人物名）
    known_face_names.append(fn)  #图片名字列表
    gender_classifier = load_model("classifier/gender_models/simple_CNN.81-0.96.hdf5")
    gender_labels = {0: 'female', 1: 'male'}


# 开启摄像头
video_capture = cv2.VideoCapture(0)

# 进行姓名识别
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # 从视频流中读取图片
    ret, frame = video_capture.read()

    # 将图片的大小进行调整
    try:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    except:
        continue
    # 把图片从BGR形式（opencv形式）转换成RGB形式（face_recognition用）
    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []

        # 将从视频中截取到的人脸的编码信息，与人脸库中的人脸编码信息进行比较，来确定姓名。
        for face_encoding in face_encodings:
            # 是否与已有的人脸图片匹配
            matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encoding, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # 显示结果
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4


        # 进行性别识别
        face = frame[(right - 60):(right + left + 60), (top - 30):(top + bottom + 30)]
        try:
            face = cv2.resize(face, (48, 48))
        except:
            continue
        face = np.expand_dims(face, 0)
        face = face / 255.0
        gender_label_arg = np.argmax(gender_classifier.predict(face))
        gender = gender_labels[gender_label_arg]

        # 画一个图框
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # 做标签
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        cv2.putText(frame, gender, (left + 6, top - 6), font, 1.0,(255, 255, 255), 1)

    # 显示结果
    cv2.imshow('Video', frame)

    # 按“q”退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
