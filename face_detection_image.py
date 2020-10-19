import cv2
import imutils


def Haarcasfacedetect(faceCascade, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        detect = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 4)
        return detect


faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


def face_detect(image):
    image = cv2.imread(image)
    resized_image = imutils.resize(image, width=min(800, image.shape[1]))

    output = Haarcasfacedetect(faceCascade, resized_image)

    cv2.imshow("Image", output)
    cv2.waitKey(0)

    return output

image = "face2.jpg"
face_detect(image)
