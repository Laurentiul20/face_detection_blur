import cv2

# load the input image and convert it to grayscale
image = cv2.imread("images/test_img.jpeg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# load the face detector
face_detector = cv2.CascadeClassifier("lib/haarcascade_frontalface_default.xml")

# detect the faces in the grayscale image
face_rects = face_detector.detectMultiScale(gray, 1.04, 5, minSize=(30, 30))

# go through the face bounding boxes 
for (x, y, w, h) in face_rects:
    
    # draw a rectangle around the face on the input image
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
cv2.imshow("Image", image)
cv2.waitKey(0)