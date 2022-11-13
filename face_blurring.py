import cv2
import sys
import os

# storing path of input file in a variable
path = sys.argv[1]


def img_processing(path):
    # load the input image and convert it to grayscale
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # load the face detector
    face_detector = cv2.CascadeClassifier(
        "lib/haarcascade_frontalface_default.xml")

    # detect the faces in the grayscale image
    face_rects = face_detector.detectMultiScale(
        gray, 1.14, 6, minSize=(65, 65))

    # go through the face bounding boxes
    for (x, y, w, h) in face_rects:

        # draw a rectangle around the face on the input image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def vid_processing(path):
    video_cap = cv2.VideoCapture(path)
    face_detector = cv2.CascadeClassifier("lib/haarcascade_frontalface_default.xml")
    while True:
        # get the next frame, resize it, and convert it to grayscale
        succes, frame = video_cap.read()
        frame = cv2.resize(frame, (640, 640))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        face_rects = face_detector.detectMultiScale(gray, 1.04, 5, minSize=(30, 30))
        
        for (x, y, w, h) in face_rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
        cv2.imshow("frame", frame)
            # wait for 1 milliseconde and if the q key is pressed, we break the loop
        if cv2.waitKey(1) == ord("q"):
            break
    # release the video capture and close all windows
    video_cap.release()
    cv2.destroyAllWindows()


def main():
    # list of image and video type extensions
    vid_extensions = ['.MP4', '.M4P', '.M4V', '.MPG', '.MP2', '.MPEG', '.MPE',
                      '.MPV', '.MOV', '.QT', '.AVI', '.WMV', '.FLV', '.SWF', '.WEBM', '.OGG']
    img_extensions = ['.JPG', '.PNG', '.GIF', '.WEBP', '.TIFF', '.PSD',
                      '.RAW', '.BMP', '.HEIF', '.INDD', '.JPEG', '.SVG', '.AI', '.EPS', '.PDF']

    # splitting path into file name and extension to get the extension
    split_path = os.path.splitext(path)
    path_ext = split_path[1]

    # if path is an image, redirect to image processing, otherwise redirect to video logic
    if path_ext.upper() in img_extensions:
        img_processing(path)
    elif path_ext.upper() in vid_extensions:
        vid_processing(path)
    else:
        print('Invalid file type! Make sure the file is an image or a video file')


if __name__ == "__main__":
    main()
