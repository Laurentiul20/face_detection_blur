import cv2
import numpy as np
import sys
import os

# storing path of input file in a variable
path = sys.argv[1]


def convolution(oldimage, kernel):

    kernel_h = kernel.shape[0]
    kernel_w = kernel.shape[1]

    if(len(oldimage.shape) == 3):
        image_pad = np.pad(oldimage, pad_width=((kernel_h // 2, kernel_h // 2), (kernel_w // 2,
                                                                                 kernel_w // 2), (0, 0)), mode='constant',
                           constant_values=0).astype(np.float32)
    elif(len(oldimage.shape) == 2):
        image_pad = np.pad(oldimage, pad_width=((kernel_h // 2, kernel_h // 2), (kernel_w // 2,
                                                                                 kernel_w // 2)), mode='constant', constant_values=0).astype(np.float32)

    h = kernel_h // 2
    w = kernel_w // 2

    image_conv = np.zeros(image_pad.shape)

    for i in range(h, image_pad.shape[0]-h):
        for j in range(w, image_pad.shape[1]-w):
            #sum = 0
            x = image_pad[i-h:i-h+kernel_h, j-w:j-w+kernel_w]
            x = x.flatten()*kernel.flatten()
            image_conv[i][j] = x.sum()
    h_end = -h
    w_end = -w

    if(h == 0):
        return image_conv[h:, w:w_end]
    if(w == 0):
        return image_conv[h:h_end, w:]

    return image_conv[h:h_end, w:w_end]


def custom_gaussian_blur(image, sigma):
    filter_size = 2 * int(4 * sigma + 0.5) + 1
    gaussian_filter = np.zeros((filter_size, filter_size), np.float32)
    m = filter_size//2
    n = filter_size//2

    for x in range(-m, m+1):
        for y in range(-n, n+1):
            x1 = 2*np.pi*(sigma**2)
            x2 = np.exp(-(x**2 + y**2)/(2 * sigma**2))
            gaussian_filter[x+m, y+n] = (1/x1)*x2

    im_filtered = np.zeros_like(image, dtype=np.float32)
    for c in range(3):
        im_filtered[:, :, c] = convolution(image[:, :, c], gaussian_filter)
    return (im_filtered.astype(np.uint8))


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

        # select the face detected area and apply the custom blurring function
        face_roi = image[y:y + h, x:x + w]
        output = custom_gaussian_blur(face_roi, 15)
        image[y:y+h, x:x+w] = output

    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def vid_processing(path):
    video_cap = cv2.VideoCapture(path)
    face_detector = cv2.CascadeClassifier(
        "lib/haarcascade_frontalface_default.xml")
    while True:
        # get the next frame, resize it, and convert it to grayscale
        succes, frame = video_cap.read()
        if succes:
            frame = cv2.resize(frame, (300, 300))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            face_rects = face_detector.detectMultiScale(
                gray, 1.02, 5, minSize=(30, 30))

            for (x, y, w, h) in face_rects:
                # Select only detected face portion for Blur
                face_color = frame[y:y + h, x:x + w]
                # Blur the Face with Gaussian Blur of Kernel
                blur = custom_gaussian_blur(face_color, 5)
                frame[y:y + h, x:x + w] = blur

            cv2.imshow("frame", frame)
        else:
            break
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
