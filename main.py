import cv2
import imutils
import numpy as np
from imutils.video import VideoStream
from PIL import Image, ImageDraw

from stream_processor import do_face_detection

rtsp_url = "rtsp://admin:admin123@192.168.1.108:554/cam/realmonitor?channel=1&subtype=1"


def main():
    vs = VideoStream(rtsp_url).start()
    while True:
        frame = vs.read()
        if frame is None:
            continue

        image = Image.fromarray(frame)
        image = do_face_detection(image)
        frame = np.asarray(image)

        frame = imutils.resize(frame, width=1200)
        cv2.imshow('Camera', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    vs.stop()


if __name__ == '__main__':
    main()
