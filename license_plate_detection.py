import io
import requests
import numpy as np
from PIL import Image, ImageDraw
from options import Options

opts = Options()


def do_license_plate_detection(image):
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    buf.seek(0)
    # how to use alpr (automatic license plate recognition)
    # https://github.com/alpr-org/alpr/blob/master/README.md
    with requests.Session() as session:
        response = session.post(opts.endpoint("vision/face/recognize"),
                                files={"image": ('image.png', buf, 'image/png')},
                                data={"min_confidence": 0.5}).json()

    predictions = []
    if response["success"]:
        predictions = response["predictions"]
    if predictions is None:
        predictions = []

    draw = ImageDraw.Draw(image)
    for item in predictions:
        label = ""
        conf = item["confidence"]
        if conf > 0.6:
            label = item["userid"]
        y_max = int(item["y_max"])
        y_min = int(item["y_min"])
        x_max = int(item["x_max"])
        x_min = int(item["x_min"])

        draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="red", width=5)
        draw.text((x_min, y_min), f"{label}")
        draw.text((x_min, y_min - 10), f"{round(conf * 100.0, 0)}")

    return image
