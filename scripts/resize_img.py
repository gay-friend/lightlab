from lightlab.data.utils import get_images
import cv2
import math

if __name__ == "__main__":
    imgsz = 320
    for name in get_images("datasets/pole-cls", recursive=True):
        im = cv2.imread(name)
        h0, w0 = im.shape[:2]  # orig hw
        r = imgsz / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            w, h = (
                min(math.ceil(w0 * r), imgsz),
                min(math.ceil(h0 * r), imgsz),
            )
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(name, im)
            print(im.shape)
