import cv2

im = cv2.imread("datasets/pole-cls/train/empty/Null_106_cam3.jpg")
h0, w0 = im.shape[:2]

print(w0 / h0 * 640)
