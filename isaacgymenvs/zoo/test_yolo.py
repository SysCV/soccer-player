from ultralytics import YOLO
from PIL import Image
from matplotlib import pyplot as plt
import cv2

# Load a model
# model = YOLO("yolov8m.yaml")  # build a new model from scratch
model = YOLO(
    "./dataset/best_93.pt"
)  # load a pretrained model (recommended for training)

results = model.predict(
    "https://ultralytics.com/images/bus.jpg", device="cuda:1", show=True
)  # predict on an image

# results = model.predict(
#     "https://ultralytics.com/images/bus.jpg", device="cuda:1", show=True
# )  # predict on an image

for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    cv2.imshow("pic", im_array)
    cv2.waitKey(1)
    # im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    # im.show()  # show image

# results = model.predict(
#     "https://ultralytics.com/images/bus.jpg", device="cuda:1", show=True
# )  # predict on an image
# results = model.predict(
#     "https://ultralytics.com/images/bus.jpg", device="cuda:1", show=True
# )  # predict on an image
# results = model.predict(
#     "https://ultralytics.com/images/bus.jpg", device="cuda:1", show=True
# )  # predict on an image
# results = model.predict(
#     "https://ultralytics.com/images/bus.jpg", device="cuda:1", show=True
# )  # predict on an image

# # Show the results
# for r in results:
#     im_array = r.plot()  # plot a BGR numpy array of predictions
#     im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
#     im.show()  # show image
#     # im.save("results.jpg")  # save image
