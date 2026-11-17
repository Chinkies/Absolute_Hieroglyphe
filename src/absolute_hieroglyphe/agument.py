import cv2 as cv
import os

dataset_path = "data/Image_DAO_Sith/"

if os.path.isdir(dataset_path):
    for img_name in os.listdir(dataset_path):
        if (img_name.endswith(".png")):
            image = cv.imread(dataset_path + img_name)
            assert image is not None
            print("img: ", img_name, ", infos: ", image.shape)
        else:
            print("ceci n'est pas mon image: ", img_name)