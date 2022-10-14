import os 
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import lane_marking

parent_dir = ""
path = os.path.join(parent_dir,"test_images/")

# create a dir to save output images (only if it doesn't exist)
save_dir = "test_images_output/"
save_dir = os.path.join(parent_dir,save_dir)
isexist = os.path.exists(save_dir)
if not isexist:
    os.mkdir(save_dir)
    print(f"Creating new directory whose path is:{save_dir}")

test_images = []
files = os.listdir(path)

for file in files:
    ext = os.path.splitext(file)[1]
    test_images.append(Image.open(os.path.join(path,file)))

for i in range(0, len(test_images)):
    test_images[i] = np.array(test_images[i])

for i,file in enumerate(files):
    image = lane_marking.run_lane_detection(test_images[i])
    # image = lane_marking_obj.run_lane_detection(test_images[i])
    image = Image.fromarray(image)
    save_name = os.path.join(save_dir,file)
    image.save(save_name)
