import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

# Loading the SAM model
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)

# Loading an image
img_path = "car.jpg" # replace this with any image you want
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
predictor.set_image(img)

# Providing the coordinates of the subject you want to segment [x, y]
# For extra context, you can also do 
# [x1,y1], foreground point 
# [x2, y2] background point
#
# point_lables -> [1, 0] 
# 1 = this is my subject
# 0 = this is not a part of my subject

# points for cat-sus.jpg
point_coords = np.array([[299, 281], [873, 138], [225, 245], [1000, 165]])
point_labels = np.array([1, 1, 0, 0])

# points for jake.jpg
point_coords_jake = np.array([[439, 493], [331, 775], [599, 482], [173, 721]])
point_labels_jake = np.array([1, 1, 0, 0])

# Creating an input box 
# [x1, y1, x2, y2]
# Where x1, y1 is the top left of the subject and x2, y2 is the bottom right.

#input box for jake.jpg
input_box = np.array([231, 341, 559, 978])

masks, scores, logits = predictor.predict(
    #---- cat-sus point inputs ----
    #point_coords=point_coords,
    #point_labels=point_labels,

    #---- jake point coords input ----
    #point_coords=point_coords_jake,
    #point_labels=point_labels_jake,

    #---- jake box input ----
    #box=input_box,

    multimask_output=True
)

# Uncomment to show figure
'''
plt.figure(figsize=(10, 10))
plt.imshow(img)
for mask in masks:
    plt.imshow(mask, alpha=0.5)
plt.axis("off")
plt.show()
'''