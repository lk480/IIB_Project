import os
import cv2
from tqdm import tqdm
from pystackreg import StackReg
from skimage import io
import numpy as np

image_folder = '/Users/lohithkonathala1/20230905_Jed_20x_final2'
video_name = 'JedEye_video2_330.mp4'

images = sorted(os.listdir(image_folder))
images = images[:330]


sr = StackReg(StackReg.AFFINE)

transformed_images = []

first_frame = io.imread(os.path.join(image_folder, images[0]))
height, width = first_frame.shape[:2]

video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'MJPG'), 60, (width, height))

# Perform the registration and write to video
for image_name in tqdm(images):
    image_path = os.path.join(image_folder, image_name)
    image = io.imread(image_path)

    # Use the first image as the reference
    if image_name == images[0]:
        ref_image = image

    transformed_image = sr.register_transform(ref_image, image)
    transformed_images.append(transformed_image)

    # Convert the transformed image to 8-bit and then to BGR color space
    transformed_image_8bit = np.clip(transformed_image, 0, 255).astype('uint8')
    transformed_image_bgr = cv2.cvtColor(transformed_image_8bit, cv2.COLOR_RGB2BGR)

    video.write(transformed_image_bgr)

cv2.destroyAllWindows()
video.release()
