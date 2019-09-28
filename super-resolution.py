import numpy as np
from PIL import Image
from ISR.models import RDN
import os
from glob import glob

img_dir = '/Users/SawS/Documents/final_year_thesis/sample-images/for-super-resolution/'
# img_dir = '/Users/SawS/Documents/final_year_thesis/sample-images/for-super-resolution/result-images/'

# For single image file
img_file = '/Users/SawS/Documents/final_year_thesis/sample-images/for-super-resolution/sample-9.png'
weight_dir = '/Users/SawS/Documents/final_year_thesis/image-super-resolution/weights/sample_weights/rdn-C6-D20-G64' \
             '-G064-x2/ArtefactCancelling/rdn-C6-D20-G64-G064-x2_ArtefactCancelling_epoch219.hdf5'
img_files = glob(os.path.join(img_dir, '*.png'))
save_img_dir = '/Users/SawS/Documents/final_year_thesis/sample-images/for-super-resolution/sr-images/'
# save_img_dir = '/Users/SawS/Documents/final_year_thesis/sample-images/for-super-resolution/sr-images/results/'

# print(img_files)
# for img_file in img_files:
#     img = Image.open(img_file)
#     lr_img = np.array(img)
#     lr_img = lr_img[:, :, :3]
#     rdn = RDN(arch_params={'C': 6, 'D': 20, 'G': 64, 'G0': 64, 'x': 2})
#     rdn.model.load_weights(weight_dir)
#     img_name = 'super-resolution-' + img_file.split('/')[-1]
#     sr_img = rdn.predict(lr_img)
#     sr_img = Image.fromarray(sr_img)
#     sr_img.save(save_img_dir + img_name)


img = Image.open(img_file)
lr_img = np.array(img)
lr_img = lr_img[:, :, :3]
rdn = RDN(arch_params={'C': 6, 'D': 20, 'G': 64, 'G0': 64, 'x': 2})
rdn.model.load_weights(weight_dir)
img_name = 'super-resolution-' + img_file.split('/')[-1]
sr_img = rdn.predict(lr_img)
sr_img = Image.fromarray(sr_img)
sr_img.save(save_img_dir + img_name)
