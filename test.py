import numpy as np
from PIL import Image

grey_image = Image.open('greyscale.png', 'r')

# Image._show(grey_image)
np_array = np.array(grey_image)

print('Original shape: ', np_array.shape)
grey_array = np_array[:, :, :1].reshape(466, 700)

print('Tuple length :', len((grey_array,) * 3))
rgb_array = np.stack((grey_array,) * 3, axis=-1)
print('Modified image : ', rgb_array.shape)

print(rgb_array.shape == (466, 700, 3))
# Image._show(Image.fromarray(rgb_array))

img_path_list = ['./wikiart/98884.jpg', './wikiart/9156.jpg', './wikiart/70619.jpg', './wikiart/30803.jpg',
                 './wikiart/70621.jpg', './wikiart/76343.jpg', './wikiart/88214.jpg', './wikiart/57282.jpg',
                 './wikiart/68782.jpg', './wikiart/61826.jpg', './wikiart/70684.jpg', './wikiart/32845.jpg',
                 './wikiart/70616.jpg', './wikiart/94790.jpg', './wikiart/90216.jpg', './wikiart/96707.jpg',
                 './wikiart/51839.jpg', './wikiart/3496.jpg', './wikiart/39916.jpg', './wikiart/77740.jpg',
                 './wikiart/96053.jpg', './wikiart/54844.jpg', './wikiart/16765.jpg', './wikiart/66437.jpg',
                 './wikiart/73450.jpg', './wikiart/69959.jpg', './wikiart/52973.jpg', './wikiart/69296.jpg',
                 './wikiart/101473.jpg', './wikiart/88640.jpg', './wikiart/74233.jpg', './wikiart/14910.jpg']
