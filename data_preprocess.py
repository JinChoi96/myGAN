import os
import matplotlib.pyplot as plt
from scipy.misc import imresize

root = 'data/celebA/img_align_celeba/'
save_dir = 'data/resized_celebA/'
resize = 64

if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
if not os.path.isdir(save_dir + 'celebA'):
    os.mkdir(save_dir + 'celebA')
img_list = os.listdir(root)

for i in range(len(img_list)):
    img = plt.imread(root + img_list[i])
    img = imresize(img, (resize, resize))
    plt.imsave(fname = save_dir + 'celebA/' + img_list[i], arr = img)

print('Image preprocessing completed')

