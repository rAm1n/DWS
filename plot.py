



from saliency.dataset import SaliencyDataset
from PIL import Image
import numpy as np
from skimage.transform import resize
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.sparse import coo_matrix
from scipy.ndimage.filters import gaussian_filter
import random





d =SaliencyDataset()
d.load('CROWD')

hts = d.get('heatmap_path', index=range(200))
times = d.get('fixation_time', index=range(200))
stim = d.get('stimuli_path', index=range(200))

path = 'duration/'




# for i in range(100):
# 	print(i)
# 	out = Image.new('RGBA', (1600, 600))
# 	ht = np.array(Image.open(hts[i]))
# 	ht = ht / ht.max()
# 	ht = cm.jet(ht)
# 	ht = ht * 255
# 	ht = ht.astype(np.uint8)
# 	st = Image.open(stim[i])
# 	time = times[i]
# 	# time = np.array(Image.open(ht))
# 	time = gaussian_filter(time, sigma=15)
# 	time = (time/time.max())
# 	time = cm.jet(time)
# 	time = time * 255
# 	time = time.astype(np.uint8)
# 	# time = Image.fromarray(time)
# 	# time = time / time.sum()
# 	out = Image.new('RGBA', (1600, 600))
# 	out.paste(Image.blend(st, Image.fromarray(ht).convert('RGB'), alpha=0.5).convert('RGB'), (0, 0))
# 	out.paste(Image.blend(st, Image.fromarray(time).convert('RGB'), alpha=0.5).convert('RGB'), (800, 000))
# 	# out.paste(Image.blend(st, ht, alpha=0.9).convert('RGB'), (0, 0))
# 	# out.paste(Image.blend(st, time, alpha=0.9).convert('RGB'), (800, 000))
# 	out.convert('RGB').save(path + str(1001 + i) + '.jpg')

im_h, im_w  = [768, 1024]


for i in range(100):
	out = Image.new('RGB', (im_w * 5, im_h*3))
	for w in range(0, im_w * 5, im_w):
		img_idx = random.randint(1,199)
		ht = np.array(Image.open(hts[img_idx]))
		ht = ht / ht.max()
		_ht, ht = ht, cm.jet(ht)
		ht = ht * 255
		ht = ht.astype(np.uint8)
		st = Image.open(stim[img_idx])
		time = times[img_idx]
		# time = (time - time.min()) / (time.max() - time.min())
		# time = np.array(Image.open(ht))
		time = gaussian_filter(time, sigma=15)
		time = (time/time.max())
		_time, time = time, cm.jet(time)
		time = time * 255
		time = time.astype(np.uint8)
		diff = _ht - _time
		diff = (diff - diff.min()) / (diff.max()-diff.min())
		diff = cm.jet(_ht - _time)
		diff = diff * 255
		diff = diff.astype(np.uint8)
		out.paste(Image.blend(st, Image.fromarray(ht).convert('RGB'), alpha=0.5).convert('RGB'), (w, 0))
		out.paste(Image.blend(st, Image.fromarray(time).convert('RGB'), alpha=0.5).convert('RGB'), (w, im_h))
		out.paste(Image.blend(st, Image.fromarray(diff).convert('RGB'), alpha=0.5).convert('RGB'), (w, 2 * im_h))
	out.save(path + str(1000 + i + 1 ) + '.jpg' )
	print(i)


