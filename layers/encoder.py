# reference : https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision.models import vgg19
import math
import torch
import torch.nn.functional as F



e_config = {
	'VGG16':{
			'arch' : [['64', '64', 'M'], ['128','128', 'M'], ['256', '256', '256', 'M'],
				['512', '512', '512', 'M'], ['512' , '512', '512', 'M']],
			'scale_factor' : 32
		},
	'DVGG16' :{
			'arch' : [['64', '64', 'M'], ['128','128', 'M'], ['256', '256', '256', 'M'],
				['512', '512', '512'], ['512d', '512d', '512d']],
			'scale_factor' : 8
		},
	'VGG16AM' :{
			'arch' : [['64', '64'], ['M', '128','128'], ['M', '256', '256', '256'],
				['M', '512', '512', '512'], ['512', '512d', '512']],
			'scale_factor' : 32
		},

	'DVGG19' : {
			'arch' : [['64', '64', 'M' , '128', '128', 'M' , '256', '256', '256', '256', 'M',
			 '512', '512', '512', '512'], ['512d', '512d', '512d', '512d']],
			 'scale_factor' : 8,
	}
}


def make_encoder(name, **kwargs):
	if name == 'DVGG19':
		model = DeepGaze(make_layers(e_config['DVGG19']['arch']))
	else:
		model = Encoder(make_layers(e_config[name]['arch']), **kwargs)

	# if pretrained:
	# 	model.load_vgg_weights()
	return model




class Encoder(nn.Module):

	def __init__(self, features):
		super(Encoder, self).__init__()

		self.features = features
		self.classifier = nn.Conv2d(512,1, kernel_size=1)
		# self.classifier_fov = nn.Conv2d(512,1, kernel_size=1)
		self.sigmoid = nn.Sigmoid()
		self.softmax = F.log_softmax

	def forward(self, x, layers=range(5)):


		feat = list()
		for name, module in self.features._modules.items():
			x = module(x)
			if int(name) in layers:
				feat.append(x)

		sal = self.classifier(feat[-1])
		# fov = self.classifier_fov(feat[-1])

		# return [feat, self.sigmoid(sal)]#, self.sigmoid(fov)]
		# return [feat, feat]
		b, c, w, h = sal.size()
		#return self.sigmoid(sal.view(b,-1)).view(b,c,w,h)

		return self.softmax(sal.view(b,-1)).view(b,c,w,h)
		return self.sigmoid(sal)
		#return sal

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()

	# def load_weights(self,w_path='weights/encoder-best-DVGG16-S3.pth.tar'):
	# def load_weights(self,w_path='weights/encoder-best-DVGG16-S3.pth.tar'):
	def load_weights(self,w_path):
		if not w_path:
			sigma = self.config['dataset']['first_blur_sigma']
			e_name = self.config['model']['name'].split('_')[0]
			filename = 'weights/encoder-best-{0}-S{1}.pth.tar'
			w_path = filename.format(e_name, sigma)

		self.load_state_dict(torch.load(w_path)['state_dict'])
		print('loading weights {0}'.format(w_path))



class DeepGaze(nn.Module):

	def __init__(self, features):
		super(DeepGaze, self).__init__()

		self.encoder = features

		self.readout = nn.Sequential(
				nn.Conv2d(512, 32, kernel_size=1),
				nn.ReLU(),
				nn.Conv2d(32, 16, kernel_size=1),
				nn.ReLU(),
				nn.Conv2d(16, 2, kernel_size=1),
				nn.ReLU(),
				nn.Conv2d(2, 1, kernel_size=1),
				nn.ReLU(),
				)

		self.sigmoid = nn.Sigmoid()

	def forward(self, x, layers=range(5)):

		features= self.encoder(x)

		return [features, self.sigmoid(self.readout(features))]

	def _initialize_weights(self):
		for m in self.encoder.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()

	def load_imagenet_weights(self):
		url = 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
		w = model_zoo.load_url(url)

		names = list([item for item in  w.keys() if 'features' in item])
		counter = 0

		for m in self.encoder.modules():
			if isinstance(m, nn.Conv2d):
				if m.dilation == (1,1):
					m.weight.data = w[names[counter]]
					if m.bias is not None:
						m.bias.data = w[names[counter+1]]
					counter+=2

		print('imagenet weights loaded.')


	def load_weights(self, w_path='weights/encoder-best-readoutfov-S3.pth.tar'):
		if not w_path:
			sigma = self.config['dataset']['first_blur_sigma']
			e_name = self.config['model']['name'].split('_')[0]
			filename = 'weights/encoder-best-{0}-S{1}.pth.tar'
			w_path = filename.format(e_name, sigma)

		self.load_state_dict(torch.load(w_path)['state_dict'])
		print('loading weights {0}'.format(w_path))




def make_layers(cfg, batch_norm=False):
	# class Layer(nn.Module):

	# 	def __init__(self, layers):
	# 		super(Layer, self).__init__()
	# 		self.layers = layers
	# 		self.build()

	# 	def build(self):
	# 		for layer in self.layers:
	# 			setattr(self, 'layer_{}'.format(i), nn.Sequential(layer))

	# 	def forward(self, x, batch_norm=True):



	network = []
	in_channels = 3
	for box in cfg:
		layers = list()
		for v in box:
			if v == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
			else:
				if 'd' in v:
					v = int(''.join(i for i in v if i.isdigit()))
					conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=2, dilation=2)
				else:
					v = int(''.join(i for i in v if i.isdigit()))
					conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
				if batch_norm:
					layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
				else:
					layers += [conv2d, nn.ReLU(inplace=True)]
				in_channels = v
		network.append(nn.Sequential(*layers))
	return nn.Sequential(*network)


def make_layers_vgg(cfg, batch_norm=False):
	layers = []
	in_channels = 3
	for v in cfg:
		if v == 'M':
			layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
		else:
			if 'd' in v:
				v = int(''.join(i for i in v if i.isdigit()))
				conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=2, dilation=2)
			else:
				v = int(''.join(i for i in v if i.isdigit()))
				conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
			if batch_norm:
				layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
			else:
				layers += [conv2d, nn.ReLU(inplace=True)]
			in_channels = v
	return nn.Sequential(*layers)


class Block(nn.Module):

	def __init__(self, scale):
		super(Block, self).__init__()
		self.scale = scale
		self.build()


	def build(self):

		self.upsample = nn.ConvTranspose2d(64, 64 , kernel_size=self.scale, stride=self.scale)

		flag = True
		for i in range(5):
			input_depth = 64 * (2**i)
			if input_depth > 512:
				input_depth = 512

			if flag:
				layer = nn.Conv2d(input_depth, 64, kernel_size=self.scale, stride=self.scale)
				if self.scale == 1:
					flag = False
				else:
					self.scale //= 2
			else:

				self.scale *=2
				layer = nn.ConvTranspose2d(input_depth, 64, kernel_size=self.scale, stride=self.scale)
			setattr(self, 'layer_{}'.format(i), nn.Sequential(layer))

		self.conv_com = nn.Sequential(
				nn.Conv2d( 5 * 64 , 64, kernel_size=1),
				nn.BatchNorm2d(64),
				nn.ReLU(inplace=True),
			)

	def forward(self, x):
		assert len(x) == 5

		features = list()
		for i in range(5):
			layer_feat = getattr(self, 'layer_{}'.format(i))(x[i])
			features.append(layer_feat)
			print(features[-1].size())


		features = torch.cat(features, 1) #concat along the depth
		features = self.conv_com(features)
		return self.upsample(features)


