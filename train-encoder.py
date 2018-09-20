
import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import shutil
import time
import numpy as np
import scipy
import skimage
import logging
import sys
from datetime import datetime
from PIL import Image
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets



from layers.encoder import e_config, make_encoder
from dataset import Saliency
from config import CONFIG

from saliency.metrics import AUC, SAUC, NSS, CC, KLdiv
import glob



random.seed(1000)
torch.manual_seed(1000)
torch.cuda.manual_seed_all(1000)



parser = argparse.ArgumentParser(description='saliency - training script')

parser.add_argument('--weights', default='/media/ramin/data/duration/weights/', metavar='DIR',
					help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='dvgg16',
					choices=['vgg16', 'dvgg16'],
					help='model architecture: ' +
						' (default: dvgg16)')

parser.add_argument('--name', '-n', metavar='NAME', default='TS-GTS',
					choices=['TS-GTS', 'TDW-GTS', 'TS-GTDW', 'TDW-GTDW',
								'TS', 'TDW'],
					help='train policy (default: TS-GTS)')

parser.add_argument('--log', default='logs/', metavar='DIR',
					help='path to dataset')
# parser.add_argument('--metric', '-m', metavar='METRIC', default='AUC',
# 					choices=['AUC', 'NSS'],
# 					help='evaluation metric')
parser.add_argument('-v','--visualize', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
					help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=15, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
					help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
					metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=3e-5, type=float,
					metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
					help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
					metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
					metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
					help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
					help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
					help='use pre-trained model')



best_prec1 = 0

RESULTS = np.zeros((2, 20, 5, 2000, 5))  # train/eval | epoch | dataset | images | metric.
RESULTS[RESULTS==0] = np.nan

SHUF_MAPS = np.load('shuf_maps.npz').tolist()

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})


def main():
	global args, best_prec1, model, train_loader, val_loaders, output, target
	args = parser.parse_args()


	logging.basicConfig(
		format="%(message)s",
		handlers=[
			# logging.FileHandler("{0}/{1}.log".format(args.log, sys.argv[0].replace('.py','') + datetime.now().strftime('_%H_%M_%d_%m_%Y'))),
			logging.FileHandler("{0}/{1}.log".format(args.log, sys.argv[0].replace('.py','') + args.name)),
			logging.StreamHandler()
		],
		level=logging.INFO)

	for sigma in range(6,16,2):
		# create model
		if args.pretrained:
			pass
			# logging.info("=> using pre-trained model '{}'".format(args.arch))
			# model = models.__dict__[args.arch](pretrained=True)
		else:
			model_name = CONFIG['model']['name']
			en_name = model_name.split('_')[0]
			logging.info("=> creating model '{}'".format(en_name))
			model = make_encoder(en_name)
			model._initialize_weights()
			# model.load_imagenet_weights()
			# CONFIG['train']['user'] = [8]
			# model = Encoder(CONFIG)
			# model._initialize_weights()
			# model.encoder.load_weights()



		model.cuda()

		# define loss function (criterion) and optimizer
		#criterion = nn.BCEWithLogitsLoss().cuda()
		#criterion = nn.BCELoss().cuda()
		criterion = nn.KLDivLoss()

		# optimizer = torch.optim.SGD(
		# 				list(model.encoder[1].parameters()) +
		# 				list(model.readout.parameters()),
		# 				# model.parameters(),
		# 				args.lr,
		# 				momentum=args.momentum,
		# 				weight_decay=args.weight_decay)

		# for param in model.encoder[0].parameters():
		# 	param.requires_grad = False


		optimizer = torch.optim.Adam(
						# list(model.encoder[1].parameters()) +
						# list(model.readout.parameters()),
						model.parameters(),
						args.lr,
						betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)



		# optionally resume from a checkpoint
		cudnn.benchmark = True


		if args.resume:
			if os.path.isfile(args.resume):
				logging.info("=> loading checkpoint '{}'".format(args.resume))
				checkpoint = torch.load(args.resume)
				args.start_epoch = checkpoint['epoch']
				best_prec1 = checkpoint['best_prec1']
				model.load_state_dict(checkpoint['state_dict'])
				optimizer.load_state_dict(checkpoint['optimizer'])
				logging.info("=> loaded checkpoint '{}' (epoch {})"
					  .format(args.resume, checkpoint['epoch']))
			else:
				logging.info("=> no checkpoint found at '{}'".format(args.resume))


		train_duration = True if (args.name == 'TDW') else False
		# eval_duration = True if (args.name.split('-')[1] == 'GTDW') else False

		print(args.name, train_duration)#, eval_duration)

		train_dataset = Saliency()
		train_dataset.load('OSIE', 'train', duration=train_duration,
		split={'train' :0.90 , 'eval' : 0.1}, sigma=sigma)
		train_loader = torch.utils.data.DataLoader(
			train_dataset, batch_size=args.batch_size, shuffle=False,
			num_workers=args.workers, pin_memory=True, sampler=None)


		# val_loaders = list()

		# for name_idx, name in enumerate(CONFIG['eval']['dataset']):
		# 	if name == 'OSIE':
		# 		split={'train' :0.75 , 'eval' : 0.25}
		# 	else:
		# 		split={'train' :0.0 , 'eval' : 1.0}
		# 	val_ds = Saliency()
		# 	# val_ds.load(name, 'eval', split=split,
		# 	# 			 duration=eval_duration, sigma=sigma)
		# 	val_ds.load(name, 'eval', split=split,
		# 				 duration=eval_duration, sigma=sigma)

		# 	val_loaders.append(
		# 		torch.utils.data.DataLoader(
		# 			val_ds,
		# 			batch_size=args.batch_size * 2, shuffle=False,
		# 			num_workers=args.workers, pin_memory=True)
		# 	)

		if args.evaluate:
			validate(val_loaders, model, criterion)
			return

		# if args.visualize:
		# 	visualize(val_loader, model)
		# 	return

		for epoch in range(args.start_epoch, args.epochs):

			try:

				adjust_learning_rate(optimizer, epoch)

				# train for one epoch
				train(train_loader, model, criterion, optimizer, epoch, sigma)

				# evaluate on validation set
				# prec1 = validate(val_loaders, model, criterion, epoch)

				# remember best prec@1 and save checkpoint
			#	is_best = prec1 > best_prec1
				# best_prec1 = max(prec1, best_prec1)
				#best_prec1 = 1
				if epoch in [4, 9, 14]:
					save_checkpoint({
						'epoch': epoch + 1,
						'arch': args.arch,
						'state_dict': model.state_dict(),
						'best_prec1': True,
						'optimizer' : optimizer.state_dict(),
					}, True, sigma)

			except Exception as x:
				print(x)


def train(train_loader, model, criterion, optimizer, epoch, sigma):
	global output, target, input
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	prec = [AverageMeter()]

	# switch to train mode
	model.train()

	end = time.time()
	logging.info('**********************************************')
	logging.info('Epoch : {0}'.format(epoch))
	# shuf_map = train_loader.dataset.d.get('fixation', size=(600,800)).sum(axis=0)
	shuf_map = SHUF_MAPS[train_loader.dataset.name]
	for batch_idx, (input, target, sal, fix) in enumerate(train_loader):
		# measure data loading time
		data_time.update(time.time() - end)

		target = target.cuda(async=True)
		# mask = mask.cuda(async=True)
		# input_var = torch.autograd.Variable(input).cuda()
		# target_var = torch.autograd.Variable(target)
		# mask_var = torch.autograd.Variable(mask)
		input = input.cuda()
		# compute output
		# features, output, fov = model(input_var)
		output = model(input)
		loss = criterion(output, target)
		# loss_2 = criterion(fov, mask_var)
		# loss = loss_1 + 0.3 * loss_2

		b_h, b_w = train_loader.dataset.d.data[0]['img_size']
		# output = torch.sigmoid(output)

		# measure accuracy and record loss
		output = torch.nn.Upsample(size=(b_h, b_w),
							mode='bilinear')(output.detach()).cpu().numpy()


		# print(output.shape, target.shape, fix.shape)
		acc_1  = accuracy(output, sal.cpu().numpy(),
			fix.cpu().numpy(), CONFIG['train']['metrics'], shuf_map,
			(0, epoch, 0, batch_idx))
		# acc_2  = accuracy(fov.data.cpu().numpy(), mask.cpu().numpy())
		losses.update(loss.item(), input.size(0))
		prec[0].update(acc_1.mean(axis=0), input.size(0))
		# top1_fov.update(acc_2.mean(), input.size(0))


		# compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if batch_idx % args.print_freq == 0:
			msg = '[{0} - {1}][{2}/{3}]\t' \
			'Time {batch_time_val:.3f} ({batch_time_avg:.3f})\t' \
			'Loss {loss_val:.4f} ({loss_avg:.4f})\n'.format(
				epoch, sigma, train_loader.dataset.name, len(train_loader),
				batch_time_val=batch_time.val[0], batch_time_avg=batch_time.avg[0],
				loss_val=losses.val[0], loss_avg=losses.avg[0])

			val = np.array2string(prec[0].val, precision=2, separator=',')
			avg = np.array2string(prec[0].avg, precision=4, separator=',')
			msg += '{0} - {1} \n'.format(val, avg)

			logging.info(msg)
	# saving4results

	# RESULTS[0, epoch, 0, :] = prec[0].avg


#
def validate(val_loaders, model, criterion, epoch):
	global prec, output, target, sal
	batch_time = AverageMeter()
	losses = [AverageMeter() for _ in range(len(val_loaders))]
	prec = [AverageMeter() for _ in range(len(val_loaders))]


	# switch to evaluate mode
	model.eval()
	end = time.time()

	with torch.no_grad():
		logging.info('Validation - Epoch : {0}'.format(epoch))
		logging.info('##############################################')
		for val_idx, val_loader in enumerate(val_loaders):
			# shuf_map = val_loader.dataset.d.get('fixation').sum(axis=0)
			shuf_map = SHUF_MAPS[val_loader.dataset.name]
			# if epoch < 3:
			# 	if val_idx >= 1:
			# 		break
			for batch_idx, (input, target, sal, fix) in enumerate(val_loader):
				# print(batch_idx)
				target = target.cuda(async=True)
				# mask = mask.cuda(async=True)
				# input_var = torch.autograd.Variable(input, volatile=True).cuda()
				# target_var = torch.autograd.Variable(target, volatile=True)
				# mask_var = torch.autograd.Variable(mask, volatile=True)
				input = input.cuda()
				# compute output
				output = model(input)
				loss = criterion(output, target)


				b_h, b_w = val_loader.dataset.d.data[0]['img_size']
				#output = torch.sigmoid(output)

				# measure accuracy and record loss
				output = torch.nn.Upsample(size=(b_h, b_w),
									mode='bilinear')(output.detach()).cpu().numpy()


				# output = [scipy.ndimage.zoom(item, (1,8,8)) for item in output.detach().cpu().numpy()]
				# output = np.array(output)

				# loss_2 = criterion(output, mask_var)
				# loss  = loss_1 + 0.3 * loss_2

				# measure accuracy and record loss
				acc_1  = accuracy(output, sal.cpu().numpy(),
					fix.cpu().numpy(), CONFIG['eval']['metrics'], shuf_map,
					(1, epoch, val_idx, batch_idx))
				# acc_2  = accuracy(fov.data.cpu().numpy(), mask.cpu().numpy())

				losses[val_idx].update(loss.item(), input.size(0))
				prec[val_idx].update(acc_1.mean(axis=0), input.size(0))
				# top1_fov.update(acc_2.mean(), input.size(0))


				# measure elapsed time
				batch_time.update(time.time() - end)
				end = time.time()

			msg = '[{0}][{1}]\t' \
			'Time {batch_time_val:.3f} ({batch_time_avg:.3f})\t' \
			'Loss {loss_val:.4f} ({loss_avg:.4f})\n'.format(
			val_loader.dataset.name, len(val_loader) * 8,
			batch_time_val=batch_time.val[0], batch_time_avg=batch_time.avg[0],
			loss_val=losses[val_idx].val[0], loss_avg=losses[val_idx].avg[0])

			# val = np.array2string(prec[val_idx].val, precision=2, separator=',')
			avg = np.array2string(prec[val_idx].avg, precision=4, separator=',')
			msg += '{0}\n'.format( avg)

			logging.info(msg)
		# RESULTS[1, epoch, val_idx, :] = prec[val_idx].avg
	return prec[0].avg[0]


def save_checkpoint(state, is_best, sigma, filename='kld-{0}-{1}-{2}.pth.tar'):
	date = datetime.now().strftime('_%y-%d-%m_%H-%M')
	filename = filename.format(args.name, state['epoch'], sigma)
	filename = os.path.join(args.weights, filename)
	torch.save(state, filename)
	if is_best:
		logging.warning('***********************saving best model *********************')
		best = os.path.join(args.weights,'encoder-best.pth.tar')
		shutil.copyfile(filename, best)

	#result_files = glob.glob('results/*.npy')
	#result_files = sorted([int(item.split('/')[-1].split('.')[0]) for item in result_files])
	#counter = 0
	#if result_files:
	#	counter = result_files[-1] + 1
	with open('results/kld-{0}-{1}.npy'.format(args.name, sigma), 'wb') as f:
		np.save(f, np.array(RESULTS))


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		if not isinstance(val, np.ndarray):
			val = np.array([val])

		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
	"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
	lr = args.lr * (0.1 ** (epoch // 30))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def accuracy(output, sal, fix, metrics=[AUC], shuf_map=None, loc=(0,0,0,0)):
	"""Computes the precision@k for the specified values of k"""
	batch_size = sal.shape[0]
	result = list()
	train, epoch, dataset_idx, batch_idx = loc

	for i in range(batch_size):
		img_idx = batch_idx * batch_size + i
		tmp = list()
		for metric_idx, metric in enumerate(metrics):
			try:
				o = output[i][0]
				o = o / o.max()
				if metric in [AUC, NSS]:
					tmp.append(metric(saliency_map=o, fixation_map=fix[i]))
				elif metric == SAUC:
					tmp.append(metric(saliency_map=o,
							fixation_map=fix[i], shuf_map=shuf_map))
				elif metric in [CC, KLdiv]:
					tmp.append(metric(output[i][0], sal[i]))
			except Exception as x:
				print(x)
			RESULTS[train, epoch, dataset_idx, img_idx, metric_idx] = tmp[-1]
		result.append(np.array(tmp))

	return np.array(result)


def visualize(loader, model):
	counter = 0
	for batch_idx, (input, target) in enumerate(loader):
		target = target.cuda(async=True)
		input_var = torch.autograd.Variable(input, volatile=True).cuda()
		target_var = torch.autograd.Variable(target, volatile=True)
		output = model(input_var).data.cpu().numpy()
		for idx, img in enumerate(input):
			img = Image.open(loader.dataset.dataset[counter][0])
			counter+=1
			w, h = img.size
			mask = np.array(output[idx][0] * 255, dtype=np.uint8)
			mask = Image.fromarray(mask).resize((w,h)).convert('RGB')
			saliency = np.array(target[idx][0] * 255, dtype=np.uint8)
			saliency = Image.fromarray(saliency).resize((w,h)).convert('RGB')

			out = Image.new('RGB', (w, h*2))
			out.paste(Image.blend(img, mask, alpha=0.9).convert('RGB'), (0,0))
			out.paste(Image.blend(img, saliency, alpha=0.9).convert('RGB'),(0,h))

			out_path = os.path.join(args.visualize, '{0}-{1}.jpg'.format(batch_idx, idx))
			out.save(out_path)



if __name__ == '__main__':
	main()
