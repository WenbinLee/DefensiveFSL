#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import os
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import grad
from torch.distributions.uniform import Uniform
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import time
from torch import autograd
from PIL import ImageFile
import pdb
import sys
sys.dont_write_bytecode = True
from sklearn.manifold import TSNE


def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.cuda.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

setup_seed(10)


# ============================ Data & Networks =====================================
import dataset.general_dataloader as GeneralDataloader
import models.network as ClassifierNet
import adversarial.attack as Attack_fun
import adversarial.attack_method as Attack_method
import utils
# ==================================================================================


ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default='/home/liwenbin/Datasets/miniImageNet--ravi', help='/miniImageNet')
parser.add_argument('--data_name', default='miniImageNet', help='miniImageNet|StanfordDog|StanfordCar|CubBird')
parser.add_argument('--method_name', default='AT', help='Clean | AT | ALP | ATDA | VAT | KLD | TCD | SKL')
parser.add_argument('--method_weight', type=float, default=1.0, help='the hyper-parameter of each method')
parser.add_argument('--version', default='Test_for_DN4', help='set the version of the current experiment')
parser.add_argument('--mode', default='train', help='train|val|test')
parser.add_argument('--outf', default='./results_2022_01_21_Baseline_AT')
parser.add_argument('--resume', default='', type=str, help='path to the lastest checkpoint (default: none)')
parser.add_argument('--encoder_model', default='Conv64F', help='Conv64F|Conv64F_Li')
parser.add_argument('--classifier_model', default='Baseline', help='Baseline|Baseline++|ProtoNet|CovaMNet|DN4')
parser.add_argument('--workers', type=int, default=4)
#  Classification parameters  #
parser.add_argument('--train_classes', type=int, default=64, help='the number of training classes')
parser.add_argument('--train_aug', action='store_true', default=True, help='Perform data augmentation or not')
parser.add_argument('--imageSize', type=int, default=84)
parser.add_argument('--epochs', default=90, type=int, help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default: 256)')
#  Few-shot parameters  #
parser.add_argument('--New_classifier_model', default='DN4', help='ProtoNet|CovaMNet|DN4')
parser.add_argument('--episodeSize', type=int, default=1, help='the mini-batch size of training')
parser.add_argument('--testepisodeSize', type=int, default=1, help='one episode is taken as a mini-batch')
parser.add_argument('--episode_train_num', type=int, default=10000, help='the total number of training episodes')
parser.add_argument('--episode_val_num', type=int, default=1000, help='the total number of evaluation episodes')
parser.add_argument('--episode_test_num', type=int, default=1000, help='the total number of testing episodes')
parser.add_argument('--way_num', type=int, default=5, help='the number of way/class')
parser.add_argument('--shot_num', type=int, default=5, help='the number of shot')
parser.add_argument('--query_num', type=int, default=15, help='the number of queries')
parser.add_argument('--aug_shot_num', type=int, default=20, help='the number of augmented support images of each class during test')
parser.add_argument('--neighbor_k', type=int, default=3, help='the number of k-nearest neighbors')
#  Attack parameters  #
parser.add_argument('--attack_mode', default='FGSM', help='FGSM|iFGSM|PGD')
parser.add_argument('--num_steps', type=int, default=10, help='the number of iterations for PGD')
parser.add_argument('--step_size', type=int, default=0.003, help='the step size of PGD')
parser.add_argument('--epsilon_max', type=int, default=0.02, help='define the scope of Epsilon, the max value of Epsilon')
#  Other optimization parameters   #
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--cosine', type=bool, default=False, help='using cosine annealing')
parser.add_argument('--lr_decay_epochs', type=list, default=[60,80], help='where to decay lr, can be a list')
parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--adam', action='store_true', default=True, help='use adam optimizer')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='the number of gpus')
parser.add_argument('--nc', type=int, default=3, help='input image channels')
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--print_freq', '-p', default=100, type=int, metavar='N', help='print frequency (default: 100)')
opt = parser.parse_args()
opt.cuda = True
# cudnn.benchmark = True




# ======================================= Define functions =============================================

def train(train_loader, model, criterion, optimizer, epoch, F_txt):
	batch_time = utils.AverageMeter()
	data_time = utils.AverageMeter()

	losses = utils.AverageMeter()
	top1 = utils.AverageMeter()
	losses_adv =  utils.AverageMeter()
	top1_adv =  utils.AverageMeter()
	F_score =  utils.AverageMeter()
	

	# switch to train mode
	model.train()
	for p in model.parameters():
		p.requires_grad = True
	end = time.time()

	for i, (input, target) in enumerate(train_loader):
		# measure data loading time
		data_time.update(time.time() - end)

		if opt.ngpu is not None:
			input_var = input.cuda()
			target = target.cuda()


		#============================= Conduct adversarial samples =============================#
		for p in model.parameters():
			p.requires_grad = False

		# Randomly sampling epsilons from [0, 0.01]
		epsilon = Uniform(torch.tensor([0.0]), torch.tensor(opt.epsilon_max)).sample().cuda()
		# epsilon = torch.tensor([0.01]).cuda()

		if opt.attack_mode=='FGSM':

			# FGSM Attack-------one step method (true targets)
			perturbed_input_var = Attack_fun.fgsm_attack2(input_var, target, model, criterion, epsilon)

		elif opt.attack_mode == 'PGD':
			
			# PGD attack------iterative method (true targets)
			perturbed_input_var = Attack_fun.pgd_attack2(opt, input_var, target, model, criterion, epsilon)

		else:
			print('The kind of attack has not been defined!')

		for p in model.parameters():
			p.requires_grad = True
		#======================================================================================#


		# Training by normal samples
		output_norm = model(input_var)
		loss_norm = criterion(output_norm, target)


		# Training by adversarial samples
		output_adv = model(perturbed_input_var)
		loss_adv = criterion(output_adv, target)
 

		# Sum all the losses
		loss = loss_norm + loss_adv


		# measure accuracy and record loss
		prec1_adv, _ = utils.accuracy(output_adv, target, topk=(1,3))
		losses_adv.update(loss_adv.item(), input_var.size(0))
		top1_adv.update(prec1_adv[0], input_var.size(0))

		prec1, _ = utils.accuracy(output_norm, target, topk=(1,3))
		losses.update(loss_norm.item(), input_var.size(0))
		top1.update(prec1[0], input_var.size(0))

		# calculate the F_0.5 score
		F1 = (1+0.25)*prec1*prec1_adv/(0.25*prec1+prec1_adv+0.001)
		F_score.update(F1[0], input_var.size(0))


		# compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()


		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()


		#============== print the intermediate results ==============#
		if i % opt.print_freq == 0 and i != 0:

			# Ouput the results without adversarial training
			print('Epoch-({0}): [{1}/{2}]\t'
				'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
				'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
				'F@0.5 {F_score.val:.3f} ({F_score.avg:.3f})'.format(
					epoch, i, len(train_loader), batch_time=batch_time, loss=losses, top1=top1, F_score=F_score))

			print('Epoch-({0}): [{1}/{2}]\t'
				'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
				'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
				'F@0.5 {F_score.val:.3f} ({F_score.avg:.3f})'.format(
					epoch, i, len(train_loader), batch_time=batch_time, loss=losses, top1=top1, F_score=F_score), file=F_txt)


			# Output the results of adversarial training
			print('Epoch-({0}): [{1}/{2}]\t'
				'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
				'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
				'F@0.5 {F_score.val:.3f} ({F_score.avg:.3f})\t'
				'Epsilon: {epsilon:.3f} (Defense)'.format(
					epoch, i, len(train_loader), epsilon=float(epsilon), batch_time=batch_time, loss=losses_adv, top1=top1_adv, F_score=F_score))

			print('Epoch-({0}): [{1}/{2}]\t'
				'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
				'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
				'F@0.5 {F_score.val:.3f} ({F_score.avg:.3f})\t'
				'Epsilon: {epsilon:.3f} (Defense)'.format(
					epoch, i, len(train_loader), epsilon=float(epsilon), batch_time=batch_time, loss=losses_adv, top1=top1_adv, F_score=F_score), file=F_txt)

	return losses.avg, losses_adv.avg




def validate(val_loader, model, criterion, epoch, best_F, epsilon, F_txt):
	batch_time = utils.AverageMeter()
	losses = utils.AverageMeter()
	top1 = utils.AverageMeter()
	losses_attack = utils.AverageMeter()
	top1_attack = utils.AverageMeter()
	F_score = utils.AverageMeter()


	# switch to evaluate mode
	model.eval()
	end = time.time()

	for i, (input, target) in enumerate(val_loader):
		if opt.ngpu is not None:
			input_var = input.cuda()
			target = target.cuda()

		#============================= Conduct adversarial samples =============================#
		for p in model.parameters():
			p.requires_grad = False

		if opt.attack_mode=='FGSM':

			# FGSM Attack-------one step method (true targets)
			perturbed_input_var = Attack_fun.fgsm_attack2(input_var, target, model, criterion, epsilon)

		elif opt.attack_mode == 'PGD':
			
			# PGD attack------iterative method (true targets)
			perturbed_input_var = Attack_fun.pgd_attack2(opt, input_var, target, model, criterion, epsilon)

		else:
			print('The kind of attack has not been defined!')

		for p in model.parameters():
			p.requires_grad = False
		#======================================================================================#


		# Validate normal samples
		output_norm = model(input_var)
		loss_norm = criterion(output_norm, target)


		# Validate adversarial samples
		output_attack = model(perturbed_input_var)
		loss_adv = criterion(output_attack, target)
 

		# measure accuracy and record loss
		prec1_attack, _ = utils.accuracy(output_attack, target, topk=(1,3))
		losses_attack.update(loss_adv.item(), input_var.size(0))
		top1_attack.update(prec1_attack[0], input_var.size(0))

		prec1, _ = utils.accuracy(output_norm, target, topk=(1,3))
		losses.update(loss_norm.item(), input_var.size(0))
		top1.update(prec1[0], input_var.size(0))

		# calculate the F_0.5 score
		F1 = (1+0.25)*prec1*prec1_attack/(0.25*prec1+prec1_attack+0.001)
		F_score.update(F1[0], input_var.size(0))

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()


		#============== print the intermediate results ==============#
		if i % opt.print_freq == 0 and i != 0:

			# Output the results before attacking
			print('Test-({0}): [{1}/{2}]\t'
				'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
				'Prec@1 {top1.val:.3f} ({top1.avg:.3f}\t'
				'F@0.5 {F_score.val:.3f} ({F_score.avg:.3f})'.format(
					epoch, i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1, F_score=F_score))


			print('Test-({0}): [{1}/{2}]\t'
				'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
				'Prec@1 {top1.val:.3f} ({top1.avg:.3f}\t'
				'F@0.5 {F_score.val:.3f} ({F_score.avg:.3f})'.format(
					epoch, i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1, F_score=F_score), file=F_txt)


			# Ouput the results after attacking
			print('Test-({0}): [{1}/{2}]\t'
				'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
				'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
				'F@0.5 {F_score.val:.3f} ({F_score.avg:.3f})\t'
				'Epsilon: {epsilon:.3f} (Attack)'.format(
					epoch, i, len(val_loader), epsilon=float(epsilon), batch_time=batch_time, loss=losses_attack, top1=top1_attack, F_score=F_score))


			print('Test-({0}): [{1}/{2}]\t'
				'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
				'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
				'F@0.5 {F_score.val:.3f} ({F_score.avg:.3f})\t'
				'Epsilon: {epsilon:.3f} (Attack)'.format(
					epoch, i, len(val_loader), epsilon=float(epsilon), batch_time=batch_time, loss=losses_attack, top1=top1_attack, F_score=F_score), file=F_txt)



	print(' * Prec@1 {top1.avg:.3f} Best_F {best_F:.3f}'.format(top1=top1, best_F=best_F))
	print(' * Prec@1 {top1.avg:.3f} Best_F {best_F:.3f}'.format(top1=top1, best_F=best_F), file=F_txt)
	print(' * Epsilon {:.3f} Prec@1 {top1.avg:.3f} (Attack)'.format(float(epsilon), top1=top1_attack))
	print(' * Epsilon {:.3f} Prec@1 {top1.avg:.3f} (Attack)'.format(float(epsilon), top1=top1_attack), file=F_txt)
	

	return top1.avg, top1_attack.avg, F_score.avg



def test(test_loader, model, criterion, epoch_index, best_prec1, epsilon, F_txt):
	batch_time = utils.AverageMeter()
	losses = utils.AverageMeter()
	top1 = utils.AverageMeter()
	losses_attack =  utils.AverageMeter()
	top1_attack =  utils.AverageMeter()
	F_score =  utils.AverageMeter()
  

	# switch to evaluate mode
	model.eval()
	accuracies = []
	accuracies_attack = []
	accuracies_F1 = []
	adv_examples = []


	end = time.time()
	for episode_index, (query_images, query_targets, support_images, support_targets) in enumerate(test_loader):


		# Convert query and support images
		input_var1 = torch.cat(query_images, 0).cuda()
		input_var2 = torch.cat(support_images, 0).squeeze(0).cuda()
		input_var2 = input_var2.contiguous().view(-1, input_var2.size(2), input_var2.size(3), input_var2.size(4))


		# Deal with the targets
		target = torch.cat(query_targets, 0).cuda()
		target_sup = torch.cat(support_targets, 0).cuda()

	
		#============================= Conduct adversarial samples =============================#
		for p in model.parameters():
			p.requires_grad = False


		if opt.attack_mode=='FGSM':

			# FGSM Attack-------one step method (true targets)
			perturbed_input_var1 = Attack_fun.fgsm_attack(input_var1, input_var2, target, model, criterion, epsilon)

		elif opt.attack_mode == 'PGD':
			
			# PGD attack------iterative method (true targets)
			perturbed_input_var1 = Attack_fun.pgd_attack(opt, input_var1, input_var2, target, model, criterion, epsilon)

		else:
			print('The kind of attack has not been defined!')

		for p in model.parameters():
			p.requires_grad = False
		#======================================================================================#


		# Training by adversarial samples
		perturbed_input_var1_fea, _, output_attack = model(perturbed_input_var1, input_var2, True)
		loss_attack = criterion(output_attack, target)


		# Training by normal samples
		input_var1_fea, input_var2_fea, output = model(input_var1, input_var2, True)
		loss = criterion(output, target)
	

		# Draw the t-SNE figure
		if episode_index == 1:

			# Store the features of clean and adversarial examples
			avgpool = nn.AdaptiveAvgPool2d(1)
			Fea_Norm = avgpool(input_var1_fea).squeeze(3).squeeze(2)	
			Fea_Adv = avgpool(perturbed_input_var1_fea).squeeze(3).squeeze(2)	
			Fea_sup = avgpool(input_var2_fea).squeeze(3).squeeze(2)

			# Draw the figure of Features by t-SNE
			utils.plot_tSNE_curve(opt, Fea_Norm, Fea_Adv, Fea_sup, target, target_sup, epoch_index, mode='train')


		# measure accuracy and record loss
		prec1, _ =  utils.accuracy(output, target, topk=(1, 3))
		losses.update(loss.item(), input_var1.size(0))
		top1.update(prec1[0], input_var1.size(0))
		accuracies.append(prec1)

		# measure accuracy and record loss again after attacking
		prec1_attack, _ =  utils.accuracy(output_attack, target, topk=(1, 3))
		losses_attack.update(loss_attack.item(), input_var1.size(0))
		top1_attack.update(prec1_attack[0], input_var1.size(0))
		accuracies_attack.append(prec1_attack)

		# calculate the F_0.5 score
		F1 = (1+0.25)*prec1*prec1_attack/(0.25*prec1+prec1_attack+0.001)
		F_score.update(F1[0], input_var1.size(0))
		accuracies_F1.append(F1)


		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()


		#============== print the intermediate results ==============#
		if episode_index % opt.print_freq == 0 and episode_index != 0:

			# Output the results before attacking
			print('Test-({0}): [{1}/{2}]\t'
				'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
				'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
				'F@0.5 {F_score.val:.3f} ({F_score.avg:.3f})'.format(
					epoch_index, episode_index, len(val_loader), batch_time=batch_time, loss=losses, top1=top1, F_score=F_score))


			print('Test-({0}): [{1}/{2}]\t'
				'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
				'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
				'F@0.5 {F_score.val:.3f} ({F_score.avg:.3f})'.format(
					epoch_index, episode_index, len(val_loader), batch_time=batch_time, loss=losses, top1=top1, F_score=F_score), file=F_txt)


			# Ouput the results after attacking
			print('Test-({0}): [{1}/{2}]\t'
				'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
				'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
				'F@0.5 {F_score.val:.3f} ({F_score.avg:.3f})\t'
				'Epsilon: {epsilon:.3f} (Attack)'.format(
					epoch_index, episode_index, len(val_loader), epsilon=float(epsilon), batch_time=batch_time, loss=losses_attack, top1=top1_attack, F_score=F_score))


			print('Test-({0}): [{1}/{2}]\t'
				'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
				'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
				'F@0.5 {F_score.val:.3f} ({F_score.avg:.3f})\t'
				'Epsilon: {epsilon:.3f} (Attack)'.format(
					epoch_index, episode_index, len(val_loader), epsilon=float(epsilon), batch_time=batch_time, loss=losses_attack, top1=top1_attack, F_score=F_score), file=F_txt)



	print(' * Prec@1 {top1.avg:.3f} best_F {best_F:.3f}'.format(top1=top1, best_F=best_F))
	print(' * Prec@1 {top1.avg:.3f} best_F {best_F:.3f}'.format(top1=top1, best_F=best_F), file=F_txt)
	print(' * Epsilon {:.3f} Prec@1 {top1.avg:.3f} (Attack)'.format(float(epsilon), top1=top1_attack))
	print(' * Epsilon {:.3f} Prec@1 {top1.avg:.3f} (Attack)'.format(float(epsilon), top1=top1_attack), file=F_txt)
	

	return top1.avg, top1_attack.avg, F_score.avg, accuracies, accuracies_attack, accuracies_F1




if __name__=='__main__':

	# save path
	opt.outf, F_txt = utils.set_save_path(opt)

	# Check if the cuda is available
	if torch.cuda.is_available() and not opt.cuda:
		print("WARNING: You have a CUDA device, so you should probably run with --cuda")

	# ========================================== Model config ===============================================
	global best_F
	best_F = 0
	model = ClassifierNet.define_model(encoder_model=opt.encoder_model, classifier_model=opt.classifier_model, norm='batch',
			class_num=opt.train_classes, way_num=opt.way_num, shot_num=opt.shot_num, init_type='normal', use_gpu=opt.cuda)


	# define loss function (criterion) and optimizer
	criterion = nn.CrossEntropyLoss().cuda()
	if opt.adam:
		optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.9), weight_decay=0.0005)
	else:
		optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, dampening=0.9, weight_decay=0.001)


	# optionally resume from a checkpoint
	if opt.resume:
		checkpoint = utils.get_resume_file(opt.resume, F_txt)
		opt.start_epoch = checkpoint['epoch']
		best_F = checkpoint['best_F']
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])

	if opt.ngpu > 1:
		model = nn.DataParallel(model, range(opt.ngpu))

	# print the parameters and architecture of the model
	print(opt)
	print(opt, file=F_txt)
	print(model) 
	print(model, file=F_txt) 


	# set cosine annealing scheduler
	if opt.cosine:
		eta_min = opt.lr * (opt.lr_decay_rate ** 3)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs, eta_min, -1)


	# ============================================ Training phase ========================================
	print('===================================== Training on the train set =====================================')
	print('===================================== Training on the train set =====================================', file=F_txt)

	Loss_total = []
	Loss_adv_total = []
	for epoch_item in range(opt.start_epoch, opt.epochs):

		print('==================== Epoch %d ====================' %epoch_item)
		print('==================== Epoch %d ====================' %epoch_item, file=F_txt)
		epsilon_test = torch.tensor([0.01]).cuda()

	
		# Loaders of Datasets 
		train_loader, val_loader, test_loader = GeneralDataloader.get_dataloader(opt, ['train', 'val', 'test'])


		# train for one epoch
		loss_norm, loss_adv = train(train_loader, model, criterion, optimizer, epoch_item, F_txt)
		Loss_total.append(loss_norm)
		Loss_adv_total.append(loss_adv)


		print('===================================== Validation on the val set =====================================')
		print('===================================== validation on the val set =====================================', file=F_txt)
		# evaluate on validation set
		prec1, prec1_attack, F_score = validate(val_loader, model, criterion, epoch_item, best_F, epsilon_test, F_txt)
		

		# Adjust the learning rates
		if opt.cosine:
			scheduler.step()
		else:
			utils.adjust_learning_rate(opt, optimizer, epoch_item, F_txt)


		# remember best prec@1 and save checkpoint
		is_best = F_score > best_F
		best_F = max(F_score, best_F)

		# save the checkpoint
		if is_best:
			utils.save_checkpoint(
				{
					'epoch_index': epoch_item,
					'encoder_model': opt.encoder_model,
					'classifier_model': opt.classifier_model,
					'model': model.state_dict(),
					'best_F': best_F,
					'optimizer' : optimizer.state_dict(),
				}, os.path.join(opt.outf, 'model_best.pth.tar'))

		if epoch_item % 10 == 0:
			filename = os.path.join(opt.outf, 'epoch_%d.pth.tar' %epoch_item)
			utils.save_checkpoint(
				{
					'epoch_index': epoch_item,
					'encoder_model': opt.encoder_model,
					'classifier_model': opt.classifier_model,
					'model': model.state_dict(),
					'best_F': best_F,
					'optimizer' : optimizer.state_dict(),
				}, filename)


	# ======================================= Plot Loss Curves =======================================
	utils.plot_loss_curve(opt, Loss_total, Loss_adv_total)
	print('======================================== Training is END ========================================\n')
	print('======================================== Training is END ========================================\n', file=F_txt)
	F_txt.close()



	# ============================================ Test phase ============================================
	# Set the save path
	F_txt_test = utils.set_save_test_path(opt)
	print('========================================== Start Test ==========================================\n')
	print('========================================== Start Test ==========================================\n', file=F_txt_test)

	# Load the trained best model
	best_model_path = os.path.join(opt.outf, 'model_best.pth.tar')
	checkpoint = utils.get_resume_file(best_model_path, F_txt_test)
	epoch_index = checkpoint['epoch_index']
	best_F = checkpoint['best_F']
	model.load_state_dict(checkpoint['model'])


	# Define a model
	New_model = ClassifierNet.Model_with_reused_Encoder(pre_trained_model=model, new_classifier=opt.New_classifier_model,
			way_num=opt.way_num, shot_num=opt.shot_num, neighbor_k=opt.neighbor_k)
	ClassifierNet.print_network(New_model)
	

	# print the parameters and architecture of the model
	print(opt)
	print(opt, file=F_txt_test)
	print(New_model) 
	print(New_model, file=F_txt_test) 


	epsilons = torch.tensor([0, 0.003, 0.007, 0.01])
	for epsilon_item in range(epsilons.size()[0]):
		epsilon = epsilons[epsilon_item].cuda()
		print('\n\n---------Attack epsilon %f--------- \n' %epsilon)
		print('\n\n---------Attack epsilon %f--------- \n' %epsilon, file=F_txt_test)

		# ============================================ Testing phase ========================================
		start_time = time.time()
		repeat_num = 5       # repeat running the testing code several times


		total_accuracy = 0.0
		total_h = np.zeros(repeat_num)
		total_accuracy_attack = 0.0
		total_h_attack = np.zeros(repeat_num)
		total_accuracy_F1 = 0.0
		total_h_F1 = np.zeros(repeat_num)

		# Loop for the repear num
		for r in range(repeat_num):
			
			print('==================== The %d-th round ====================' %r)
			print('==================== The %d-th round ====================' %r, file=F_txt_test)

			# ======================================= Loaders of Datasets =======================================
			opt.current_epoch = epoch_item
			_, _, test_loader = GeneralDataloader.get_dataloader(opt, ['train', 'val', 'test'])


			# evaluate on validation/test set
			prec1, prec1_attack, F_score, accuracies, accuracies_attack, accuracies_F1 = test(test_loader, New_model, criterion, epoch_index, best_F, epsilon, F_txt_test)

			# No attack
			test_accuracy, h = utils.mean_confidence_interval(accuracies)
			total_h[r] = h

			# Attack
			test_accuracy_attack, h_attack = utils.mean_confidence_interval(accuracies_attack)
			total_h_attack[r] = h_attack

			# F0.5 score
			test_accuracy_f1, h_f1 = utils.mean_confidence_interval(accuracies_F1)
			total_h_F1[r] = test_accuracy_f1


			print('\nTest accuracy: %f h: %f' %(test_accuracy, h))
			print('\nTest accuracy: %f h: %f' %(test_accuracy, h), file=F_txt_test)

			
			print('Test accuracy: %f h: %f (Attack)' %(test_accuracy_attack, h_attack))
			print('Test accuracy: %f h: %f (Attack)' %(test_accuracy_attack, h_attack), file=F_txt_test)


			print('Test F0.5 score: %f h: %f' %(test_accuracy_f1, h_f1))
			print('Test F0.5 score: %f h: %f' %(test_accuracy_f1, h_f1), file=F_txt_test)

			print("\n")
			total_accuracy += test_accuracy
			total_accuracy_attack += test_accuracy_attack
			total_accuracy_F1 += test_accuracy_f1
			

		print('\nMean_accuracy: %f h: %f' %(total_accuracy/repeat_num, total_h.mean()))
		print('\nMean_accuracy: %f h: %f' %(total_accuracy/repeat_num, total_h.mean()), file=F_txt_test)
		print('Mean_accuracy: %f h: %f Epsilon: %f' %(total_accuracy_attack/repeat_num, total_h_attack.mean(), epsilon))
		print('Mean_accuracy: %f h: %f Epsilon: %f' %(total_accuracy_attack/repeat_num, total_h_attack.mean(), epsilon), file=F_txt_test)
		print('Mean_F0.5: %f h: %f Epsilon: %f' %(total_accuracy_F1/repeat_num, total_h_F1.mean(), epsilon))
		print('Mean_F0.5: %f h: %f Epsilon: %f' %(total_accuracy_F1/repeat_num, total_h_F1.mean(), epsilon), file=F_txt_test)

	print('===================================== Test is END =====================================\n')
	print('===================================== Test is END =====================================\n', file=F_txt_test)
	F_txt_test.close()

