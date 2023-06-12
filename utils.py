import torch
import os
import pdb
import scipy as sp
import scipy.stats
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE



def adjust_learning_rate(opt, optimizer, epoch, F_txt):
	"""Sets the learning rate to the initial LR decayed by 2 every 10 epoches"""
	if opt.classifier_model == 'Baseline':
		lr = opt.lr * (0.5 ** (epoch // 30))
	else:
		lr = opt.lr * (0.1 ** (epoch // 10))
	print('Learning rate: %f' %lr)
	print('Learning rate: %f' %lr, file=F_txt)
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def adjust_learning_rate2(opt, optimizer, epoch, F_txt):
	"""Sets the learning rate to the initial LR decayed by decay rate every steep step"""
	steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
	if steps > 0:
		new_lr = opt.lr * (opt.lr_decay_rate ** steps)
		print('Learning rate: %f' %new_lr)
		print('Learning rate: %f' %new_lr, file=F_txt)
		for param_group in optimizer.param_groups:
			param_group['lr'] = new_lr


def count_parameters(model):
	"""Count the total number of parameters in one model"""
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
	torch.save(state, filename)
	

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
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	with torch.no_grad():
		maxk = max(topk)
		batch_size = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target.view(1, -1).expand_as(pred))

		res = []
		for k in topk:
			correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
			res.append(correct_k.mul_(100.0 / batch_size))
		return res


def mean_confidence_interval(data, confidence=0.95):
	a = [1.0*np.array(data[i].cpu()) for i in range(len(data))]
	n = len(a)
	m, se = np.mean(a), scipy.stats.sem(a)
	h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
	return m,h


def set_save_path(opt):
	'''
		Settings of the save path
	'''

	if opt.train_aug:
		opt.outf = opt.outf+'/'+opt.data_name+'/'+'DA_'+opt.classifier_model+'_DFSL_'+opt.method_name+'_'+str(opt.method_weight)+'_'\
		+opt.encoder_model+'_'+opt.data_name+'_'+str(opt.way_num)+'Way_'+str(opt.shot_num)+'Shot_'+opt.version
	else:
		opt.outf = opt.outf+'/'+opt.data_name+'/'+opt.classifier_model+'_DFSL_'+opt.method_name+'_'+str(opt.method_weight)+'_'\
		+opt.encoder_model+'_'+opt.data_name+'_'+str(opt.way_num)+'Way_'+str(opt.shot_num)+'Shot_'+opt.version

	if not os.path.exists(opt.outf):
		os.makedirs(opt.outf)

	if torch.cuda.is_available() and not opt.cuda:
		print("WARNING: You have a CUDA device, so you should probably run with --cuda")

	# save the opt and results to txt file
	txt_save_path = os.path.join(opt.outf, 'opt_results.txt')
	F_txt = open(txt_save_path, 'a+')

	return opt.outf, F_txt


def set_save_test_path(opt, finetune=False):
	'''
		Settings of the save path
	'''

	if not os.path.exists(opt.outf):
		os.makedirs(opt.outf)

	if torch.cuda.is_available() and not opt.cuda:
		print("WARNING: You have a CUDA device, so you should probably run with --cuda")

	# save the opt and results to txt file
	if finetune:
		txt_save_path = os.path.join(opt.outf, 'Test_Finetune_results.txt')
	else:
		txt_save_path = os.path.join(opt.outf, 'Test_results.txt')
	F_txt_test = open(txt_save_path, 'a+')

	return F_txt_test


def get_resume_file(checkpoint_dir, F_txt):

	if os.path.isfile(checkpoint_dir):
		print("=> loading checkpoint '{}'".format(checkpoint_dir))
		print("=> loading checkpoint '{}'".format(checkpoint_dir), file=F_txt)
		checkpoint = torch.load(checkpoint_dir)
		print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_dir, checkpoint['epoch_index']))
		print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_dir, checkpoint['epoch_index']), file=F_txt)

		return checkpoint
	else:
		print("=> no checkpoint found at '{}'".format(checkpoint_dir))
		print("=> no checkpoint found at '{}'".format(checkpoint_dir), file=F_txt)

		return None


def plot_loss_curve(opt, train_loss, val_loss, test_loss=None):

	if test_loss:
		train_loss = np.array(train_loss)
		val_loss = np.array(val_loss)
		test_loss = np.array(test_loss)


		# Save lossed to txt
		np.savetxt(os.path.join(opt.outf, 'train_loss.txt'), train_loss)
		np.savetxt(os.path.join(opt.outf, 'val_loss.txt'), val_loss)
		np.savetxt(os.path.join(opt.outf, 'test_loss.txt'), test_loss)

		# Plot the loss curves
		fig, ax = plt.subplots()
		ax.plot(range(0, opt.epochs), train_loss, label='Train loss')
		ax.plot(range(0, opt.epochs), val_loss, label='Val loss')
		ax.plot(range(0, opt.epochs), test_loss, label='Test loss')
		legend = ax.legend(loc='upper right', fontsize='medium')
		plt.savefig(os.path.join(opt.outf, 'Loss.png'), bbox_inches='tight')
		# plt.show()
	else:
		train_loss = np.array(train_loss)
		val_loss = np.array(val_loss)


		# Save lossed to txt
		np.savetxt(os.path.join(opt.outf, 'train_loss.txt'), train_loss)
		np.savetxt(os.path.join(opt.outf, 'val_loss.txt'), val_loss)


		# Plot the loss curves
		fig, ax = plt.subplots()
		ax.plot(range(0, opt.epochs), train_loss, label='Train loss')
		ax.plot(range(0, opt.epochs), val_loss, label='Val loss')
		legend = ax.legend(loc='upper right', fontsize='medium')
		plt.savefig(os.path.join(opt.outf, 'Loss.png'), bbox_inches='tight')
		# plt.show()


def plot_vat_wass_loss_curve(opt, loss1, loss2):


	loss1 = np.array(loss1)
	loss2 = np.array(loss2)


	# Save lossed to txt
	np.savetxt(os.path.join(opt.outf, 'Loss_Wass.txt'), loss1)
	np.savetxt(os.path.join(opt.outf, 'Loss_VAT.txt'), loss2)


	# Plot the loss curves
	fig, ax = plt.subplots()
	ax.plot(range(0, opt.epochs), loss1, label='Wass loss')
	ax.plot(range(0, opt.epochs), loss2, label='VAT loss')
	legend = ax.legend(loc='upper right', fontsize='medium')
	plt.savefig(os.path.join(opt.outf, 'Wass_VAT_Loss.png'), bbox_inches='tight')
	# plt.show()


def plot_clean_adv_loss_curve(opt, clean_loss, adv_loss):

	clean_loss = np.array(clean_loss)
	adv_loss = np.array(adv_loss)


	# Save lossed to txt
	np.savetxt(os.path.join(opt.outf, 'Loss_clean.txt'), clean_loss)
	np.savetxt(os.path.join(opt.outf, 'Loss_adv.txt'), adv_loss)


	# Plot the loss curves
	fig, ax = plt.subplots()
	ax.plot(range(0, opt.epochs), clean_loss, label='Clean loss')
	ax.plot(range(0, opt.epochs), adv_loss, label='Adv loss')
	legend = ax.legend(loc='upper right', fontsize='medium')
	plt.savefig(os.path.join(opt.outf, 'Clean_Adv_Loss.png'), bbox_inches='tight')
	# plt.show()


def plot_tSNE_curve(opt, Fea_Norm, Fea_Adv, Fea_sup, Target_norm_adv, Target_sup, epoch_index, mode='train'):

	# Set the save path
	if mode == 'train':
		fea_file_path = os.path.join(opt.outf, 'Train_Feature_tSNE')
	else:
		fea_file_path = os.path.join(opt.outf, 'Val_Feature_tSNE')

	if not os.path.exists(fea_file_path):
		os.makedirs(fea_file_path)
	
	# pdb.set_trace()
	Fea_total = torch.cat((Fea_Norm, Fea_Adv, Fea_sup), 0)
	Fea_total_New = TSNE(n_components=2).fit_transform(Fea_total.cpu().detach().numpy())

	Fea_Norm_New = Fea_total_New[0:Fea_Norm.size(0)]
	Fea_Adv_New = Fea_total_New[Fea_Norm.size(0):]
	Fea_Sup_New = Fea_total_New[Fea_Norm.size(0)+Fea_Adv.size(0):]


	# pdb.set_trace()	
	# Plot the loss curves
	plt.figure(figsize=(10,10))
	color = {0: 'red', 1: 'blue', 2: 'green', 3: 'cyan', 4: 'yellow'}

	target_unique = torch.unique(Target_norm_adv, sorted=True)
	for item in range(4):
		current_target = target_unique[item].unsqueeze(0)
		current_target_index = (Target_norm_adv==current_target).nonzero().squeeze(1)
		current_sup_index = (Target_sup==current_target).nonzero().squeeze(1)
		current_target = current_target.cpu().detach().numpy()


		current_Fea_Norm = Fea_Norm_New[current_target_index.cpu().detach().numpy()]
		current_Fea_Adv = Fea_Adv_New[current_target_index.cpu().detach().numpy()]
		current_Fea_sup = Fea_Sup_New[current_sup_index.cpu().detach().numpy()]


		# Plot the curves
		p1 = plt.scatter(current_Fea_Norm[:,0], current_Fea_Norm[:,1], s=80, c=color[item], marker='+', linewidths=3)
		p2 = plt.scatter(current_Fea_Adv[:,0], current_Fea_Adv[:,1], s=30, c=color[item], marker='o', linewidths=1)  
		p3 = plt.scatter(current_Fea_sup[:,0], current_Fea_sup[:,1], s=120, c=color[item], marker='*', linewidths=2)
		legend1 = plt.legend([p1, p2, p3], ['class_'+str(current_target[0])+'_Clean', 'class_'+str(current_target[0])+'_Adv', 'class_'+str(current_target[0])+'_Sup'], 
			loc=item+1, fontsize='medium', scatterpoints=1)
		plt.gca().add_artist(legend1)
		

	plt.axis([-20, 20, -20, 20])
	plt.savefig(os.path.join(fea_file_path, 'Epoch_'+str(epoch_index)+'_tSNE.jpg'), bbox_inches='tight')
	plt.close()



