import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import functools
import random
import pdb
import copy
import sys
sys.dont_write_bytecode = True



# ============================ Backbone & Classifier ===============================
import models.backbone as backbone
import models.classifier as classifier
# ==================================================================================
''' 
	All models consist of two parts: backbone module and classifier module.
'''


###############################################################################
# Functions
###############################################################################

encoder_dict = dict(
			Conv64F    = backbone.Conv64F,
			Conv64F_Local = backbone.Conv64F_Local,
			ResNet12   = backbone.ResNet12) 


classifier_dict = dict(
			Baseline = classifier.Baseline_Metric,
			Baseline_plus = classifier.Baseline_Metric,
			ProtoNet = classifier.Prototype_Metric,
			CovaMNet = classifier.Covariance_Metric,
			RelationNet = classifier.LearnToCompare_Metric,
			DN4      = classifier.ImgtoClass_Metric,
			CAN      = classifier.CAN_Metric,
			DeepEMD  = classifier.DeepEMD_Metric) 


def weights_init_normal(m):
	classname = m.__class__.__name__
	
	# print(classname)
	if classname.find('Conv') != -1:
		init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('Linear') != -1:
		init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('BatchNorm2d') != -1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
	classname = m.__class__.__name__
	# print(classname)
	if classname.find('Conv') != -1:
		init.xavier_normal_(m.weight.data, gain=0.02)
	elif classname.find('Linear') != -1:
		init.xavier_normal_(m.weight.data, gain=0.02)
	elif classname.find('BatchNorm2d') != -1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
	classname = m.__class__.__name__
	# print(classname)
	if classname.find('Conv') != -1:
		init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
	elif classname.find('Linear') != -1:
		init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
	elif classname.find('BatchNorm2d') != -1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
	classname = m.__class__.__name__
	print(classname)
	if classname.find('Conv') != -1:
		init.orthogonal_(m.weight.data, gain=1)
	elif classname.find('Linear') != -1:
		init.orthogonal_(m.weight.data, gain=1)
	elif classname.find('BatchNorm2d') != -1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
	print('initialization method [%s]' % init_type)
	if init_type == 'normal':
		net.apply(weights_init_normal)
	elif init_type == 'xavier':
		net.apply(weights_init_xavier)
	elif init_type == 'kaiming':
		net.apply(weights_init_kaiming)
	elif init_type == 'orthogonal':
		net.apply(weights_init_orthogonal)
	else:
		raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
	if norm_type == 'batch':
		norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
	elif norm_type == 'instance':
		norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
	elif norm_type == 'none':
		norm_layer = None
	else:
		raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
	return norm_layer


def print_network(net):
	num_params = 0
	for param in net.parameters():
		num_params += param.numel()
	
	print('Total number of parameters: %d' % num_params)



def define_model(pretrained=False, model_root=None, encoder_model='Conv64F', classifier_model='ProtoNet', norm='batch', init_type='normal', use_gpu=True, **kwargs):
	model = None
	norm_layer = get_norm_layer(norm_type=norm)

	if use_gpu:
		assert(torch.cuda.is_available())

	if   classifier_model in ['Baseline', 'Baseline_plus']:
		model = General_model(encoder_model=encoder_model, classifier_model=classifier_model, **kwargs)

	elif classifier_model in ['ProtoNet', 'RelationNet', 'CovaMNet', 'DN4', 'CAN', 'DeepEMD']:
		model = Fewshot_model(encoder_model=encoder_model, classifier_model=classifier_model, **kwargs)

	else:
		raise NotImplementedError('Model name [%s] is not recognized' % classifier_model)
	
	print_network(model)

	if use_gpu:
		model.cuda()

	if pretrained:
		model.load_state_dict(model_root)

	return model




class Fewshot_model(nn.Module):
	'''
		Define a few-shot learning model, which consists of an embedding module and a classifier moduel.
	'''
	def __init__(self, encoder_model='Conv64F', classifier_model='ProtoNet', class_num=64, way_num=5, shot_num=5, query_num=10, neighbor_k=3):
		super(Fewshot_model, self).__init__()
		self.encoder_model = encoder_model
		self.classifier_model = classifier_model
		self.way_num = way_num
		self.shot_num = shot_num
		self.query_num = query_num
		self.neighbor_k = neighbor_k
		self.loss_type = 'softmax'

		if   encoder_model in ['Conv64F', 'Conv64F_Local']:
			self.feature_dim = 64
		elif encoder_model in ['ResNet12', 'ResNet12_Local']:
			self.feature_dim = 640

		
		encoder_module    = encoder_dict[self.encoder_model]
		classifier_module = classifier_dict[self.classifier_model]

		self.features   = encoder_module()
		self.classifier = classifier_module(way_num=self.way_num, shot_num=self.shot_num, neighbor_k=self.neighbor_k)


	def forward(self, input1, input2, is_feature=False):
		
		x1 = self.features(input1)      # query 
		x2 = self.features(input2)      # support set
		out = self.classifier(x1, x2)

		if is_feature:
			return x1, x2, out
		else:
			return out



class General_model(nn.Module):
	'''
		Define a general image classification model, which consists of an embedding module and a classifier moduel.
	'''
	def __init__(self, encoder_model='Conv64F', classifier_model='Baseline', class_num=64, way_num=5, shot_num=5, query_num=10, neighbor_k=3):
		super(General_model, self).__init__()
		self.class_num = class_num
		self.way_num = way_num
		self.shot_num = shot_num

		if   encoder_model in ['Conv64F', 'Conv64F_Local']:
			self.feature_dim = 64
		elif encoder_model in ['ResNet12', 'ResNet12_Local']:
			self.feature_dim = 640


		if   classifier_model == 'Baseline':
			self.loss_type = 'softmax'
		elif classifier_model == 'Baseline_plus':
			self.loss_type = 'dist'
		elif classifier_model == 'LR':
			self.loss_type = 'LR'

		encoder_module    = encoder_dict[encoder_model]
		classifier_module = classifier_dict[classifier_model]

		self.features   = encoder_module()
		self.classifier = classifier_module(class_num=self.class_num, feature_dim=self.feature_dim, loss_type=self.loss_type)


	def forward(self, x, is_feature=False):
	
		x   = self.features(x)
		out = self.classifier(x)

		if is_feature:
			return x, out
		else:
			return out



class Model_with_reused_Encoder(nn.Module):
	'''
		Construct a new few-shot model by reusing a pre-trained embedding module.
	'''
	def __init__(self, pre_trained_model, new_classifier='ProtoNet', way_num=5, shot_num=5, neighbor_k=3):
		super(Model_with_reused_Encoder, self).__init__()
		self.way_num = way_num
		self.shot_num = shot_num
		self.neighbor_k = neighbor_k
		self.model = pre_trained_model

		# Only use the features module
		self.features = nn.Sequential(
			*list(self.model.features.children())
			)

		classifier_module = classifier_dict[new_classifier]
		self.classifier = classifier_module(way_num=self.way_num, shot_num=self.shot_num, neighbor_k=self.neighbor_k)


	def forward(self, input1, input2, is_feature=False):
		
		x1 = self.features(input1)      # query 
		x2 = self.features(input2)      # support set
		out = self.classifier(x1, x2)

		if is_feature:
			return x1, x2, out
		else:
			return out