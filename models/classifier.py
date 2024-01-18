import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import torch.nn.functional as F
import utils
import pdb
from abc import abstractmethod
from torch.nn.utils.weight_norm import WeightNorm
from .emd_utils import *
import models.can as CAN_model


# =========================== Fine tune a new classifier model =========================== #
class Finetune_Classifier(nn.Module):
	'''
		Construct a new classifier module for a new task.
	'''
	def __init__(self, class_num=5, feature_dim=64, loss_type = 'softmax'):
		super(Finetune_Classifier, self).__init__()
		self.class_num = class_num
		self.feature_dim = feature_dim
		self.loss_type = loss_type       #'softmax' or #'dist'

		if loss_type == 'softmax':  # Baseline
			self.avgpool = nn.AdaptiveAvgPool2d(1)
			self.classifier = nn.Linear(self.feature_dim, self.class_num)

		elif loss_type == 'dist':   # Baseline ++
			self.avgpool = nn.AdaptiveAvgPool2d(1)
			self.classifier = distLinear(self.feature_dim, self.class_num)


	def forward(self, x):
	
		x = self.avgpool(x).squeeze(3).squeeze(2)	
		# x = x.view(x.size(0), -1)             
		scores = self.classifier(x)

		return scores



# =========================== General classification method: Baseline =========================== #
class Baseline_Metric(nn.Module):
	'''
		Classifier module of the general image classification task.
		1. Baseline
		2. Baseline ++ 
		   Note that Both of them are parametric classifiers.
		   "A Closer Look at Few-shot Classification. ICLR 2019."
	'''
	def __init__(self, class_num=64, feature_dim=64, loss_type = 'softmax'):
		super(Baseline_Metric, self).__init__()
		self.class_num = class_num
		self.feature_dim = feature_dim
		self.loss_type = loss_type       #'softmax' or #'dist'

		if loss_type == 'softmax':  # Baseline
			self.avgpool = nn.AdaptiveAvgPool2d(1)
			self.classifier = nn.Linear(self.feature_dim, self.class_num)

		elif loss_type == 'dist':   # Baseline ++
			self.avgpool = nn.AdaptiveAvgPool2d(1)
			self.classifier = distLinear(self.feature_dim, self.class_num)


	def forward(self, x):
		# pdb.set_trace()
		x = self.avgpool(x).squeeze(3).squeeze(2)	 # 64 * 64
		# x = x.view(x.size(0), -1)                  # 64 * 1600        
		scores = self.classifier(x)

		return scores


class distLinear(nn.Module):
	'''
		Coming from "A Closer Look at Few-shot Classification. ICLR 2019."
		https://github.com/wyharveychen/CloserLookFewShot.git
	'''
	def __init__(self, indim, outdim):
		super(distLinear, self).__init__()
		self.L = nn.Linear( indim, outdim, bias = False)
		self.class_wise_learnable_norm = True  #See the issue#4&8 in the github 
		if self.class_wise_learnable_norm:      
			WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm      

		if outdim <=200:
			self.scale_factor = 2; #a fixed scale factor to scale the output of cos value into a reasonably large input for softmax
		else:
			self.scale_factor = 10; #in omniglot, a larger scale factor is required to handle >1000 output classes.

	def forward(self, x):
		x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
		x_normalized = x.div(x_norm+ 0.00001)
		if not self.class_wise_learnable_norm:
			L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
			self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
		cos_dist = self.L(x_normalized) #matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm
		scores = self.scale_factor* (cos_dist) 

		return scores



# =========================== Few-shot learning method: ProtoNet =========================== #
class Prototype_Metric(nn.Module):
	'''
		The classifier module of ProtoNet by using the mean prototype and Euclidean distance,
		which is also Non-parametric.
		"Prototypical networks for few-shot learning. NeurIPS 2017."
	'''
	def __init__(self, way_num=5, shot_num=5, neighbor_k=3):
		super(Prototype_Metric, self).__init__()
		self.way_num = way_num
		self.avgpool = nn.AdaptiveAvgPool2d(1)


	# Calculate the Euclidean distance between the query and the mean prototype of the support class.
	def cal_EuclideanDis(self, input1, input2):
		'''
		 input1 (query images): 75 * 64 * 5 * 5
		 input2 (support set):  25 * 64 * 5 * 5
		'''
	
		# input1---query images
		query = input1.view(input1.size(0), -1)                                      # 75 * 1600     (Conv64F)
		# query = self.avgpool(input1).squeeze(3).squeeze(2)
		query = query.unsqueeze(1)                                                   # 75 * 1 * 1600 (Conv64F)
   

		# input2--support set
		# input2 = self.avgpool(input2).squeeze(3).squeeze(2)                        # 25 * 64
		input2 = input2.view(input2.size(0), -1)                                     # 25 * 1600     
		support_set = input2.contiguous().view(self.way_num, -1, input2.size(1))     # 5 * 5 * 1600    
		support_set = torch.mean(support_set, 1)                                     # 5 * 1600


		# Euclidean distances between a query set and a support set
		proto_dis = -torch.pow(query-support_set, 2).sum(2)                          # 75 * 5 
		

		return proto_dis


	def forward(self, x1, x2):

		proto_dis = self.cal_EuclideanDis(x1, x2)

		return proto_dis



# =========================== Few-shot learning method: DN4 =========================== #
class ImgtoClass_Metric(nn.Module):
	'''
		Image-to-class classifier module for DN4, which is a Non-parametric classifier.
		"Revisiting local descriptor based image-to-class measure for few-shot learning. CVPR 2019."
	'''
	def __init__(self, way_num=5, shot_num=5, neighbor_k=3):
		super(ImgtoClass_Metric, self).__init__()
		self.neighbor_k = neighbor_k
		self.shot_num = shot_num


	# Calculate the Image-to-class similarity between the query and support class via k-NN.
	def cal_cosinesimilarity(self, input1, input2):
		'''
		 input1 (query images):  75 * 64 * 21 * 21
		 input2 (support set):   25 * 64 * 21 * 21
		'''

		# input1---query images
		input1 = input1.contiguous().view(input1.size(0), input1.size(1), -1)         # 75 * 64 * 441 (Conv64F_Li)
		input1 = input1.permute(0, 2, 1)                                              # 75 * 441 * 64 (Conv64F_Li)

		
		# input2--support set
		input2 = input2.contiguous().view(input2.size(0), input2.size(1), -1)         # 25 * 64 * 441
		input2 = input2.permute(0, 2, 1)                                              # 25 * 441 * 64


		# L2 Normalization
		input1_norm = torch.norm(input1, 2, 2, True)                                  # 75 * 441 * 1
		query = input1/input1_norm                                                    # 75 * 441 * 64
		query = query.unsqueeze(1)                                                    # 75 * 1 * 441 *64


		input2_norm = torch.norm(input2, 2, 2, True)                                  # 25 * 441 * 1 
		support_set = input2/input2_norm                                              # 25 * 441 * 64
		support_set = support_set.contiguous().view(-1,
				self.shot_num*support_set.size(1), support_set.size(2))               # 5 * 2205 * 64    
		support_set = support_set.permute(0, 2, 1)                                    # 5 * 64 * 2205     


		# cosine similarity between a query set and a support set
		innerproduct_matrix = torch.matmul(query, support_set)                        # 75 * 5 * 441 * 2205


		# choose the top-k nearest neighbors
		topk_value, topk_index = torch.topk(innerproduct_matrix, self.neighbor_k, 3)  # 75 * 5 * 441 * 3
		img2class_sim = torch.sum(torch.sum(topk_value, 3), 2)                        # 75 * 5 


		return img2class_sim


	def forward(self, x1, x2):

		img2class_sim = self.cal_cosinesimilarity(x1, x2)

		return img2class_sim



# =========================== Few-shot learning method: CovaMNet =========================== #
class Covariance_Metric(nn.Module):
	'''
		Covariance metric classifier module of CovaMNet.
		"Distribution Consistency based Covariance Metric Networks for Few-shot Learning. AAAI 2019."
	'''
	def __init__(self, way_num=5, shot_num=5, neighbor_k=3):
		super(Covariance_Metric, self).__init__()
		self.shot_num = shot_num

		self.conv1d_layer = nn.Sequential(
			nn.LeakyReLU(0.2, True),
			nn.Dropout(),
			nn.Conv1d(1, 1, kernel_size=441, stride=441),
		)

		for m in self.modules():
			if isinstance(m, nn.Conv1d):
				init.normal_(m.weight.data, 0.0, 0.02)
			elif isinstance(m, nn.Linear):
				init.normal_(m.weight.data, 0.0, 0.02)
			elif isinstance(m, nn.BatchNorm2d):
				init.normal_(m.weight.data, 1.0, 0.02)
				init.constant_(m.bias.data, 0.0)


	def cal_covariance_Batch(self, feature):     
		'''
		   Calculate the Covariance Matrices based on the local descriptors for a mini-batch images.  
		   feature (support set): 25 * 64 * 21 * 21
		'''

		feature = feature.contiguous().view(feature.size(0), feature.size(1), -1)     # 25 * 64 * 441
		feature = feature.permute(0, 2, 1)                                            # 25 * 441 * 64
		feature = feature.contiguous().view(-1,
				self.shot_num*feature.size(1), feature.size(2))                       # 5 * 2205 * 64    
	

		n_local_descriptor = torch.tensor(feature.size(1)).cuda()
		feature_mean = torch.mean(feature, 1, True)                                   # 5 * 1 * 64
		feature = feature - feature_mean                                              # 5 * 2205 * 64
		cov_matrix = torch.matmul(feature.permute(0, 2, 1), feature)                  # 5 * 64 * 64
		cov_matrix = torch.div(cov_matrix, n_local_descriptor - 1)                    # 5 * 64 * 64

		return feature_mean, cov_matrix 


	# calculate the similarity  
	def cal_covasimilarity(self, input, CovaMatrix):
	
		B, C, h, w = input.size()
		Cova_Sim = []
	
		for i in range(B):
			query_sam = input[i]
			query_sam = query_sam.view(C, -1)
			mean_query = torch.mean(query_sam, 1, True)
			query_sam = query_sam-mean_query

			if torch.cuda.is_available():
				mea_sim = torch.zeros(1, CovaMatrix.size(0)*h*w).cuda()

			for j in range(CovaMatrix.size(0)):
				temp_dis = torch.transpose(query_sam, 0, 1)@CovaMatrix[j]@query_sam
				mea_sim[0, j*h*w:(j+1)*h*w] = temp_dis.diag()

			Cova_Sim.append(mea_sim.unsqueeze(0))

		Cova_Sim = torch.cat(Cova_Sim, 0) # get Batch*1*(h*w*num_classes)

		return Cova_Sim 


	def forward(self, x1, x2):

		Mean_support, CovaMatrix = self.cal_covariance_Batch(x2)
		Cova_Sim = self.cal_covasimilarity(x1, CovaMatrix)
		scores = self.conv1d_layer(Cova_Sim).squeeze(1)

		return scores



# =========================== Few-shot learning method: RelationNet =========================== #
class LearnToCompare_Metric(nn.Module):
	'''
		Learn-to-compare classifier module for RelationNet, which is a parametric classifier.
		"Learning to Compare: Relation Network for Few-Shot Learning. CVPR 2018."
	'''
	def __init__(self, way_num=5, shot_num=5, neighbor_k=3):
		super(LearnToCompare_Metric, self).__init__()
		self.neighbor_k = neighbor_k
		self.shot_num = shot_num
		self.way_num = way_num


		# Relation Block of RelationNet
		self.RelationNetwork = nn.Sequential(
			nn.Conv2d(64*2,64,kernel_size=3,padding=0),
			nn.BatchNorm2d(64, momentum=1, affine=True),
			nn.ReLU(),
			nn.MaxPool2d(2),

			nn.Conv2d(64,64,kernel_size=3,padding=0),
			nn.BatchNorm2d(64, momentum=1, affine=True),
			nn.ReLU(),
			nn.MaxPool2d(2)
		)

		self.fc1 = nn.Linear(64*3*3, 8)
		self.fc2 = nn.Linear(8,1)


		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.Linear):
				init.normal_(m.weight.data, 0.0, 0.02)
			elif isinstance(m, nn.BatchNorm2d):
				init.normal_(m.weight.data, 1.0, 0.02)
				init.constant_(m.bias.data, 0.0)


	# Calculate the relationships between the query and support class via a Deep Metric Network.
	def cal_relationships(self, input1, input2):
		'''
		 input1 (query images):  75 * 64 * 21 * 21
		 input2 (support set):   25 * 64 * 21 * 21
		'''
		# pdb.set_trace()
		# input1---query images
		input1 = input1.unsqueeze(0).repeat(1*self.way_num,1,1,1,1)              # 5 * 75 * 64 * 21 * 21 (Conv64F_Li)
		query = input1.permute(1, 0, 2, 3, 4)                                    # 75 * 5 * 64 * 21 * 21 (Conv64F_Li)

		
		# input2--support set
		input2 = input2.contiguous().view(self.way_num, self.shot_num, input2.size(1), input2.size(2), input2.size(3))   # 5 * 5 * 64 * 21 * 21
		support = torch.sum(input2, 1).squeeze(1)                                                                        # 5 * 64 * 21 * 21
		support = support.unsqueeze(0).repeat(query.size(0), 1, 1, 1, 1)         # 75 * 5 * 64 * 21 * 21 (Conv64F_Li)


		# Concatenation 
		relation_pairs = torch.cat((support, query), 2).view(-1,input2.size(2)*2, input2.size(3), input2.size(4))
		out = self.RelationNetwork(relation_pairs)
		out = out.view(out.size(0), -1)
		out = F.relu(self.fc1(out))
		relations = self.fc2(out).view(-1, self.way_num).cuda()

		return relations


	def forward(self, x1, x2):

		relations = self.cal_relationships(x1, x2)

		return relations




# =========================== Few-shot learning method: DeepEMD =========================== #
class DeepEMD_Metric(nn.Module):
	'''
		DeeepEMD_Metric classifier module for DeepEMD.
		"DeepEMD: Differentiable Earth Mover's Distance for Few-Shot Learning. CVPR 2020."
	'''
	def __init__(self, way_num=5, shot_num=5, neighbor_k=3):
		super(DeepEMD_Metric, self).__init__()
		self.neighbor_k = neighbor_k
		self.way_num = way_num
		self.shot_num = shot_num
		self.norm = 'center'
		self.metric = 'cosine'
		self.solver = 'opencv'
		self.temperature = 12.5
		self.form = 'L2'
		self.l2_strength = 0.000001
		self.sfc_lr = 0.1
		self.sfc_wd = 0
		self.sfc_update_step = 2
		self.sfc_bs = 4


	def forward(self, x1, x2):

		if self.shot_num == 1:
			logits = self.emd_forward_1shot(x2, x1)
			return logits
		
		elif self.shot_num == 5:
			# x2 = self.get_sfc(x2)
			x2 = x2.contiguous().view(self.way_num, self.shot_num, x2.size(1), x2.size(2), x2.size(3))
			x2 = torch.mean(x2, 1)
			logits = self.emd_forward_1shot(x2, x1)
			return logits
		else:
			raise ValueError('Unknown mode')
		

	def emd_forward_1shot(self, proto, query):
		# pdb.set_trace()
		proto = proto.squeeze(0)

		weight_1 = self.get_weight_vector(query, proto)
		weight_2 = self.get_weight_vector(proto, query)

		proto = self.normalize_feature(proto)
		query = self.normalize_feature(query)

		similarity_map = self.get_similiarity_map(proto, query)
		if self.solver == 'opencv' or (not self.training):
			logits = self.get_emd_distance(similarity_map, weight_1, weight_2, solver='opencv')
		else:
			logits = self.get_emd_distance(similarity_map, weight_1, weight_2, solver='qpth')
   
		return logits


	def get_sfc(self, support):
		support = support.squeeze(0)
		# init the proto
		SFC = support.view(self.shot_num, -1, 64, support.shape[-2], support.shape[-1]).mean(dim=0).clone().detach()
		SFC = nn.Parameter(SFC.detach(), requires_grad=True)

		optimizer = torch.optim.SGD([SFC], lr=self.sfc_lr, momentum=0.9, dampening=0.9, weight_decay=0)

		# crate label for finetune
		label_shot = torch.arange(self.way_num).repeat(self.shot_num)
		label_shot = label_shot.type(torch.cuda.LongTensor)

		with torch.enable_grad():
			for k in range(0, self.sfc_update_step):
				rand_id = torch.randperm(self.way_num * self.shot_num).cuda()
				for j in range(0, self.way_num * self.shot_num, self.sfc_bs):
					selected_id = rand_id[j: min(j + self.sfc_bs, self.way_num * self.shot_num)]
					batch_shot = support[selected_id, :]
					batch_label = label_shot[selected_id]
					optimizer.zero_grad()
					logits = self.emd_forward_1shot(SFC, batch_shot.detach())
					loss = F.cross_entropy(logits, batch_label)
					loss.backward()
					optimizer.step()
		return SFC


	def get_weight_vector(self, A, B):

		M = A.shape[0]
		N = B.shape[0]

		B = F.adaptive_avg_pool2d(B, [1, 1])
		B = B.repeat(1, 1, A.shape[2], A.shape[3])

		A = A.unsqueeze(1)
		B = B.unsqueeze(0)

		A = A.repeat(1, N, 1, 1, 1)
		B = B.repeat(M, 1, 1, 1, 1)

		combination = (A * B).sum(2)
		combination = combination.view(M, N, -1)
		combination = F.relu(combination) + 1e-3
		return combination


	def normalize_feature(self, x):
		if self.norm == 'center':
			x = x - x.mean(1).unsqueeze(1)
			return x
		else:
			return x

	def get_similiarity_map(self, proto, query):
		way = proto.shape[0]
		num_query = query.shape[0]
		query = query.view(query.shape[0], query.shape[1], -1)
		proto = proto.view(proto.shape[0], proto.shape[1], -1)

		proto = proto.unsqueeze(0).repeat([num_query, 1, 1, 1])
		query = query.unsqueeze(1).repeat([1, way, 1, 1])
		proto = proto.permute(0, 1, 3, 2)
		query = query.permute(0, 1, 3, 2)
		feature_size = proto.shape[-2]

		if self.metric == 'cosine':
			proto = proto.unsqueeze(-3)
			query = query.unsqueeze(-2)
			query = query.repeat(1, 1, 1, feature_size, 1)
			similarity_map = F.cosine_similarity(proto, query, dim=-1)
		if self.metric == 'l2':
			proto = proto.unsqueeze(-3)
			query = query.unsqueeze(-2)
			query = query.repeat(1, 1, 1, feature_size, 1)
			similarity_map = (proto - query).pow(2).sum(-1)
			similarity_map = 1 - similarity_map

		return similarity_map


	def get_emd_distance(self, similarity_map, weight_1, weight_2, solver='opencv'):
		num_query = similarity_map.shape[0]
		num_proto = similarity_map.shape[1]
		num_node=weight_1.shape[-1]

		if solver == 'opencv':  # use openCV solver

			for i in range(num_query):
				for j in range(num_proto):
					_, flow = emd_inference_opencv(1 - similarity_map[i, j, :, :], weight_1[i, j, :], weight_2[j, i, :])

					similarity_map[i, j, :, :] =(similarity_map[i, j, :, :])*torch.from_numpy(flow).cuda()

			temperature=(self.temperature/num_node)
			logitis = similarity_map.sum(-1).sum(-1) *  temperature
			return logitis

		elif solver == 'qpth':
			weight_2 = weight_2.permute(1, 0, 2)
			similarity_map = similarity_map.view(num_query * num_proto, similarity_map.shape[-2],
												 similarity_map.shape[-1])
			weight_1 = weight_1.view(num_query * num_proto, weight_1.shape[-1])
			weight_2 = weight_2.reshape(num_query * num_proto, weight_2.shape[-1])

			_, flows = emd_inference_qpth(1 - similarity_map, weight_1, weight_2,form=self.form, l2_strength=self.l2_strength)

			logitis=(flows*similarity_map).view(num_query, num_proto,flows.shape[-2],flows.shape[-1])
			temperature = (self.temperature / num_node)
			logitis = logitis.sum(-1).sum(-1) *  temperature
		else:
			raise ValueError('Unknown Solver')

		return logitis




# =========================== Few-shot learning method: CAN =========================== #
class CAN_Metric(nn.Module):
	'''
		CAN_Metric classifier module for CAN.
		"Cross Attention Network for Few-shot Classification (NeurIPS 2019)."
	'''
	def __init__(self, way_num=5, shot_num=5, neighbor_k=3, num_classes=64, scale_cls=7, iter_num_prob=35.0/75):
		super(CAN_Metric, self).__init__()
		self.neighbor_k = neighbor_k
		self.way_num = way_num
		self.shot_num = shot_num
		self.cam_layer = CAN_model.CAMLayer(scale_cls, iter_num_prob, num_classes, nFeat=64, HW=5)


	def forward(self, x1, x2, query_targets, support_targets):
		'''
		 x1 (query images):  75 * 64 * 5 * 5
		 x2 (support set):   25 * 64 * 5 * 5
		'''
		query_feat   = x1.unsqueeze(0)
		support_feat = x2.unsqueeze(0)
		episode_size = 1


		# convert to one-hot
		support_targets_one_hot = CAN_model.one_hot(
			support_targets.reshape(episode_size * self.way_num * self.shot_num), self.way_num
		)
		support_targets_one_hot = support_targets_one_hot.reshape(
			episode_size, self.way_num * self.shot_num, self.way_num
		)
		query_targets_one_hot = CAN_model.one_hot(
			query_targets.reshape(episode_size * x1.size(0)), self.way_num
		)
		query_targets_one_hot = query_targets_one_hot.reshape(
			episode_size, x1.size(0), self.way_num
		)

		cls_scores = self.cam_layer(support_feat, query_feat, support_targets_one_hot, query_targets_one_hot)
		cls_scores = torch.sum(cls_scores.reshape(*cls_scores.size()[:2], -1), dim=-1)

		return cls_scores