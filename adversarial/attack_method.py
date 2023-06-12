import torch
import torch.nn as nn
import pdb


# define loss function (criterion) 
criterion_mse = nn.MSELoss(reduction='mean').cuda()
criterion_kl = nn.KLDivLoss(reduction='batchmean').cuda()



# calculate the covariance matrix 
def cal_covariance(input_sam):

	B, C, h, w = input_sam.size()

	input_sam = input_sam.permute(1, 0, 2, 3)
	input_sam = input_sam.contiguous().view(C, -1)
	mean_vector = torch.mean(input_sam, 1, True)
	input_sam = input_sam-mean_vector

	covariance_matrix = input_sam@torch.transpose(input_sam, 0, 1)
	covariance_matrix = torch.div(covariance_matrix, h*w*B-1)
	covariance_matrix = covariance_matrix + 0.01*torch.eye(covariance_matrix.size(1)).cuda()

	return mean_vector, covariance_matrix  



def cal_covariance_Batch(feature):     # feature: 75 * 64 * 21 * 21
	
	feature = feature.contiguous().view(feature.size(0), feature.size(1), -1)     # 75 * 64 * 441
	feature = feature.permute(0, 2, 1)                                            # 75 * 441 * 64
	

	n_local_descriptor = torch.tensor(feature.size(1)).cuda()
	feature_mean = torch.mean(feature, 1, True)   # Batch * 1 * 64
	feature = feature - feature_mean
	cov_matrix = torch.matmul(feature.permute(0, 2, 1), feature)
	cov_matrix = torch.div(cov_matrix, n_local_descriptor - 1)
	cov_matrix = cov_matrix + 0.01*torch.eye(cov_matrix.size(1)).cuda()

	return feature_mean, cov_matrix



def cal_covariance_Batch_Class(feature):     # feature: 75 * 64 * 21 * 21
	
	feature = feature.contiguous().view(feature.size(0), feature.size(1), -1)     # 75 * 64 * 441
	feature = feature.permute(0, 2, 1)                                            # 75 * 441 * 64
	feature = feature.contiguous().view(-1, opt.query_num*feature.size(1), feature.size(2))   # 5 * 6615 * 64    
	

	n_local_descriptor = torch.tensor(feature.size(1)).cuda()
	feature_mean = torch.mean(feature, 1, True)   # Batch * 1 * 64
	feature = feature - feature_mean
	cov_matrix = torch.matmul(feature.permute(0, 2, 1), feature)
	cov_matrix = torch.div(cov_matrix, n_local_descriptor - 1)
	cov_matrix = cov_matrix + 0.01*torch.eye(cov_matrix.size(1)).cuda()

	return feature_mean, cov_matrix



# calculate the covariance matrix over the entire support set
def cal_covariance_S(input_S):

	# input2--support set
	input_S = input_S.contiguous().view(input_S.size(0), input_S.size(1), -1)     # 25 * 64 * 441
	input_S = input_S.permute(0, 2, 1)                                            # 25 * 441 * 64
	input_matrix = input_S.contiguous().view(-1, input_S.size(2))                 # 11025 * 64 for 5-shot


	# pdb.set_trace()
	mean_input = torch.mean(input_matrix, 0, True)   # 1*64
	input_matrix = input_matrix-mean_input
	covariance_matrix = torch.matmul(input_matrix.permute(1,0), input_matrix)  # 64*64
	covariance_matrix = torch.div(covariance_matrix, input_matrix.size(0)-1)   # 64*64
	covariance_matrix_inverse = torch.inverse(covariance_matrix + 0.01*torch.eye(input_matrix.size(1)).cuda()) # Inverse matrix

	return covariance_matrix, covariance_matrix_inverse



def KL_distance_Batch(mean1, cov1, mean2, cov2, triplet_loss=False):
	'''
	   mean1: 75 * 1 * 64
	   cov1:  75 * 64 * 64
	   mean2: 75 * 1 * 64
	   cov2: 75 * 64 * 64
	'''

	cov2_inverse = torch.inverse(cov2)            # 75 * 64 * 64
	cov1_det = torch.det(cov1)                    # 75 
	cov2_det = torch.det(cov2)                    # 75 
	mean_diff = mean2 - mean1                     # 75 * 1 * 64

	# Calculate the trace
	matrix_product = torch.matmul(cov2_inverse, cov1)         # 75 * 64 * 64
	trace_dis = [torch.trace(matrix_product[i]).unsqueeze(0) for i in range(matrix_product.size(0))]
	trace_dis = torch.cat(trace_dis, 0).unsqueeze(1)          # 75 * 1
	
	# Calcualte the Mahalanobis Distance
	maha_product = torch.matmul(mean_diff, cov2_inverse)                              # 75 * 1 * 64
	maha_product = torch.matmul(maha_product, mean_diff.permute(0, 2, 1)).squeeze(2)  # 75 * 1

	# Calcualte the Matrix Det
	matrix_det = torch.logdet(cov2) - torch.logdet(cov1)  # 75
	matrix_det = matrix_det.unsqueeze(1)

	KL_dis = (trace_dis + maha_product + matrix_det - mean1.size(2))/2.

	if triplet_loss:
		return KL_dis
	else:
		return torch.mean(KL_dis)


def Wass_distance_Batch(mean1, cov1, mean2, cov2, S_cov_inv):
	'''
		   mean1: 75 * 1 * 64
		   cov1:  75 * 64 * 64
		   mean2: 75 * 1 * 64
		   cov2: 75 * 64 * 64
		   S_cov_inv: 64 * 64
	'''

	mean_diff = mean1-mean2                         # 75 * 1 * 64
	Maha_dis = torch.matmul(mean_diff, S_cov_inv)   # 75 * 1 * 64
	Maha_dis = torch.matmul(Maha_dis, mean_diff.permute(0, 2, 1)).squeeze(2).squeeze(1) # 75 


	# Trace distance between cov1 and cov2
	cov1_new = torch.matmul(cov1, S_cov_inv)   # 75 * 64 * 64
	cov2_new = torch.matmul(cov2, S_cov_inv)   # 75 * 64 * 64
	cov1_1 = torch.matmul(cov1_new, cov1_new)  # 75 * 64 * 64
	cov1_2 = torch.matmul(cov1_new, cov2_new)  # 75 * 64 * 64
	cov2_2 = torch.matmul(cov2_new, cov2_new)  # 75 * 64 * 64
	Trace_dis = [(torch.trace(cov1_1[i])+torch.trace(cov2_2[i])-2*torch.trace(cov1_2[i])).unsqueeze(0) for i in range(cov1.size(0))]

	Trace_dis = torch.cat(Trace_dis, 0)
	Wass_dis = Maha_dis + Trace_dis

	return torch.mean(Wass_dis)


def UDA_distance_Batch(mean1, cov1, mean2, cov2):
	'''
	   mean1: 75 * 1 * 64
	   cov1:  75 * 64 * 64
	   mean2: 75 * 1 * 64
	   cov2: 75 * 64 * 64
	'''

	# Calculate the l1 mean and l1 covariance
	l1_norm_mean = torch.norm((mean1-mean2), p=1, dim=2)
	l1_norm_cova = torch.norm((cov1-cov2), p=1, dim=(1,2), keepdim=True).squeeze(2)

	l1_norm_mean = l1_norm_mean/mean1.size(0)
	l1_norm_cova = l1_norm_cova/(mean1.size(0)*mean1.size(0))

	UDA_dis = l1_norm_mean + l1_norm_cova

	return torch.mean(UDA_dis)

