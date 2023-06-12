import torch
import pdb
import math
import numpy as np
import copy
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
from .deepfool import DeepFool
from .carlini import CarliniWagnerL2



# FGSM attack for two inputs
def fgsm_attack(input_var1, input_var2, target, model, criterion, epsilon):
	
	# pdb.set_trace()
	# Copy the input1 
	perturbed_input1 = input_var1.clone()
	perturbed_input1.requires_grad = True


	# output and loss
	model.zero_grad() 
	output = model(perturbed_input1, input_var2)
	loss = criterion(output, target)


	input1_grad = torch.autograd.grad(loss, perturbed_input1, only_inputs=True)[0]

	# Collect the element-wise sign of the data gradient
	sign_data_grad = input1_grad.sign()

	# Create the perturbed image by adjusting each pixel of the input image
	perturbed_input1 = perturbed_input1 + epsilon*sign_data_grad

	# Adding clipping to maintain [0,1] range
	perturbed_input1 = torch.clamp(perturbed_input1, 0.0, 1.0)

	return perturbed_input1.detach()



# FGSM attack for one input
def fgsm_attack2(input_var, target, model, criterion, epsilon):
	
	# pdb.set_trace()
	# Copy the input1 
	perturbed_input_var = input_var.clone()
	perturbed_input_var.requires_grad = True
	model.zero_grad() 
	

	# output and loss
	output = model(perturbed_input_var)
	loss = criterion(output, target)

	input_grad = torch.autograd.grad(loss, perturbed_input_var, only_inputs=True)[0]

	# Collect the element-wise sign of the data gradient
	sign_data_grad = input_grad.sign()

	# Create the perturbed image by adjusting each pixel of the input image
	perturbed_input_var = perturbed_input_var + epsilon*sign_data_grad

	# Adding clipping to maintain [0,1] range
	perturbed_input_var = torch.clamp(perturbed_input_var, 0, 1)

	return perturbed_input_var.detach()



# FGSM attack for few-shot method CAN
def fgsm_attack_can(input_var1, input_var2, target, target_sup, model, criterion, epsilon):
	
	# pdb.set_trace()
	# Copy the input1 
	perturbed_input1 = input_var1.clone()
	perturbed_input1.requires_grad = True


	# output and loss
	model.zero_grad() 
	output = model(perturbed_input1, input_var2, target, target_sup)
	loss = criterion(output, target)


	input1_grad = torch.autograd.grad(loss, perturbed_input1, only_inputs=True)[0]

	# Collect the element-wise sign of the data gradient
	sign_data_grad = input1_grad.sign()

	# Create the perturbed image by adjusting each pixel of the input image
	perturbed_input1 = perturbed_input1 + epsilon*sign_data_grad

	# Adding clipping to maintain [0,1] range
	perturbed_input1 = torch.clamp(perturbed_input1, 0.0, 1.0)

	return perturbed_input1.detach()



# PGD attack code
def pgd_attack(opt, input_var1, input_var2, target, model, criterion, epsilon, iteration=False):
	
	# Copy the input1 
	perturbed_input1 = input_var1.clone()
	# pdb.set_trace()
	
	# Add random noise
	random_noise = torch.FloatTensor(*input_var1.shape).uniform_(-float(epsilon.cpu()), float(epsilon.cpu())).cuda()
	perturbed_input1 = perturbed_input1.data + random_noise 
	perturbed_input1.requires_grad = True


	alpha = 1.0/255.0
	# alpha = opt.step_size
 
	if iteration:
		iter_num = opt.num_steps
	else:
		iter_num = torch.min(torch.tensor([epsilon*255+4, epsilon*255*1.25])) # min(epsilon+4, epsilon*1.25)
		iter_num = math.ceil(iter_num)

	# print(iter_num)
	for _ in range(iter_num):

		# compute output & loss
		model.zero_grad() 
		output = model(perturbed_input1, input_var2)
		loss = criterion(output, target)
		input1_grad = torch.autograd.grad(loss, perturbed_input1, only_inputs=True)[0]
		
		# Create the perturbed image by adjusting each pixel of the input image
		perturbed_input1 = perturbed_input1 + alpha*input1_grad.sign()

		# Clipping to maintain [X-epsilon, X+epsilon] range
		perturbed_input1 = linfball_proj(input_var1, epsilon.cuda(), perturbed_input1)

		# Adding clipping to maintain [-1,1] range
		perturbed_input1 = torch.clamp(perturbed_input1, 0.0, 1.0)
	
	return perturbed_input1.detach()



# Clamp all elements in input into the range [min, max] and return a resulting tensor
def tensor_clamp(t, min, max, in_place=True):
	# print(min, max)
	if not in_place:
		res = t.clone()
	else:
		res = t
	idx = res.data < min
	res.data[idx] = min[idx]
	idx = res.data > max
	res.data[idx] = max[idx]

	return res


def linfball_proj(center, radius, t, in_place=True):
	return tensor_clamp(t, min=center - radius, max=center + radius, in_place=in_place)



# Deepfool attack code
# refer to https://github.com/tobylyf/adv-attack/blob/master/run_deepfool.py
def deepfool_attack(input_var1, input_var2, model):
	

	# pdb.set_trace()
	deepfool = DeepFool(nb_candidate=5, max_iter=30)
	perturbed_input1 = deepfool.attack(model, input_var1, input_var2)
	
	return perturbed_input1.detach()



# C&W attack code
# refer to https://github.com/tobylyf/adv-attack/blob/master/run_cw.py
def cw_attack(input_var1, input_var2, model, target):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	
	# pdb.set_trace()
	CW_attacker = CarliniWagnerL2((0.0, 1.0), 5, learning_rate=0.001, search_steps=4, max_iterations=10, initial_const=10, quantize=False, device=device)
	perturbed_input1 = CW_attacker.attack(model, input_var1, input_var2, target, targeted=False)
	
	return perturbed_input1.detach()