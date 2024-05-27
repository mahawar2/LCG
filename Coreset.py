import os
import math
import torch
from thop import profile

class ParamertricFilter(torch.nn.Module):

	def __init__(self, k, epsilon, dim=1, classes=2):
		super(ParamertricFilter, self).__init__()
		self.k = k
		self.dim = dim
		self.classes = classes
		self.lmbda = torch.tensor(float('inf'))
		self.nu = torch.tensor(0.)
		self.mu = torch.tensor(0.)
		self.index = 0 #torch.zeros(0)
		self.meanX = torch.zeros(1,dim)
		self.sensitivity = torch.tensor(0.)
		self.epsilon = epsilon
		self.completeCoresetIndexs = []
		self.elements = [torch.empty(0, dim)] * classes
		self.meanPerClass = torch.zeros(classes)
		self.sdPerClass = torch.zeros(classes)

	def forward(self, data, labels):
		data = torch.reshape(data, (-1,self.dim))
		labels = torch.reshape(labels, (-1,self.classes))
		valid = 0
		self.lmbda = torch.min(self.lmbda, torch.min(torch.abs(data)))
		self.nu = torch.max(self.nu, torch.max(torch.abs(data)))
		self.mu = (self.lmbda / self.nu)
		self.meanX = torch.div((torch.mul(self.meanX, self.index) + data), (self.index + 1))
		MDdist = (torch.exp(self.nu)/2) * torch.pow((self.meanX - data),2).sum()
		self.sensitivity += MDdist
		if self.index == 0:
			self.p = torch.tensor(1)
		else:
			self.p = torch.min(torch.tensor(1), (((2*self.epsilon*MDdist)/(self.sensitivity*self.mu)) + ((8*self.epsilon)/((self.index)*self.mu))))
		self.index = self.index + 1
		if torch.rand(1) < self.p:
			for i in range(labels.shape[1]):
				if labels[0,i] == 1:
					if self.elements[i].shape[0] == 0:
						class_mean = torch.zeros((1,self.dim))
					else:
						class_mean = torch.mean(self.elements[i], dim=0)
					distance = torch.sqrt(torch.sum(torch.pow(torch.subtract(class_mean, data), 2)))
					if distance >= (self.meanPerClass[i] + self.k * self.sdPerClass[i]):
						valid = 1
			if valid == 1:
				for i in range(labels.shape[1]):
					if labels[0,i] == 1:
						self.elements[i] = torch.cat((self.elements[i],data), dim=0)					
						distance_mat = []
						for j in range(self.elements[i].shape[0]):
							distance_mat += [torch.sqrt(torch.sum(torch.pow(torch.subtract(self.elements[i][j], class_mean), 2)))]
						self.meanPerClass[i] = sum(distance_mat)/len(distance_mat)
						self.sdPerClass[i] = sum([((x - self.meanPerClass[i]) ** 2) for x in distance_mat]) / len(distance_mat) ** 0.5
				self.completeCoresetIndexs += [self.index-1]
				return self.completeCoresetIndexs
			else:
				return 0
				
macs, params = profile(model, inputs=(torch.randn(1,32),torch.bernoulli(torch.randn(1,15).uniform_(0,1)), ))
print(macs)

