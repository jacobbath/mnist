import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_default_dtype(torch.float64)

class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()
		self.l1 = nn.Linear(784, 256)
		self.l2 = nn.Linear(256, 64)
		self.l3 = nn.Linear(64, 10)

	def forward(self, x):
		x = F.relu(self.l1(x))
		x = F.relu(self.l2(x))
		x = F.softmax(self.l3(x), dim=0)
		return x

