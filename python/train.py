import numpy as np
from keras.datasets import mnist
import torch
from model import Net
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


def process_data():
	(train_X, train_y), (test_X, test_y) = mnist.load_data()
	X = np.zeros((len(train_X), len(train_X[0])**2))
	X_test = np.zeros((len(test_X), len(test_X[0])**2))

	for i in range(len(X)):
		X[i] = train_X[i].flatten()

	for i in range(len(X_test)):
		X_test[i] = test_X[i].flatten()

	return torch.from_numpy(X/256), torch.from_numpy(train_y), \
			torch.from_numpy(X_test/256), torch.from_numpy(test_y)


def train(net, train_X, train_y):
	criterion = nn.MSELoss()
	optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
	running_loss = 0.0
	loss_tracker = []
	for i, X in enumerate(train_X):
		label = train_y[i]
		one_hot = torch.zeros(10)
		one_hot[label.item()] = 1
		optimizer.zero_grad()
		outputs = net(X)
		loss = criterion(outputs, one_hot)
		loss.backward()
		optimizer.step()
		
		running_loss += loss.item()

		if not i % 100:
			loss_tracker.append(running_loss)
			print(i, running_loss)
			running_loss = 0

	return net, loss_tracker


def test(net, test_X, test_y):
	correct_count = 0
	tot_count = 0
	for i, X in enumerate(test_X):
		label = test_y[i].item()
		pred = torch.argmax(net(X)).item()
		correct = label == pred
		correct_count += correct
		tot_count += 1

	print('Accuracy:', correct_count/tot_count)


if __name__ == '__main__':
	train_X, train_y, test_X, test_y = process_data()
	net = Net()
	net, loss_tracker = train(net, train_X, train_y)
	test(net, test_X, test_y)
	plt.plot(loss_tracker)
	plt.show()

	