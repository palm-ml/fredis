import numpy as np
import torch
import math
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder

def binarize_class(y):  
    label = y.reshape(len(y), -1)
    enc = OneHotEncoder(categories='auto') 
    enc.fit(label)
    label = enc.transform(label).toarray().astype(np.float32)     
    label = torch.from_numpy(label)
    return label

def accuracy_check(loader, model, device):
    with torch.no_grad():
        total, num_samples = 0, 0
        for images, labels in loader:
            labels, images = labels.to(device), images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            # _, y = torch.max(labels.data, 1)
            # print(predicted, labels)
            total += (predicted == labels).sum().item()
            num_samples += labels.size(0)

    return 100*(total/num_samples)

# def accuracy_check0(loader, model, device):
#     with torch.no_grad():
#         total, num_samples = 0, 0
#         truew = 0.0
#         for images, labels in loader:
#             labels, images = labels.to(device), images.to(device)
#             outputs = model(images)
#             outsoft = F.softmax(outputs, dim=1)
#             w, predicted = torch.max(outsoft.data, 1)
#             _, y = torch.max(labels.data, 1)
#             total += (predicted == y).sum().item()
#             num_samples += labels.size(0)
            
#             truew += w[predicted == y].sum().item()

#     return 100*(total/num_samples), (truew/total)

def getnewList(newlist):
	d = []
	for element in newlist:
		if not isinstance(element,list):
			d.append(element)
		else:
			d.extend(getnewList(element))
	
	return d
	
def generate_unreliable_candidate_labels(train_labels, partial_rate, noisy_rate):
    K = (torch.max(train_labels) - torch.min(train_labels) + 1).item()
    n = train_labels.shape[0]

    # Categorical Distribution
    Categorical_Matrix = torch.ones(n, K) * (noisy_rate / (K-1))
    Categorical_Matrix[torch.arange(n), train_labels] = 1 - noisy_rate
    noisy_label_sampler = torch.distributions.Categorical(probs=Categorical_Matrix)
    noisy_labels = noisy_label_sampler.sample()

    # Bernoulli Distribution
    Bernoulli_Matrix = torch.ones(n, K) * partial_rate
    Bernoulli_Matrix[torch.arange(n), train_labels] = 0
    incorrect_labels = torch.zeros(n, K)
    for i in range(n):
        incorrect_labels_sampler = torch.distributions.Bernoulli(probs=Bernoulli_Matrix[i])
        incorrect_labels_row = incorrect_labels_sampler.sample()
        while incorrect_labels_row.sum() < 1:
            incorrect_labels_row = incorrect_labels_sampler.sample()
        incorrect_labels[i] = incorrect_labels_row.clone().detach()
    # check
    partial_labels = incorrect_labels.clone().detach()
    partial_labels[torch.arange(n), noisy_labels] = 1.0
    return partial_labels


    
