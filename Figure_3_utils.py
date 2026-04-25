import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from scipy.stats import norm
from scipy.stats import mannwhitneyu

import copy
from copy import deepcopy
import random

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

from src.model import SSCNetwork
from src.utils.general import make_input, LatentSpace, get_ordered_indices
from network_parameters import network_parameters
  

# Create a simple linear classifier model
class LinearClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        #return F.softmax(self.linear(x), dim=1)
        return self.linear(x)
    

def dual_task_cross_entropy(output, targets):
    """
    output: Tensor of shape (batch_size, 10)
    target_A: Tensor of shape (batch_size,) with values in [0, 4]
    target_B: Tensor of shape (batch_size,) with values in [0, 4]
    """
    logits_A = output[:, :5]   # First 5 neurons
    logits_B = output[:, 5:]   # Last 5 neurons

    loss_A = F.cross_entropy(logits_A, targets[:, 0])
    loss_B = F.cross_entropy(logits_B, targets[:, 1])

    return loss_A + loss_B

def train_model(state_dicts, train_loader, X_test, Y_test, num_trials, num_epochs):

  trials_accuracy = torch.zeros((num_trials, num_epochs, 2))
  trained_state_dicts = []


  for trial in range(num_trials):
    input_size = 100
    num_classes = 10
    model = LinearClassifier(input_size, num_classes)
    model.load_state_dict(state_dicts[trial])

    optimizer = optim.SGD(model.parameters(), lr=10)

    eval_acc = []
    for epoch in range(num_epochs):
        #if epoch%50 == 0:
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = dual_task_cross_entropy(outputs, labels.long())
            loss.backward()
            optimizer.step()

        # Evaluate on the test set
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            predicted = torch.max(test_outputs[:, :5], 1)[1], torch.max(test_outputs[:, 5:], 1)[1]                         
            accuracy_0 = accuracy_score(Y_test[:, 0].numpy(), predicted[0].numpy())
            accuracy_1 = accuracy_score(Y_test[:, 1].numpy(), predicted[1].numpy())
            eval_acc.append((accuracy_0, accuracy_1))
    trials_accuracy[trial] = torch.tensor(eval_acc)
    trained_state_dicts.append(deepcopy(model.state_dict()))
  return trials_accuracy, trained_state_dicts


def test_model(state_dicts, X_test, Y_test):
  input_size = 100
  num_classes = 10
  trials_accuracy = torch.zeros((len(state_dicts), 2))
  for trial, state_dict in enumerate(state_dicts):
    model = LinearClassifier(input_size, num_classes)
    model.load_state_dict(state_dict)
    # Evaluate on the test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        predicted = torch.max(test_outputs[:, :5], 1)[1], torch.max(test_outputs[:, 5:], 1)[1]                         
        accuracy_0 = accuracy_score(Y_test[:, 0].numpy(), predicted[0].numpy())
        accuracy_1 = accuracy_score(Y_test[:, 1].numpy(), predicted[1].numpy())
        trials_accuracy[trial] = torch.tensor((accuracy_0, accuracy_1))
    return trials_accuracy
  

def test_network(net, input_params, sleep=False, print_rate=1):
  input, input_episodes, input_latents = make_input(**input_params)
  with torch.no_grad():
    for day in range(input_params["num_days"]):
      if day%print_rate == 0:
        print(day)
      net(input[day], debug=False)
      if sleep:
        net.sleep()
  return input, input_episodes, input_latents, net
  

def test_network_permuted(net, input_params, sleep=False, print_rate=1):
  input, input_episodes, input_latents = make_input(**input_params)
  permutation = torch.randperm(net.sen_size)
  with torch.no_grad():
    for day in range(input_params["num_days"]):
      if day%print_rate == 0:
        print(day)
      net(input[day][:, permutation], debug=False)
      if sleep:
        net.sleep()
  return input, input_episodes, input_latents, net




def get_one_shot_accuracy(net_0, recording_parameters, input_params, latent_specs, num_days, permuted=False):

    seed_everything(0)

    net = copy.deepcopy(net_0)

    net.init_recordings(recording_parameters)
    net.frozen = True
    net.activity_recordings_rate = 1
    net.connectivity_recordings_rate = np.inf

    input_params["num_days"] = num_days
    input_params["latent_space"] = LatentSpace(**latent_specs)

    if permuted:
        input, input_episodes, input_latents, network = test_network_permuted(net, input_params, sleep=False, print_rate=50)
    else:
        input, input_episodes, input_latents, network = test_network(net, input_params, sleep=False, print_rate=50)


    X_mtl_sensory = torch.tensor(np.array(net.activity_recordings["mtl_sensory"][-input_params["num_days"]*input_params["day_length"]:]))
    X_mtl_semantic =  torch.tensor(np.array(net.activity_recordings["mtl_semantic"][-input_params["num_days"]*input_params["day_length"]:]))

    X_latent_A = torch.repeat_interleave(F.one_hot(input_latents[:, :, 0].long(), num_classes=latent_specs["dims"][0]), dim=2, repeats=5).flatten(start_dim=0, end_dim=1)
    X_latent_B = torch.repeat_interleave(F.one_hot(input_latents[:, :, 1].long(), num_classes=latent_specs["dims"][1]), dim=2, repeats=5).flatten(start_dim=0, end_dim=1)
    X_latent_AB = torch.cat((X_latent_A, X_latent_B), axis=1)
    X_latent = torch.zeros((X_latent_AB.shape[0], 100))
    X_latent[:X_latent_AB.shape[0], :X_latent_AB.shape[1]] = X_latent_AB


    # Split the data into training and testing sets
    X_mtl_sensory_train, X_mtl_sensory_test, Y_train, Y_test = train_test_split(X_mtl_sensory, input_latents.flatten(start_dim=0, end_dim=1), test_size=0.995, random_state=42)
    X_mtl_semantic_train, X_mtl_semantic_test, Y_train, Y_test = train_test_split(X_mtl_semantic, input_latents.flatten(start_dim=0, end_dim=1), test_size=0.995, random_state=42)
    X_latent_train, X_latent_test, Y_train, Y_test = train_test_split(X_latent, input_latents.flatten(start_dim=0, end_dim=1), test_size=0.995, random_state=42)

    # Create DataLoader for training set
    train_dataset_mtl_sensory = TensorDataset(X_mtl_sensory_train, Y_train)
    train_loader_mtl_sensory = DataLoader(train_dataset_mtl_sensory, batch_size=32, shuffle=True)

    train_dataset_mtl_semantic = TensorDataset(X_mtl_semantic_train, Y_train)
    train_loader_mtl_semantic = DataLoader(train_dataset_mtl_semantic, batch_size=32, shuffle=True)


    train_dataset_latent = TensorDataset(X_latent_train, Y_train)
    train_loader_latent = DataLoader(train_dataset_latent, batch_size=32, shuffle=True)


    num_trials = 50
    num_epochs = 1

    input_size = 100
    num_classes = 10
    state_dicts = [deepcopy(LinearClassifier(input_size, num_classes).state_dict()) for trial in range(num_trials)]


    acc_mtl_sensory, models_sensory = train_model(state_dicts, train_loader_mtl_sensory,  X_mtl_sensory_test, Y_test, num_trials, num_epochs)
    acc_mtl_semantic, models_semantic = train_model(state_dicts, train_loader_mtl_semantic,  X_mtl_semantic_test, Y_test, num_trials, num_epochs)
    acc_latent, models_latent = train_model(state_dicts, train_loader_latent, X_latent_test, Y_test, num_trials, num_epochs)


    return acc_mtl_sensory, acc_mtl_semantic, acc_latent