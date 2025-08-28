import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset

import itertools
import random

from collections import OrderedDict

def get_ordered_weights(weights, start_idx=0 ,end_idx=None, order_columns=False):

    end_idx = weights.shape[0] if end_idx is None else end_idx


    synapses = weights.clone()


    # Extract that subset of rows (shape ~ [num_rows x D], e.g. 500 x D)
    submatrix = synapses[start_idx:end_idx, :]  



    # -- 1. Compute pairwise cosine similarities among those rows
    #    The result is shape [num_rows, num_rows], e.g. 500 x 500
    cosine_sim = F.cosine_similarity(
        submatrix.unsqueeze(1),   # [num_rows,   1, D]
        submatrix.unsqueeze(0),   # [   1     ,num_rows, D]
        dim=2
    )

    cosine_sim.fill_diagonal_(0)
    new_order = []
    prev_row = 0

    for _ in range(end_idx):
        new_row = np.argmax(cosine_sim[prev_row])
        new_order.append(prev_row)
        cosine_sim[:, prev_row] = -1
        prev_row = new_row

    # -- 3. Reorder rows in [start_idx:end_idx]
    synapses[start_idx:end_idx, :] = synapses[new_order, :]

    if order_columns:
        synapses[:, start_idx:end_idx] = synapses[:, new_order]

    return cosine_sim, new_order, synapses


def test_network(net, input_params, print_rate=1):
  input, input_latents = make_input(**input_params)
  with torch.no_grad():
    for day in range(input_params["num_days"]):
      if day%print_rate == 0:
        print(day)
      net(input[day])
      net.sleep()
  return input, input_latents, net


def get_Hopfield_Tsodyks(input):
  W = torch.zeros((input.shape[2], input.shape[2]))
  processed_input = input.squeeze(1) - torch.mean(input)
  for pattern in processed_input:
      W += torch.outer(pattern, pattern)
  return W


class ModernHopfieldNetwork(nn.Module):
    def __init__(self, pattern_dim, num_patterns, beta=1.0):
        """
        Modern Hopfield Network
        Args:
            pattern_dim: Dimension of the patterns
            num_patterns: Number of patterns to store
            beta: Inverse temperature parameter (sharpness of attraction basin)
        """
        super(ModernHopfieldNetwork, self).__init__()
        self.pattern_dim = pattern_dim
        self.num_patterns = num_patterns
        self.beta = beta

        # Storage for the patterns
        self.patterns = nn.Parameter(torch.zeros(num_patterns, pattern_dim), requires_grad=False)

    def store_patterns(self, new_patterns):
        """
        Store patterns into the Hopfield Network
        Args:
            new_patterns: Tensor of shape (num_patterns, pattern_dim)
        """
        if new_patterns.shape != (self.num_patterns, self.pattern_dim):
            raise ValueError(f"Expected shape ({self.num_patterns}, {self.pattern_dim}), got {new_patterns.shape}")
        self.patterns.data = new_patterns.clone()

    def forward(self, query):
        """
        Query the Hopfield network to retrieve the closest pattern.
        Args:
            query: Tensor of shape (batch_size, pattern_dim)
        Returns:
            The closest patterns (retrieved), shape (batch_size, pattern_dim)
        """
        # Compute pairwise similarities between query and stored patterns
        similarity = torch.matmul(query, self.patterns.T)  # Shape: (batch_size, num_patterns)
        # Apply softmax to retrieve patterns (energy minimization)
        attention_weights = F.softmax(self.beta * similarity, dim=1)  # Shape: (batch_size, num_patterns)
        # Weighted sum to reconstruct the pattern
        retrieved_patterns = torch.matmul(attention_weights, self.patterns)  # Shape: (batch_size, pattern_dim)
        return retrieved_patterns
    


def get_ff(N, train_loader, num_epochs, lr):
  class SelfSupervisedModel(nn.Module):
    def __init__(self, N):
        super(SelfSupervisedModel, self).__init__()
        self.linear = nn.Linear(N, N)  # Single linear layer with N neurons
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.linear(x))
    

  model = SelfSupervisedModel(N)
  criterion = nn.MSELoss()  # Mean squared error loss for reconstruction
  optimizer = optim.Adam(model.parameters(), lr=lr)
  loss_curve = []     
    
  for epoch in range(num_epochs):
      for data, labels in train_loader:
          # Zero the parameter gradients
          optimizer.zero_grad()

          # Forward pass
          outputs = model(data)
          loss = criterion(outputs, labels)

          # Backward pass and optimize
          loss.backward()
          optimizer.step()
          loss_curve.append(loss.item())  # Store the loss value

  return model, loss_curve

def get_Boltzmann(N, train_loader, num_epochs, lr):
  # Define the Boltzmann Machine model
  class BoltzmannMachine(nn.Module):
      def __init__(self, n_visible):
          super(BoltzmannMachine, self).__init__()
          self.weights = nn.Parameter(torch.randn(n_visible, n_visible) * 0.01)
          nn.init.zeros_(self.weights.diagonal())  # No self-loops

      def forward(self, v):
          activation = torch.matmul(v, self.weights)
          prob = torch.sigmoid(activation)
          return torch.bernoulli(prob)

  # Initialize the model
  model = BoltzmannMachine(N)
  optimizer = optim.SGD(model.parameters(), lr=lr)

  # Training loop
  for epoch in range(num_epochs):
      for v_batch, label in train_loader:
          batch_size = v_batch.size(0)
          # Positive phase
          pos_phase = torch.matmul(v_batch.T, v_batch) / batch_size

          # Negative phase
          gibbs_samples = model(v_batch)

          neg_phase = torch.matmul(gibbs_samples.T, gibbs_samples) / batch_size

          # Update weights
          weight_update = lr * (pos_phase - neg_phase)
          with torch.no_grad():
              model.weights += weight_update

      # Log training progress
      if (epoch + 1) % 100 == 0:
          with torch.no_grad():
              # Compute reconstruction error
              recon_error = torch.mean((v_batch - gibbs_samples) ** 2).item()
          print(f"Epoch {epoch + 1}/{num_epochs}, Reconstruction Error: {recon_error:.4f}")
  return model


def get_Hebbian(input):
  W = torch.zeros((input.shape[2], input.shape[2]))
  processed_input = input.squeeze(1)
  for pattern in processed_input:
      W += torch.outer(pattern, pattern)
  return W


def get_cos_sim_torch(x1, x2):
  return torch.dot(x1, x2)/(torch.norm(x1)*torch.norm(x2))
def get_cos_sim_np(x1, x2):
  return np.dot(x1, x2)/(np.linalg.norm(x1)*np.linalg.norm(x2))


def get_cond_matrix(latent_space, weights, eta):
  num_subs = len(latent_space.sub_index_to_neuron_index)
  sim_cond_matrix = np.zeros((num_subs, num_subs))
  th_cond_matrix = np.zeros((num_subs, num_subs))
  for conditioned_sub_index, ((conditioned_latent, conditioned_sub), conditioned_neuron_index) in enumerate(zip(latent_space.sub_index_to_latent_sub, latent_space.sub_index_to_neuron_index)):
    for condition_sub_index, ((condition_latent, condition_sub), condition_neuron_index) in  enumerate(zip(latent_space.sub_index_to_latent_sub, latent_space.sub_index_to_neuron_index)):
      if conditioned_sub != condition_sub:
        sim_cond_matrix[conditioned_sub_index][condition_sub_index] = np.mean(weights[conditioned_neuron_index][:, condition_neuron_index])
      else:
        #sim_cond_matrix[conditioned_sub_index][condition_sub_index] = np.mean(weights[conditioned_neuron_index][:, condition_neuron_index][~np.eye(condition_neuron_index.shape[0], dtype=bool)])
        sim_cond_matrix[conditioned_sub_index][condition_sub_index] = np.mean(weights[conditioned_neuron_index][:, condition_neuron_index])
      if conditioned_latent != condition_latent:
        label = [0, 0]
        label[conditioned_latent] = conditioned_sub
        label[condition_latent] = condition_sub
        try:
          th_cond_matrix[conditioned_sub_index][condition_sub_index] = latent_space.label_to_probs[tuple(label)]/(eta*latent_space.sub_index_to_marginal[condition_sub_index] + (1 - eta)*latent_space.sub_index_to_marginal[conditioned_sub_index])
        except:
          th_cond_matrix[conditioned_sub_index][condition_sub_index] = 0

      elif conditioned_sub == condition_sub:
        th_cond_matrix[conditioned_sub_index][condition_sub_index] = 1

      else:
        th_cond_matrix[conditioned_sub_index][condition_sub_index] = 0
  return sim_cond_matrix, th_cond_matrix

'''
def get_sample_from_num_swaps(x_0, num_swaps, regions):
  x = x_0.clone().detach()
  #get on and off index
  on_index = x_0.nonzero().squeeze(1)
  off_index = (x_0 ==0).nonzero().squeeze(1)
  #choose at random num_flips indices
  flip_off = on_index[torch.randperm(len(on_index))[:int(num_swaps/2)]]
  flip_on = off_index[torch.randperm(len(off_index))[:int(num_swaps/2)]]
  #flip on to off and off to on
  x[flip_off] = 0
  x[flip_on] = 1
  return x
'''

def get_sample_from_num_swaps(x_0, num_swaps, regions=None):
    if regions == None:
      x = x_0.clone().detach()
      #get on and off index
      on_index = x_0.nonzero().squeeze(1)
      off_index = (x_0 ==0).nonzero().squeeze(1)
      #choose at random num_flips indices
      flip_off = on_index[torch.randperm(len(on_index))[:int(num_swaps/2)]]
      flip_on = off_index[torch.randperm(len(off_index))[:int(num_swaps/2)]]
      #flip on to off and off to on
      x[flip_off] = 0
      x[flip_on] = 1
      return x
    
    else:

      x = x_0.clone().detach()

      total_size = sum([len(region) for region in regions])  # Total size of all regions

      for region in regions:
          # Get the size of the region
          region_size = len(region)

          # Determine the number of swaps for this region
          num_swaps_region = round(num_swaps * region_size / total_size)

          # Get on and off indices for this region
          on_index = region[x_0[region] == 1]
          off_index = region[x_0[region] == 0]

          # Choose at random num_swaps_region indices
          flip_off = on_index[torch.randperm(len(on_index))[:num_swaps_region // 2]]
          flip_on = off_index[torch.randperm(len(off_index))[:num_swaps_region // 2]]

          # Flip on to off and off to on within this region
          x[flip_off] = 0
          x[flip_on] = 1

      return x


class LatentSpace():
    def __init__(self, num, total_sizes, act_sizes, dims, prob_list, random_neurons=False):
      self.num_latents = num
      self.dims = dims
      self.total_sizes = total_sizes
      self.total_size = sum(total_sizes)
      self.act_sizes = act_sizes
      self.random_neurons = random_neurons
      self.latent_patterns = [[self.get_sub_latent(latent, sub_dim) for sub_dim in range(self.dims[latent])] for latent in range(self.num_latents)]
      self.sub_index_to_neuron_index = [self.latent_patterns[latent][sub_dim].nonzero().squeeze(1).detach().numpy() + sum(self.total_sizes[:latent]) for latent in range(self.num_latents) for sub_dim in range(self.dims[latent])]
      self.sub_index_to_latent_sub = [(latent, sub_dim) for latent in range(self.num_latents) for sub_dim in range(self.dims[latent])]
      self.index_to_label = list(itertools.product(*[[i for i in range(dim)] for dim in self.dims]))
      self.label_to_index = {label: index for index, label in enumerate(self.index_to_label)}
      self.label_to_neurons = OrderedDict({label: self.get_neurons_from_label(label) for label in self.index_to_label})
      self.label_to_probs = OrderedDict({label: prob for label, prob in zip(self.index_to_label, prob_list)})
      self.sub_index_to_marginal = [self.get_marginal(latent, sub) for (latent, sub) in self.sub_index_to_latent_sub]

    def get_sub_latent(self, latent, sub_dim):
      if self.random_neurons:
        sub_latent = torch.zeros((self.total_sizes[latent]))
        sub_latent[torch.randperm(self.total_sizes[latent])[:self.act_sizes[latent]]] = 1
      else:
        sub_latent = torch.zeros((self.total_sizes[latent]))
        sub_latent[sub_dim*self.act_sizes[latent]:(sub_dim+1)*self.act_sizes[latent]] = 1
      return sub_latent

    def get_neurons_from_label(self, label):
        return torch.cat(tuple(self.latent_patterns[latent][index] for latent, index in enumerate(label)))

    def get_marginal(self, latent, sub):
      marginal = 0
      for label, prob in self.label_to_probs.items():
        if label[latent] == sub:
          marginal += prob
      return marginal

    def sample(self):
      return random.choices(list(self.label_to_neurons.items()), weights=list(self.label_to_probs.values()))[0]
    

class SatelliteSpace():
  def __init__(self, neoctx_size):
    self.neoctx_size = neoctx_size
    self.total_size = self.neoctx_size
    self.names = ["gavan", "volar", "motar", "nivex", "sorex", "denor", "sopra", "funda", "bacta", "gondo", "malar", "benin", "colar", "nodon", "praxa"]
    self.alpha_names = ["gavan", "volar", "motar", "nivex", "sorex"]
    self.beta_names = ["denor", "sopra", "funda", "bacta", "gondo"]
    self.gamma_names = ["malar", "benin", "colar", "nodon", "praxa"]
    self.classes = ["alpha", "beta", "gamma"]
    self.num_classes = len(self.classes)
    self.num_names = len(self.names)
    self.alpha_prototype = []
    self.name_to_class = {
        name: class_
        for class_names, class_ in zip([self.alpha_names, self.beta_names, self.gamma_names], self.classes)
        for name in class_names
    }
    self.num_attributes = 5
    self.name_to_attributes = {
        name: self.get_attributes_from_name(name)
        for name in self.names
    }
    self.name_to_neural = {
        name: self.get_neural_from_name(name)
        for name in self.names
    }
    self.element_to_neural = self.get_elements_to_neural()

  def get_elements_to_neural(self):
      elements_to_neural = {}
      attributes = ["name", "class", "attribute1", "attribute2", "attribute3", "attribute4", "attribute5"]
      elements_per_attribute = [15, 3, 3, 6, 6, 6, 6]
      n_act = 25
      starting = 0
      for attribute_index, attribute in enumerate(attributes):
        neurals = []
        for element in range(elements_per_attribute[attribute_index]):
          neural = torch.zeros(self.neoctx_size)
          neural[starting:starting+n_act] = 1
          starting += n_act
          neurals.append(neural)
        elements_to_neural[attribute] = neurals
      return elements_to_neural
  def get_attributes_from_name(self, name):
    #make a tensor num of attributes (5) times num_classes*2 (6)
    attributes_0 = torch.zeros((1, self.num_classes))
    class_index = self.classes.index(self.name_to_class[name])
    attributes_0[0, class_index] = 1
    attributes_1 = torch.zeros((self.num_attributes - 1, self.num_classes*2))
    #get name index (0 to 14)
    name_index = self.names.index(name)
    #get name class index --> this is within class index, tells you which of the elements
    name_class_index = name_index%(self.num_attributes)
    for att_num in range(self.num_attributes - 1):
      if name_class_index - 1 == att_num:
        attributes_1[att_num, class_index+self.num_classes] = 1
      else:
        attributes_1[att_num, class_index] = 1


    #attributes_0 = torch.zeros_like(attributes_0)
    attributes = torch.cat([attributes_0.flatten(), attributes_1.flatten()])

    return attributes.flatten()

  def get_neural_from_name(self, name):
    neural = torch.zeros((self.neoctx_size))
    neural_name = torch.nn.functional.one_hot(torch.tensor(self.names.index(name)), num_classes=self.num_names)
    neural_class = torch.nn.functional.one_hot(torch.tensor(self.classes.index(self.name_to_class[name])), num_classes=self.num_classes)
    neural_attributes = self.name_to_attributes[name]
    min_neural = torch.cat((neural_name, neural_class, neural_attributes))
    min_neural_size = len(min_neural)
    min_neural_repeats = self.neoctx_size//min_neural_size
    if min_neural_repeats == 0:
      print("neoctx is not big enough for minimal neural representation, should be at least {}".format(min_neural_size))
    neural[:min_neural_size*min_neural_repeats] = min_neural.repeat_interleave(min_neural_repeats)
    return neural

  def sample(self):
    name = random.choice(self.names)
    label = self.names.index(name)
    return label, self.name_to_neural[name]
    
def get_cos_sim_torch(x1, x2):
  return torch.dot(x1, x2)/(torch.norm(x1)*torch.norm(x2))
def get_cos_sim_np(x1, x2):
  return np.dot(x1, x2)/(np.linalg.norm(x1)*np.linalg.norm(x2))


def make_input(num_days, day_length, mean_duration, fixed_duration, num_swaps, latent_space, regions=None, satellite=False):
  def get_partial_circle():
    delta_theta = 45
    image = np.zeros((40, 25), dtype=np.uint8)

    # Define the circle parameters
    x_center, y_center = 15, 20  # Center coordinates
    radius = 7  # Radius
    thickness = 2 # Thickness

    # Define the start and end angles for the partial circle
    theta = np.random.uniform(0, 360)
    start_angle = theta - delta_theta/2
    end_angle = theta + delta_theta/2


    # Draw the partial circle on the image
    color = 255  # White color for grayscale
    cv2.ellipse(image, (x_center, y_center), (radius, radius), 0, -start_angle, -end_angle, color, thickness)

    return int(theta), torch.tensor(image.flatten())

  def get_sample_from_err_rate(pattern):
    p_size = len(pattern)
    error_index = torch.randperm(p_size)[:int(p_size*error_rate)]
    pattern[error_index] = pattern[error_index]^ torch.ones_like(pattern[error_index])
    return pattern

  #initialize input tensor
  #input = torch.zeros((num_days, day_length, sum(latent_space.total_sizes)))
  input = torch.zeros((num_days, day_length, latent_space.total_size))
  input_episodes = torch.zeros((num_days, day_length), dtype=torch.int32)
  input_latents = torch.zeros((num_days, day_length, latent_space.num_latents), dtype=torch.int32)

  #create input from noisy patterns
  for day in range(num_days):
    day_timestep = 0
    while day_timestep < day_length:
      #pattern_duration = pattern_duration if (day_timestep + pattern_duration <= day_length) else day_length - day_timestep
      pattern_duration = mean_duration if fixed_duration else int(torch.poisson(mean_duration*torch.ones(1))[0])
      if satellite:
        latent_index, pattern = latent_space.sample()
        input_latents[day, day_timestep:day_timestep+pattern_duration] = latent_index
      else:
        label, pattern = latent_space.sample()
        input_episodes[day, day_timestep:day_timestep+pattern_duration] = latent_space.label_to_index[label]
        input_latents[day, day_timestep:day_timestep+pattern_duration, :] = torch.tensor(label, dtype=torch.int32)

      #label, pattern =  get_partial_circle()
      #input_latents[day, day_timestep:day_timestep+pattern_duration] = label
      for pattern_timestep in range(pattern_duration):
        input[day, (day_timestep + pattern_timestep):(day_timestep+pattern_timestep + 1)] = get_sample_from_num_swaps(pattern, num_swaps, regions)
      day_timestep += pattern_duration

  return input, input_episodes, input_latents


def selectivity_indices(connectivity_matrix, presynaptic_patterns, threshold=0.):
    """
    Computes the selectivity of postsynaptic neurons to presynaptic activity patterns.

    Parameters:
        connectivity_matrix (np.ndarray): Matrix of shape (n_post, n_pre) representing
                                          the connectivity strengths.
        presynaptic_patterns (np.ndarray): Matrix of shape (m, n_pre) representing
                                           m different presynaptic activity patterns.
        threshold (float): A minimum selectivity value to consider a postsynaptic neuron selective.
                           Below this value, the neuron is considered non-selective and gets NaN.

    Returns:
        np.ndarray: A vector of length n_post with the index of the most selective
                    presynaptic pattern or NaN if below the threshold.
    """
    # Normalize the presynaptic patterns and the connectivity matrix
    normalized_patterns = presynaptic_patterns / np.linalg.norm(presynaptic_patterns, axis=1, keepdims=True)
    normalized_connectivity = connectivity_matrix / np.linalg.norm(connectivity_matrix, axis=1, keepdims=True)

    # Calculate the dot product (cosine similarity) between connectivity and patterns
    dot_product_matrix = np.dot(normalized_connectivity, normalized_patterns.T)

    # Find the index of the maximum selectivity value for each postsynaptic neuron
    max_indices = np.argmax(dot_product_matrix, axis=1)
    max_values = np.max(dot_product_matrix, axis=1)
    # Apply threshold to determine if a neuron is selective
    max_indices[max_values < threshold] = -1
    result = np.where(max_indices == -1, np.nan, max_indices)

    return result


def get_distribution_num_overlaps(K, N, num_samples):
  overlaps = []
  for sample in range(num_samples):
    draw_1 = torch.randperm(N)[:K]
    draw_2 = torch.randperm(N)[:K]
    # Convert tensors to sets
    set_a = set(draw_1.tolist())
    set_b = set(draw_2.tolist())


    # Find intersection and count common values
    common_values = set_a.intersection(set_b)
    overlaps.append(len(common_values))
  return overlaps




def get_selectivity_accuracy(network, input_latents, input_episodes, input_params, latent_specs):
    X_ctx = torch.stack(network.activity_recordings["ctx"], dim=0)[-100*input_params["day_length"]:]
    X_mtl = torch.stack(network.activity_recordings["mtl"], dim=0)[-100*input_params["day_length"]:]
    X_mtl_dense = torch.stack(network.activity_recordings["mtl_dense"], dim=0)[-100*input_params["day_length"]:]
    X_mtl_sparse = torch.stack(network.activity_recordings["mtl_sparse"], dim=0)[-100*input_params["day_length"]:]

    X_latent_A = F.one_hot(input_latents[-100:, :, 0].long(), num_classes=latent_specs["dims"][0])
    X_latent_B = F.one_hot(input_latents[-100:, :, 1].long(), num_classes=latent_specs["dims"][1])
    X_latent_AB = torch.cat((X_latent_A, X_latent_B), axis=2)
    X_latent_episode = F.one_hot(input_episodes[-100:].long(), num_classes=np.prod(latent_specs["dims"]))


    # Flatten time
    days, day_length, num_latents = X_latent_AB.shape
    _, num_neurons_ctx = X_ctx.shape
    _, num_neurons_mtl = X_mtl.shape
    _, num_neurons_mtl_dense = X_mtl_dense.shape
    _, num_neurons_mtl_sparse = X_mtl_sparse.shape

    latents_flat = X_latent_AB.reshape(-1, num_latents).float()    # shape: (days * day_length, num_latents)
    neurons_ctx_flat = X_ctx.reshape(-1, num_neurons_ctx).float()    # shape: (days * day_length, num_neurons)
    neurons_mtl_flat = X_mtl.reshape(-1, num_neurons_mtl).float()    # shape: (days * day_length, num_neurons)
    neurons_mtl_dense_flat = X_mtl_dense.reshape(-1, num_neurons_mtl_dense).float()    # shape: (days * day_length, num_neurons)
    neurons_mtl_sparse_flat = X_mtl_sparse.reshape(-1, num_neurons_mtl_sparse).float()    # shape: (days * day_length, num_neurons)

    # Normalize (zero mean, unit variance)
    latents_norm = (latents_flat - latents_flat.mean(dim=0)) / latents_flat.std(dim=0)
    neurons_ctx_norm = (neurons_ctx_flat - neurons_ctx_flat.mean(dim=0)) / neurons_ctx_flat.std(dim=0)
    neurons_mtl_norm = (neurons_mtl_flat - neurons_mtl_flat.mean(dim=0)) / neurons_mtl_flat.std(dim=0)
    neurons_mtl_dense_norm = (neurons_mtl_dense_flat - neurons_mtl_dense_flat.mean(dim=0)) / neurons_mtl_dense_flat.std(dim=0)
    neurons_mtl_sparse_norm = (neurons_mtl_sparse_flat - neurons_mtl_sparse_flat.mean(dim=0)) / neurons_mtl_sparse_flat.std(dim=0)



    # Compute correlation (selectivity): (num_neurons, num_latents)
    print( neurons_ctx_norm.T.shape, latents_norm.shape)
    selectivity_ctx = neurons_ctx_norm.T @ latents_norm / latents_norm.shape[0]
    selectivity_mtl = neurons_mtl_norm.T @ latents_norm / latents_norm.shape[0]
    selectivity_mtl_dense = neurons_mtl_dense_norm.T @ latents_norm / latents_norm.shape[0]
    selectivity_mtl_sparse = neurons_mtl_sparse_norm.T @ latents_norm / latents_norm.shape[0]



    threshold = 0.7
    num_neurons_ctx, num_latents = selectivity_ctx.shape

    # Identify the latent with the highest selectivity per neuron
    max_selectivity, max_latent_idx = selectivity_ctx.max(dim=1)

    # Assign neurons to latents if selectivity >= threshold, else assign -1 (non-selective)
    assigned_latents = torch.where(max_selectivity >= threshold, max_latent_idx, -1)

    # Generate reordered indices
    ordered_indices = []

    # Iterate over each latent to collect selective neurons
    for latent in range(num_latents):
        latent_neurons = (assigned_latents == latent).nonzero(as_tuple=True)[0]
        ordered_indices.append(latent_neurons)

    # Add non-selective neurons at the end
    non_selective_neurons = (assigned_latents == -1).nonzero(as_tuple=True)[0]
    ordered_indices.append(non_selective_neurons)

    # Concatenate indices into a single ordering
    ordered_indices_ctx = torch.cat(ordered_indices)



    threshold = 0.7
    num_neurons, num_latents = selectivity_mtl.shape

    # Identify the latent with the highest selectivity per neuron
    max_selectivity, max_latent_idx = selectivity_mtl.max(dim=1)

    # Assign neurons to latents if selectivity >= threshold, else assign -1 (non-selective)
    assigned_latents = torch.where(max_selectivity >= threshold, max_latent_idx, -1)

    # Generate reordered indices
    ordered_indices = []

    # Iterate over each latent to collect selective neurons
    for latent in range(num_latents):
        latent_neurons = (assigned_latents == latent).nonzero(as_tuple=True)[0]
        ordered_indices.append(latent_neurons)

    # Add non-selective neurons at the end
    non_selective_neurons = (assigned_latents == -1).nonzero(as_tuple=True)[0]
    ordered_indices.append(non_selective_neurons)

    # Concatenate indices into a single ordering
    ordered_indices_mtl = torch.cat(ordered_indices)


    threshold = 0.7
    num_neurons_mtl_dense, num_latents = selectivity_mtl_dense.shape

    # Identify the latent with the highest selectivity per neuron
    max_selectivity, max_latent_idx = selectivity_mtl_dense.max(dim=1)

    # Assign neurons to latents if selectivity >= threshold, else assign -1 (non-selective)
    assigned_latents = torch.where(max_selectivity >= threshold, max_latent_idx, -1)

    # Generate reordered indices
    ordered_indices = []

    # Iterate over each latent to collect selective neurons
    for latent in range(num_latents):
        latent_neurons = (assigned_latents == latent).nonzero(as_tuple=True)[0]
        ordered_indices.append(latent_neurons)

    # Add non-selective neurons at the end
    non_selective_neurons = (assigned_latents == -1).nonzero(as_tuple=True)[0]
    ordered_indices.append(non_selective_neurons)

    # Concatenate indices into a single ordering
    ordered_indices_mtl_dense = torch.cat(ordered_indices)

    threshold = 0.7
    num_neurons_mtl_sparse, num_latents = selectivity_mtl_sparse.shape

    # Identify the latent with the highest selectivity per neuron
    max_selectivity, max_latent_idx = selectivity_mtl_sparse.max(dim=1)

    # Assign neurons to latents if selectivity >= threshold, else assign -1 (non-selective)
    assigned_latents = torch.where(max_selectivity >= threshold, max_latent_idx, -1)

    # Generate reordered indices
    ordered_indices = []

    # Iterate over each latent to collect selective neurons
    for latent in range(num_latents):
        latent_neurons = (assigned_latents == latent).nonzero(as_tuple=True)[0]
        ordered_indices.append(latent_neurons)

    # Add non-selective neurons at the end
    non_selective_neurons = (assigned_latents == -1).nonzero(as_tuple=True)[0]
    ordered_indices.append(non_selective_neurons)

    # Concatenate indices into a single ordering
    ordered_indices_mtl_sparse = torch.cat(ordered_indices)



    # -- Accuracy of mtl_sparse as classifier of A and B --

    # For each neuron, what class (latent) is it assigned to?
    # assigned_latents is shape (num_neurons_mtl_sparse,)
    # For samples, neurons_mtl_sparse_flat is (num_samples, num_neurons_mtl_sparse)

    # Only consider selective neurons
    selective_mask = assigned_latents != -1
    assigned_classes = assigned_latents[selective_mask]
    if selective_mask.sum() == 0:
        print("No selective neurons in mtl_sparse, cannot compute accuracy.")
        accuracy_A, accuracy_B = 0, 0
    else:
        # Filter activity to only selective neurons
        mtl_sparse_activity_selective = neurons_mtl_sparse_flat[:, selective_mask]  # (samples, n_selective_neurons)

        # For each sample, find the most active neuron among selective neurons
        winner_idx = mtl_sparse_activity_selective.argmax(dim=1)  # shape: (samples,)
        # Assign the predicted class for each sample (the class the winning neuron codes for)
        predicted_class = assigned_classes[winner_idx]  # shape: (samples,)

        # True classes: For AB (joint), it's the latent index with 1 in latents_flat for each sample
        # For one-hot, argmax gives the class index.
        true_class = latents_flat.argmax(dim=1)  # shape: (samples,)

        # Only keep predictions where the neuron is selective (should always be the case here)
        accuracy = (predicted_class == true_class).float().mean().item()
        print(f"mtl_sparse classification accuracy (A+B joint): {accuracy:.4f}")

        # -- If you want accuracy for A and B separately --
        # A: latents_flat[:, :latent_specs["dims"][0]]
        # B: latents_flat[:, latent_specs["dims"][0]:]

        # Assign each neuron to A or B, based on its assigned latent index
        # A_latents = range(latent_specs["dims"][0])
        # B_latents = range(latent_specs["dims"][0], latent_specs["dims"][0] + latent_specs["dims"][1])
        A_range = torch.arange(latent_specs["dims"][0], device=assigned_latents.device)
        B_range = torch.arange(latent_specs["dims"][0], latent_specs["dims"][0]+latent_specs["dims"][1], device=assigned_latents.device)

        # Only use neurons selective for A (if any)
        A_neuron_mask = (assigned_classes < latent_specs["dims"][0])
        if A_neuron_mask.any():
            winner_A_idx = mtl_sparse_activity_selective[:, A_neuron_mask].argmax(dim=1)
            predicted_class_A = assigned_classes[A_neuron_mask][winner_A_idx]
            true_class_A = latents_flat[:, :latent_specs["dims"][0]].argmax(dim=1)
            accuracy_A = (predicted_class_A == true_class_A).float().mean().item()
            print(f"mtl_sparse classification accuracy (A only): {accuracy_A:.4f}")
        else:
            accuracy_A = 0
            print("No selective neurons for A in mtl_sparse.")

        # Only use neurons selective for B (if any)
        B_neuron_mask = (assigned_classes >= latent_specs["dims"][0])
        if B_neuron_mask.any():
            winner_B_idx = mtl_sparse_activity_selective[:, B_neuron_mask].argmax(dim=1)
            predicted_class_B = assigned_classes[B_neuron_mask][winner_B_idx] - latent_specs["dims"][0]
            true_class_B = latents_flat[:, latent_specs["dims"][0]:].argmax(dim=1)
            accuracy_B = (predicted_class_B == true_class_B).float().mean().item()
            print(f"mtl_sparse classification accuracy (B only): {accuracy_B:.4f}")
        else:
           accuracy_B = 0
            #print("No selective neurons for B in mtl_sparse.")

    return ordered_indices_ctx, ordered_indices_mtl, ordered_indices_mtl_dense, ordered_indices_mtl_sparse, selectivity_ctx, selectivity_mtl, selectivity_mtl_dense, selectivity_mtl_sparse, accuracy_A, accuracy_B






def get_selectivity(network, input_latents, input_episodes, input_params, latent_specs):

    X_ctx = torch.stack(network.activity_recordings["ctx"], dim=0)[network.awake_indices][-100*input_params["day_length"]:]
    X_mtl = torch.stack(network.activity_recordings["mtl"], dim=0)[network.awake_indices][-100*input_params["day_length"]:]
    X_mtl_dense = torch.stack(network.activity_recordings["mtl_dense"], dim=0)[network.awake_indices][-100*input_params["day_length"]:]
    X_mtl_sparse = torch.stack(network.activity_recordings["mtl_sparse"], dim=0)[network.awake_indices][-100*input_params["day_length"]:]

    X_latent_A = F.one_hot(input_latents[-100:, :, 0].long(), num_classes=latent_specs["dims"][0])
    X_latent_B = F.one_hot(input_latents[-100:, :, 1].long(), num_classes=latent_specs["dims"][1])
    X_latent_AB = torch.cat((X_latent_A, X_latent_B), axis=2)
    X_latent_episode = F.one_hot(input_episodes[-100:].long(), num_classes=np.prod(latent_specs["dims"]))


    # Flatten time
    days, day_length, num_latents = X_latent_AB.shape
    _, num_neurons_ctx = X_ctx.shape
    _, num_neurons_mtl = X_mtl.shape
    _, num_neurons_mtl_dense = X_mtl_dense.shape
    _, num_neurons_mtl_sparse = X_mtl_sparse.shape

    latents_flat = X_latent_AB.reshape(-1, num_latents).float()    # shape: (days * day_length, num_latents)
    neurons_ctx_flat = X_ctx.reshape(-1, num_neurons_ctx).float()    # shape: (days * day_length, num_neurons)
    neurons_mtl_flat = X_mtl.reshape(-1, num_neurons_mtl).float()    # shape: (days * day_length, num_neurons)
    neurons_mtl_dense_flat = X_mtl_dense.reshape(-1, num_neurons_mtl_dense).float()    # shape: (days * day_length, num_neurons)
    neurons_mtl_sparse_flat = X_mtl_sparse.reshape(-1, num_neurons_mtl_sparse).float()    # shape: (days * day_length, num_neurons)

    # Normalize (zero mean, unit variance)
    latents_norm = (latents_flat - latents_flat.mean(dim=0)) / latents_flat.std(dim=0)
    neurons_ctx_norm = (neurons_ctx_flat - neurons_ctx_flat.mean(dim=0)) / neurons_ctx_flat.std(dim=0)
    neurons_mtl_norm = (neurons_mtl_flat - neurons_mtl_flat.mean(dim=0)) / neurons_mtl_flat.std(dim=0)
    neurons_mtl_dense_norm = (neurons_mtl_dense_flat - neurons_mtl_dense_flat.mean(dim=0)) / neurons_mtl_dense_flat.std(dim=0)
    neurons_mtl_sparse_norm = (neurons_mtl_sparse_flat - neurons_mtl_sparse_flat.mean(dim=0)) / neurons_mtl_sparse_flat.std(dim=0)

    # Compute correlation (selectivity): (num_neurons, num_latents)
    selectivity_ctx = neurons_ctx_norm.T @ latents_norm / latents_norm.shape[0]
    selectivity_mtl = neurons_mtl_norm.T @ latents_norm / latents_norm.shape[0]
    selectivity_mtl_dense = neurons_mtl_dense_norm.T @ latents_norm / latents_norm.shape[0]
    selectivity_mtl_sparse = neurons_mtl_sparse_norm.T @ latents_norm / latents_norm.shape[0]



    threshold = 0.7
    num_neurons_ctx, num_latents = selectivity_ctx.shape

    # Identify the latent with the highest selectivity per neuron
    max_selectivity, max_latent_idx = selectivity_ctx.max(dim=1)

    # Assign neurons to latents if selectivity >= threshold, else assign -1 (non-selective)
    assigned_latents = torch.where(max_selectivity >= threshold, max_latent_idx, -1)

    # Generate reordered indices
    ordered_indices = []

    # Iterate over each latent to collect selective neurons
    for latent in range(num_latents):
        latent_neurons = (assigned_latents == latent).nonzero(as_tuple=True)[0]
        ordered_indices.append(latent_neurons)

    # Add non-selective neurons at the end
    non_selective_neurons = (assigned_latents == -1).nonzero(as_tuple=True)[0]
    ordered_indices.append(non_selective_neurons)

    # Concatenate indices into a single ordering
    ordered_indices_ctx = torch.cat(ordered_indices)



    threshold = 0.7
    num_neurons, num_latents = selectivity_mtl.shape

    # Identify the latent with the highest selectivity per neuron
    max_selectivity, max_latent_idx = selectivity_mtl.max(dim=1)

    # Assign neurons to latents if selectivity >= threshold, else assign -1 (non-selective)
    assigned_latents = torch.where(max_selectivity >= threshold, max_latent_idx, -1)

    # Generate reordered indices
    ordered_indices = []

    # Iterate over each latent to collect selective neurons
    for latent in range(num_latents):
        latent_neurons = (assigned_latents == latent).nonzero(as_tuple=True)[0]
        ordered_indices.append(latent_neurons)

    # Add non-selective neurons at the end
    non_selective_neurons = (assigned_latents == -1).nonzero(as_tuple=True)[0]
    ordered_indices.append(non_selective_neurons)

    # Concatenate indices into a single ordering
    ordered_indices_mtl = torch.cat(ordered_indices)


    threshold = 0.7
    num_neurons_mtl_dense, num_latents = selectivity_mtl_dense.shape

    # Identify the latent with the highest selectivity per neuron
    max_selectivity, max_latent_idx = selectivity_mtl_dense.max(dim=1)

    # Assign neurons to latents if selectivity >= threshold, else assign -1 (non-selective)
    assigned_latents = torch.where(max_selectivity >= threshold, max_latent_idx, -1)

    # Generate reordered indices
    ordered_indices = []

    # Iterate over each latent to collect selective neurons
    for latent in range(num_latents):
        latent_neurons = (assigned_latents == latent).nonzero(as_tuple=True)[0]
        ordered_indices.append(latent_neurons)

    # Add non-selective neurons at the end
    non_selective_neurons = (assigned_latents == -1).nonzero(as_tuple=True)[0]
    ordered_indices.append(non_selective_neurons)

    # Concatenate indices into a single ordering
    ordered_indices_mtl_dense = torch.cat(ordered_indices)

    threshold = 0.7
    num_neurons_mtl_sparse, num_latents = selectivity_mtl_sparse.shape

    # Identify the latent with the highest selectivity per neuron
    max_selectivity, max_latent_idx = selectivity_mtl_sparse.max(dim=1)

    # Assign neurons to latents if selectivity >= threshold, else assign -1 (non-selective)
    assigned_latents = torch.where(max_selectivity >= threshold, max_latent_idx, -1)

    # Generate reordered indices
    ordered_indices = []

    # Iterate over each latent to collect selective neurons
    for latent in range(num_latents):
        latent_neurons = (assigned_latents == latent).nonzero(as_tuple=True)[0]
        ordered_indices.append(latent_neurons)

    # Add non-selective neurons at the end
    non_selective_neurons = (assigned_latents == -1).nonzero(as_tuple=True)[0]
    ordered_indices.append(non_selective_neurons)

    # Concatenate indices into a single ordering
    ordered_indices_mtl_sparse = torch.cat(ordered_indices)



    # -- Accuracy of mtl_sparse as classifier of A and B --

    # For each neuron, what class (latent) is it assigned to?
    # assigned_latents is shape (num_neurons_mtl_sparse,)
    # For samples, neurons_mtl_sparse_flat is (num_samples, num_neurons_mtl_sparse)

    # Only consider selective neurons
    selective_mask = assigned_latents != -1
    assigned_classes = assigned_latents[selective_mask]
    if selective_mask.sum() == 0:
        return ordered_indices_ctx, ordered_indices_mtl, ordered_indices_mtl_dense, ordered_indices_mtl_sparse, selectivity_ctx, selectivity_mtl, selectivity_mtl_dense, selectivity_mtl_sparse, None, None, None
        print("No selective neurons in mtl_sparse, cannot compute accuracy.")

    else:
        # Filter activity to only selective neurons
        mtl_sparse_activity_selective = neurons_mtl_sparse_flat[:, selective_mask]  # (samples, n_selective_neurons)

        # For each sample, find the most active neuron among selective neurons
        winner_idx = mtl_sparse_activity_selective.argmax(dim=1)  # shape: (samples,)
        # Assign the predicted class for each sample (the class the winning neuron codes for)
        predicted_class = assigned_classes[winner_idx]  # shape: (samples,)

        # True classes: For AB (joint), it's the latent index with 1 in latents_flat for each sample
        # For one-hot, argmax gives the class index.
        true_class = latents_flat.argmax(dim=1)  # shape: (samples,)

        # Only keep predictions where the neuron is selective (should always be the case here)
        accuracy = (predicted_class == true_class).float().mean().item()
        print(f"mtl_sparse classification accuracy (A+B joint): {accuracy:.4f}")

        # -- If you want accuracy for A and B separately --
        # A: latents_flat[:, :latent_specs["dims"][0]]
        # B: latents_flat[:, latent_specs["dims"][0]:]

        # Assign each neuron to A or B, based on its assigned latent index
        # A_latents = range(latent_specs["dims"][0])
        # B_latents = range(latent_specs["dims"][0], latent_specs["dims"][0] + latent_specs["dims"][1])
        A_range = torch.arange(latent_specs["dims"][0], device=assigned_latents.device)
        B_range = torch.arange(latent_specs["dims"][0], latent_specs["dims"][0]+latent_specs["dims"][1], device=assigned_latents.device)

        # Only use neurons selective for A (if any)
        A_neuron_mask = (assigned_classes < latent_specs["dims"][0])
        if A_neuron_mask.any():
            winner_A_idx = mtl_sparse_activity_selective[:, A_neuron_mask].argmax(dim=1)
            predicted_class_A = assigned_classes[A_neuron_mask][winner_A_idx]
            true_class_A = latents_flat[:, :latent_specs["dims"][0]].argmax(dim=1)
            accuracy_A = (predicted_class_A == true_class_A).float().mean().item()
            print(f"mtl_sparse classification accuracy (A only): {accuracy_A:.4f}")
        else:
            print("No selective neurons for A in mtl_sparse.")
            accuracy_A = 0

        # Only use neurons selective for B (if any)
        B_neuron_mask = (assigned_classes >= latent_specs["dims"][0])
        if B_neuron_mask.any():
            winner_B_idx = mtl_sparse_activity_selective[:, B_neuron_mask].argmax(dim=1)
            predicted_class_B = assigned_classes[B_neuron_mask][winner_B_idx] - latent_specs["dims"][0]
            true_class_B = latents_flat[:, latent_specs["dims"][0]:].argmax(dim=1)
            accuracy_B = (predicted_class_B == true_class_B).float().mean().item()
            print(f"mtl_sparse classification accuracy (B only): {accuracy_B:.4f}")
        else:
            print("No selective neurons for B in mtl_sparse.")
            accuracy_B = 0


    # --- Compute episode selectivity for ctx ---
    # Flatten time (match ctx shape)
    timesteps, num_neurons_ctx = X_ctx.shape
    # episodes: (days, day_length) --> flatten to (days * day_length,)
    episodes_flat = input_episodes[-100:].reshape(-1)  # shape (days * day_length,)
    # one-hot episodes: (days * day_length, n_episodes)
    n_episodes = np.prod(latent_specs["dims"])
    episodes_onehot = F.one_hot(episodes_flat.long(), num_classes=n_episodes).float()  # (days*day_length, n_episodes)

    # activity: (days, day_length, n_ctx) --> (days*day_length, n_ctx)
    neurons_ctx_flat = X_ctx.reshape(-1, num_neurons_ctx).float()
    neurons_ctx_norm = (neurons_ctx_flat - neurons_ctx_flat.mean(dim=0)) / (neurons_ctx_flat.std(dim=0) + 1e-6)
    episodes_onehot_norm = (episodes_onehot - episodes_onehot.mean(dim=0)) / (episodes_onehot.std(dim=0) + 1e-6)

    # (n_ctx, n_episodes): correlation of each neuron with each episode
    episode_selectivity_ctx = neurons_ctx_norm.T @ episodes_onehot_norm / episodes_onehot_norm.shape[0]

    return ordered_indices_ctx, ordered_indices_mtl, ordered_indices_mtl_dense, ordered_indices_mtl_sparse, selectivity_ctx, selectivity_mtl, selectivity_mtl_dense, selectivity_mtl_sparse, accuracy_A, accuracy_B, episode_selectivity_ctx





