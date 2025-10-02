import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy



class SSCNetwork(nn.Module):
    def __init__(self, net_params, rec_params):

      super(SSCNetwork, self).__init__()
      self.init_network(net_params)
      self.init_recordings(rec_params)

    def forward(self, input, debug=False):
        #this resests the network quick connections at the beginning of each day
        if self.reset_dayly:
            self.mtl_mtl = torch.zeros((self.mtl_size, self.mtl_size))
            self.mtl_semantic_mtl_semantic = torch.zeros((self.mtl_semantic_size, self.mtl_semantic_size))
            self.mtl_sensory_mtl_sensory =  torch.zeros((self.mtl_sensory_size, self.mtl_sensory_size))

          
        #iterate over every timestep in the input sequence (day)
        for timestep in range(input.shape[0]):
            
            #activate sensory
            self.sen, _ = self.activation(input[timestep], 'sen')

            #sensory to mtl_sensory
            self.mtl_sensory_hat = F.linear(self.sen, self.mtl_sensory_sen)
            self.mtl_sensory, _ = self.activation(self.mtl_sensory_hat, 'mtl_sensory')

            #mtl_semantic is initialized at random
            self.mtl_semantic_hat = torch.randn(self.mtl_semantic_size)
            self.mtl_semantic, _ = self.activation(self.mtl_semantic_hat, 'mtl_semantic')

            #set full mtl
            self.mtl[:self.mtl_sensory_size] = self.mtl_sensory
            self.mtl[self.mtl_sensory_size:] = self.mtl_semantic

            #mtl to ctx
            self.ctx_hat = F.linear(self.mtl[:self.mtl_sensory_size], self.ctx_mtl[:, :self.mtl_sensory_size]) + self.ctx_b*self.ctx_IM
            self.ctx, _ = self.activation(self.ctx_hat, 'ctx')



            #if CTX is developed (after phase A), pattern complete ctx and ctx to mtl_semantic
            if self.day >= self.duration_phase_A:
               #self.ctx = self.pattern_complete('ctx', self.ctx)
               self.mtl_semantic_hat = F.linear(self.ctx, self.mtl_semantic_ctx) + self.mtl_semantic_b*self.mtl_semantic_IM
               self.mtl_semantic, _ = self.activation(self.mtl_semantic_hat, 'mtl_semantic')
               self.mtl[self.mtl_sensory_size:] = self.mtl_semantic

              #only after phase A mtl_semantic_mtl_semantic is used
               self.hebbian('mtl_semantic', 'mtl_semantic')
               self.homeostasis('mtl_semantic', 'mtl_semantic')

            #if MTL is developed (after phase B), mtl_semantic to ctx
            if self.day >= self.duration_phase_B:
               self.ctx_hat = F.linear(self.mtl, self.ctx_mtl)
               self.ctx, _ = self.activation(self.ctx_hat, 'ctx')


            #learn mtl_mtl, mtl_sensory_mtl_sensory and ctx_ctx

            self.hebbian('mtl', 'mtl')
            self.homeostasis('mtl', 'mtl')

            self.hebbian('mtl_sensory', 'mtl_sensory')
            self.homeostasis('mtl_sensory', 'mtl_sensory')

            self.hebbian('ctx', 'ctx')
            self.homeostasis('ctx', 'ctx')

            self.record()
            self.time_index += 1
            self.awake_indices.append(self.time_index)
        self.day += 1
    

    def sleep(self):

      #sleep has two phases, episodic replay and semantic replay
      
      #we start with episodic replay
      for timestep in range(self.sleep_duration_A):
        #if mtl_semantic is not developed (before duration phase B), we use mtl_sensory
        if self.day <= self.duration_phase_B:
          mtl_sensory_random = torch.randn(self.mtl_sensory_size)
          #mtl_sensory_random = torch.randn(self.mtl_sensory_size)
          self.mtl_sensory = self.pattern_complete('mtl_sensory', h_0=mtl_sensory_random, sleep=True)
          self.mtl[:self.mtl_sensory_size] = self.mtl_sensory
          self.mtl[self.mtl_sensory_size:] = 0
          self.ctx_hat = F.linear(self.mtl, self.ctx_mtl) + self.ctx_b*self.ctx_IM
          self.ctx, _ = self.activation(self.ctx_hat, 'ctx', subregion_index=0, sleep=True)

          self.hebbian('ctx', 'mtl')
          self.homeostasis('ctx', 'mtl')

        else:
          semantic_charge = torch.randint(low=1, high=self.max_semantic_charge_replay+1, size=(1,))[0]
          self.mtl = self.mtl_generate(semantic_charge)
          self.ctx_hat = F.linear(self.mtl, self.ctx_mtl) + self.ctx_b*self.ctx_IM
          self.ctx, _ = self.activation(self.ctx_hat, 'ctx', subregion_index=semantic_charge-1, sleep=True)

          self.hebbian('ctx', 'mtl')
          self.homeostasis('ctx', 'mtl')

        self.record()
        self.time_index += 1
        self.sleep_indices_A.append(self.time_index)

      if self.day >= self.duration_phase_A:

        for timestep in range(self.sleep_duration_B):

          ctx_random = torch.randn(self.ctx_size)
          self.ctx = self.pattern_complete('ctx', h_0=ctx_random, subregion_index=0, sleep=True)

          self.mtl_semantic_hat = F.linear(self.ctx, self.mtl_semantic_ctx) + self.mtl_semantic_b*self.mtl_semantic_IM
          self.mtl_semantic, _ = self.activation(self.mtl_semantic_hat, 'mtl_semantic', sleep=True)
          self.mtl[self.mtl_sensory_size:] = self.mtl_semantic

          self.hebbian('mtl_semantic', 'ctx')
          self.homeostasis('mtl_semantic', 'ctx')

          self.record()
          self.time_index += 1
          self.sleep_indices_B.append(self.time_index)

    def activation(self, x, region, x_conditioned=None, subregion_index=None, sleep=False, sparsity=None):
      #we add a small variance to the network pre-activation in case some activities are even, so these are activated randomly.
      x = x + (1e-10 + torch.max(x) - torch.min(x))/100000*torch.randn(x.shape)

      if x_conditioned is not None:
         x[x_conditioned==1] = torch.max(x) + 1
         
      x_prime = torch.zeros(x.shape)
      x_sparsity = getattr(self, region + '_sparsity') if not sleep else getattr(self, region + '_sparsity_sleep')
      x_sparsity = x_sparsity if sparsity is None else sparsity 
      x_subregions = getattr(self, region + '_subregions')

      if sleep:
        subregional_input = [x[subregion].sum() for subregion in x_subregions]
        subregion_index = torch.topk(torch.tensor(subregional_input), 1).indices.int() if subregion_index is None else subregion_index
        subregion = x_subregions[subregion_index]
        x_subregion = torch.zeros_like(subregion).float()
        top_indices = torch.topk(x[subregion], int(len(subregion)*x_sparsity[subregion_index])).indices
        x_subregion[top_indices] = 1
        x_prime[subregion]  = x_subregion

      else:
        for subregion_index, subregion in enumerate(x_subregions):
          x_subregion = torch.zeros_like(subregion).float()
          top_indices = torch.topk(x[subregion], int(len(subregion)*x_sparsity[subregion_index])).indices
          x_subregion[top_indices] = 1
          x_prime[subregion]  = x_subregion

      return x_prime, subregion_index
    

    def pattern_complete(self, region, h_0=None, h_conditioned=None, subregion_index=None, sleep=False, num_iterations=None, sparsity=None):
        num_iterations = num_iterations  if num_iterations != None else getattr(self, region + '_pattern_complete_iterations')
        h = h_0 if h_0 is not None else getattr(self, region)
        w = getattr(self, region + '_' + region)
        for iteration in range(num_iterations):
            h, subregion_index = self.activation(F.linear(h, w), region, h_conditioned, subregion_index, sleep=sleep, sparsity=sparsity)
        return h
    

    def mtl_generate(self, semantic_charge, num_iterations=None):
        num_iterations = num_iterations  if num_iterations != None else getattr(self, 'mtl_generate_pattern_complete_iterations')
        mtl_sparsity = (semantic_charge/self.max_semantic_charge_input)*self.mtl_sparsity.clone()
        h_random = torch.randn(self.mtl_size)
        h = self.pattern_complete('mtl', h_0=h_random, h_conditioned=None, num_iterations=num_iterations, sparsity=mtl_sparsity)
        return h


    def hebbian(self, post_region, pre_region, sleep=False, quick=False, inh=False):
        if self.frozen:
           pass
        else:
          w_name = post_region + '_' + pre_region
          w =  getattr(self, w_name)
          lmbda = getattr(self, w_name + '_lmbda')

          if w_name in {'ctx_mtl', 'mtl_semantic_ctx'}:
            IM = getattr(self, post_region + '_IM')
            IM_lmbda = getattr(self, 'max_post_' + w_name)/ torch.sum(getattr(self, pre_region))
            lmbda = lmbda*(1 - IM) + IM_lmbda*IM
            lmbda = lmbda[:, None]

          if w_name == 'ctx_ctx':
            IM = getattr(self, post_region + '_IM')
            lmbda = lmbda*torch.outer(1 - IM, 1 - IM)

          delta_w = torch.outer(getattr(self, post_region), getattr(self, pre_region))
          w += lmbda*delta_w

          setattr(self, w_name, w)


    def homeostasis(self, post_region, pre_region, quick=False, inh=False):
        if self.frozen:
            pass
        else:
          w_name = post_region + '_' + pre_region
          w =  getattr(self, w_name)

          max_post_connectivity = getattr(self, 'max_post_' + w_name)
          total_post_connectivity = torch.sum(w, dim=1)
          post_exceeding_mask = total_post_connectivity > max_post_connectivity

          post_scaling_factors = torch.where(
              post_exceeding_mask,
              max_post_connectivity / total_post_connectivity,
              torch.ones_like(total_post_connectivity)
          )

          w *= post_scaling_factors.unsqueeze(1)
          setattr(self, w_name, w)

          if w_name in {"ctx_mtl", "mtl_semantic_ctx"}:
            setattr(self, post_region + '_IM', (total_post_connectivity.round() < max_post_connectivity).float())

          w = getattr(self, w_name)
          max_pre_connectivity = getattr(self, 'max_pre_' + w_name)
          total_pre_connectivity = torch.sum(w, dim=0)
          pre_exceeding_mask = total_pre_connectivity > max_pre_connectivity
          pre_scaling_factors = torch.where(
                pre_exceeding_mask,
                max_pre_connectivity / total_pre_connectivity,
                torch.ones_like(total_pre_connectivity)
            )

          w *= pre_scaling_factors
          setattr(self, w_name, w)


    def init_recordings(self, rec_params):
      self.activity_recordings = {}
      for region in rec_params["regions"]:
        self.activity_recordings[region] = [getattr(self, region)]
      self.activity_recordings_rate = rec_params["rate_activity"]
      self.activity_recordings_time = []
      self.connectivity_recordings = {}
      for connection in rec_params["connections"]:
        self.connectivity_recordings[connection] = [getattr(self, connection)]
      self.connectivity_recordings_time = []
      self.connectivity_recordings_rate = rec_params["rate_connectivity"]

      self.time_index = 0
      self.awake_indices = []
      self.sleep_indices_A = []
      self.sleep_indices_B = []

    def record(self):
      if self.time_index%self.activity_recordings_rate == 0:
        for region in self.activity_recordings:
          layer_activity = getattr(self, region)
          self.activity_recordings[region].append(deepcopy(layer_activity.detach().clone()))
          self.activity_recordings_time.append(self.time_index)
      if self.time_index%self.connectivity_recordings_rate == 0:
        for connection in self.connectivity_recordings:
          connection_state = getattr(self, connection)
          self.connectivity_recordings[connection].append(deepcopy(connection_state.detach().clone()))
          self.connectivity_recordings_time.append(self.time_index)

    def init_network(self, net_params):

      #initialize network parameters
      for key, value in net_params.items():
        setattr(self, key, value)

      for region in self.regions:
         num_subregions = getattr(self, region + "_num_subregions")
         size_subregions =  getattr(self, region + "_size_subregions")
         region_size = torch.sum(size_subregions)
         setattr(self, region + "_size", region_size)
         subregions = []
         for subregion_index in range(num_subregions):
            start, end = sum(size_subregions[:subregion_index]), sum(size_subregions[:subregion_index+1])
            subregions.append(torch.arange(start, end))
         setattr(self, region + "_subregions", subregions)
         

      self.frozen = False

      #define subnetworks
      self.sen = torch.zeros((self.sen_size))
      self.mtl_sensory_hat = torch.zeros((self.mtl_sensory_size))
      self.mtl_semantic_hat = torch.zeros((self.mtl_semantic_size))
      self.mtl_semantic = torch.zeros((self.mtl_semantic_size))
      self.mtl_sensory = torch.zeros((self.mtl_sensory_size))
      self.mtl_hat = torch.zeros((self.mtl_size))
      self.mtl = torch.zeros((self.mtl_size))
      self.ctx_hat = torch.zeros((self.ctx_size))
      self.ctx = torch.zeros((self.ctx_size))

      #define connectivity
      self.mtl_sensory_mtl_sensory = torch.zeros((self.mtl_sensory_size, self.mtl_sensory_size))
      self.mtl_semantic_mtl_semantic = torch.zeros((self.mtl_semantic_size, self.mtl_semantic_size))
      self.mtl_mtl = torch.zeros((self.mtl_size, self.mtl_size))

      self.ctx_ctx = torch.zeros((self.ctx_size, self.ctx_size))

      if self.mtl_sensory_sen_projection:
        self.mtl_sensory_sen = torch.zeros((self.mtl_sensory_size, self.sen_size))
        for post_neuron in range(self.mtl_sensory_size):
          self.mtl_sensory_sen[post_neuron, torch.randperm(self.sen_size)[:self.mtl_sensory_sen_size]] = self.max_post_mtl_sensory_sen/self.mtl_sensory_sen_size
        self.mtl_sensory_sen = self.mtl_sensory_sen.clone()

      else:
        self.mtl_sensory_sen = torch.eye(self.sen_size)


      self.ctx_IM = torch.ones((self.ctx_size)) 
      self.ctx_b = torch.zeros((self.ctx_size))
      for subregion, subregion_b in zip(self.ctx_subregions, self.ctx_subregions_b):
        self.ctx_b[subregion] = subregion_b
      self.ctx_mtl_IM_lmbda = self.ctx_mtl_lmbda
      self.ctx_mtl = torch.randn((self.ctx_size, self.mtl_size))*self.ctx_mtl_std
    
      self.mtl_semantic_IM = torch.ones((self.mtl_semantic_size)) 
      self.mtl_semantic_b = torch.zeros((self.mtl_semantic_size))
      for subregion, subregion_b in zip(self.mtl_semantic_subregions, self.mtl_semantic_subregions_b):
        self.mtl_semantic_b[subregion] = subregion_b
      self.mtl_semantic_ctx = torch.randn((self.mtl_semantic_size, self.ctx_size))*self.mtl_semantic_ctx_std

      #initialize day count
      self.day = 0