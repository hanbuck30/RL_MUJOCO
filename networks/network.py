import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import *
from networks.base import Network

class Actor(Network):
    def __init__(self, layer_num, input_dim, output_dim, hidden_dim, activation_function = torch.tanh,last_activation = None, trainable_std = False):
        super(Actor, self).__init__(layer_num, input_dim, output_dim, hidden_dim, activation_function ,last_activation)
        self.trainable_std = trainable_std
        if self.trainable_std == True:
            self.logstd = nn.Parameter(torch.zeros(1, output_dim))
    def forward(self, x):
        mu = self._forward(x)
        if self.trainable_std == True:
            std = torch.exp(self.logstd)
        else:
            logstd = torch.zeros_like(mu)
            std = torch.exp(logstd)
        return mu,std

class Critic(Network):
    def __init__(self, layer_num, input_dim, output_dim, hidden_dim, activation_function, last_activation = None):
        super(Critic, self).__init__(layer_num, input_dim, output_dim, hidden_dim, activation_function ,last_activation)
        
    def forward(self, *x):
        x = torch.cat(x,-1)
        return self._forward(x)
    


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
from abc import ABCMeta

class HebbianActor(nn.Module, metaclass=ABCMeta):
    def __init__(self, layer_num, input_dim, output_dim, hidden_dim, activation_function=torch.tanh, last_activation=None, trainable_std=False, eta=0.0001):
        super(HebbianActor, self).__init__()
        self.trainable_std = trainable_std
        self.activation = activation_function
        self.last_activation = last_activation
        self.eta = eta
        self.clip = 0.01
        layers_unit = [input_dim] + [hidden_dim] * (layer_num - 1)
        layers = [nn.Linear(layers_unit[idx], layers_unit[idx + 1]) for idx in range(len(layers_unit) - 1)]
        self.layers = nn.ModuleList(layers)
        self.last_layer = nn.Linear(layers_unit[-1], output_dim)
        if self.trainable_std:
            self.logstd = nn.Parameter(torch.zeros(1, output_dim))
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.last_layer(x)
        if self.last_activation is not None:
            x = self.last_activation(x)
        mu = x
        if self.trainable_std:
            std = torch.exp(self.logstd)
        else:
            std = torch.exp(torch.zeros_like(mu))
        return mu, std

    def hebbian_update(self, inputs, delta_mu):
        x = inputs
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                pre_synaptic = x
                x = self.activation(layer(x))
                post_synaptic = x
                layer.weight.data += torch.clamp(self.eta * torch.mm(pre_synaptic.T, (post_synaptic)).T, -self.clip, self.clip)
        pre_synaptic = x
        post_synaptic = self.last_layer(x)
        if self.last_activation is not None:
            post_synaptic = self.last_activation(post_synaptic)
        self.last_layer.weight.data  += torch.clamp(self.eta * torch.mm(pre_synaptic.T, delta_mu).T, -self.clip, self.clip)
        return 

class HebbianCritic(nn.Module, metaclass=ABCMeta):
    def __init__(self, layer_num, input_dim, output_dim, hidden_dim, activation_function, last_activation=None, eta=0.0001):
        super(HebbianCritic, self).__init__()
        self.activation = activation_function
        self.last_activation = last_activation
        self.eta = eta
        self.clip = 0.01
        layers_unit = [input_dim] + [hidden_dim] * (layer_num - 1)
        layers = [nn.Linear(layers_unit[idx], layers_unit[idx + 1]) for idx in range(len(layers_unit) - 1)]
        self.layers = nn.ModuleList(layers)
        self.last_layer = nn.Linear(layers_unit[-1], output_dim)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.last_layer(x)
        if self.last_activation is not None:
            x = self.last_activation(x)
        return x

    def hebbian_update(self, inputs, delta_v):
        self.hebbian_weights = [torch.zeros_like(layer.weight) for layer in self.layers if isinstance(layer, nn.Linear)]
        self.hebbian_weights.append(torch.zeros_like(self.last_layer.weight))
        x = inputs
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                pre_synaptic = x
                x = self.activation(layer(x))
                post_synaptic = x
                self.hebbian_weights[i] = self.hebbian_weights[i].to(self.device)
                self.hebbian_weights[i] += torch.clamp(self.eta * torch.mm(pre_synaptic.T, (post_synaptic)).T, self.clip, -self.clip)
                layer.weight.data += self.hebbian_weights[i]
        pre_synaptic = x
        post_synaptic = self.last_layer(x)
        if self.last_activation is not None:
            post_synaptic = self.last_activation(post_synaptic)
        self.last_layer.weight.data += torch.clamp(self.eta * torch.mm(pre_synaptic.T, (delta_v)).T, self.clip, -self.clip)
        return 