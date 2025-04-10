import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from agents.routing_brain import RoutingBrain
import dfjsp_routing as routing
from networks.ddqn_network import DQNNetwork

class DRL_routing(RoutingBrain):
    def __init__(self, env, job_creator, wc_list, address_seed, **kwargs):
        # initialize the environment and the workcenter to be controlled
        self.env = env
        self.job_creator = job_creator
        self.wc_list = wc_list
        for wc in self.wc_list:
            wc.job_routing = self.action_DRL_machine
        # retrive the data of chosen workcenter
        self.m_per_wc = len(self.wc_list[0].m_list)
        self.input_size = self.m_per_wc*3 + 3
        # action space, consists of all selectable machines
        self.output_size = self.m_per_wc
        # specify the path to store the model
        self.path = sys.path[0]
        # specify the ANN and state function
        if 'validated' in kwargs and kwargs['validated']:

            self.action_NN = DQNNetwork(self.input_size, self.output_size)

            self.build_state = self.state_deeper
            self.action_NN.load_state_dict(torch.load(address_seed))
            self.action_NN.eval()  # must have this if you're loading a model, unnecessray for loading state_dict

    def action_DRL_machine(self, job_idx, routing_data, job_pt, ttd, job_slack, wc_idx, *args):
        s_t = self.build_state(routing_data, job_pt, ttd, job_slack, wc_idx)
        value = self.action_NN.forward(s_t.reshape(1,1,self.input_size), wc_idx)
        a_t = torch.argmax(value)
        return a_t
