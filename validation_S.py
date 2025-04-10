import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from agents.sequcing_brain import SequencingBrain
from networks.ddqn_network import SA_network
import sequencing
import os
class DRL_sequencing(SequencingBrain):
    def __init__(self, env, machine_list, job_creator, address_seed, *args, **kwargs):
        # initialize the environment and the workcenter to be controlled
        self.env = env
        # get list of alll machines, for collecting the global data
        self.m_list = machine_list
        self.job_creator = job_creator
        # each item is a network
        self.net_dict = {}
        self.address_seed = address_seed


        self.kwargs = kwargs
        # build action NN for each target machine
        if 'validated' in kwargs and kwargs['validated']:
            self.func_list = [sequencing.SPT,sequencing.WINQ,sequencing.MS,sequencing.CR]
            self.output_size = len(self.func_list)

            self.input_size = 25
            # build a network
            self.network = SA_network(self.input_size, self.output_size)
            # retrive the parameters
            self.network.load_state_dict(torch.load(self.address_seed))
            # just in case
            self.network.eval()  # must have this if you're loading a model, unnecessray for loading state_dict
            self.build_state = self.state_multi_channel
            for m in self.m_list:
                m.job_sequencing = self.action_sqc_rule
        else:
            print("WARNING: ANN TYPE NOT SPECIFIED !!!!")

        print('--------------------------')

    def action_sqc_rule(self, local_data):
        s_t = self.build_state(local_data)
        value = self.network.forward(s_t.reshape([1,1,self.input_size]))
        a_t = torch.argmax(value)
        job_position = self.func_list[a_t](local_data)
        return job_position
