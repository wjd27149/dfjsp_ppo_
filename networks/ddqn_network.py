import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

class DQNNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 48)
        self.fc4 = nn.Linear(48, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, 16)
        self.fc7 = nn.Linear(16, output_size)
        
        self.tanh = nn.Tanh()
        self.instancenorm = nn.InstanceNorm1d(input_size)
        self.flatten = nn.Flatten()
        self.loss_func = F.smooth_l1_loss

    def forward(self, x, *args):
        x = self.instancenorm(x)
        x = self.flatten(x)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.fc3(x))
        x = self.tanh(self.fc4(x))
        x = self.tanh(self.fc5(x))
        x = self.tanh(self.fc6(x))
        return self.fc7(x)
    

class ValidatedNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.lr = 0.001
        self.no_size = 3
        self.pt_size = 6
        self.m_available_size = 9  
        self.ttd_slack_size = 11
        
        # 分通道归一化层
        self.normlayer_no = nn.Sequential(
            nn.InstanceNorm1d(3), nn.Flatten())
        self.normlayer_pt = nn.Sequential(
            nn.InstanceNorm1d(3), nn.Flatten())
        self.normlayer_m_available = nn.Sequential(
            nn.InstanceNorm1d(3), nn.Flatten())
        self.normlayer_ttd_slack = nn.Sequential(
            nn.InstanceNorm1d(2), nn.Flatten())
        
        # 共享网络层
        self.subsequent_module = nn.Sequential(
            nn.Linear(input_size, 48), nn.Tanh(),
            nn.Linear(48, 36), nn.Tanh(),
            nn.Linear(36, 36), nn.Tanh(),
            nn.Linear(36, 24), nn.Tanh(),
            nn.Linear(24, 24), nn.Tanh(),
            nn.Linear(24, 12), nn.Tanh(),
            nn.Linear(12, output_size)
        )
        
        self.loss_func = F.smooth_l1_loss
        self.optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)

    def forward(self, x, *args):
        x_no = x[:, :, :self.no_size]
        x_pt = x[:, :, self.no_size:self.pt_size]
        x_m_available = x[:, :, self.pt_size:self.m_available_size]
        x_ttd_slack = x[:, :, self.m_available_size:self.ttd_slack_size]
        x_rest = x[:, :, self.ttd_slack_size:].squeeze(1)
        
        x = torch.cat([
            self.normlayer_no(x_no),
            self.normlayer_pt(x_pt),
            self.normlayer_m_available(x_m_available),
            self.normlayer_ttd_slack(x_ttd_slack),
            x_rest
        ], dim=1)
        return self.subsequent_module(x)
    

class SA_network(nn.Module):
    def __init__(self, input_size, output_size):
        super(SA_network, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        # for slicing the data
        self.no_size = 3
        self.pt_size = 6
        self.remaining_pt_size = 11
        self.ttd_slack_size = 16
        # FCNN parameters
        layer_1 = 48
        layer_2 = 36
        layer_3 = 36
        layer_4 = 24
        layer_5 = 24
        layer_6 = 12
        # normalization modules
        self.normlayer_no = nn.Sequential(
                                nn.InstanceNorm1d(3),
                                nn.Flatten()
                                )
        self.normlayer_pt = nn.Sequential(
                                nn.InstanceNorm1d(3),
                                nn.Flatten()
                                )
        self.normlayer_remaining_pt = nn.Sequential(
                                nn.InstanceNorm1d(5),
                                nn.Flatten()
                                )
        self.normlayer_ttd_slack = nn.Sequential(
                                nn.InstanceNorm1d(5),
                                nn.Flatten()
                                )
        # shared layers of machines
        self.subsequent_module = nn.Sequential(
                                nn.Linear(self.input_size, layer_1),
                                nn.Tanh(),
                                nn.Linear(layer_1, layer_2),
                                nn.Tanh(),
                                nn.Linear(layer_2, layer_3),
                                nn.Tanh(),
                                nn.Linear(layer_3, layer_4),
                                nn.Tanh(),
                                nn.Linear(layer_4, layer_5),
                                nn.Tanh(),
                                nn.Linear(layer_5, layer_6),
                                nn.Tanh(),
                                nn.Linear(layer_6, output_size)
                                )
        # Huber loss function
        self.loss_func = F.smooth_l1_loss
        # the universal network for all scheudling agents
        self.network = nn.ModuleList([self.normlayer_no, self.normlayer_pt, self.normlayer_remaining_pt, self.normlayer_ttd_slack, self.subsequent_module])

    def forward(self, x, *args):
        #print('original',x)
        # slice the data
        x_no = x[:,:, : self.no_size]
        x_pt = x[:,:, self.no_size : self.pt_size]
        x_remaining_pt = x[:,:, self.pt_size : self.remaining_pt_size]
        x_ttd_slack = x[:,:, self.remaining_pt_size : self.ttd_slack_size]
        x_rest = x[:,:, self.ttd_slack_size :].squeeze(1)
        # normalize data in multiple channels
        x_normed_no = self.network[0](x_no)
        x_normed_pt = self.network[1](x_pt)
        x_normed_remaining_pt = self.network[2](x_remaining_pt)
        x_normed_ttd_slack = self.network[3](x_ttd_slack)
        #print('normalized',x_normed_no)
        # concatenate all data
        #print(x_normed_no, x_normed_pt, x_normed_remaining_pt, x_normed_ttd, x_normed_slack, x_rest)
        x = torch.cat([x_normed_no, x_normed_pt, x_normed_remaining_pt, x_normed_ttd_slack, x_rest], dim=1)
        #print('combined',x)
        # the last, independent part of module
        x = self.network[4](x)
        #print('output',x)
        return x