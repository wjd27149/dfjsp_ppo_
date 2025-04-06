
import torch.nn as nn
import torch.nn.functional as F
import torch

#1 = ENABLED, 0=DISABLED
DEBUG_MODE = 0

class ppo_network(nn.Module):
    def __init__(self, input_size, output_size):
        super(ppo_network, self).__init__()
        self.lr = 0.001
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
    
    def get_log_prob(self, obs, actions):
        """离散动作版本"""
        logits = self.forward(obs)  # 网络输出各动作的logits
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions.squeeze(-1))  # 确保actions是正确形状
    

import torch.nn as nn
import torch.nn.functional as F

class StateProcessor(nn.Module):
    """共享的状态预处理模块"""
    def __init__(self, input_size):
        if DEBUG_MODE == 1:
            print("===============Into StateProcessor Init()================")
        super().__init__()
        self.no_size = 3
        self.pt_size = 6
        self.remaining_pt_size = 11
        self.ttd_slack_size = 16
        
        # 归一化模块（与原始结构相同）
        self.normlayer_no = nn.Sequential(
            nn.InstanceNorm1d(3), nn.Flatten())
        self.normlayer_pt = nn.Sequential(
            nn.InstanceNorm1d(3), nn.Flatten())
        self.normlayer_remaining_pt = nn.Sequential(
            nn.InstanceNorm1d(5), nn.Flatten())
        self.normlayer_ttd_slack = nn.Sequential(
            nn.InstanceNorm1d(5), nn.Flatten())
        if DEBUG_MODE == 1:
            print("===============StateProcessor Init() completed================")
        
    def forward(self, x):
        #print("输入 x 的设备:", x.device) 在GPU上
        # 切片处理（与原始结构相同）
        x_no = x[:, :, :self.no_size]
        #print("x_no 的设备:", x_no.device) 在GPU上
        x_pt = x[:, :, self.no_size:self.pt_size]
        x_remaining_pt = x[:, :, self.pt_size:self.remaining_pt_size]
        x_ttd_slack = x[:, :, self.remaining_pt_size:self.ttd_slack_size]
        x_rest = x[:, :, self.ttd_slack_size:].squeeze(1)
        
        # 归一化处理
        x_normed_no = self.normlayer_no(x_no)
        x_normed_pt = self.normlayer_pt(x_pt)
        x_normed_remaining_pt = self.normlayer_remaining_pt(x_remaining_pt)
        x_normed_ttd_slack = self.normlayer_ttd_slack(x_ttd_slack)

        # 拼接特征
        return torch.cat([
            x_normed_no, x_normed_pt, 
            x_normed_remaining_pt, x_normed_ttd_slack, 
            x_rest
        ], dim=1)

class ActorNetwork(nn.Module):
    """策略网络 Actor"""
    def __init__(self, input_size, output_size):
        super().__init__()
        self.state_processor = StateProcessor(input_size)
        
        # 独立策略头
        self.policy_net = nn.Sequential(
            nn.Linear(input_size, 48),
            nn.Tanh(),
            nn.Linear(48, 36),
            nn.Tanh(),
            nn.Linear(36, 36),
            nn.Tanh(),
            nn.Linear(36, output_size)  # 输出各动作的logits
        )
        
        # 初始化最后一层（重要！）
        nn.init.orthogonal_(self.policy_net[-1].weight, gain=0.01)
        nn.init.constant_(self.policy_net[-1].bias, 0)

    def forward(self, x):
        if DEBUG_MODE == 1:
            print("===============Into Actor forward()================")
            print("actor network forward x device: ", x.device)
        state_features = self.state_processor(x)
        if DEBUG_MODE == 1:
            print("state_features device: ", state_features.device)
        if DEBUG_MODE == 1:
            print("===============Going to leave Actor forward()================")
        # ret = self.policy_net(state_features)
        # if DEBUG_MODE == 1:
        #    print("ret device: ", ret.device) #ret也在GPU上，正常
        return self.policy_net(state_features)
    
    def get_log_prob(self, obs, actions):
        """计算动作对数概率"""
        if DEBUG_MODE == 1:
            print(f"actornet->get_log_prob->obs device: {obs.device}")
            print(f"actornet->get_log_prob->actions device: {actions.device}")
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions.squeeze(-1))

class CriticNetwork(nn.Module):
    """价值网络 Critic """
    def __init__(self, input_size,output_size):
        super().__init__()
        self.state_processor = StateProcessor(input_size)
        
        # 更深的价值网络
        self.value_net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Tanh(),
            nn.Linear(64, 48),
            nn.Tanh(),
            nn.Linear(48, 36),
            nn.Tanh(),
            nn.Linear(36, output_size)  # 输出单个状态价值
        )
        
        # 初始化最后一层
        nn.init.orthogonal_(self.value_net[-1].weight, gain=1.0)
        nn.init.constant_(self.value_net[-1].bias, 0)

    def forward(self, x):
        state_features = self.state_processor(x)
        return self.value_net(state_features)