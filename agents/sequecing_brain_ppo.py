import numpy as np
import sys


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import sequencing
import os

from networks.ppo_network import ppo_network,ActorNetwork, CriticNetwork
from utils.ppo_buffer import PPOTrajectoryBuffer
from utils.shop_floor import shopfloor
from utils.record_output import plot_loss, plot_tard

import time

import simpy
from torch.distributions import MultivariateNormal

#1. Init CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
OBSERVE_CUDA = 0 # 1观察内存信息

# GPU性能信息：
# sequecing_brain_ppo 在__init__中将actor网络和critic网络移动到GPU上，调用了两次to(device)
# 项目工程中例如PPOBuffer中的其他张量都尽可能在GPU上创建了

#1 = ENABLE, 0 = DISABLED
DEBUG_MODE = 0

class Sequencing_brain:
    def __init__(self, span, *args, **kwargs):

        # 2. Init Multi-Channel
        self.build_state = self.state_multi_channel      
        print("---> Multi-Channel (MC) mode ON <---")

        self.span = span
        self.input_size = 25
        # list that contains available rules, and use SPT for the first phase of warmup
        self.func_list = [sequencing.SPT,sequencing.WINQ,sequencing.MS,sequencing.CR]
        # action space, consists of all selectable rules
        self.output_size = len(self.func_list)

        '''
        specify new address seed for storing the trained parameters
        '''
        self.address_seed = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ppo_models')
        # initialize initial replay memory, a dictionary that contains empty lists of replay memory for machines
        self.rep_memo = []
        # some training-related parameters
        self.minibatch_size = 4
        self.buffer_size = 512
        self.buffer = PPOTrajectoryBuffer(self.buffer_size, self.input_size)  # Initialize the experience replay buffer

        # Initialize actor and critic networks
        self.actor = ActorNetwork(self.input_size, self.output_size).to(device)                                                   # ALG STEP 1
        self.critic = CriticNetwork(self.input_size, 1).to(device)

        # Initialize optimizers for actor and critic
        # actor, critic网络已经移动到GPU上后，才可以建立optim
        self.lr = 0.0005
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr)

        # Initialize the covariance matrix used to query the actor for actions
        self.cov_var = torch.full(size=(self.output_size,), fill_value=0.5, device=device) # 从to_device 形式变更为直接在GPU上创建
        self.cov_mat = torch.diag(self.cov_var) #to device 是不必要的，将会在GPU上创建，可能是因为跟随cov_var

        self.n_updates_per_iteration = 5                # Number of times to update actor/critic per iteration
        self.clip_ratio = 0.2                     # Clipping ratio for PPO
        #self.gamma = 0.95                      # Discount factor for future rewards
        self.gamma = 0.99                      # Discount factor for future rewards, long-term task
        self.gae_lambda = 0.95                  # Lambda for GAE
        self.save_freq = 10                             # How often we save in number of iterations
        self.n_trajectories = 5

        self.tard = []
        self.actor_losses = []
        self.critic_losses = []
        if DEBUG_MODE == 1:
            print("===========BrainPPO Init Done==============")

    def reset(self, job_creator, m_list, env):
        if DEBUG_MODE == 1:
            print("===============Into reset()=================")
        # initialize the environment and the workcenter to be controlled
        self.env = env
        self.job_creator = job_creator
        self.m_list = m_list
        self.m_no = len(self.m_list)

        self.job_creator.build_sqc_experience_repository(self.m_list)

        for m in self.m_list:
            m.sequencing_learning_event.succeed()
            m.job_sequencing = self.action_DRL
            m.reward_function = m.get_reward13
            m.build_state = self.state_multi_channel
        if DEBUG_MODE == 1:
            print("===============reset() complted===============")
            
    def collect_trajectories(self, n_trajectories):
        """收集新轨迹并更新经验池"""
        if DEBUG_MODE == 1:
            print("===============Into collect_trajectories()================")
        for _ in range(n_trajectories):
            # create the shop floor instance
            m = 6
            wc = 3
            length_list = [2, 2, 2]  
            env = simpy.Environment()
            spf = shopfloor(env, self.span, m, wc, length_list)

            self.reset(spf.job_creator, spf.m_list, env=env)

            env.run()
            if OBSERVE_CUDA == 1:
                print(f"Allocated Memory: {torch.cuda.memory_allocated() / 1024 / 1024} MBs") #观察显存占用情况
                print(f"Cached Memory: {torch.cuda.memory_reserved() / 1024 /1024} MBs")
            # Collect the trajectory data from the job creator
            self.buffer.finalize_trajectory(spf.job_creator.rep_memo_ppo)
            output_time, cumulative_tard, tard_mean, tard_max, tard_rate = spf.job_creator.tardiness_output()
            self.tard.append(cumulative_tard[-1])
        if DEBUG_MODE == 1:
            print("===============collect_trajectories() completed================")

    def compute_returns(self, rewards, next_obs):
        """计算n-step回报"""
        if DEBUG_MODE == 1:
            print("===============Into compute_returns()================")
        returns = torch.zeros_like(rewards)
        R = 0
        for t in reversed(range(len(rewards))):
            R = rewards[t] + self.gamma * R
            returns[t] = R
        if DEBUG_MODE == 1:
            print("===============compute_returns() completed================")
        return returns

    def update(self):
        if DEBUG_MODE == 1:
            print("===============Into update()================")
        """使用全经验池数据进行mini-batch更新"""
        if len(self.buffer) < self.minibatch_size:
            return
        
        # 获取所有经验数据
        batch = self.buffer.sample_batch(self.minibatch_size)
        obs = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_obs = batch['next_states']
        old_log_probs = batch['log_probs']

        # 检查张量所处设备
        if DEBUG_MODE == 1:
            print(f"sequencing_brain->update->obs device: {obs.device}")
            print(f"sequencing_brain->update->actions device: {actions.device}")
            print(f"sequencing_brain->update->rewards device: {rewards.device}")
            print(f"sequencing_brain->update->next_obs device: {next_obs.device}")
            print(f"sequencing_brain->update->old_log_probs device: {old_log_probs.device}")
        
        # 计算回报（不需要梯度）
        with torch.no_grad():
            returns = self.compute_returns(rewards, next_obs)
            returns = returns.squeeze(-1)  # 去掉最后一个维度
        # 多轮更新
        for _ in range(self.n_updates_per_iteration):
            # --- 1. 先更新Critic ---
            # 计算当前critic的values（需要梯度）
            values = self.critic(obs).squeeze()
            
            # Critic损失
            critic_loss = F.mse_loss(values, returns)
            
            # Critic反向传播
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()
            
            # --- 2. 再更新Actor ---
            # 重新计算values（不需要梯度）
            with torch.no_grad():
                values = self.critic(obs).squeeze()
                advantages = returns - values
            
            # 标准化优势
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # 计算新策略的对数概率
            new_log_probs = self.actor.get_log_prob(obs, actions)
            ratios = torch.exp(new_log_probs - old_log_probs)
            
            # PPO损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.clip_ratio, 1+self.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Actor反向传播
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            self.actor_losses.append(actor_loss.item())
            self.critic_losses.append(critic_loss.item())
        if DEBUG_MODE == 1:
            print("===============update() completed ================")

    def compute_gae(self, rewards, values, next_values, dones=None):
        if DEBUG_MODE == 1:
            print("===============Into compute_gae()================")
        """计算广义优势估计(GAE)
        
        Args:
            rewards: 形状[batch_size]或[batch_size, 1]的张量
            values: 形状与rewards相同的当前状态价值估计
            next_values: 形状与rewards相同的下一状态价值估计
            dones: 可选 终止标志 形状与rewards相同
            
        Returns:
            advantages: 形状与输入相同的优势值
            returns: 形状与输入相同的回报值
        """
        # 将输入移动到GPU计算
        #rewards = rewards.to(device)
        #values = values.to(device) 已经在GPU上
        #next_values = next_values.to(device) 已经在GPU上

        # 确保输入是一维的
        rewards = rewards.squeeze(-1) if rewards.dim() > 1 else rewards
        values = values.squeeze(-1) if values.dim() > 1 else values
        next_values = next_values.squeeze(-1) if next_values.dim() > 1 else next_values
        
        # 检查这几个张量的设备
        if DEBUG_MODE == 1:
            print(f"rewards device: {rewards.device}")
            print(f"values device: {values.device}")
            print(f"next_values device: {next_values.device}")

        if dones is None:
            dones = torch.zeros_like(rewards) # zeros_like会继承rewards张量设备
        else:
            dones = dones.squeeze(-1) if dones.dim() > 1 else dones
        
        if DEBUG_MODE == 1:
            print(f"dones device: {dones.device}")

        advantages = torch.zeros_like(rewards, device=device) # 在GPU上创建advantages
        gae = 0
        
        # 反向计算
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = next_values[t]
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t+1]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        if DEBUG_MODE == 1:
            print(f"advantages device: {advantages.device}")
            print(f"values device: {values.device}")
        
        returns = advantages + values

        if DEBUG_MODE == 1:
            print("===============compute_gae() completed================")
        
        return advantages, returns
    
    def update_with_gae(self):
        """使用GAE进行mini-batch更新"""
        if DEBUG_MODE == 1:
            print("===============Into update_with_gae()================")
        if len(self.buffer) < self.minibatch_size:
            return
        
        # 获取所有经验数据
        batch = self.buffer.sample_batch(self.minibatch_size)
        obs = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_obs = batch['next_states']
        old_log_probs = batch['log_probs']

        # 检查经验数据所在设备
        if DEBUG_MODE == 1:
            print(f"obs device: {obs.device}")
            print(f"actions device: {actions.device}")
            print(f"rewards device: {rewards.device}")
            print(f"next_obs device: {next_obs.device}")
            print(f"old_log_probs device: {old_log_probs.device}")

        # 计算GAE（不需要梯度）
        with torch.no_grad():
            values = self.critic(obs).squeeze()
            next_values = self.critic(next_obs).squeeze()
            advantages, returns = self.compute_gae(rewards, values, next_values)
        
        # 多轮更新
        for _ in range(self.n_updates_per_iteration):
            # --- 1. 更新Critic ---
            # 计算当前critic的values（需要梯度）
            values = self.critic(obs).squeeze(-1)
            returns = returns.squeeze(-1)  # 去掉最后一个维度
            # Critic损失
            critic_loss = F.mse_loss(values, returns)
            
            # Critic反向传播
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()
            
            # --- 2. 更新Actor ---
            # 重新计算values（不需要梯度）
            with torch.no_grad():
                values = self.critic(obs).squeeze()
            
            # 计算新策略的对数概率
            new_log_probs = self.actor.get_log_prob(obs, actions)
            ratios = torch.exp(new_log_probs - old_log_probs)
            
            # PPO损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.clip_ratio, 1+self.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Actor反向传播
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            
            self.actor_losses.append(actor_loss.item())
            self.critic_losses.append(critic_loss.item())
        if DEBUG_MODE == 1:
            print("===============update_with_gae() completed================")

    
    def train(self, total_steps):
        if DEBUG_MODE == 1:
            print("===============Into train()================")
        """训练循环"""
        step = 0
        while step < total_steps:
            start_time = time.time()
            # 1. 收集新轨迹并更新经验池
            # 收集新数据
            self.collect_trajectories(n_trajectories = self.n_trajectories)
            
            # 更新策略
            # self.update()
            self.update_with_gae()
            end_time = time.time()
            print(f"Episode {step+1}/{total_steps} took {end_time - start_time:.2f} seconds")
            if step % self.save_freq == 0:
                # 保存模型
                print(f"Saving model at step {step}")
                self.save_model(self.address_seed)
            step += 1
        if DEBUG_MODE == 1:
            print("===============train() completed================")

    def save_model(self, save_dir):
        """保存Actor和Critic模型"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        """保存模型参数"""
        torch.save(self.actor.state_dict(), os.path.join(save_dir, 'ppo_actor.pt'))
        torch.save(self.critic.state_dict(), os.path.join(save_dir, 'ppo_critic.pt'))

    def action_DRL(self, sqc_data):
        if DEBUG_MODE == 1:
            print("===============Into action_DRL()================")
        
        m_idx = sqc_data[-1] # sqc_data为list列表, m_idx 为int
        s_t = self.build_state(sqc_data)    # build_state()中创建s_t张量时指定设备device
        if DEBUG_MODE == 1:
            print("s_t device: ", s_t.device)
        
        # 将状态转换为适合网络的格式
        state_tensor = s_t.reshape([1,1,self.input_size]) # shape: [1, state_dim]
        
        if DEBUG_MODE == 1:
            print(f"===============the device of state_tensor is:{state_tensor.device}================")

        # 使用 actor 网络生成动作分布
        with torch.no_grad():
            # actor 网络输出均值 (假设网络直接输出均值)
            action_mean = self.actor.forward(state_tensor)
            
            # 创建动作分布 (假设协方差矩阵是固定的或由另一网络输出)
            dist = MultivariateNormal(action_mean, self.cov_mat)
            
            # 采样动作
            action = dist.sample()
            log_prob = dist.log_prob(action)
            if DEBUG_MODE == 1:
                print("===============actor net action sampled================")
        
        # 使用 actor 网络选择的动作 (转换为离散动作)
        a_t = torch.argmax(action).item()  # 假设动作空间是离散的
        
        # the decision is made by one of the available sequencing rule
        job_position = self.func_list[a_t](sqc_data)
        j_idx = sqc_data[-2][job_position]
        # 记录经验 (包括 log_prob 用于 PPO 更新)
        self.build_experience(j_idx, m_idx, s_t, a_t,  log_prob=log_prob)

        if DEBUG_MODE == 1:
            print("===============action_DRL() completed================")

        return job_position
    '''
    2. downwards are functions used for building the state of the experience (replay memory)
    '''
    '''
    local data consists of:
    0            1                 2         3        4
    [current_pt, remaining_job_pt, due_list, env.now, completion_rate,
    5              6      7     8     9        10         11           12/-3   13/-2  14/-1
    time_till_due, slack, winq, avlm, next_pt, rem_no_op, waited_time, wc_idx, queue, m_idx]
    '''

    def state_multi_channel(self, sqc_data):
        if DEBUG_MODE == 1:
            print("===============Into state_multi_channel()================")
        # information in job number, global and local
        in_system_job_no = self.job_creator.in_system_job_no
        local_job_no = len(sqc_data[0])
        # the information of coming job (currently being processed by other machines)
        #print('coming jobs:',self.job_creator.next_wc_list, self.job_creator.arriving_job_slack_list, self.job_creator.release_time_list, sqc_data[-3])
        arriving_jobs = np.where(self.job_creator.next_wc_list == sqc_data[-3])[0] # return the index of coming jobs
        arriving_job_no = arriving_jobs.size  # expected arriving job number
        if arriving_job_no: # if there're jobs coming at your workcenter
            arriving_job_time = (self.job_creator.release_time_list[arriving_jobs] - self.env.now).mean() # average time from now when next job arrives at workcenter
            arriving_job_slack = (self.job_creator.arriving_job_slack_list[arriving_jobs]).mean() # what's the average slack time of the arriving job
        else:
            arriving_job_time = 0
            arriving_job_slack = 0
        #print(arriving_jobs, arriving_job_no, arriving_job_time, arriving_job_slack, self.env.now, sqc_data[-3])
        # information of progression of jobs, get from the job creator
        global_comp_rate = self.job_creator.comp_rate
        global_realized_tard_rate = self.job_creator.realized_tard_rate
        global_exp_tard_rate = self.job_creator.exp_tard_rate
        available_time = (self.job_creator.available_time_list - self.env.now).clip(0,None)
        # get the pt of all remaining jobs in system
        rem_pt = []
        # need loop here because remaining_pt have different length
        for m in self.m_list:
            for x in m.remaining_pt_list:
                rem_pt += x.tolist()
        # processing time related data
        total_time = sum(available_time)
        pt_share = available_time[sqc_data[-1]] / total_time if total_time > 0 else 0

        global_pt_CV = np.std(rem_pt) / np.mean(rem_pt)
        # information of queuing jobs in queue
        local_pt_sum = np.sum(sqc_data[0])
        local_pt_mean = np.mean(sqc_data[0])
        local_pt_min = np.min(sqc_data[0])
        local_pt_CV = np.std(sqc_data[0]) / local_pt_mean
        # information of queuing jobs in remaining processing time
        local_remaining_pt_sum = np.sum(sqc_data[1])
        local_remaining_pt_mean = np.mean(sqc_data[1])
        local_remaining_pt_max = np.max(sqc_data[1])
        local_remaining_pt_CV = np.std(sqc_data[1]) / local_remaining_pt_mean if local_remaining_pt_mean > 0 else 0
        # information of WINQ
        avlm_mean = np.mean(sqc_data[8])
        avlm_min = np.min(sqc_data[8])
        avlm_CV = np.std(sqc_data[8]) / avlm_mean if avlm_mean > 0 else 0
        # time-till-due related data:
        time_till_due = sqc_data[5]
        realized_tard_rate = time_till_due[time_till_due<0].size / local_job_no # ratio of tardy jobs
        ttd_sum = time_till_due.sum()
        ttd_mean = time_till_due.mean()
        ttd_min = time_till_due.min()
        ttd_CV = (time_till_due.std() / ttd_mean).clip(-2,2) if ttd_mean > 0 else 0
        # slack-related data:
        slack = sqc_data[6]
        exp_tard_rate = slack[slack<0].size / local_job_no # ratio of jobs expect to be tardy
        slack_sum = slack.sum()
        slack_mean = slack.mean()
        slack_min = slack.min()
        slack_CV = (slack.std() / slack_mean).clip(-2,2) if slack_mean > 0 else 0
        # use raw data, and leave the magnitude adjustment to normalization layers
        no_info = [in_system_job_no, arriving_job_no, local_job_no] # info in job number
        pt_info = [local_pt_sum, local_pt_mean, local_pt_min] # info in processing time
        remaining_pt_info = [local_remaining_pt_sum, local_remaining_pt_mean, local_remaining_pt_max, avlm_mean, avlm_min] # info in remaining processing time
        ttd_slack_info = [ttd_mean, ttd_min, slack_mean, slack_min, arriving_job_slack] # info in time till due
        progression = [pt_share, global_comp_rate, global_realized_tard_rate, global_exp_tard_rate] # progression info
        heterogeneity = [global_pt_CV, local_pt_CV, ttd_CV, slack_CV, avlm_CV] # heterogeneity
        # concatenate the data input
        s_t = np.nan_to_num(np.concatenate([no_info, pt_info, remaining_pt_info, ttd_slack_info, progression, heterogeneity]),nan=0,posinf=1,neginf=-1)
        # convert to tensor
        s_t = torch.tensor(s_t, dtype=torch.float, device=device)
        if DEBUG_MODE == 1:
            print("===============state_multi_channel() completed================")
        return s_t
    
    # add the experience to job creator's incomplete experiece memory
    def build_experience(self,j_idx,m_idx,s_t,a_t, log_prob):
        self.job_creator.incomplete_rep_memo[m_idx][self.env.now] = [s_t, a_t, log_prob]


if __name__ == '__main__':
    # create the shop floor instance
    m = 6
    wc = 3
    length_list = [2, 2, 2]   

    total_episode = 5
    span = 1000

    sequencing_brain = Sequencing_brain(span= span)
    sequencing_brain.train(total_steps = total_episode)


    # print(sequencing_brain.tard)
    # plot_loss(sequencing_brain.tard)
    # plot_loss(sequencing_brain.actor_losses)
    # plot_loss(sequencing_brain.critic_losses)
