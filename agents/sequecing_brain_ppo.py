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
# from utils.ppo_buffer import PPOTrajectoryBuffer
from utils.shop_floor import shopfloor
from utils.record_output import plot_loss, plot_tard
from utils.chart import generate_gannt_chart

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
	def __init__(self, m, wc, length_list, tightness, add_job, **hyperparameters):
		self.m = m
		self.wc = wc
		self.length_list = length_list
		self.tightness = tightness
		self.add_job = add_job
		# 1. Init hyperpara
		self._init_hyperparameters(hyperparameters)

		# 2. Init Multi-Channel
		self.build_state = self.state_multi_channel      
		print("---> Multi-Channel (MC) mode ON <---")

		# list that contains available rules, and use SPT for the first phase of warmup
		self.func_list = [sequencing.SPT,sequencing.WINQ,sequencing.MS,sequencing.CR]
		# action space, consists of all selectable rules
		self.output_size = len(self.func_list)

		
		# specify new address seed for storing the trained parameters
		self.address_seed = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ppo_models')

		# initialize initial replay memory, a dictionary that contains empty lists of replay memory for machines
		self.rep_memo = []
		# some training-related parameters
		self.minibatch_size = self.timespan #直接按一把模拟来采样
		# self.buffer_size = 512
		# self.buffer = PPOTrajectoryBuffer(self.input_size)  # Initialize the experience replay buffer

		# Initialize actor and critic networks
		self.actor = ActorNetwork(self.input_size, self.output_size).to(device)                                                   # ALG STEP 1
		self.critic = CriticNetwork(self.input_size, 1).to(device)

		# Initialize optimizers for actor and critic
		# actor, critic网络已经移动到GPU上后，才可以建立optim
		self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr)
		self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr)

		# Initialize the covariance matrix used to query the actor for actions
		self.cov_var = torch.full(size=(self.output_size,), fill_value=0.5, device=device) # 从to_device 形式变更为直接在GPU上创建
		self.cov_mat = torch.diag(self.cov_var) #to device 是不必要的，将会在GPU上创建，可能是因为跟随cov_var

		self.gae_lambda = 0.95                  # Lambda for GAE
		self.save_freq = 20                             # How often we save in number of iterations
		self.n_trajectories = 5					# 每次rollout模拟5次环境

		# below are data used for debug
		self.tard = []
		self.actor_losses = []
		self.critic_losses = []
		if DEBUG_MODE == 1:
			print("===========BrainPPO Init Done==============")

	# 用来重置模拟环境
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

	def collect_trajectories(self, n_trajectories): # equal to rollout()
		"""收集新轨迹并更新经验池"""
		# state, next_state, log_prob已经是GPU（device）上的张量, action和reward本身是标量
		batch_state =[]	#
		batch_acts = []
		batch_log_probs = []	#
		batch_next_state = []	#
		batch_rews = []
		batch_rtgs = []
		batch_lens = []

		ep_rews = []
		total_len = 0
		if DEBUG_MODE == 1:
			print("===============Into collect_trajectories()================")
		for _ in range(n_trajectories):
			ep_rews = []
			# create the shop floor instance
			env = simpy.Environment()
			spf = shopfloor(env, self.timespan, self.m, self.wc, self.length_list, self.tightness, self.add_job)

			self.reset(spf.job_creator, spf.m_list, env=env)
			env.run()
			if OBSERVE_CUDA == 1:
				print(f"Allocated Memory: {torch.cuda.memory_allocated() / 1024 / 1024} MBs") #观察显存占用情况
				print(f"Cached Memory: {torch.cuda.memory_reserved() / 1024 /1024} MBs")

			# generate_gannt_chart(spf.job_creator.production_record, spf.m_list) # 画图                     

			#Collect the trajectory data from the job creator
			# self.buffer.finalize_trajectory(spf.job_creator.rep_memo_ppo)
			total_traj = spf.job_creator.rep_memo_ppo #一条完整轨迹
			if not total_traj or len(total_traj) < 1:
				print("[ERROR] total_traj len < 0 or empty.")
				return
			print("length of total_trajectory: ", len(total_traj)) #检查一整条traj的长度
			total_len += len(total_traj) #计算累计长度，即n条轨迹加起来的总长
			for step in total_traj:
				if len(step) != 5:
					raise ValueError(f"Each step should contain 5 elements, but got {len(step)}")
				state, action, log_prob, next_state, reward = step
				# state, next_state, log_prob已经是GPU（device）上的张量, action和reward本身是标量, 但会在下面的sample_batch()中被统一为张量形式
				batch_state.append(state)
				batch_acts.append(action)
				batch_log_probs.append(log_prob)
				batch_next_state.append(next_state)
				ep_rews.append(reward)
			ep_t = len(total_traj) #一整条完整轨迹的长度就是当前episode总共用掉的timesteps

			batch_lens.append(ep_t + 1)
			batch_rews.append(ep_rews)

		batch_state = torch.stack(batch_state).reshape(total_len, 1, self.input_size)
		batch_acts = torch.tensor(batch_acts, dtype=torch.long, device=device).reshape(total_len, 1)
		batch_log_probs = torch.stack(batch_log_probs).reshape(total_len, 1)
		batch_rtgs = self.compute_rtgs(batch_rews).reshape(total_len)
		# batch_rews = torch.tensor(batch_rews, dtype=torch.float32, device=device).reshape(total_len, 1)
		

		# collect data for debug purpose
		output_time, cumulative_tard, tard_mean, tard_max, tard_rate = spf.job_creator.tardiness_output()
		self.tard.append(cumulative_tard[-1])
		if DEBUG_MODE == 1:
			print("===============collect_trajectories() completed================")
		return batch_state, batch_acts, batch_log_probs, batch_rtgs, batch_lens

	def compute_rtgs(self, batch_rews): # rewards to go 返回的rtgs为tensor
		batch_rtgs = []

		# Iterate through each episode
		for ep_rews in reversed(batch_rews):

			discounted_reward = 0 # The discounted reward so far

			# Iterate through all rewards in the episode. We go backwards for smoother calculation of each
			# discounted return (think about why it would be harder starting from the beginning)
			for rew in reversed(ep_rews):
				discounted_reward = rew + discounted_reward * self.gamma
				batch_rtgs.insert(0, discounted_reward)

		# Convert the rewards-to-go into a tensor
		batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float, device=device)

		return batch_rtgs
	
	def evaluate(self, batch_obs, batch_acts):
		# Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
		#print(f"type of batch_obs:{type(batch_obs)}")
		#_ = input()
		V = self.critic(batch_obs).squeeze() # 报错

		# Calculate the log probabilities of batch actions using most recent actor network.
		# This segment of code is similar to that in get_action()
		log_probs = self.actor.get_log_prob(batch_obs, batch_acts)

		# Return the value vector V of each observation in the batch
		# and log probabilities log_probs of each action in the batch
		return V, log_probs

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

		# rewards, values, next_values都已经在GPU上
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
				next_non_terminal = 1.0 - dones[t].float()
				next_value = next_values[t]
			else:
				next_non_terminal = 1.0 - dones[t].float()
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
	
	def train(self, total_episodes):
		if DEBUG_MODE == 1:
			print("===============Into train()================")
		"""训练循环"""
		episode = 0
		while episode < total_episodes:
			start_time = time.time()
			# 1. 收集新轨迹，不再使用经验池模式，改为返回batch data
			# batch data为n_trajectories条完整轨迹的数据
			batch_state, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.collect_trajectories(n_trajectories = self.n_trajectories) # 运行n_trajectories次模拟并获得n条完整轨迹存放在buffer中
			V, _ = self.evaluate(batch_state, batch_acts)
			A_k = batch_rtgs - V.detach()
			A_k = (A_k - A_k.mean()) / A_k.std() + 1e-10
			# 2. 更新策略
			for _ in range(self.n_updates_per_iteration):
				V, curr_log_probs = self.evaluate(batch_state, batch_acts)
				# print("V.shape=",V.shape)
				ratios = torch.exp(curr_log_probs - batch_log_probs)
				surr1 = ratios * A_k
				surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * A_k
				actor_loss = (-torch.min(surr1, surr2)).mean()
				#print(f"V shape:{V.shape}, batch_rtgs shape:{batch_rtgs.shape}")
				critic_loss = nn.MSELoss()(V, batch_rtgs)
				print(f"actor_loss = {actor_loss}")
				print(f"critic_loss = {critic_loss}")
				self.actor_optim.zero_grad()
				actor_loss.backward(retain_graph=True)
				self.actor_optim.step()
				self.critic_optim.zero_grad()
				critic_loss.backward()
				self.critic_optim.step()
			end_time = time.time()
			print(f"Episode {episode+1}/{total_episodes} took {end_time - start_time:.2f} seconds")
			if episode % self.save_freq == 0:
				# 保存模型
				print(f"Saving model at step {episode}")
				self.save_model(self.address_seed)
			episode += 1

		if DEBUG_MODE == 1:
			print("===============train() completed================")

	def action_DRL(self, sqc_data):	# 询问actor网络并获取策略
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
			action_mean = self.actor(state_tensor)
			
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
		# print(f"job {j_idx} on machine {m_idx} at time {self.env.now} has been added to the experience memory")
	
	def _init_hyperparameters(self, hyperparameters):
		self.timespan = 1000
		self.timesteps_per_batch = 2048
		self.gamma = 0.95
		self.n_updates_per_iteration = 5
		self.lr = 0.005
		self.clip_ratio = 0.2
		self.input_size = 25

		# Change any default values to custom values for specified hyperparameters
		for param, val in hyperparameters.items():
			exec('self.' + param + ' = ' + str(val))
			'''hyperparameters = {
				'timespan': 1000,
				'timesteps_per_batch': 2048, 
				'max_timesteps_per_episode': 200, 
				'gamma': 0.99, 
				'n_updates_per_iteration': 10,
				'lr': 3e-4, 
				'clip_ratio': 0.2,
				'input_size': 25
			  }'''
	
	def save_model(self, save_dir):
		"""保存Actor和Critic模型"""
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		"""保存模型参数"""
		torch.save(self.actor.state_dict(), os.path.join(save_dir, f"{self.m}_{self.wc}_{self.tightness}_{self.add_job}_ppo_actor.pt"))
		torch.save(self.critic.state_dict(), os.path.join(save_dir, f"{self.m}_{self.wc}_{self.tightness}_{self.add_job}_ppo_critic.pt"))