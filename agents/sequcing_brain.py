from networks.ddqn_network import SA_network
import sequencing
import sys
import torch.optim as optim
import torch
import random
import numpy as np
import copy
import os

class SequencingBrain:
    def __init__(self, span, total_episode, input_size, output_size, *args, **kwargs):

        self.span = span
        self.total_episode = total_episode

        # state space, eah machine generate 3 types of data, along with the processing time of arriving job, and job slack
        self.input_size = int(input_size)         
        self.func_list = [sequencing.SPT,sequencing.WINQ,sequencing.MS,sequencing.CR]
        self.output_size = int(output_size)

        # specify the path to store the model
        self.path = sys.path[0]

        # specify the address to store the model
        print("---X TEST mode ON X---")
        self.sequencing_action_NN = SA_network(self.input_size, self.output_size)
        self.sequencing_target_NN = copy.deepcopy(self.sequencing_action_NN)
        self.build_state = self.state_multi_channel
        self.train = self.train_DDQN

        # initialize the synchronization time 学习率和探索率
        self.current_episode = 0

        ''' specify the optimizer '''
        # 初始化参数
        self.initial_lr = 0.01       # 初始学习率
        self.min_lr = 0.0001         # 最小学习率
        self.sequencing_action_NN.lr = self.initial_lr
        self.initial_epsilon = 0.5  # 初始探索率
        self.epsilon = self.initial_epsilon
        self.epsilon_min = 0.01      # 最小探索率
        self.epsilon_decay = 0.98  # 每步衰减系数（0.9995^1000≈0.6）

        self.optimizer = optim.SGD(self.sequencing_action_NN.parameters(), 
                                 lr=self.initial_lr, 
                                 momentum=0.9)

        # Initialize the parameters for training
        self.minibatch_size = 64 * 2
        self.rep_memo_size = 512 * 8
        self.discount_factor = 0.8 # how much agent care long-term rewards

        # initialize the list for deta memory
        self.rep_memo = []
        self.exploration_record = []
        self.time_record = []
        self.loss_record = []

        self.sequencing_target_NN_update_interval = 1000 # the interval of updating the target network
        self.sequencing_action_NN_training_interval = 20 # the interval of training the action NN
        self.training_step = 0 # the step of training the action NN
        
    def reset(self, env,span, job_creator, m_list, wc_list):
        # initialize the environment and the workcenter to be controlled
        self.env = env
        self.span = span
        self.job_creator = job_creator
        self.m_list = m_list
        self.wc_list = wc_list
        # and build dicts that equals number of machines to be controlled in job creator
        self.job_creator.build_sqc_experience_repository(self.m_list)
        for m in m_list:
            m.sequencing_learning_event.succeed()
            m.reward_function = m.get_reward13
            m.build_state = self.state_multi_channel
            m.job_sequencing = self.action_drl
            

        self.env.process(self.training_process_parameter_sharing())
        self.env.process(self.update_rep_memo_parameter_sharing_process())
        self.env.process(self.update_training_setting_process())

    # train the action NN periodically
    def training_process_parameter_sharing(self):
        # 1. 等待初始数据  
        while len(self.rep_memo) < self.minibatch_size:
            # print(f"Waiting for data: {len(self.rep_memo)}/{self.minibatch_size}")
            # print(f"Current time: {self.env.now}")
            yield self.env.timeout(50)  # 逐步推进时间

        # 2. 正式训练循环

        while self.env.now < self.span:
            self.train()
            # print(f"Training at t={self.env.now}, buffer={len(self.rep_memo)}")
            yield self.env.timeout(max(1, self.sequencing_action_NN_training_interval))  # 至少1时间步

    def update_rep_memo_parameter_sharing_process(self):
        while self.env.now < self.span:
            # print(f"Updating replay memory at t={self.env.now}, buffer={len(self.rep_memo)}")
            for m in self.m_list:
                # add new memoery from corresponding rep_memo from job creator
                self.rep_memo += self.job_creator.rep_memo[m.m_idx].copy()
                # and clear the replay memory in job creator, keep it updated
                self.job_creator.rep_memo[m.m_idx] = []
            # clear the obsolete experience periodically
            if len(self.rep_memo) > self.rep_memo_size:
                truncation = len(self.rep_memo)-self.rep_memo_size
                self.rep_memo = self.rep_memo[truncation:]
            yield self.env.timeout(20)

    def action_drl(self, sqc_data):
        s_t = self.build_state(sqc_data)
        m_idx = sqc_data[-1]
        # Epsilon-greedy policy
        if random.random() < self.epsilon:
            a_t = torch.randint(0,self.output_size,[])
            #print('Random Action / By Brain')
        else:
            # input state to policy network, produce the state-action value
            value = self.sequencing_action_NN.forward(s_t.reshape([1,1,self.input_size]), m_idx)
            # greedy policy
            a_t = torch.argmax(value)
            #print("State is:", s_t)
            #print('State-Action Values:', value)
            #print('Sequencing NN, action %s / By Brain'%(a_t))
        # the decision is made by one of the available sequencing rule
        job_position = self.func_list[a_t](sqc_data)
        j_idx = sqc_data[-2][job_position]
        # add the state at sequencing to job creator's corresponding repository
        self.build_experience(j_idx,m_idx,s_t,a_t)
        return job_position
    
    # add the experience to job creator's incomplete experiece memory
    def build_experience(self,j_idx,m_idx,s_t,a_t):
        self.job_creator.incomplete_rep_memo[m_idx][self.env.now] = [s_t,a_t]
        # print('incomplete_rep_memo:', self.job_creator.incomplete_rep_memo[m_idx],'env.now:', self.env.now)
    '''
    2. downwards are functions used for building the state of the experience (replay memory)
    '''

    def state_multi_channel(self, sqc_data):
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
        s_t = torch.tensor(s_t, dtype=torch.float)
        return s_t
    
    '''
    4. downwards are functions used in the training of DRL, including the dynamic training process
       dynamic training parameters update and the optimization funciton of ANN
       the class for builidng the ANN is at the bottom
    '''
    def save_model(self, address_seed):
        # save the model
        model_path = (os.path.join(os.path.dirname(sys.path[0]), 'ddqn_models','SA'))
        os.makedirs(model_path, exist_ok=True)
        torch.save(self.sequencing_action_NN.state_dict(), os.path.join(model_path, address_seed))
        print('model saved at %s' % (os.path.join(self.path, 'ddqn_models', address_seed)))

    def update_training_parameters(self):
        """
        基于训练步数动态调整学习率和探索率（带最小值保护）
        更新逻辑：
        - 学习率 线性衰减 initial_lr → initial_lr*0.1
        - 探索率 指数衰减 initial_epsilon → epsilon_min
        """
        # 计算当前进度（基于step而非episode）
        current_progress = min(self.current_episode / self.total_episode, 1.0)
        
        # ===== 1. 学习率更新 =====
        # 线性衰减：从initial_lr降到initial_lr*0.1
        new_lr = self.initial_lr * (1 - 0.9 * current_progress)
        self.sequencing_action_NN.lr = max(new_lr, self.min_lr)  # 确保不低于最小值
    
        # ===== 2. 探索率更新 =====
        # 指数衰减：epsilon = initial_epsilon * decay^current_episode
        self.epsilon = max(
            self.epsilon_min,
            self.initial_epsilon * (self.epsilon_decay ** self.current_episode)
        )
        
        # 更新优化器的学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.sequencing_action_NN.lr
        
        # print('-' * 50)
        # print(f'Step {self.current_episode}/{self.total_episode}:')
        # print(f'Learning Rate = {self.sequencing_action_NN.lr:.6f}')
        # print(f'Exploration Rate (ε) = {self.epsilon:.4f}')
        # print('-' * 50)

    # synchronize the ANN and TNN, and some settings
    def update_training_setting_process(self):
        while self.env.now < self.span:
            # synchronize the parameter of policy and target network
            self.sequencing_target_NN = copy.deepcopy(self.sequencing_action_NN)
            # print('--------------------------------------------------------')
            # print('the target network is updated at time %s' % self.env.now)
            # print('--------------------------------------------------------')
            
            yield self.env.timeout(self.sequencing_target_NN_update_interval)        # 每隔 1000 同步一次

    # the function that draws minibatch and trains the action NN
    def train_DDQN(self):
        # print(".............TRAINING .............")
        """
        draw the random minibatch to train the network, randomly
        every element in the replay menory is a list [s_0, a_0, r_0, s_1]
        all element of this list are tensors
        """
        size = min(len(self.rep_memo),self.minibatch_size)
        minibatch = random.sample(self.rep_memo,size)
        '''
        slice, and stack the 1D tensors to several 2D tensors (batches)
        the "torch.stack" is only applicable when the augment is a list of tensors, or multi-dimensional tensor
        '''
        # reshape so we can meet the requiremnt of input of instancenorm
        sample_s0_batch = torch.stack([data[0] for data in minibatch], dim=0).reshape(size,1,self.input_size)
        sample_s1_batch = torch.stack([data[2] for data in minibatch], dim=0).reshape(size,1,self.input_size)
        # reshape so we can use .gather() method
        sample_a0_batch = torch.stack([data[1] for data in minibatch], dim=0).reshape(size,1)
        sample_r0_batch = torch.stack([data[3] for data in minibatch], dim=0).reshape(size,1)
        '''
        the size of these batches:
        sample_s0_batch = sample_s1_batch = minibatch size * 1 * input_size
        sample_a0_batch = sample_r0_batch = minibatch size * m_no
        sample_r0_batch = minibatch size
        '''
        # get the Q value (current value of state-action pair) of s0
        Q_0 = self.sequencing_action_NN.forward(sample_s0_batch)
        #print('Q_0 is:\n', Q_0)
        #print('a_0 is:', sample_a0_batch)

        # get the current state-action value of actions that would have been taken
        current_value = Q_0.gather(1, sample_a0_batch)
        #print('current value is:', current_value)

        '''
        get the Q Value of s_1 in both action and target network, to estimate the state value
        architecture is DDQN, NOT DQN !!!
        evaluate the greedy policy according to action network, but using the target network to estimate the value
        '''
        Q_1_action = self.sequencing_action_NN.forward(sample_s1_batch)
        Q_1_target = self.sequencing_target_NN.forward(sample_s1_batch)
        #print('Q_1_action is:\n', Q_1_action)
        #print('Q_1_target is:\n', Q_1_target)

        '''
        size of Q_0, Q_1_action and Q_1_target = minibatch size * m_no
        they're 2D tensors
        '''
        max_Q_1_action, max_Q_1_action_idx = torch.max(Q_1_action, dim=1) # use action network to get action, rather than max operation
        # print('max value of Q_1_action is:\n', max_Q_1_action)

        max_Q_1_action_idx = max_Q_1_action_idx.reshape([size,1])
        # print('max idx of Q_1_action is:\n', max_Q_1_action_idx)
        
        # adjust the max_Q of s_0 by the discount factor (refer to Bellman Equation and TD)
        next_state_value = Q_1_target.gather(1, max_Q_1_action_idx)
        # print('estimated value of next state is:\n', next_state_value)

        discounted_next_state_value = self.discount_factor * next_state_value
        # print('discounted next state value is:\n', discounted_next_state_value)

        '''
        the sum of reward and discounted max_Q is the target value
        target value is 2D matrix, size = minibatch_size * m_no
        '''
        #print('reward batch is:', sample_r0_batch)
        target_value = (sample_r0_batch + discounted_next_state_value)
        #print('target value is:', target_value)
        '''
        the loss is difference between current state-action value and the target value
        '''
        # calculate the loss
        loss = self.sequencing_action_NN.loss_func(current_value, target_value)
        if self.env.now % 1000 == 0:
            print(f"Episode {self.current_episode + 1}/{self.total_episode}  loss: {loss.item():.4f}  time: {self.env.now}")

        self.loss_record.append(float(loss))
        # clear the gradient
        self.optimizer.zero_grad()  # zero the gradient buffers
        # calculate gradient
        loss.backward(retain_graph=True)
        # set the optimizer
        # print('perform the optimization of parameters')
        # optimize the parameters
        self.optimizer.step()


