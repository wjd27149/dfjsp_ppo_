from networks.ddqn_network import ValidatedNetwork, DQNNetwork
import dfjsp_routing
import sys
import torch.optim as optim
import torch
import random
import numpy as np
import copy
import os

class RoutingBrain:
    def __init__(self, span, total_episode, input_size, output_size, *args, **kwargs):

        self.span = span
        self.total_episode = total_episode

        self.input_size = int(input_size)         
        self.func_list = [dfjsp_routing.TT, dfjsp_routing.ET, dfjsp_routing.EA, dfjsp_routing.SQ]
        self.output_size = len(self.func_list)
        
        self.output_size = int(output_size)
        print('output size:', self.output_size)

        # specify the path to store the model
        self.path = sys.path[0]

        # specify the address to store the model
        print("---X TEST mode ON X---")
        self.routing_action_NN = DQNNetwork(self.input_size, self.output_size)
        self.routing_target_NN = copy.deepcopy(self.routing_action_NN)
        self.build_state = self.state_deeper
        self.train = self.train_DDQN

        # initialize the synchronization time 学习率和探索率
        self.current_episode = 0

        ''' specify the optimizer '''
        # 初始化参数
        self.initial_lr = 0.01       # 初始学习率
        self.min_lr = 0.0001         # 最小学习率
        self.routing_action_NN.lr = self.initial_lr
        self.initial_epsilon = 0.5  # 初始探索率
        self.epsilon = self.initial_epsilon
        self.epsilon_min = 0.01      # 最小探索率
        self.epsilon_decay = 0.98  # 每步衰减系数（0.9995^1000≈0.6）

        self.optimizer = optim.SGD(self.routing_action_NN.parameters(), 
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
        self.routing_action_NN_training_interval = 20 # the interval of training the action NN
        self.training_step = 0 # the step of training the action NN
        
    def reset(self, env,span, job_creator, m_list, wc_list):
        # initialize the environment and the workcenter to be controlled
        self.env = env
        self.span = span
        self.job_creator = job_creator
        self.m_list = m_list
        self.wc_list = wc_list
        # specify the constituent machines, and activate their routing learning
        self.m_per_wc = len(self.wc_list[0].m_list)
        

        for m in m_list:
            m.routing_learning_event.succeed()
            
        # action space, consists of all selectable machines
        for wc in self.wc_list:
            wc.job_routing = self.action_DRL_machine
            wc.build_state = self.state_deeper

        self.env.process(self.training_process_parameter_sharing())
        self.env.process(self.update_rep_memo_parameter_sharing_process())
        self.env.process(self.update_training_setting_process())

    # train the action NN periodically
    def training_process_parameter_sharing(self):
        # 1. 等待初始数据  
        while len(self.rep_memo) < self.minibatch_size:
            print(f"Waiting for data: {len(self.rep_memo)}/{self.minibatch_size}")
            print(f"Current time: {self.env.now}")
            yield self.env.timeout(50)  # 逐步推进时间

        # 2. 正式训练循环

        while self.env.now < self.span:
            self.train()
            # print(f"Training at t={self.env.now}, buffer={len(self.rep_memo)}")
            yield self.env.timeout(max(1, self.routing_action_NN_training_interval))  # 至少1时间步

    def update_rep_memo_parameter_sharing_process(self):
        while self.env.now < self.span:
            for wc in self.wc_list:
                self.rep_memo += wc.rep_memo.copy()
                # and clear workcenter's replay memory
                wc.rep_memo = []
            # print(self.rep_memo)
            # and clear obsolete memory
            # print('not Truncate replay memory to size:',len(self.rep_memo), "time:", self.env.now)
            if len(self.rep_memo) > self.rep_memo_size:
                
                truncation = len(self.rep_memo)-self.rep_memo_size
                self.rep_memo = self.rep_memo[truncation:]
                # print('Truncate replay memory to size:',len(self.rep_memo), "time:", self.env.now)
            yield self.env.timeout(20)

    # random exploration is for collecting more experience
    # routing_data 代表了 routing 方法 选择哪个机器
    # args[2] 代表了那些需要被处理进网络训练的数据
    def action_DRL(self, job_idx, routing_data, job_pt, ttd, job_slack, wc_idx, *args):
        # s_t = self.build_state(routing_data, job_pt, job_slack, wc_idx)
        s_t = self.build_state(routing_data, job_pt, ttd, job_slack, wc_idx)
        # generate the action
        if random.random() < self.epsilon:
            # a_t = torch.randint(0,self.m_per_wc,[])
            a_t = torch.randint(0,len(self.func_list),[])
            machine_position = self.func_list[a_t](job_idx, routing_data, ttd, job_pt, job_slack, wc_idx)
            # print('RANDOM ROUTING: wc {} assign job {} to m {}'.format(wc_idx, job_idx,self.wc_list[wc_idx].m_list[machine_position].m_idx))
        else:
            # input state to policy network, produce the state-action value
            value = self.routing_action_NN.forward(s_t.reshape(1,1,self.input_size), wc_idx)
            # print("State:",s_t, 'State-Action Values:', value)
            a_t = torch.argmax(value)
            machine_position = self.func_list[a_t](job_idx, routing_data, ttd, job_pt, job_slack, wc_idx)
            # print('DRL ROUTING: wc {} assign job {} to m {}'.format(wc_idx, job_idx,self.wc_list[wc_idx].m_list[machine_position].m_idx))
        # add current state and action to target wc's incomplete experience
        # this is the first half of a single record of experience
        # the second half will be appended to first half when routed job complete its operation
        self.build_experience(job_idx, s_t, a_t, wc_idx)
        return machine_position
    
    def action_DRL_machine(self, job_idx, routing_data, job_pt, ttd, job_slack, wc_idx, *args):
        # s_t = self.build_state(routing_data, job_pt, job_slack, wc_idx)
        s_t = self.build_state(routing_data, job_pt, ttd, job_slack, wc_idx)
        # generate the action
        if random.random() < self.epsilon:
            a_t = torch.randint(0,self.m_per_wc,[])
            # a_t = torch.randint(0,len(self.func_list),[])
            # machine_position = self.func_list[a_t](job_idx, routing_data, ttd, job_pt, job_slack, wc_idx)
            # print('RANDOM ROUTING: wc {} assign job {} to m {}'.format(wc_idx, job_idx,self.wc_list[wc_idx].m_list[machine_position].m_idx))
        else:
            # input state to policy network, produce the state-action value
            value = self.routing_action_NN.forward(s_t.reshape(1,1,self.input_size), wc_idx)
            # print("State:",s_t, 'State-Action Values:', value)
            a_t = torch.argmax(value)
            # machine_position = self.func_list[a_t](job_idx, routing_data, ttd, job_pt, job_slack, wc_idx)
            # print('DRL ROUTING: wc {} assign job {} to m {}'.format(wc_idx, job_idx,self.wc_list[wc_idx].m_list[machine_position].m_idx))
        # add current state and action to target wc's incomplete experience
        # this is the first half of a single record of experience
        # the second half will be appended to first half when routed job complete its operation
        self.build_experience(job_idx, s_t, a_t, wc_idx)
        return a_t
    
    def build_experience(self, job_idx, s_t, a_t, wc_idx):
        self.wc_list[wc_idx].incomplete_experience[job_idx] = [s_t, a_t]
    '''
    2. downwards are functions used for building the state of the experience (replay memory)
    '''

    def state_deeper(self, routing_data, job_pt, ttd, job_slack, wc_idx):
        coming_job_idx = np.where(self.job_creator.next_wc_list == wc_idx)[0]  # return the index of coming jobs
        coming_job_no = coming_job_idx.size  # expected arriving job number
        if coming_job_no:  # if there're jobs coming at your workcenter
            next_job = self.job_creator.release_time_list[coming_job_idx].argmin()  # the index of next job
            coming_job_time = (self.job_creator.release_time_list[coming_job_idx] - self.env.now)[next_job]  # time from now when next job arrives at workcenter
            coming_job_slack = self.job_creator.arriving_job_slack_list[coming_job_idx][next_job]  # what's the average slack time of the arriving job
        else:
            coming_job_time = 0
            coming_job_slack = 0

        # Ensure all inputs are numpy arrays with consistent dtype  取前两个数
        m_state = np.array([a[:2] for a in routing_data], dtype=np.float32)  # Convert to 2D array
        job_pt = np.array([job_pt], dtype=np.float32)  # Convert to 1D array
        job_slack = np.array([job_slack], dtype=np.float32)  # Convert to 1D array
        coming_job_time = np.array([coming_job_time], dtype=np.float32)  # Convert to 1D array
        coming_job_slack = np.array([coming_job_slack], dtype=np.float32)  # Convert to 1D array
        # print("m_state:", m_state, m_state.shape, m_state.dtype)
        # print("job_pt:", job_pt, job_pt.shape, job_pt.dtype)
        # print("job_slack:", job_slack, job_slack.shape, job_slack.dtype)
        # print("coming_job_time:", coming_job_time, coming_job_time.shape, coming_job_time.dtype)
        # print("coming_job_slack:", coming_job_slack, coming_job_slack.shape, coming_job_slack.dtype)
        # Flatten all arrays
        m_state_flat = m_state.flatten()  # 展平为 (4,)
        job_pt_flat = job_pt.flatten()  # 展平为 (2,)
        job_slack_flat = job_slack.flatten()  # 展平为 (1,)
        coming_job_time_flat = coming_job_time.flatten()  # 展平为 (1,)
        coming_job_slack_flat = coming_job_slack.flatten()  # 展平为 (1,)

        # Concatenate all arrays
        s_t_np = np.concatenate([m_state_flat, job_pt_flat, job_slack_flat, coming_job_time_flat, coming_job_slack_flat])

        # Convert to PyTorch tensor
        s_t = torch.tensor(s_t_np, dtype=torch.float32)        
        # print(s_t, s_t.shape, s_t.dtype, s_t.numel())
        return s_t
    
    def state_mutil_channel(self, routing_data, job_pt, ttd, job_slack, wc_idx):
        coming_job_idx = np.where(self.job_creator.next_wc_list == wc_idx)[0]  # return the index of coming jobs
        coming_job_no = coming_job_idx.size  # expected arriving job number
        in_system_job_no = self.job_creator.in_system_job_no
        buffer_num = sum(sublist[2] for sublist in routing_data)
        # print('buffer_num:', buffer_num, 'routing_data:', routing_data)
        buffer_num_mean = buffer_num / len(routing_data)

        job_pt = np.array([job_pt], dtype=np.float32)  # Convert to 1D array
        job_pt_sum = np.sum(job_pt)
        job_pt_mean = np.mean(job_pt)
        job_pt_min = np.min(job_pt)
        job_pt_std_dev = np.std(job_pt)
        job_pt_cv = job_pt_std_dev / job_pt_mean

        m_available_list = [sublist[1] for sublist in routing_data]
        m_available_sum = sum(m_available_list)
        m_available_min = min(m_available_list)
        m_available_mean = np.mean(m_available_list)
        m_available_std_dev = np.std(m_available_list)
        # 检查 m_available_mean 是否为 0 或 NaN
        if m_available_mean == 0 or np.isnan(m_available_mean):
            m_available_cv = 0  # 或其他默认值
        else:
            m_available_cv = m_available_std_dev / m_available_mean

        # job_slack = np.array([job_slack], dtype=np.float32)  # Convert to 1D array
        # ttd = np.array([ttd], dtype=np.float32)  # Convert to 1D array

        m_ur_list = [sublist[4] for sublist in routing_data]
        # print('m_ur_list:', m_ur_list)
        ur_mean = np.mean(m_ur_list)
        ur_std_dev = np.std(m_ur_list)
        # 计算 ur_cv
        if ur_mean == 0 or np.isnan(ur_mean):
            ur_cv = 0  # 或其他默认值
        else:
            ur_cv = ur_std_dev / ur_mean

        # information of progression of jobs, get from the job creator
        global_comp_rate = self.job_creator.comp_rate
        global_realized_tard_rate = self.job_creator.realized_tard_rate
        global_exp_tard_rate = self.job_creator.exp_tard_rate

        # use raw data, and leave the magnitude adjustment to normalization layers
        no_info = [in_system_job_no, buffer_num_mean, coming_job_no] # info in job number
        pt_info = [job_pt_sum, job_pt_mean, job_pt_min] # info in processing time
        m_available_info = [m_available_sum, m_available_mean, m_available_min] # info in machine available time
        ttd_slack_info = [job_slack, ttd] # info in time till due
        progression = [global_comp_rate, global_realized_tard_rate, global_exp_tard_rate] # progression info
        heterogeneity = [job_pt_cv, m_available_cv, ur_cv] # heterogeneity
        # print('no_info:', no_info, 'pt_info:', pt_info, 'm_available_info:', m_available_info, 'ttd_slack_info:', ttd_slack_info, 'progression:', progression, 'heterogeneity:', heterogeneity)
        # concatenate the data input
        s_t = np.nan_to_num(np.concatenate([no_info, pt_info, m_available_info, ttd_slack_info, progression, heterogeneity]),nan=0,posinf=1,neginf=-1)
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
        model_path = (os.path.join(os.path.dirname(sys.path[0]), 'ddqn_models','RA'))
        os.makedirs(model_path, exist_ok=True)
        torch.save(self.routing_action_NN.state_dict(), os.path.join(model_path, address_seed))
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
        self.routing_action_NN.lr = max(new_lr, self.min_lr)  # 确保不低于最小值
    
        # ===== 2. 探索率更新 =====
        # 指数衰减：epsilon = initial_epsilon * decay^current_episode
        self.epsilon = max(
            self.epsilon_min,
            self.initial_epsilon * (self.epsilon_decay ** self.current_episode)
        )
        
        # 更新优化器的学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.routing_action_NN.lr
        
        # print('-' * 50)
        # print(f'Step {self.current_episode}/{self.total_episode}:')
        # print(f'Learning Rate = {self.routing_action_NN.lr:.6f}')
        # print(f'Exploration Rate (ε) = {self.epsilon:.4f}')
        # print('-' * 50)

    # synchronize the ANN and TNN, and some settings
    def update_training_setting_process(self):
        while self.env.now < self.span:
            # synchronize the parameter of policy and target network
            self.routing_target_NN = copy.deepcopy(self.routing_action_NN)
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
        sample_s1_batch = torch.stack([data[3] for data in minibatch], dim=0).reshape(size,1,self.input_size)
        # reshape so we can use .gather() method
        sample_a0_batch = torch.stack([data[1] for data in minibatch], dim=0).reshape(size,1)
        sample_r0_batch = torch.stack([data[2] for data in minibatch], dim=0).reshape(size,1)
        '''
        the size of these batches:
        sample_s0_batch = sample_s1_batch = minibatch size * 1 * input_size
        sample_a0_batch = sample_r0_batch = minibatch size * m_no
        sample_r0_batch = minibatch size
        '''
        # get the Q value (current value of state-action pair) of s0
        Q_0 = self.routing_action_NN.forward(sample_s0_batch)
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
        Q_1_action = self.routing_action_NN.forward(sample_s1_batch)
        Q_1_target = self.routing_target_NN.forward(sample_s1_batch)
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
        loss = self.routing_action_NN.loss_func(current_value, target_value)
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


