import torch
import random
from collections import deque

#Init CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PPO Buffer Using device: {device}")


class PPOTrajectoryBuffer:
    def __init__(self, buffer_size, input_size):
        self.input_size = input_size
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def finalize_trajectory(self, total_trajectory):
        """接收一整条轨迹，添加到 buffer 中"""
        if not total_trajectory or len(total_trajectory) < 1:
            return

        # 检查并转换每一项为标准格式
        processed = []
        for step in total_trajectory:
            if len(step) != 5:
                raise ValueError(f"Each step should contain 5 elements, but got {len(step)}")

            #print("press Any Key to continue...")
            #_ = input() #等待用户按下任意按钮
            state, action, log_prob, next_state, reward = step
            # state, next_state, log_prob已经是GPU（device）上的张量, action和reward本身是标量, 但会在下面的sample_batch()中被统一为张量形式
            processed.append((state, action, log_prob, next_state, reward))
        
        self.buffer.append(processed)
        print(f"Trajectory of length {len(processed)} added to buffer. Current buffer size: {len(self.buffer)}")

    def sample_batch(self, batch_size):
        """随机从某一条轨迹采样 batch"""
        if len(self.buffer) == 0:
            raise ValueError("Experience buffer is empty")

        # 随机选一条轨迹
        traj = random.choice(self.buffer)
        traj_len = len(traj)

        if traj_len < 2:
            raise ValueError("Trajectory is too short")

        # 实际采样大小
        actual_batch_size = min(batch_size, traj_len)

        indices = random.sample(range(traj_len), actual_batch_size)

        # 提取对应的数据,tensor方法创建的张量，直接指定在GPU上创建，stack方法创建的，经验证已经在GPU上
        states = torch.stack([traj[i][0] for i in indices]).reshape(actual_batch_size,1,self.input_size)
        actions = torch.tensor([traj[i][1] for i in indices], dtype=torch.long, device=device).reshape(actual_batch_size,1)
        log_probs = torch.stack([traj[i][2] for i in indices]).reshape(actual_batch_size, 1)
        next_states = torch.stack([traj[i][3] for i in indices]).reshape(actual_batch_size,1,self.input_size)
        rewards = torch.tensor([traj[i][4] for i in indices], dtype=torch.float32, device=device).reshape(actual_batch_size,1)
        dones = torch.zeros(actual_batch_size, dtype=torch.bool, device=device)
        dones[-1] = True  # 可以按需设置终止状态

        return {
            'states': states,
            'actions': actions,
            'log_probs': log_probs,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }

    def __len__(self):
        return len(self.buffer)
