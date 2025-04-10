import simpy
import sys
sys.path #sometimes need this to refresh the path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import numpy as np
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent_machine
import agent_workcenter
import job_creation

from agents.routing_brain import RoutingBrain

import random
import time

from tools import generate_length_list, get_group_index
from utils.record_output import plot_loss, plot_tard
from utils.shop_floor import shopfloor


if __name__ == '__main__':
    m = [6,12,24]
    wc = [3, 4, 6]
    length_list = [[2, 2, 2],[3, 3, 3, 3],[4, 4, 4, 4, 4, 4]]
    tightness = [0.6, 1.0, 1.6]
    add_job = [100, 200]
    
    total_episode = 800
    
    
for i in range(len(tightness)):
    for j in range(len(length_list)):
        for k in range(len(add_job)):
                if (i, j, k)  in [(0, 0, 0), (0, 0, 1)]:
                # if i == 2:
                    print(f'm[j], wc[j], length_list[j], tightness[i], add_job[k]: {m[j]}, {wc[j]}, {length_list[j]}, {tightness[i]}, {add_job[k]}')
                    span = 1000
                    tard = []
                    routing_brain = RoutingBrain(span = span , total_episode= total_episode, input_size = int(m[j]/wc[j]) * 3 + 3, output_size = m[j]/wc[j])
                    start_time = time.time()
                    for step in range(total_episode):
                        routing_brain.update_training_parameters()
                        env = simpy.Environment()
                        spf = shopfloor(env, span, m[j], wc[j], length_list[j], tightness[i], add_job[k], add_flag = False)

                        routing_brain.reset(env,span, spf.job_creator, spf.m_list, spf.wc_list)
                        '''STEP 5: run the simulaiton'''
                        env.run()
                        output_time, cumulative_tard, tard_mean, tard_max, tard_rate = spf.job_creator.tardiness_output()
                        # 选择最大的makespan 当时间输入
                        makespan = output_time[-1]
                        # print(makespan)
                        span = makespan
                        tard.append(cumulative_tard[-1])
                        routing_brain.current_episode += 1

                        if routing_brain.current_episode % 100 == 0:
                            routing_brain.save_model(address_seed = f"{m[j]}_{wc[j]}_{tightness[i]}_{add_job[k]}"+".pt")

                    end_time = time.time()
                    print(f"Simulation time: {end_time - start_time:.2f} seconds")
                    save_path = (os.path.join(os.path.dirname(sys.path[0]), 'photo_record','RA'))
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    plot_loss(routing_brain.loss_record, os.path.join(save_path, f"loss_{m[j]}_{wc[j]}_{tightness[i]}_{add_job[k]}_loss"+".png"))
                    plot_tard(tard, os.path.join(save_path, f"tard_{m[j]}_{wc[j]}_{tightness[i]}_{add_job[k]}_tard"+".png"))
                    # print("tardiness:",tard)
    