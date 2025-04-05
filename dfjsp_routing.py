import simpy
import random
import numpy as np
import torch

'''
routing data
job_idx, routing_data, machine_condition, job_pt, job slack, wc_idx, remaining_pt
routing_data = [cumulative_pt, time till available, que_size, cumulative_run_time, m_ur]
cumulative_pt : 当前机器上的所有工作的总加工时间   
time till available : 恢复到可用状态 还需要多少时间
que_size : 当前机器上的所有工作的总数量
cumulative_run_time : 当前机器已经运行的时间
m_ur : 当前机器的利用率
'''
# Benchmark, as the worst possible case
def random_routing(idx, data, job_pt, job_slack, wc_idx, *args):
    machine_idx = np.random.randint(len(job_pt))
    return machine_idx

def TT(idx, data, job_pt, ttd, job_slack, wc_idx, *args): # shortest total waiting time
    # axis=0 means choose along columns
    # print("routing data:", data)
    rank = np.argmin(data, axis=0)
    machine_idx = rank[0]
    return machine_idx

def ET(idx, data, job_pt, ttd, job_slack, wc_idx, *args): # minimum exceution time
    machine_idx = np.argmin(job_pt)
    return machine_idx

def EA(idx, data, job_pt, ttd, job_slack, wc_idx, *args):# earliest available
    # print("routing data:",data, np.transpose(data))
    rank = np.argmin(data, axis=0)
    # print("rank:",rank)
    machine_idx = rank[1]
    return machine_idx

def SQ(idx, data, job_pt, ttd, job_slack, wc_idx, *args):# shortest queue
    rank = np.argmin(data, axis=0)
    machine_idx = rank[2]
    return machine_idx

def CT(idx, data, job_pt, ttd, job_slack, wc_idx, *args): # earliest completion time
    #print(data,job_pt)
    completion_time = np.array(data)[:,1].clip(0) + np.array(job_pt)
    machine_idx = completion_time.argmin()
    return machine_idx

def UT(idx, data, job_pt, ttd, job_slack, wc_idx, *args): # lowest utilization rate
    rank = np.argmin(data, axis=0)
    machine_idx = rank[3]
    return machine_idx

def GP_R1(idx, data, job_pt, job_slack, wc_idx, *args): # genetic programming
    data = np.transpose(data)
    sec1 = min(2 * data[2] * np.max([data[2]*job_pt/data[1] , job_pt*data[0]*data[0]], axis=0))
    sec2 = data[2] * job_pt - data[1]
    sum = sec1 + sec2
    machine_idx = sum.argmin()
    return machine_idx

def GP_R2(idx, data, job_pt, job_slack, wc_idx, *args): # genetic programming
    data = np.transpose(data)
    sec1 = data[2]*data[2], (data[2]+job_pt)*data[2]
    sec2 = np.min([data[1],args[0]/(data[1]*args[0]-1)],axis=0)
    sec3 = -data[2] * args[0]
    sec4 = data[2] * job_pt * np.max([data[0], np.min([data[1],job_pt],axis=0)/(args[0])],axis=0)
    sec5 = np.max([data[2]*data[2], np.ones_like(data[2])*(args[1]-args[0]-1), (data[2]+job_pt)*np.min([data[2],np.ones_like(data[2])*args[1]],axis=0)],axis=0)
    sum = sec1 - sec2 * np.max([sec3+sec4/sec5],axis=0)
    machine_idx = sum.argmin()
    return machine_idx
