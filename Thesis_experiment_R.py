import simpy
import sys
sys.path #sometimes need this to refresh the path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import numpy as np
from tabulate import tabulate
import pandas as pd
from pandas import DataFrame
import random

import agent_machine
import agent_workcenter
import sequencing
import dfjsp_routing as routing
import job_creation
import os

from tools import generate_length_list, get_group_index

import pandas as pd
from pandas import DataFrame

import validation_R
class shopfloor:
    def __init__(self, env, span, m_no, wc_no, m_per_wc, tightness,  add_job,**kwargs):
        '''STEP 1: create environment instances and specifiy simulation span '''
        self.env=env
        self.span = span
        self.m_no = m_no
        self.m_list = []
        self.wc_no = wc_no
        self.wc_list = []
        '''STEP 2.1: create instances of machines'''
        for i in range(m_no):
            expr1 = '''self.m_{} = agent_machine.machine(env, {}, print = 0)'''.format(i,i) # create machines
            exec(expr1)
            expr2 = '''self.m_list.append(self.m_{})'''.format(i) # add to machine list
            exec(expr2)
        # print(self.m_list)
        '''STEP 2.2: create instances of work centers'''
        cum_m_idx = 0
        for i in range(wc_no):
            x = [self.m_list[m_idx] for m_idx in range(cum_m_idx, cum_m_idx + m_per_wc[i])]
            #print(x)
            expr1 = '''self.wc_{} = agent_workcenter.workcenter(env, {}, x)'''.format(i,i) # create work centers
            exec(expr1)
            expr2 = '''self.wc_list.append(self.wc_{})'''.format(i) # add to machine list
            exec(expr2)
            cum_m_idx += m_per_wc[i]
        # print(self.wc_list)

        '''STEP 3: initialize the job creator'''
        # env, span, machine_list, workcenter_list, number_of_jobs, pt_range, due_tightness, E_utliz
        self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
        pt_range=[10,50], due_tightness= tightness, add_num= add_job, mpc_max=1.3, length_list = m_per_wc, beta= [10,15], random_seed = True)
        # self.job_creator.output()

        '''STEP 4: initialize machines and work centers'''
        for wc in self.wc_list:
            wc.print_info = 0
            wc.log_info = 0
            wc.initialization(self.job_creator)
        for i,m in enumerate(self.m_list):
            m.print_info = 0
            m.log_info = 0
            wc_idx = get_group_index(m_per_wc, i)
            m.initialization(self.m_list,self.wc_list,self.job_creator,self.wc_list[wc_idx])

        '''STEP 5: set sequencing or routing rules, and DRL'''
        # check if need to reset sequencing rule
        if 'sequencing_rule' in kwargs:
            print("Taking over: machines use {} sequencing rule".format(kwargs['sequencing_rule']))
            for m in self.m_list:
                order = "m.job_sequencing = sequencing." + kwargs['sequencing_rule']
                try:
                    exec(order)
                except:
                    print("Rule assigned to machine {} is invalid !".format(m.m_idx))
                    raise Exception

        # check if need to reset routing rule
        if 'routing_rule' in kwargs:
            print("Taking over: workcenters use {} routing rule".format(kwargs['routing_rule']))
            for wc in self.wc_list:
                order = "wc.job_routing = routing." + kwargs['routing_rule']
                try:
                    exec(order)
                except:
                    print("Rule assigned to workcenter {} is invalid !".format(wc.wc_idx))
                    raise Exception

        # specify the architecture of DRL
        if 'address'in kwargs and kwargs['address'] :
            self.routing_brain = validation_R.DRL_routing(self.env, self.job_creator, self.wc_list, address_seed= kwargs['address'], validated=1)


    def simulation(self):
        self.env.run()


# dictionary to store shopfloors and production record
spf_dict = {}
production_record = {}
# list of experiments
benchmark = ['EA','UT','SQ','CT','random_routing']
title = benchmark + ['deep MARL-MC']
#benchmark = ['EA','CT']
sum_record = []
benchmark_record = []
max_record = []
rate_record = []
iteration = 50
export_result = 1

m = [6,12,24]
wc = [3, 4, 6]
# lst = [2 for _ in range(3)]
length_list = [[2, 2, 2],[3, 3, 3, 3],[4, 4, 4, 4, 4, 4]]
tightness = [0.6, 1.0, 1.6]
add_job = [100, 200]
span = 1000

for i in range(len(tightness)):
    for j in range(len(length_list)):
        for k in range(len(add_job)):
            for run in range(iteration):
                print('******************* ITERATION-{} *******************'.format(run))
                sum_record.append([])
                benchmark_record.append([])
                max_record.append([])
                rate_record.append([])
                seed = np.random.randint(2000000000)
                # run simulation with different rules                
                for idx,rule in enumerate(benchmark):
                    # create the environment instance for simulation
                    # if i == 0 and j == 0 and k == 0:
                    if True:
                        env = simpy.Environment()  
                        spf = shopfloor(env, span, m[j], wc[j], length_list[j], tightness[i], add_job[k], routing_rule = rule)
                        spf.simulation()
                        output_time, cumulative_tard, tard_mean, tard_max, tard_rate = spf.job_creator.tardiness_output()
                        sum_record[run].append(cumulative_tard[-1])
                        benchmark_record[run].append(cumulative_tard[-1])
                        max_record[run].append(tard_max)
                        rate_record[run].append(tard_rate)


                # if i == 0 and j == 0 and k == 0:
                if True:
                    # and extra run with deep MARL-MC
                    env = simpy.Environment()
                    path = (os.path.join(sys.path[0], 'ddqn_models','RA', f"{m[j]}_{wc[j]}_{tightness[i]}_{add_job[k]}" + ".pt"))
                    print(path)
                    spf = shopfloor(env, span, m[j], wc[j], length_list[j], tightness[i], add_job[k], address = path)
                    spf.simulation()
                    output_time, cumulative_tard, tard_mean, tard_max, tard_rate = spf.job_creator.tardiness_output()
                    sum_record[run].append(cumulative_tard[-1])
                    max_record[run].append(tard_max)
                    rate_record[run].append(tard_rate)
            print('-------------- Complete Record --------------')
            print(tabulate(sum_record, headers=title))
            print('-------------- Average Performance --------------')

            # get the performnce without DRL
            avg_b = np.mean(benchmark_record,axis=0)
            ratio_b = np.around(avg_b/avg_b.max()*100,2)
            winning_rate_b = np.zeros(len(title))
            for idx in np.argmin(benchmark_record,axis=1):
                winning_rate_b[idx] += 1
            winning_rate_b = np.around(winning_rate_b/iteration*100,2)

            # get the overall performance (include DRL)
            avg = np.mean(sum_record,axis=0)
            std = np.std(sum_record, axis=0)   # 沿轴0计算每列的标准差
            max = np.mean(max_record,axis=0)
            tardy_rate = np.around(np.mean(rate_record,axis=0)*100,2)
            ratio = np.around(avg/avg.min()*100,2)
            rank = np.argsort(ratio)
            winning_rate = np.zeros(len(title))
            for idx in np.argmin(sum_record,axis=1):
                winning_rate[idx] += 1
            winning_rate = np.around(winning_rate/iteration*100,2)
            for rank,rule in enumerate(rank):
                print("{}, avg.: {} | max: {} | %: {}% | tardy %: {}% | winning rate: {}/{}%"\
                .format(title[rule],avg[rule],max[rule],ratio[rule],tardy_rate[rule],winning_rate_b[rule],winning_rate[rule]))

            if export_result:
                df_win_rate = DataFrame([winning_rate], columns=title)
                df_average = DataFrame([avg], columns=title)
                df_std = DataFrame([std], columns=title)
                # print(df_win_rate)
                df_sum = DataFrame(sum_record, columns=title)
                # print(df_sum)
                df_tardy_rate = DataFrame(rate_record, columns=title)
                # print(df_tardy_rate)
                df_max = DataFrame(max_record, columns=title)
                # print(df_max)
                df_before_win_rate = DataFrame([winning_rate_b], columns=title)
                parent_dir = os.path.join(sys.path[0], 'Thesis_result_figure','RAW_SA_experiment')
                if not os.path.exists(parent_dir):
                    os.makedirs(parent_dir)
                address = os.path.join(parent_dir, f"{m[j]}_{wc[j]}_{tightness[i]}_{add_job[k]}" + '.xlsx')
                Excelwriter = pd.ExcelWriter(address,engine="xlsxwriter")
                dflist = [df_win_rate, df_sum, df_tardy_rate, df_max, df_before_win_rate]
                sheetname = ['win rate','sum', 'tardy rate', 'maximum','before win rate']

                for line,df in enumerate(dflist):
                    df.to_excel(Excelwriter, sheet_name=sheetname[line], index=False)
                Excelwriter._save()

                raw_address = os.path.join(parent_dir , 'raw_RA.xlsx')
                    # 检查文件是否存在
                if os.path.exists(raw_address):
                    # 读取现有数据并追加新行
                    existing_win = pd.read_excel(raw_address, sheet_name='win rate')
                    existing_avg = pd.read_excel(raw_address, sheet_name='average')
                    existing_std = pd.read_excel(raw_address, sheet_name='std')

                    updated_win = pd.concat([existing_win, df_win_rate], ignore_index=True)
                    updated_avg = pd.concat([existing_avg, df_average], ignore_index=True)
                    updated_std = pd.concat([existing_std, df_std], ignore_index=True)
                    
                    # 保存更新后的数据
                    with pd.ExcelWriter(raw_address, engine='openpyxl') as writer:
                        updated_win.to_excel(writer, sheet_name='win rate', index=False)
                        updated_avg.to_excel(writer, sheet_name='average', index=False)
                        updated_std.to_excel(writer, sheet_name='std', index=False)
                else:
                    # 文件不存在时，创建新文件
                    with pd.ExcelWriter(raw_address, engine='openpyxl') as writer:
                        df_win_rate.to_excel(writer, sheet_name='win rate', index=False)
                        df_average.to_excel(writer, sheet_name='average', index=False)
                        df_std.to_excel(writer, sheet_name='std', index=False)
                print('export to {}'.format(address))

            # reset the record for next iteration
            sum_record = []
            benchmark_record = []
            max_record = []
            rate_record = []
