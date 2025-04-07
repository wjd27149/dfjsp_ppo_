
from matplotlib import pyplot as plt

def generate_gannt_chart(production_record, m_list):
    # 获取作业数量
    extracted_data = production_record
    n_jobs = len(extracted_data)
    # 提取每个作业的数据
    data = {}
    for job_idx, job_data in extracted_data.items():
        operations = job_data[0]  # 操作时间点和加工时间
        machines = job_data[1]    # 机器索引
        job_name = job_data[6]  # 工件名字
        data[job_idx] = [operations, machines, job_name]
    # 使用 matplotlib 的 colormap 生成颜色列表
    colors = plt.cm.get_cmap('tab20').colors  # 获取 'tab20' 的所有颜色
    n_colors = len(colors)  # 颜色数量

    # 初始化绘图
    fig, ax = plt.subplots(figsize=(10, 6))

    # 遍历每个作业
    for job_idx, (operations, machines, job_name) in data.items():
        color = colors[job_idx % n_colors]  # 循环使用颜色
        for i, ((start_time, duration, cmt, flag), machine_idx) in enumerate(zip(operations, machines)):
            # 对时间值进行四舍五入，保留两位小数
            start_time_rounded = round(start_time, 2)
            duration_rounded = round(duration, 2)
            cmt_rounded = round(cmt, 2)

            if flag == 0:
                # 正常绘制条形图
                ax.barh(
                    machine_idx,  # 机器索引作为纵坐标
                    duration_rounded,  # 加工时间作为条形宽度
                    left=start_time_rounded,  # 操作开始时间作为条形起始位置
                    color=color,  # 作业颜色
                    edgecolor='black',  # 条形边框颜色
                    label=f'Job {job_name}_{job_idx}' if i == 0 else ""  # 仅第一次添加图例
                )
                # 在条形中间添加标签
                label = (
                    f"Name {job_name}\n"  # 工件名字
                    f"Index {job_idx}\n"  # 工件序号
                    f"Op {i+1}\n"  # 工件序号和加工工序数
                    f"Start {start_time_rounded}\n"  # 开始时间
                    f"End {round(start_time_rounded + duration_rounded, 2)}\n"  # 结束时间
                    f"Dur {duration_rounded}"  # 加工时长
                )
                ax.text(
                    start_time_rounded + duration_rounded / 2,  # 标签的横坐标（条形中间）
                    machine_idx,  # 标签的纵坐标（机器索引）
                    label,  # 标签内容
                    ha='center',  # 水平居中
                    va='center',  # 垂直居中
                    fontsize=8,  # 字体大小
                    color='black'  # 字体颜色
                )
            else:
                # 将 cmt 拆分为 cmt - pt 和 pt
                pt_rounded = duration_rounded  # 假设 duration 是 pt
                cmt_minus_pt_rounded = round(cmt_rounded - pt_rounded, 2)  # cmt - pt 两位小数

                # 绘制 cmt - pt 部分（红色边框）
                ax.barh(
                    machine_idx,  # 机器索引作为纵坐标
                    cmt_minus_pt_rounded,  # cmt - pt 作为条形宽度
                    left=start_time_rounded,  # 操作开始时间作为条形起始位置
                    color='red',  # 作业颜色
                    edgecolor='black',  # 条形边框颜色
                )
                # # 在条形中间添加标签
                # label = (
                #     f"Name {job_name}\n"  # 工件名字
                #     f"Index {job_idx}\n"  # 工件序号
                #     f"Op {i+1}\n"  # 工件序号和加工工序数
                #     f"Start {start_time_rounded}\n"  # 开始时间
                #     f"End {round(start_time_rounded + cmt_minus_pt_rounded, 2)}\n"  # 结束时间
                #     f"Dur {cmt_minus_pt_rounded}"  # 加工时长
                # )
                # ax.text(
                #     start_time_rounded + cmt_minus_pt_rounded / 2,  # 标签的横坐标（条形中间）
                #     machine_idx,  # 标签的纵坐标（机器索引）
                #     label,  # 标签内容
                #     ha='center',  # 水平居中
                #     va='center',  # 垂直居中
                #     fontsize=8,  # 字体大小
                #     color='black'  # 字体颜色
                # )
                # 绘制 pt 部分
                ax.barh(
                    machine_idx,  # 机器索引作为纵坐标
                    pt_rounded,  # pt 作为条形宽度
                    left=round(start_time_rounded + cmt_minus_pt_rounded, 2),  # 操作开始时间 + cmt - pt
                    color=color,  # 作业颜色
                    edgecolor='black',  # 条形边框颜色
                    label=f'Job {job_name}_{job_idx}' if i == 0 else ""  # 仅第一次添加图例
                )
                # 在条形中间添加标签
                label = (
                    f"Name {job_name}\n"  # 工件名字
                    f"Index {job_idx}\n"  # 工件序号
                    f"Op {i+1}\n"  # 工件序号和加工工序数
                    f"Start {round(start_time_rounded + cmt_minus_pt_rounded, 2)}\n"  # 开始时间
                    f"End {round(start_time_rounded + cmt_rounded, 2)}\n"  # 结束时间
                    f"Dur {pt_rounded}"  # 加工时长
                )
                ax.text(
                    round(start_time_rounded + cmt_minus_pt_rounded + pt_rounded / 2, 2),  # 标签的横坐标（条形中间）
                    machine_idx,  # 标签的纵坐标（机器索引）
                    label,  # 标签内容
                    ha='center',  # 水平居中
                    va='center',  # 垂直居中
                    fontsize=8,  # 字体大小
                    color='black'  # 字体颜色
                )

    # 获取所有 machines 的最大值
    max_machine = len(m_list)

    # 设置纵坐标标签（机器索引）
    ax.set_yticks(range(max_machine))
    ax.set_yticklabels([f'Machine {i}' for i in range(max_machine)])

    # 设置横坐标标签（时间）
    ax.set_xlabel('Time')
    ax.set_ylabel('Machine')

    # 添加图例
    ax.legend(loc='upper right')

    # 设置标题
    ax.set_title('Gantt Chart')

    # 显示网格
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)

    # 显示图形
    plt.tight_layout()
    # 保存图片到本地文件
    plt.savefig('output.png')  # 保存为PNG格式，文件名是 output.png        
    plt.show()