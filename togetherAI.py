import os
from together import Together

client = Together(api_key="35ba5bebf6288e43fdc8989965161592e3335d7067c772c0c6995cdc0e60cd88")

conversation_history = []

# send a message and get a response
def ask_question(question, history):
    history.append({"role": "user", "content": question})
    
    response = client.chat.completions.create(
        model="meta-llama/Llama-3-70b-chat-hf",
        messages=history
    )
    
    answer = response.choices[0].message.content
    
    history.append({"role": "assistant", "content": answer})
    
    return answer, history


question_1 = """ I have a dataset containing training and inference jobs running state-of-the-art ML workloads. It is collected from a large production cluster with over 6,500 GPUs (on ~1800 machines) in Alibaba PAI (Platform for Artificial Intelligence).

I would like you to write Python code for a graph that I will specify as a task eventually. You have access to the following SQL data and utils functions. Please strictly use the provided data and functions when writing the code. The data is provided as a CSV file.:

pai_job_table:
job launch information.
Columns
Example Entry
job_name
4b3f04b66a525d2df903eb16
inst_id
8cb3bec23d14dbde320b6613452e768cbbf35b8bd64ee28fcceb77d3c47d
user
58540f191766
status
Terminated
start_time
4550879.0
end_time
4551416.0

job_name: name of users' submit jobs. It has been desensitized to protect users' privacy (similar to user_name, worker_name, inst_name, etc. below).
inst_id: please treat or revise it as job_id, since each job_name corresponds to one inst_id. It can be joined with inst_id in pai_sensor_table and pai_group_tag_table.
user: user name.
status: job status, including 'Running', 'Terminated', 'Failed', 'Waiting'; only 'Terminated' tasks are successful.
start_time: timestamp of job submission time.
end_time: timestamp of job completion time.
Note of time: Both start_time and end_time are in seconds and have been deducted by a constant number for desensitization. Still, if translated to Unix time in UTC+8 timezone ("Asia/Shanghai"), they will have the same time of the day (e.g., "08:59:59 UTC+8") and the same day of the week (e.g., "Sunday") as the original traces, while having fake dates, months, and years.
pai_task_table:
task launch information.
Columns
Example Entry
job_name
4f057009a5d481acec67b088
task_name
tensorflow
inst_num
1.0
status
Terminated
start_time
1739162.0
end_time
1739200.0
plan_cpu
600.0
plan_mem
29.296875
plan_gpu
50.0
gpu_type
MISC

job_name: job name; same as the entry in pai_job_table.
task_name: most jobs have only one task, but some may launch multiple tasks of different names (roles), e.g., ps, worker, evaluator.
inst_num: number of instances launched by the task.
status: task status.
start_time: timestamp of task launch time. The gap between job.start_time and the earliest task.start_time in the job implies its wait time before launching (scheduling latency).
end_time: timestamp of task completion time.
plan_cpu: number of CPU cores requested in percentage (i.e., 600.0 is 6 vCPU cores) .
plan_mem: GB of main memory requested.
plan_gpu: number of GPUs requested in percentage (i.e., 50.0 is 50% GPU).
gpu_type: type of GPUs assigned to this task. MISC is short for "miscellaneous", indicating GPUs of older generations, e.g., NVIDIA Tesla K40m, K80, M60.
pai_instance_table:
instance launch information.
Columns
Example Entry
job_name
af724763f4f5d0beef445849
task_name
worker
inst_name
0d39aa867a79c16eff67daa8f6248f09af8346b177c9e3e23645c48354a8
worker_name
54dbcd2db287841c03d0639b2a93e783a090ea085348f8cdb8e603d8b96f
inst_id
e387fbc18d80cc3c9ca4f1f13ff1d46778c9a25eaaeca2a95314fdf20d8e
status
Terminated
start_time
2081336.0
end_time
2083889.0
machine
471dda9ed84965451e042145

job_name: job name; same as the entry in pai_job_table.
task_name: task name; same as the entry in pai_task_table.
inst_name: name of instance in each task.
worker_name: information to distinguish instances; it is more detailed than inst_name and to be joined with worker_name in pai_sensor_table and pai_machine_metric.
inst_id: please treat or revise it as job_id, since each job_name corresponds to one inst_id; same as the entry in pai_job_table
status: instance status.
start_time: timestamp of instance launch time.
end_time: timestamp of instance completion time.
machine: the name of machine that the instance resides on, to be joined with machine in pai_machine_spec and pai_machine_metric.
pai_sensor_table:
instance resource sensor information.
Columns
Example Entry
job_name
a9449d475665e3bf0512520b
task_name
worker
worker_name
bcecec52225d6b4ae6bc724ce0269a02026195a364f54cf4850c2cca0054
inst_id
589de47b56f88129837f506134b874e0356dc0931732a687bcf907fb8325
machine
6884752e3565b15cafe14218
gpu_name
/dev/nvidia0
cpu_usage
140.1451612903226
gpu_wrk_util
16.0625
avg_mem
1.4627511160714286
max_mem
2.3935546875
avg_gpu_wrk_mem
1.2446746826171875
max_gpu_wrk_mem
2.3994140625
read
21271328.384615384
write
16376189.815384615
read_count
2922.4461538461537
write_count
3419.7846153846153

job_name: job name; same as the entry in pai_job_table.
task_name: task name; same as the entry in pai_task_table.
worker_name: worker name; same as the entry in pai_instance_table.
inst_id: please treat or revise it as job_id, since each job_name corresponds to one inst_id. Same as the entry in pai_job_table
machine: machine name; same as the entry in pai_instance_table.
gpu_name: name of the GPU on that machine (not gpu_type).
cpu_usage: number of CPU cores used in percentage (i.e., 600.0 is 6 vCPU cores) (c.f. plan_cpu in pai_task_table).
gpu_wrk_util: number of GPUs used in percentage (i.e., 50.0 is 50% GPU) (c.f., plan_gpu in pai_task_table).
avg_mem: GB of main memory used (in average) (c.f., plan_mem in pai_task_table).
max_mem: GB of main memory used (maximum) (c.f., plan_mem in pai_task_table).
avg_gpu_wrk_mem: GB of GPU memory used (in average).
max_gpu_wrk_mem: GB of GPU memory used (maximum).
read: Bytes of network input.
write: Bytes of network output.
read_count: Number of times of network read input.
write_count: Number of times of network write output.
Note of sensor: all the sensor metrics (CPU, GPU, Memory, I/O) in this table are collected for each instance (indexed by worker_name) but not task, taking the average of all data in the instance's lifetime (except for max_mem and max_gpu_wrk_mem being the maximum).
pai_group_tag_table:
instance semantic information.
Columns
Example Entry
inst_id
f7f6218cb5cb82e00b85476691d15d5055c143a351396d8f81737421dbd6
user
d2d3b77d342e
gpu_type_spec
V100M32
group
fbeb14d671c629b6e82bee889fe4bb4c
workload
nmt

inst_id: please treat or revise it as job_id, since each job_name corresponds to one inst_id. Same as the entry in pai_job_table
user: user name; same as the entry in pai_job_table.
gpu_type_spec: being empty if the instance does not specify GPU type requirements, else being one of the gpu_type in pai_task_table.
group: a semantic tag that indicates some instances have similar customized inputs, e.g., entry scripts, command-line parameters, data sources and sinks; consequently, instances with the same group tag are considered as repeated instances. Please refer to the trace analysis paper for detailed discussion.
workload: we study some Deep Learning tasks by investigating their customized inputs (mentioned above) and record their workload in this field; around 9% instances have this tag, including graphlearn, ctr, bert, etc.
pai_machine_spec:
machine specification.
Columns
Example Entry
machine
82e5af3cd5c4af3f56c62c54
gpu_type
T4
cap_cpu
96
cap_mem
512
cap_gpu
2

machine: machine name; same as the entry in pai_instance_table.
gpu_type: GPU type; same as the entry in pai_task_table.
cap_cpu: CPU capacity; number of CPU cores in the machine.
cap_mem: memory capacity; GB of main memory capacity in the machine.
cap_gpu: GPU capacity; number of GPU in the machine.
pai_machine_metric:
machine resource metrics with respect to the instance.
Columns
Example Entry
worker_name
b739d0d058e0db100aaf47e48a4d61320c95c2f5a334a8262d5e830d849c
machine
74e1c8457b01c76b314b22bb
start_time
6150435
end_time
6150689
machine_cpu_iowait
0.0028667003281999497
machine_cpu_kernel
3.583656890012642
machine_cpu_usr
14.928745108999438
machine_gpu
87.82875849911859
machine_load_1
18.298592909228066
machine_net_receive
111649584.57135652
machine_num_worker
5.053068410462776
machine_cpu
18.515268699340282

worker_name: worker name; same as the entry in pai_instance_table.
machine: machine name; same as the entry in pai_instance_table.
start_time: timestamp of instance launch time; same as the entry in pai_instance_table.
end_time: timestamp of instance completion; same as the entry in pai_instance_table.
machine_cpu_iowait : machine-level metrics of CPU I/O wait.
machine_cpu_kernel : machine-level metrics of CPU kernel usage.
machine_cpu_usr : machine-level metrics of CPU user usage.
machine_gpu : machine-level metrics of GPU utilization.
machine_load_1 : machine-level metrics of 1-min load average.
machine_net_receive : machine-level metrics of network received bytes.
machine_num_worker : machine-level metrics of number of co-located instances (workers).
machine_cpu : machine-level metrics of CPU overall usage.
Note of machine_ metrics: these metrics are machine-level metrics, taking average of the sensor data during the instance's (indexed by worker_name) lifetime.

I have joined this data as follows:
DATA_DIR = './'
dfj = get_df(DATA_DIR + 'pai_job_table.csv')
dft = get_df(DATA_DIR + 'pai_task_table.csv')
dfi = get_df(DATA_DIR + 'pai_instance_table.csv')
dfs = get_df(DATA_DIR + 'pai_sensor_table.csv')
dfg = get_df(DATA_DIR + 'pai_group_tag_table.csv')
dfp = get_df(DATA_DIR + 'pai_machine_spec.csv')
dfm = get_df(DATA_DIR + 'pai_machine_metric.csv')

# dfa: dataframe of tasks
dfa = get_dfa(dft, dfj, dfi, dfg)
#dfw: dataframe of workers
dfw = get_dfw(dfi, dft, dfg)
# dfws: DataFrame of Worker with sensor data
dfws = dfw.merge(dfp.drop(columns={'gpu_type'}), on='machine', how='left')
dfws = dfws.merge(dfs.drop(columns=['job_name','task_name','inst_id','machine']), on='worker_name')
# dfas: DataFrame of Task with sensor data
dfas = dfws.groupby(['job_name','task_name'])[['cpu_usage','gpu_wrk_util','avg_mem','avg_gpu_wrk_mem','plan_cpu','plan_gpu','plan_mem','cap_cpu','cap_gpu','cap_mem']].sum()
dfas = dfa.drop(columns=['plan_cpu','plan_mem','plan_gpu']).merge(dfas, on=['job_name','task_name'])

"Here are some functions you may need to use when generating the graphs:
import os
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

########### Data Constants ###########
DATA_DIR = '../data/'
if not os.access('/tmp/figures', os.F_OK):
    os.mkdir('/tmp/figures')
if not os.access('/tmp/figures', os.W_OK):
    print('Cannot write to /tmp/figures, please fix it.')
    exit()
else:
    print('figures saved to /tmp/figures')

########### Prepare Functions ###########
def get_df(file, header=None):
    df = pd.read_csv(file, header=None)
    # df.columns = DF_HEADER.get(key, df.columns)
    df.columns = pd.read_csv("{}.header".format(file.split('.csv')[0])).columns if header is None else header
    return df

def load_all_df():
    dfj = get_df(DATA_DIR + 'pai_job_table.csv')
    dft = get_df(DATA_DIR + 'pai_task_table.csv')
    dfi = get_df(DATA_DIR + 'pai_instance_table.csv')
    dfs = get_df(DATA_DIR + 'pai_sensor_table.csv')
    dfg = get_df(DATA_DIR + 'pai_group_tag_table.csv')
    dfp = get_df(DATA_DIR + 'pai_machine_spec.csv')
    dfm = get_df(DATA_DIR + 'pai_machine_metric.csv')
    return dfj,dft,dfi,dfs,dfg,dfp,dfm

def get_dfiw(dfi):
    dfiw = dfi.sort_values(['status','start_time','end_time'])
    dfiw.drop_duplicates(subset=['worker_name'], keep='last', inplace=True)
    dfiw.dropna(subset=['worker_name'], inplace=True)
    dfiw['runtime'] = dfiw[(dfiw.start_time>0)&(dfiw.end_time>0)]['end_time'] \
                    - dfiw[(dfiw.start_time>0)&(dfiw.end_time>0)]['start_time']
    dfiw.loc[dfiw.start_time==0, 'start_time'] = np.nan
    dfiw.loc[dfiw.start_time==0, 'end_time'] = np.nan
    return dfiw

def get_dfw(dfi, dft, dfg):
    dfw = get_dfiw(dfi)
    dfw['start_date']=dfw.start_time.apply(pd.Timestamp, unit='s', tz='Asia/Shanghai')
    print('dfi + dft ...')
    dfw = dfw.merge(dft, on=['job_name','task_name'], how='left', suffixes=['', '_t'])
    print('dfi + dft + dfg ...')
    dfw = dfw.merge(dfg, on='inst_id', how='left')  # reserve NaN ones by how='left'
    dfw.loc[dfw.group.isnull(),'group'] = dfw.loc[dfw.group.isnull(), 'user']  # fill group==NaN ones with user
    return dfw

def get_dfia(dfi):
    dfi_s = dfi[dfi.start_time > 0][['job_name','task_name','start_time']].groupby(['job_name','task_name']).min()  # start_time
    dfi_e = dfi[dfi.end_time > 0][['job_name','task_name','end_time']].groupby(['job_name','task_name']).max()  # end_time
    dfi_m = dfi[(dfi.start_time > 0) & (dfi.end_time > 0)][['job_name','task_name','end_time','start_time']]
    dfi_m['runtime'] = dfi_m.end_time-dfi_m.start_time
    dfi_m = dfi_m.groupby(['job_name','task_name']).mean()[['runtime']].reset_index() # runtime
    dfi_u = dfi[['job_name','task_name','status']].drop_duplicates().groupby(['job_name','task_name']).max() # status
    dfia = dfi_u
    for df in [dfi_s, dfi_e, dfi_m]:
        dfia = dfia.merge(df, on=['job_name','task_name'], how='left')
    return dfia

def get_dfa(dft, dfj, dfi, dfg):
    print('dft + dfj ...')
    dfa = dft.merge(dfj, on=['job_name'], suffixes = ['','_j'])
    dfa.loc[dfa.start_time==0, 'start_time'] = np.nan
    dfa.loc[dfa.start_time==0, 'end_time'] = np.nan
    dfa['runtime'] = dfa.end_time - dfa.start_time
    print('dft + dfj + dfi ...')
    dfia = get_dfia(dfi)
    dfa = dfa.merge(dfia, on=['job_name','task_name'], suffixes=['','_i'])
    dfa['duration_min'] = dfa.runtime_i / 60  # duration of instances
    dfa['wait_time'] = dfa.start_time_i - dfa.start_time # task wait time
    dfa['start_date']=dfa.start_time.apply(pd.Timestamp, unit='s', tz='Asia/Shanghai') # task start time
    # dfa = dfa[dfa.status=='Terminated']
    print('dft + dfj + dfi + dfg ...')
    dfa = dfa.merge(dfg[[x for x in dfg.columns if x != 'user']], on='inst_id', how='left')  # reserve NaN ones by how='left'
    dfa.loc[dfa.group.isnull(),'group'] = dfa.loc[dfa.group.isnull(), 'user']  # fill group==NaN ones with user
    return dfa

def get_dfwitm(dfwit, csv_file='intermediate_data/machine_metric_shennong_machine_all.csv'):
    res_df = pd.read_csv(csv_file, index_col=0)
    dfwitm = dfwit.merge(res_df.loc[:, ~res_df.columns.isin(['start_time','end_time','machine'])], on='worker_name', how='left')
    return dfwitm

########### Plot Functions ###########
linestyle_list = [
     ('solid', 'solid'),       # Same as (0, ()) or '-'
     ('dotted', 'dotted'),     # Same as (0, (1, 1)) or '.'
     ('dashed', 'dashed'),     # Same as '--'
     ('dashdot', 'dashdot'),   # Same as '-.'
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),
     ('densely dotted',        (0, (1, 1))),
     ('densely dashed',        (0, (5, 1))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashed',        (0, (5, 10))),
     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('loosely dotted',        (0, (1, 10))),
     ('dashed',                (0, (5, 5))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('dotted',                (0, (1, 1))),
]

def get_cdf(data, inverse=False):
    sorted_data = sorted(data)
    p = 100. * np.arange(len(sorted_data))/(len(sorted_data)-1)
    p = 100. - p if inverse else p # CCDF
    return sorted_data, p

def plot_data_cdf(data, inverse=False, datalabel=None, xlabel=None, title=None, xlog=False, xlim=None, ylog=False, xticks=None, figsize=(4,3), dpi=120, savefig=None, ylabel=None):
    plt.figure(figsize=figsize, dpi=dpi)
    if type(data) == pd.DataFrame:
        data.dropna(inplace=True)
    x, y = get_cdf(data, inverse)
    plt.plot(x, y, label=datalabel, color='green', linestyle='-')
    if datalabel is not None: plt.legend(loc='lower right')
    if xlog: plt.xscale('log')
    if ylog: plt.yscale('log')
    if xlim is not None: plt.xlim(xlim)
    plt.ylim(0, 100)
    if xlabel is not None: plt.xlabel(xlabel)
    plt.ylabel(ylabel) if ylabel is not None else plt.ylabel('CCDF') if inverse is True else plt.ylabel('CDF')
    if title is not None: plt.title(title)
    if xticks is not None: plt.xticks(xticks)
    plt.grid(alpha=.3, linestyle='--')
    if savefig is not None:
        plt.savefig('/tmp/figures/{}.pdf'.format(savefig),bbox_inches='tight')
    else:
        plt.show()

def plot_data_cdfs(data, datalabel=None, inverse=False, xlabel=None, title=None, xlog=False, ylog=False, xticks=None, figsize=(4,3), dpi=120, xlim=None, ylim=None, ylabel=None, yticks=None, savefig=None, loc='best', fontsize=None):
    plt.figure(figsize=figsize, dpi=dpi)
    for i, d in enumerate(data):
        if type(data) == pd.DataFrame:
            d.dropna(inplace=True)
        x, y = get_cdf(d, inverse)
        label = datalabel[i] if datalabel is not None else None
        plt.plot(x, y, label=label, linestyle=linestyle_list[i % len(linestyle_list)][1])
    if datalabel is not None: plt.legend(loc=loc, fontsize=fontsize)
    if xlog: plt.xscale('log')
    if ylog: plt.yscale('log')
    plt.ylim(0, 100) if ylim is None else plt.ylim(ylim)
    if xlim is not None: plt.xlim(xlim)
    if xlabel is not None: plt.xlabel(xlabel)
    if ylabel is None:
        plt.ylabel('CCDF') if inverse is True else plt.ylabel('CDF')
    else:
        plt.ylabel(ylabel)
    if title is not None: plt.title(title)
    if xticks is not None: plt.xticks(xticks)
    if yticks is not None: plt.yticks(yticks)
    plt.grid(alpha=.3, linestyle='--')
    if savefig is not None:
        plt.savefig('/tmp/figures/{}.pdf'.format(savefig),bbox_inches='tight')
    else:
        plt.show()

def draw_bar_plot(odf, col, figsize=(4,4), dpi=120, portion=False, title=None, limit=30):
    dfout=odf.reset_index().groupby(col).count()[['index']].sort_values('index', ascending=False).head(limit)
    dfout['portion'] = 100 * dfout['index'] / dfout['index'].sum()
    plt.figure(figsize=figsize, dpi=dpi)
    if portion:
        plt.barh(y=dfout.index, width=dfout['portion'])
        plt.xlabel('Percentage (total: %.2f)'%(dfout['index'].sum()))
    else:
        plt.barh(y=dfout.index, width=dfout['index'])
    plt.grid(alpha=.3, linestyle='--')
    return dfout

########### Process Functions ###########

def get_inst_task_num_ratio(dfa, inst_num_list=[2, 8, 20, 64, 100, 256, 512]):
    total_num_task, total_num_inst = len(dfa), sum(dfa['inst_num'])
    data_df = []
    for i in inst_num_list:
        temp_df = dfa[dfa['inst_num'] >= i]
        task_num_ratio = len(temp_df) / total_num_task
        inst_num_ratio = sum(temp_df['inst_num']) / total_num_inst
        data_df.append([task_num_ratio, inst_num_ratio])
    out_df = pd.DataFrame(data_df, columns=['num_task_ratio','num_inst_ratio'])
    out_df = out_df.T.rename(columns=dict(zip(range(len(inst_num_list)), inst_num_list)))
    return out_df


def add_hour_date(df):
    if 'start_date' not in df:
        if 'start_time_t' in df:
            target_col = 'start_time_t'
        elif 'start_time' in df:
            target_col = 'start_time'
        else:
            print('start_time, start_time_t, dayofyear unfound in df')
            return None
        df['start_date'] = df[target_col].apply(lambda x: pd.Timestamp(x, unit='s', tz='Asia/Shanghai'))
    if 'date' not in df:
        df['date'] = df['start_date'].apply(lambda x: x.date())
    if 'hour' not in df:
        df['hour'] = df['start_date'].apply(lambda x: x.hour)
    return df

def get_hourly_task_request(df): # df = dftjkix
    sum_df_list = []
    df = add_hour_date(df.copy())
    # for day in sorted(df.dayofyear.unique()):
    for date in sorted(df.date.unique()):
        # tempdf = df[df.dayofyear==day]
        tempdf = df[df.date==date]
        res_df = tempdf.groupby('hour').count()[['job_name']]
        res_df.rename(columns={'job_name':date}, inplace=True)
        sum_df_list.append(res_df.T)
    out_df = pd.DataFrame().append(sum_df_list)
    return out_df.dropna() # if a day contains hours of NaN, it is not a typical day

def get_hourly_task_resource_request(df, metrics='cpu'): # df = dftjkix
    sum_df_list = []
    df = add_hour_date(df)
    if metrics == 'cpu':
        df['plan_resource'] = df.plan_cpu.apply(lambda x: x/100)
    elif metrics == 'gpu':
        df['plan_resource'] = df.plan_gpu.apply(lambda x: x/100)
    elif metrics == 'mem':
        df['plan_resource'] = df.plan_mem.apply(lambda x: x/1000)
    else:
        exit()
    # for day in sorted(df.dayofyear.unique()):
    for date in sorted(df.date.unique()):
        # tempdf = df[df.dayofyear==day]
        tempdf = df[df.date==date]
        res_df = tempdf.groupby('hour').sum()[['plan_resource']]
        res_df.rename(columns={'job_name':date}, inplace=True)
        sum_df_list.append(res_df.T)
    out_df = pd.DataFrame().append(sum_df_list)
    return out_df.dropna() # if a day contains hours of NaN, it is not a typical day

def plan_minus_usg_over_cap_task(dfas):
    dfas['plan_gpu_minus_usage_over_capacity'] = (dfas['plan_gpu'] - dfas['gpu_wrk_util']) / (100 * dfas['cap_gpu'])
    dfas['plan_cpu_minus_usage_over_capacity'] = (dfas['plan_cpu'] - dfas['cpu_usage']) / (100 * dfas['cap_cpu'] )
    dfas['plan_mem_minus_usage_over_capacity'] = (dfas['plan_mem'] - dfas['avg_mem']) / dfas['cap_mem']

    dfas_task = dfas.groupby(['job_name','task_name'])[['plan_gpu_minus_usage_over_capacity','plan_cpu_minus_usage_over_capacity','plan_mem_minus_usage_over_capacity']].mean()

    pgu_datas, pgu_label, ugp_datas, ugp_label = [], [], [], []
    for device in ['cpu','gpu','mem']:
        apu = dfas_task[~dfas_task['plan_{}_minus_usage_over_capacity'.format(device)].isnull()]
        pgu = dfas_task[dfas_task['plan_{}_minus_usage_over_capacity'.format(device)] > 0]
        ugp = dfas_task[dfas_task['plan_{}_minus_usage_over_capacity'.format(device)] < 0]
        print("{}: plan > usage: {:.2f}%, plan < usage: {:.2f}%".format(
            device, 100 * len(pgu) / len(apu), 100 * len(ugp) / len(apu)    ))
        pgu_label.append("{} {:.2f}%".format(device, 100 * len(pgu) / len(apu)))
        pgu_datas.append(pgu['plan_{}_minus_usage_over_capacity'.format(device)])
        ugp_label.append("{} {:.2f}%".format(device, 100 * len(ugp) / len(apu)))
        ugp_datas.append(-ugp['plan_{}_minus_usage_over_capacity'.format(device)])

    return pgu_datas, ugp_datas, pgu_label, ugp_label

Here is an example. I am using this data to create a graph that has CDF on the y-axis and ‘Num of tasks submitted per User’ on the x-axis. This is the code:
user_task_count = dfa.groupby('user').count()[['job_name']]

plt.figure(figsize=(4,3), dpi=120)
plot_data_cdf(user_task_count['job_name'], xlog=True,
              ylabel='CDF',xlabel='Num of tasks submitted per User',
              xticks=[1,10,100,10**3,10**4,10**5])

Now I will ask you a series of tasks, please remember to use any functions/dataframes you see fit that we have defined so far. Here is your first task: generate code for a graph that has ‘Num of Instances in the Task” on the x-axis and ‘CCDF’ on the y-axis.  Use existing functions and convert start_date to hour of the week ensure that you don't get the following error: TypeError: Cannot compare NaT with datetime.date object"""
answer_1, conversation_history = ask_question(question_1, conversation_history)
print("Assistant A1:", answer_1)

question_x = "Please generate code for a graph that has ‘Num of Instances in the Task” on the x-axis and 'PDF' on the y-axis.  Use existing functions and convert start_date to hour of the week ensure that you don't get the following error: TypeError: Cannot compare NaT with datetime.date object"
answer_x, conversation_history = ask_question(question_x, conversation_history)
print("Assistant Ax:", answer_x)


# question_2 = "Your next task: Generate code for a line-plot graph with ‘'Hours from the beginning of a week (Sun. to Sat.)' on the x-axis and 'Num of tasks' on the y-axis. The plot should show the number of tasks submitted at each hour of the day, with the x-axis ranging from 0 to 168 (representing 7 days). The data is stored in a dataframe, `dfa` ,with columns `start_date` and `job_name`. The `start_date` column should be converted to a numerical representation of the hour of the year, and then grouped by hour to calculate the number of tasks."
# answer_2, conversation_history = ask_question(question_2, conversation_history)
# print("Assistant A2:", answer_2)

# question_3 = "Your next task: generate Python code to create a boxplot showing the number of tasks submitted per hour of the day using the dfa dataframe, with hours on the x-axis and number of tasks on the y-axis. The boxplot should have a custom color palette with hours below the mean value in a default color and hours above the mean value in a highlighted color. The graph should also have a grid, custom y-ticks, and a specific font size. Finally, save the graph as a PDF file in the /tmp/figures directory."
# answer_3, conversation_history = ask_question(question_3, conversation_history)
# print("Assistant A3:", answer_3)

# question_4 = "Your next task: Write Python code to generate a graph that has 'Total resource requests' on the y-axis and 'Hours from the beginning of a week (Sun. to Sat.)' on the x-axis. The graph should show the total resource requests for CPU, GPU, and Memory over a period of 7 days. The data for resource requests is available in the dataframe dfw. The x-axis should be divided into hours, and the y-axis should be divided into appropriate units (e.g., 10 CPU cores, 1 GPU, 100 GB Memory)."
# answer_4, conversation_history = ask_question(question_4, conversation_history)
# print("Assistant A4:", answer_4)

# question_5 = """Your next task: Generate a Python code to plot the CDF of task wait times for different GPU usage ranges. The ranges are: less than 25%%, between 25%% and 50%%, between 50%% and 100%%, exactly 100%%, and more than 100%%. Use the `dfa` dataframe and plot the CDFs with logarithmic x-axis, x-ticks at [1, 10, 100, 1000, 10^4, 10^5], and x-limit from 1 to 10^5. Label the x-axis as 'Task wait time (sec)' and provide a legend with the corresponding GPU usage ranges. Ensure that any dataframe fields/keys you use are defined, I've listed the available columns below for you.
# Here are the columns in dfa: ['job_name', 'task_name', 'inst_num', 'status', 'start_time', 'end_time', 'plan_cpu', 'plan_mem', 'plan_gpu', 'gpu_type', 'inst_id', 'user', 'status_j', 'start_time_j', 'end_time_j', 'runtime', 'status_i', 'start_time_i', 'end_time_i', 'runtime_i', 'duration_min', 'wait_time', 'start_date', 'gpu_type_spec', 'group', 'workload']
# Please first make a copy of the data frame like data_df = dfa."""
# answer_5, conversation_history = ask_question(question_5, conversation_history)
# print("Assistant A5:", answer_5)

# question_6 = """Your nexy task: Can you write a Python code using the plot_data_cdfs function to plot a line graph with CCDF on the y-axis and 'Task wait time (sec)' on the x-axis, comparing the wait times of different GPU types (T4, MISC, P100, V100, V100M32) in the dfa dataframe, with a logarithmic x-axis and x-ticks at [1, 10, 100, 1000, 10**4, 10**5]? Please first make a copy of the data frame like data_df = dfa."""
# answer_6, conversation_history = ask_question(question_6, conversation_history)
# print("Assistant A6:", answer_6)

# question_7 = "Can you generate a Python code to plot the CDF of 'plan_cpu' and 'cpu_usage' columns from the 'dfas' dataframe, with 'CPU Request' and 'CPU Usage' as labels, 'percent of CPU' as the x-axis label from 0 to 6000, 'CDF' as the y-axis label, and specific x-axis ticks and limits?"
# answer_7, conversation_history = ask_question(question_7, conversation_history)
# print("Assistant A7:", answer_7)

# question_8 = "generate Python code to plot the CDF of GPU Request and GPU usage from 'dfas' dataframe. Have 'percent of GPU' as the x-axis label from 0 to 600, 'CDF' as the y-axis label, and specific x-axis ticks and limits. Plot it on the same graph."
# answer_8, conversation_history = ask_question(question_8, conversation_history)
# print("Assistant A8:", answer_8)

# question_9 = "generate Python code to plot the CDF of Mem Request and Mem usage from 'dfas' dataframe. Have 'GB of Main Memory' as the x-axis label, 'CDF' as the y-axis label, and specific x-axis ticks and limits. Plot both on the same graph."
# answer_9, conversation_history = ask_question(question_9, conversation_history)
# print("Assistant A9:", answer_9)


# import os
# from together import Together

# client = Together(api_key="35ba5bebf6288e43fdc8989965161592e3335d7067c772c0c6995cdc0e60cd88")

# conversation_history = []

# # send a message and get a response
# def ask_question(question, history):
#     history.append({"role": "user", "content": question})
    
#     response = client.chat.completions.create(
#         model="meta-llama/Llama-3-70b-chat-hf",
#         messages=history
#     )
    
#     answer = response.choices[0].message.content
    
#     history.append({"role": "assistant", "content": answer})
    
#     return answer, history


# context_1 = """ I have a dataset containing training and inference jobs running state-of-the-art ML workloads. It is collected from a large production cluster with over 6,500 GPUs (on ~1800 machines) in Alibaba PAI (Platform for Artificial Intelligence).

# I would like you to write Python code for a graph that I will specify as a task eventually. You have access to the following SQL data and utils functions. Please strictly use the provided data and functions when writing the code. The data is provided as a CSV file.:

# pai_job_table:
# job launch information.
# Columns
# Example Entry
# job_name
# 4b3f04b66a525d2df903eb16
# inst_id
# 8cb3bec23d14dbde320b6613452e768cbbf35b8bd64ee28fcceb77d3c47d
# user
# 58540f191766
# status
# Terminated
# start_time
# 4550879.0
# end_time
# 4551416.0

# job_name: name of users' submit jobs. It has been desensitized to protect users' privacy (similar to user_name, worker_name, inst_name, etc. below).
# inst_id: please treat or revise it as job_id, since each job_name corresponds to one inst_id. It can be joined with inst_id in pai_sensor_table and pai_group_tag_table.
# user: user name.
# status: job status, including 'Running', 'Terminated', 'Failed', 'Waiting'; only 'Terminated' tasks are successful.
# start_time: timestamp of job submission time.
# end_time: timestamp of job completion time.
# Note of time: Both start_time and end_time are in seconds and have been deducted by a constant number for desensitization. Still, if translated to Unix time in UTC+8 timezone ("Asia/Shanghai"), they will have the same time of the day (e.g., "08:59:59 UTC+8") and the same day of the week (e.g., "Sunday") as the original traces, while having fake dates, months, and years.
# pai_task_table:
# task launch information.
# Columns
# Example Entry
# job_name
# 4f057009a5d481acec67b088
# task_name
# tensorflow
# inst_num
# 1.0
# status
# Terminated
# start_time
# 1739162.0
# end_time
# 1739200.0
# plan_cpu
# 600.0
# plan_mem
# 29.296875
# plan_gpu
# 50.0
# gpu_type
# MISC

# job_name: job name; same as the entry in pai_job_table.
# task_name: most jobs have only one task, but some may launch multiple tasks of different names (roles), e.g., ps, worker, evaluator.
# inst_num: number of instances launched by the task.
# status: task status.
# start_time: timestamp of task launch time. The gap between job.start_time and the earliest task.start_time in the job implies its wait time before launching (scheduling latency).
# end_time: timestamp of task completion time.
# plan_cpu: number of CPU cores requested in percentage (i.e., 600.0 is 6 vCPU cores) .
# plan_mem: GB of main memory requested.
# plan_gpu: number of GPUs requested in percentage (i.e., 50.0 is 50% GPU).
# gpu_type: type of GPUs assigned to this task. MISC is short for "miscellaneous", indicating GPUs of older generations, e.g., NVIDIA Tesla K40m, K80, M60.
# pai_instance_table:
# instance launch information.
# Columns
# Example Entry
# job_name
# af724763f4f5d0beef445849
# task_name
# worker
# inst_name
# 0d39aa867a79c16eff67daa8f6248f09af8346b177c9e3e23645c48354a8
# worker_name
# 54dbcd2db287841c03d0639b2a93e783a090ea085348f8cdb8e603d8b96f
# inst_id
# e387fbc18d80cc3c9ca4f1f13ff1d46778c9a25eaaeca2a95314fdf20d8e
# status
# Terminated
# start_time
# 2081336.0
# end_time
# 2083889.0
# machine
# 471dda9ed84965451e042145

# job_name: job name; same as the entry in pai_job_table.
# task_name: task name; same as the entry in pai_task_table.
# inst_name: name of instance in each task.
# worker_name: information to distinguish instances; it is more detailed than inst_name and to be joined with worker_name in pai_sensor_table and pai_machine_metric.
# inst_id: please treat or revise it as job_id, since each job_name corresponds to one inst_id; same as the entry in pai_job_table
# status: instance status.
# start_time: timestamp of instance launch time.
# end_time: timestamp of instance completion time.
# machine: the name of machine that the instance resides on, to be joined with machine in pai_machine_spec and pai_machine_metric.
# pai_sensor_table:
# instance resource sensor information.
# Columns
# Example Entry
# job_name
# a9449d475665e3bf0512520b
# task_name
# worker
# worker_name
# bcecec52225d6b4ae6bc724ce0269a02026195a364f54cf4850c2cca0054
# inst_id
# 589de47b56f88129837f506134b874e0356dc0931732a687bcf907fb8325
# machine
# 6884752e3565b15cafe14218
# gpu_name
# /dev/nvidia0
# cpu_usage
# 140.1451612903226
# gpu_wrk_util
# 16.0625
# avg_mem
# 1.4627511160714286
# max_mem
# 2.3935546875
# avg_gpu_wrk_mem
# 1.2446746826171875
# max_gpu_wrk_mem
# 2.3994140625
# read
# 21271328.384615384
# write
# 16376189.815384615
# read_count
# 2922.4461538461537
# write_count
# 3419.7846153846153

# job_name: job name; same as the entry in pai_job_table.
# task_name: task name; same as the entry in pai_task_table.
# worker_name: worker name; same as the entry in pai_instance_table.
# inst_id: please treat or revise it as job_id, since each job_name corresponds to one inst_id. Same as the entry in pai_job_table
# machine: machine name; same as the entry in pai_instance_table.
# gpu_name: name of the GPU on that machine (not gpu_type).
# cpu_usage: number of CPU cores used in percentage (i.e., 600.0 is 6 vCPU cores) (c.f. plan_cpu in pai_task_table).
# gpu_wrk_util: number of GPUs used in percentage (i.e., 50.0 is 50% GPU) (c.f., plan_gpu in pai_task_table).
# avg_mem: GB of main memory used (in average) (c.f., plan_mem in pai_task_table).
# max_mem: GB of main memory used (maximum) (c.f., plan_mem in pai_task_table).
# avg_gpu_wrk_mem: GB of GPU memory used (in average).
# max_gpu_wrk_mem: GB of GPU memory used (maximum).
# read: Bytes of network input.
# write: Bytes of network output.
# read_count: Number of times of network read input.
# write_count: Number of times of network write output.
# Note of sensor: all the sensor metrics (CPU, GPU, Memory, I/O) in this table are collected for each instance (indexed by worker_name) but not task, taking the average of all data in the instance's lifetime (except for max_mem and max_gpu_wrk_mem being the maximum).
# pai_group_tag_table:
# instance semantic information.
# Columns
# Example Entry
# inst_id
# f7f6218cb5cb82e00b85476691d15d5055c143a351396d8f81737421dbd6
# user
# d2d3b77d342e
# gpu_type_spec
# V100M32
# group
# fbeb14d671c629b6e82bee889fe4bb4c
# workload
# nmt

# inst_id: please treat or revise it as job_id, since each job_name corresponds to one inst_id. Same as the entry in pai_job_table
# user: user name; same as the entry in pai_job_table.
# gpu_type_spec: being empty if the instance does not specify GPU type requirements, else being one of the gpu_type in pai_task_table.
# group: a semantic tag that indicates some instances have similar customized inputs, e.g., entry scripts, command-line parameters, data sources and sinks; consequently, instances with the same group tag are considered as repeated instances. Please refer to the trace analysis paper for detailed discussion.
# workload: we study some Deep Learning tasks by investigating their customized inputs (mentioned above) and record their workload in this field; around 9% instances have this tag, including graphlearn, ctr, bert, etc.
# pai_machine_spec:
# machine specification.
# Columns
# Example Entry
# machine
# 82e5af3cd5c4af3f56c62c54
# gpu_type
# T4
# cap_cpu
# 96
# cap_mem
# 512
# cap_gpu
# 2

# machine: machine name; same as the entry in pai_instance_table.
# gpu_type: GPU type; same as the entry in pai_task_table.
# cap_cpu: CPU capacity; number of CPU cores in the machine.
# cap_mem: memory capacity; GB of main memory capacity in the machine.
# cap_gpu: GPU capacity; number of GPU in the machine.
# pai_machine_metric:
# machine resource metrics with respect to the instance.
# Columns
# Example Entry
# worker_name
# b739d0d058e0db100aaf47e48a4d61320c95c2f5a334a8262d5e830d849c
# machine
# 74e1c8457b01c76b314b22bb
# start_time
# 6150435
# end_time
# 6150689
# machine_cpu_iowait
# 0.0028667003281999497
# machine_cpu_kernel
# 3.583656890012642
# machine_cpu_usr
# 14.928745108999438
# machine_gpu
# 87.82875849911859
# machine_load_1
# 18.298592909228066
# machine_net_receive
# 111649584.57135652
# machine_num_worker
# 5.053068410462776
# machine_cpu
# 18.515268699340282

# worker_name: worker name; same as the entry in pai_instance_table.
# machine: machine name; same as the entry in pai_instance_table.
# start_time: timestamp of instance launch time; same as the entry in pai_instance_table.
# end_time: timestamp of instance completion; same as the entry in pai_instance_table.
# machine_cpu_iowait : machine-level metrics of CPU I/O wait.
# machine_cpu_kernel : machine-level metrics of CPU kernel usage.
# machine_cpu_usr : machine-level metrics of CPU user usage.
# machine_gpu : machine-level metrics of GPU utilization.
# machine_load_1 : machine-level metrics of 1-min load average.
# machine_net_receive : machine-level metrics of network received bytes.
# machine_num_worker : machine-level metrics of number of co-located instances (workers).
# machine_cpu : machine-level metrics of CPU overall usage.
# Note of machine_ metrics: these metrics are machine-level metrics, taking average of the sensor data during the instance's (indexed by worker_name) lifetime.

# I have joined this data as follows:
# DATA_DIR = './'
# dfj = get_df(DATA_DIR + 'pai_job_table.csv')
# dft = get_df(DATA_DIR + 'pai_task_table.csv')
# dfi = get_df(DATA_DIR + 'pai_instance_table.csv')
# dfs = get_df(DATA_DIR + 'pai_sensor_table.csv')
# dfg = get_df(DATA_DIR + 'pai_group_tag_table.csv')
# dfp = get_df(DATA_DIR + 'pai_machine_spec.csv')
# dfm = get_df(DATA_DIR + 'pai_machine_metric.csv')

# # dfa: dataframe of tasks
# dfa = get_dfa(dft, dfj, dfi, dfg)
# #dfw: dataframe of workers
# dfw = get_dfw(dfi, dft, dfg)
# # dfws: DataFrame of Worker with sensor data
# dfws = dfw.merge(dfp.drop(columns={'gpu_type'}), on='machine', how='left')
# dfws = dfws.merge(dfs.drop(columns=['job_name','task_name','inst_id','machine']), on='worker_name')
# # dfas: DataFrame of Task with sensor data
# dfas = dfws.groupby(['job_name','task_name'])[['cpu_usage','gpu_wrk_util','avg_mem','avg_gpu_wrk_mem','plan_cpu','plan_gpu','plan_mem','cap_cpu','cap_gpu','cap_mem']].sum()
# dfas = dfa.drop(columns=['plan_cpu','plan_mem','plan_gpu']).merge(dfas, on=['job_name','task_name'])

# Can you tell me what the different dfs are what their columns are/represent?"""

# context_ans_1, conversation_history = ask_question(context_1, conversation_history)
# print("Assistant:", context_ans_1)

# context_2 = """Here are some functions you may need to use when generating the graphs:
# import os
# import datetime
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib
# import matplotlib.pyplot as plt

# ########### Data Constants ###########
# DATA_DIR = '../data/'
# if not os.access('/tmp/figures', os.F_OK):
#     os.mkdir('/tmp/figures')
# if not os.access('/tmp/figures', os.W_OK):
#     print('Cannot write to /tmp/figures, please fix it.')
#     exit()
# else:
#     print('figures saved to /tmp/figures')

# ########### Prepare Functions ###########
# def get_df(file, header=None):
#     df = pd.read_csv(file, header=None)
#     # df.columns = DF_HEADER.get(key, df.columns)
#     df.columns = pd.read_csv("{}.header".format(file.split('.csv')[0])).columns if header is None else header
#     return df

# def load_all_df():
#     dfj = get_df(DATA_DIR + 'pai_job_table.csv')
#     dft = get_df(DATA_DIR + 'pai_task_table.csv')
#     dfi = get_df(DATA_DIR + 'pai_instance_table.csv')
#     dfs = get_df(DATA_DIR + 'pai_sensor_table.csv')
#     dfg = get_df(DATA_DIR + 'pai_group_tag_table.csv')
#     dfp = get_df(DATA_DIR + 'pai_machine_spec.csv')
#     dfm = get_df(DATA_DIR + 'pai_machine_metric.csv')
#     return dfj,dft,dfi,dfs,dfg,dfp,dfm

# def get_dfiw(dfi):
#     dfiw = dfi.sort_values(['status','start_time','end_time'])
#     dfiw.drop_duplicates(subset=['worker_name'], keep='last', inplace=True)
#     dfiw.dropna(subset=['worker_name'], inplace=True)
#     dfiw['runtime'] = dfiw[(dfiw.start_time>0)&(dfiw.end_time>0)]['end_time'] \
#                     - dfiw[(dfiw.start_time>0)&(dfiw.end_time>0)]['start_time']
#     dfiw.loc[dfiw.start_time==0, 'start_time'] = np.nan
#     dfiw.loc[dfiw.start_time==0, 'end_time'] = np.nan
#     return dfiw

# def get_dfw(dfi, dft, dfg):
#     dfw = get_dfiw(dfi)
#     dfw['start_date']=dfw.start_time.apply(pd.Timestamp, unit='s', tz='Asia/Shanghai')
#     print('dfi + dft ...')
#     dfw = dfw.merge(dft, on=['job_name','task_name'], how='left', suffixes=['', '_t'])
#     print('dfi + dft + dfg ...')
#     dfw = dfw.merge(dfg, on='inst_id', how='left')  # reserve NaN ones by how='left'
#     dfw.loc[dfw.group.isnull(),'group'] = dfw.loc[dfw.group.isnull(), 'user']  # fill group==NaN ones with user
#     return dfw

# def get_dfia(dfi):
#     dfi_s = dfi[dfi.start_time > 0][['job_name','task_name','start_time']].groupby(['job_name','task_name']).min()  # start_time
#     dfi_e = dfi[dfi.end_time > 0][['job_name','task_name','end_time']].groupby(['job_name','task_name']).max()  # end_time
#     dfi_m = dfi[(dfi.start_time > 0) & (dfi.end_time > 0)][['job_name','task_name','end_time','start_time']]
#     dfi_m['runtime'] = dfi_m.end_time-dfi_m.start_time
#     dfi_m = dfi_m.groupby(['job_name','task_name']).mean()[['runtime']].reset_index() # runtime
#     dfi_u = dfi[['job_name','task_name','status']].drop_duplicates().groupby(['job_name','task_name']).max() # status
#     dfia = dfi_u
#     for df in [dfi_s, dfi_e, dfi_m]:
#         dfia = dfia.merge(df, on=['job_name','task_name'], how='left')
#     return dfia

# def get_dfa(dft, dfj, dfi, dfg):
#     print('dft + dfj ...')
#     dfa = dft.merge(dfj, on=['job_name'], suffixes = ['','_j'])
#     dfa.loc[dfa.start_time==0, 'start_time'] = np.nan
#     dfa.loc[dfa.start_time==0, 'end_time'] = np.nan
#     dfa['runtime'] = dfa.end_time - dfa.start_time
#     print('dft + dfj + dfi ...')
#     dfia = get_dfia(dfi)
#     dfa = dfa.merge(dfia, on=['job_name','task_name'], suffixes=['','_i'])
#     dfa['duration_min'] = dfa.runtime_i / 60  # duration of instances
#     dfa['wait_time'] = dfa.start_time_i - dfa.start_time # task wait time
#     dfa['start_date']=dfa.start_time.apply(pd.Timestamp, unit='s', tz='Asia/Shanghai') # task start time
#     # dfa = dfa[dfa.status=='Terminated']
#     print('dft + dfj + dfi + dfg ...')
#     dfa = dfa.merge(dfg[[x for x in dfg.columns if x != 'user']], on='inst_id', how='left')  # reserve NaN ones by how='left'
#     dfa.loc[dfa.group.isnull(),'group'] = dfa.loc[dfa.group.isnull(), 'user']  # fill group==NaN ones with user
#     return dfa

# def get_dfwitm(dfwit, csv_file='intermediate_data/machine_metric_shennong_machine_all.csv'):
#     res_df = pd.read_csv(csv_file, index_col=0)
#     dfwitm = dfwit.merge(res_df.loc[:, ~res_df.columns.isin(['start_time','end_time','machine'])], on='worker_name', how='left')
#     return dfwitm

# ########### Plot Functions ###########
# linestyle_list = [
#      ('solid', 'solid'),       # Same as (0, ()) or '-'
#      ('dotted', 'dotted'),     # Same as (0, (1, 1)) or '.'
#      ('dashed', 'dashed'),     # Same as '--'
#      ('dashdot', 'dashdot'),   # Same as '-.'
#      ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),
#      ('densely dashdotted',    (0, (3, 1, 1, 1))),
#      ('densely dotted',        (0, (1, 1))),
#      ('densely dashed',        (0, (5, 1))),
#      ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
#      ('loosely dashed',        (0, (5, 10))),
#      ('loosely dashdotted',    (0, (3, 10, 1, 10))),
#      ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
#      ('loosely dotted',        (0, (1, 10))),
#      ('dashed',                (0, (5, 5))),
#      ('dashdotted',            (0, (3, 5, 1, 5))),
#      ('dotted',                (0, (1, 1))),
# ]

# def get_cdf(data, inverse=False):
#     sorted_data = sorted(data)
#     p = 100. * np.arange(len(sorted_data))/(len(sorted_data)-1)
#     p = 100. - p if inverse else p # CCDF
#     return sorted_data, p

# def plot_data_cdf(data, inverse=False, datalabel=None, xlabel=None, title=None, xlog=False, xlim=None, ylog=False, xticks=None, figsize=(4,3), dpi=120, savefig=None, ylabel=None):
#     plt.figure(figsize=figsize, dpi=dpi)
#     if type(data) == pd.DataFrame:
#         data.dropna(inplace=True)
#     x, y = get_cdf(data, inverse)
#     plt.plot(x, y, label=datalabel, color='green', linestyle='-')
#     if datalabel is not None: plt.legend(loc='lower right')
#     if xlog: plt.xscale('log')
#     if ylog: plt.yscale('log')
#     if xlim is not None: plt.xlim(xlim)
#     plt.ylim(0, 100)
#     if xlabel is not None: plt.xlabel(xlabel)
#     plt.ylabel(ylabel) if ylabel is not None else plt.ylabel('CCDF') if inverse is True else plt.ylabel('CDF')
#     if title is not None: plt.title(title)
#     if xticks is not None: plt.xticks(xticks)
#     plt.grid(alpha=.3, linestyle='--')
#     if savefig is not None:
#         plt.savefig('/tmp/figures/{}.pdf'.format(savefig),bbox_inches='tight')
#     else:
#         plt.show()

# def plot_data_cdfs(data, datalabel=None, inverse=False, xlabel=None, title=None, xlog=False, ylog=False, xticks=None, figsize=(4,3), dpi=120, xlim=None, ylim=None, ylabel=None, yticks=None, savefig=None, loc='best', fontsize=None):
#     plt.figure(figsize=figsize, dpi=dpi)
#     for i, d in enumerate(data):
#         if type(data) == pd.DataFrame:
#             d.dropna(inplace=True)
#         x, y = get_cdf(d, inverse)
#         label = datalabel[i] if datalabel is not None else None
#         plt.plot(x, y, label=label, linestyle=linestyle_list[i % len(linestyle_list)][1])
#     if datalabel is not None: plt.legend(loc=loc, fontsize=fontsize)
#     if xlog: plt.xscale('log')
#     if ylog: plt.yscale('log')
#     plt.ylim(0, 100) if ylim is None else plt.ylim(ylim)
#     if xlim is not None: plt.xlim(xlim)
#     if xlabel is not None: plt.xlabel(xlabel)
#     if ylabel is None:
#         plt.ylabel('CCDF') if inverse is True else plt.ylabel('CDF')
#     else:
#         plt.ylabel(ylabel)
#     if title is not None: plt.title(title)
#     if xticks is not None: plt.xticks(xticks)
#     if yticks is not None: plt.yticks(yticks)
#     plt.grid(alpha=.3, linestyle='--')
#     if savefig is not None:
#         plt.savefig('/tmp/figures/{}.pdf'.format(savefig),bbox_inches='tight')
#     else:
#         plt.show()

# def draw_bar_plot(odf, col, figsize=(4,4), dpi=120, portion=False, title=None, limit=30):
#     dfout=odf.reset_index().groupby(col).count()[['index']].sort_values('index', ascending=False).head(limit)
#     dfout['portion'] = 100 * dfout['index'] / dfout['index'].sum()
#     plt.figure(figsize=figsize, dpi=dpi)
#     if portion:
#         plt.barh(y=dfout.index, width=dfout['portion'])
#         plt.xlabel('Percentage (total: %.2f)'%(dfout['index'].sum()))
#     else:
#         plt.barh(y=dfout.index, width=dfout['index'])
#     plt.grid(alpha=.3, linestyle='--')
#     return dfout

# ########### Process Functions ###########

# def get_inst_task_num_ratio(dfa, inst_num_list=[2, 8, 20, 64, 100, 256, 512]):
#     total_num_task, total_num_inst = len(dfa), sum(dfa['inst_num'])
#     data_df = []
#     for i in inst_num_list:
#         temp_df = dfa[dfa['inst_num'] >= i]
#         task_num_ratio = len(temp_df) / total_num_task
#         inst_num_ratio = sum(temp_df['inst_num']) / total_num_inst
#         data_df.append([task_num_ratio, inst_num_ratio])
#     out_df = pd.DataFrame(data_df, columns=['num_task_ratio','num_inst_ratio'])
#     out_df = out_df.T.rename(columns=dict(zip(range(len(inst_num_list)), inst_num_list)))
#     return out_df


# def add_hour_date(df):
#     if 'start_date' not in df:
#         if 'start_time_t' in df:
#             target_col = 'start_time_t'
#         elif 'start_time' in df:
#             target_col = 'start_time'
#         else:
#             print('start_time, start_time_t, dayofyear unfound in df')
#             return None
#         df['start_date'] = df[target_col].apply(lambda x: pd.Timestamp(x, unit='s', tz='Asia/Shanghai'))
#     if 'date' not in df:
#         df['date'] = df['start_date'].apply(lambda x: x.date())
#     if 'hour' not in df:
#         df['hour'] = df['start_date'].apply(lambda x: x.hour)
#     return df

# def get_hourly_task_request(df): # df = dftjkix
#     sum_df_list = []
#     df = add_hour_date(df.copy())
#     # for day in sorted(df.dayofyear.unique()):
#     for date in sorted(df.date.unique()):
#         # tempdf = df[df.dayofyear==day]
#         tempdf = df[df.date==date]
#         res_df = tempdf.groupby('hour').count()[['job_name']]
#         res_df.rename(columns={'job_name':date}, inplace=True)
#         sum_df_list.append(res_df.T)
#     out_df = pd.DataFrame().append(sum_df_list)
#     return out_df.dropna() # if a day contains hours of NaN, it is not a typical day

# def get_hourly_task_resource_request(df, metrics='cpu'): # df = dftjkix
#     sum_df_list = []
#     df = add_hour_date(df)
#     if metrics == 'cpu':
#         df['plan_resource'] = df.plan_cpu.apply(lambda x: x/100)
#     elif metrics == 'gpu':
#         df['plan_resource'] = df.plan_gpu.apply(lambda x: x/100)
#     elif metrics == 'mem':
#         df['plan_resource'] = df.plan_mem.apply(lambda x: x/1000)
#     else:
#         exit()
#     # for day in sorted(df.dayofyear.unique()):
#     for date in sorted(df.date.unique()):
#         # tempdf = df[df.dayofyear==day]
#         tempdf = df[df.date==date]
#         res_df = tempdf.groupby('hour').sum()[['plan_resource']]
#         res_df.rename(columns={'job_name':date}, inplace=True)
#         sum_df_list.append(res_df.T)
#     out_df = pd.DataFrame().append(sum_df_list)
#     return out_df.dropna() # if a day contains hours of NaN, it is not a typical day

# def plan_minus_usg_over_cap_task(dfas):
#     dfas['plan_gpu_minus_usage_over_capacity'] = (dfas['plan_gpu'] - dfas['gpu_wrk_util']) / (100 * dfas['cap_gpu'])
#     dfas['plan_cpu_minus_usage_over_capacity'] = (dfas['plan_cpu'] - dfas['cpu_usage']) / (100 * dfas['cap_cpu'] )
#     dfas['plan_mem_minus_usage_over_capacity'] = (dfas['plan_mem'] - dfas['avg_mem']) / dfas['cap_mem']

#     dfas_task = dfas.groupby(['job_name','task_name'])[['plan_gpu_minus_usage_over_capacity','plan_cpu_minus_usage_over_capacity','plan_mem_minus_usage_over_capacity']].mean()

#     pgu_datas, pgu_label, ugp_datas, ugp_label = [], [], [], []
#     for device in ['cpu','gpu','mem']:
#         apu = dfas_task[~dfas_task['plan_{}_minus_usage_over_capacity'.format(device)].isnull()]
#         pgu = dfas_task[dfas_task['plan_{}_minus_usage_over_capacity'.format(device)] > 0]
#         ugp = dfas_task[dfas_task['plan_{}_minus_usage_over_capacity'.format(device)] < 0]
#         print("{}: plan > usage: {:.2f}%, plan < usage: {:.2f}%".format(
#             device, 100 * len(pgu) / len(apu), 100 * len(ugp) / len(apu)    ))
#         pgu_label.append("{} {:.2f}%".format(device, 100 * len(pgu) / len(apu)))
#         pgu_datas.append(pgu['plan_{}_minus_usage_over_capacity'.format(device)])
#         ugp_label.append("{} {:.2f}%".format(device, 100 * len(ugp) / len(apu)))
#         ugp_datas.append(-ugp['plan_{}_minus_usage_over_capacity'.format(device)])

#     return pgu_datas, ugp_datas, pgu_label, ugp_label

# Can you give me a very brief overview of each function you have at your disposal?"""
# context_ans_2, conversation_history = ask_question(context_2, conversation_history)
# print("Assistant:", context_ans_2)

# question_1 = """Here is an example. I am using this data to create a graph that has CDF on the y-axis and ‘Num of tasks submitted per User’ on the x-axis. This is the code:
# user_task_count = dfa.groupby('user').count()[['job_name']]

# plt.figure(figsize=(4,3), dpi=120)
# plot_data_cdf(user_task_count['job_name'], xlog=True,
#               ylabel='CDF',xlabel='Num of tasks submitted per User',
#               xticks=[1,10,100,10**3,10**4,10**5])

# Now I will ask you a series of tasks, please remember to use any functions/dataframes you see fit that we have defined so far.Here is your first task: generate code for a graph that has ‘Num of Instances in the Task” on the x-axis and ‘CCDF’ on the y-axis."""
# answer_1, conversation_history = ask_question(question_1, conversation_history)
# print("Assistant:", answer_1)

# question_2 = "Your next task: Generate code for a line-plot graph with ‘'Hours from the beginning of a week (Sun. to Sat.)' on the x-axis and 'Num of tasks' on the y-axis. The plot should show the number of tasks submitted at each hour of the day, with the x-axis ranging from 0 to 168 (representing 7 days). The data is stored in a dataframe, `dfa` ,with columns `start_date` and `job_name`. The `start_date` column should be converted to a numerical representation of the hour of the year, and then grouped by hour to calculate the number of tasks."
# answer_2, conversation_history = ask_question(question_2, conversation_history)
# print("Assistant:", answer_2)

# question_3 = "Your next task: generate Python code to create a boxplot showing the number of tasks submitted per hour of the day using the dfa dataframe, with hours on the x-axis and number of tasks on the y-axis. The boxplot should have a custom color palette with hours below the mean value in a default color and hours above the mean value in a highlighted color. The graph should also have a grid, custom y-ticks, and a specific font size. Finally, save the graph as a PDF file in the /tmp/figures directory."
# answer_3, conversation_history = ask_question(question_3, conversation_history)
# print("Assistant:", answer_3)

# question_4 = "Your next task: Write Python code to generate a graph that has 'Total resource requests' on the y-axis and 'Hours from the beginning of a week (Sun. to Sat.)' on the x-axis. The graph should show the total resource requests for CPU, GPU, and Memory over a period of 7 days. The data for resource requests is available in the dataframe dfw. The x-axis should be divided into hours, and the y-axis should be divided into appropriate units (e.g., 10 CPU cores, 1 GPU, 100 GB Memory)."
# answer_4, conversation_history = ask_question(question_4, conversation_history)
# print("Assistant:", answer_4)


# question = """What query would I need to give you to generate the following python code.:
# data_df = dfas
# plot_data_cdfs([data_df['plan_cpu'].dropna(), data_df['cpu_usage'].dropna()], ['CPU Request', 'CPU Usage'],
#                xlabel='% of CPU', xlim=(0, 6000), ylabel='CDF', xticks=[0,600,2000,4000,6000], dpi=120)"""
# answer, conversation_history = ask_question(question, conversation_history)
# print("Assistant A:", answer)