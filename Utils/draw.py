import matplotlib.pyplot as plt
import pandas as pd
import csv


def readCSV(filePath):
    with open(filePath) as csvfile:
        reader = csv.DictReader(csvfile)
        column = [row for row in reader]
    print({'duration (s)' : column[0]['resources/duration (s)']})
    print({'gpu:0/gpu_utilization (%)/mean': column[0]['resources/gpu:0/gpu_utilization (%)/mean']})
    print({'gpu:0/memory_utilization (%)/mean': column[0]['resources/gpu:0/memory_utilization (%)/mean']})
    print({'gpu:0/power_usage (W)/mean': column[0]['resources/gpu:0/power_usage (W)/mean']})
    print({'host/cpu_percent (%)/mean': column[0]['resources/host/cpu_percent (%)/mean']})
    print({'host/memory_percent (%)/mean': column[0]['resources/host/memory_percent (%)/mean']})


"""
metrics-daemon/host/cpu_percent (%)/mean,
metrics-daemon/host/cpu_percent (%)/min,
metrics-daemon/host/cpu_percent (%)/max,
metrics-daemon/host/cpu_percent (%)/last,
metrics-daemon/host/memory_percent (%)/mean,
metrics-daemon/host/memory_percent (%)/min,
metrics-daemon/host/memory_percent (%)/max,
metrics-daemon/host/memory_percent (%)/last,
metrics-daemon/host/swap_percent (%)/mean,
metrics-daemon/host/swap_percent (%)/min,
metrics-daemon/host/swap_percent (%)/max,
metrics-daemon/host/swap_percent (%)/last,
metrics-daemon/host/memory_used (GiB)/mean,
metrics-daemon/host/memory_used (GiB)/min,
metrics-daemon/host/memory_used (GiB)/max,
metrics-daemon/host/memory_used (GiB)/last,
metrics-daemon/host/load_average (%) (1 min)/mean,
metrics-daemon/host/load_average (%) (1 min)/min,
metrics-daemon/host/load_average (%) (1 min)/max,
metrics-daemon/host/load_average (%) (1 min)/last,
metrics-daemon/host/load_average (%) (5 min)/mean,
metrics-daemon/host/load_average (%) (5 min)/min,
metrics-daemon/host/load_average (%) (5 min)/max,
metrics-daemon/host/load_average (%) (5 min)/last,
metrics-daemon/host/load_average (%) (15 min)/mean,
metrics-daemon/host/load_average (%) (15 min)/min,
metrics-daemon/host/load_average (%) (15 min)/max,
metrics-daemon/host/load_average (%) (15 min)/last,
metrics-daemon/gpu:0/memory_used (MiB)/mean,
metrics-daemon/gpu:0/memory_used (MiB)/min,
metrics-daemon/gpu:0/memory_used (MiB)/max,
metrics-daemon/gpu:0/memory_used (MiB)/last,
metrics-daemon/gpu:0/memory_free (MiB)/mean,
metrics-daemon/gpu:0/memory_free (MiB)/min,
metrics-daemon/gpu:0/memory_free (MiB)/max,
metrics-daemon/gpu:0/memory_free (MiB)/last,
metrics-daemon/gpu:0/memory_total (MiB)/mean,
metrics-daemon/gpu:0/memory_total (MiB)/min,
metrics-daemon/gpu:0/memory_total (MiB)/max,
metrics-daemon/gpu:0/memory_total (MiB)/last,
metrics-daemon/gpu:0/memory_percent (%)/mean,
metrics-daemon/gpu:0/memory_percent (%)/min,
metrics-daemon/gpu:0/memory_percent (%)/max,
metrics-daemon/gpu:0/memory_percent (%)/last,
metrics-daemon/gpu:0/gpu_utilization (%)/mean,
metrics-daemon/gpu:0/gpu_utilization (%)/min,
metrics-daemon/gpu:0/gpu_utilization (%)/max,
metrics-daemon/gpu:0/gpu_utilization (%)/last,
metrics-daemon/gpu:0/memory_utilization (%)/mean,
metrics-daemon/gpu:0/memory_utilization (%)/min,
metrics-daemon/gpu:0/memory_utilization (%)/max,
metrics-daemon/gpu:0/memory_utilization (%)/last,
metrics-daemon/gpu:0/fan_speed (%)/mean,
metrics-daemon/gpu:0/fan_speed (%)/min,
metrics-daemon/gpu:0/fan_speed (%)/max,
metrics-daemon/gpu:0/fan_speed (%)/last,
metrics-daemon/gpu:0/temperature (C)/mean,
metrics-daemon/gpu:0/temperature (C)/min,
metrics-daemon/gpu:0/temperature (C)/max,
metrics-daemon/gpu:0/temperature (C)/last,
metrics-daemon/gpu:0/power_usage (W)/mean,
metrics-daemon/gpu:0/power_usage (W)/min,
metrics-daemon/gpu:0/power_usage (W)/max,
metrics-daemon/gpu:0/power_usage (W)/last,
metrics-daemon/duration (s),
metrics-daemon/timestamp,metrics-daemon/last_timestamp
"""
def readCSV_v2(filePath):
    with open(filePath) as csvfile:
        reader = csv.DictReader(csvfile)
        column = [row for row in reader]
    for idx in range(len(column)):
        print('iter:', idx)
        print({'duration (s)': column[idx]['metrics-daemon/duration (s)']})
        print({'gpu_utilization (%)/mean': column[idx]['metrics-daemon/gpu:0/gpu_utilization (%)/mean']})
        print({'memory_utilization (%)/mean': column[idx]['metrics-daemon/gpu:0/memory_utilization (%)/mean']})
        print({'power_usage (W)/mean': column[idx]['metrics-daemon/gpu:0/power_usage (W)/mean']})
        print({'host/cpu_percent (%)/mean': column[idx]['metrics-daemon/host/cpu_percent (%)/mean']})
        print({'host/memory_percent (%)/mean': column[idx]['metrics-daemon/host/memory_percent (%)/mean']})


def draw(filePath):
    plt.clf()
    df = pd.read_csv(filePath)
    duration_label = 'metrics-daemon/duration (s)'
    gpu_utilization_label = 'metrics-daemon/gpu:0/gpu_utilization (%)/mean'
    gpu_memory_utilization_label = 'metrics-daemon/gpu:0/memory_utilization (%)/mean'
    host_cpu_percent_label = 'metrics-daemon/host/cpu_percent (%)/mean'
    host_memory_percent_label = 'metrics-daemon/host/memory_percent (%)/mean'
    interval = 20
    x_axis_data = df[duration_label][::interval]
    y_gpu_utilization_data = df[gpu_utilization_label][::interval]
    y_gpu_memory_utilization_data = df[gpu_memory_utilization_label][::interval]
    y_host_cpu_percent_data = df[host_cpu_percent_label][::interval]
    y_host_memory_percent_data = df[host_memory_percent_label][::interval]

    plt.ylim((0, 100))
    plt.plot(x_axis_data, y_gpu_utilization_data, marker='s', markersize=4, color='tomato',
             linestyle='-', label='gpu_utilization', alpha=0.8)
    plt.plot(x_axis_data, y_gpu_memory_utilization_data, marker='o', markersize=4, color='y',
             linestyle='-', label='gpu_memory_utilization', alpha=0.8)
    plt.plot(x_axis_data, y_host_cpu_percent_data, marker='*', markersize=4, color='m',
             linestyle='--', label='host_cpu_percent', alpha=0.8)
    plt.plot(x_axis_data, y_host_memory_percent_data, marker='x', markersize=4, color='g',
             linestyle='--', label='host_memory_percent', alpha=0.8)
    # plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，
    plt.tight_layout()

    # 创建图例，并将其放置在图的右下角外部
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, borderaxespad=0., fontsize=15)
    plt.subplots_adjust(top=0.8)
    plt.xlabel('duration(s)')  # 数据库的单位得改一下
    plt.ylabel('utilization/precent(%)')  # y_label
    # plt.title('BlackBoxResourceUtilization')

    plt.savefig('Matformer_BlackBoxResource.jpg')
    plt.show()


def drawForFrame(filePath1, filePath2, name1, name2, savePath):
    plt.clf()
    df1 = pd.read_csv(filePath1)
    df2 = pd.read_csv(filePath2)
    duration_label = 'metrics-daemon/duration (s)'
    gpu_utilization_label = 'metrics-daemon/gpu:0/gpu_utilization (%)/mean'
    gpu_memory_utilization_label = 'metrics-daemon/gpu:0/memory_utilization (%)/mean'
    host_cpu_percent_label = 'metrics-daemon/host/cpu_percent (%)/mean'
    host_memory_percent_label = 'metrics-daemon/host/memory_percent (%)/mean'
    interval = 2

    x1_axis_data = df1[duration_label][::interval]
    y1_gpu_utilization_data = df1[gpu_utilization_label][::interval]
    y1_gpu_memory_utilization_data = df1[gpu_memory_utilization_label][::interval]
    y1_host_cpu_percent_data = df1[host_cpu_percent_label][::interval]
    y1_host_memory_percent_data = df1[host_memory_percent_label][::interval]

    x2_axis_data = df2[duration_label][::interval]
    y2_gpu_utilization_data = df2[gpu_utilization_label][::interval]
    y2_gpu_memory_utilization_data = df2[gpu_memory_utilization_label][::interval]
    y2_host_cpu_percent_data = df2[host_cpu_percent_label][::interval]
    y2_host_memory_percent_data = df2[host_memory_percent_label][::interval]

    x_max = max(x1_axis_data.iloc[-1], x2_axis_data.iloc[-1])
    plt.xlim((0, x_max*1.05))
    plt.ylim((0, 100))
    plt.plot(x1_axis_data, y1_gpu_utilization_data, marker='s', markersize=4, color='tomato',
             linestyle='-', label='gpu_utilization_' + name1, alpha=0.8)
    plt.plot(x2_axis_data, y2_gpu_utilization_data, marker='s', markersize=4, color='tomato',
             linestyle='--', label='gpu_utilization_' + name2, alpha=0.8)

    plt.plot(x1_axis_data, y1_gpu_memory_utilization_data, marker='o', markersize=4, color='y',
             linestyle='-', label='gpu_memory_utilization_' + name1, alpha=0.8)
    plt.plot(x2_axis_data, y2_gpu_memory_utilization_data, marker='o', markersize=4, color='y',
             linestyle='--', label='gpu_memory_utilization_' + name2, alpha=0.8)

    plt.plot(x1_axis_data, y1_host_cpu_percent_data, marker='*', markersize=4, color='m',
             linestyle='-', label='host_cpu_percent_' + name1, alpha=0.8)
    plt.plot(x2_axis_data, y2_host_cpu_percent_data, marker='*', markersize=4, color='m',
             linestyle='--', label='host_cpu_percent_' + name2, alpha=0.8)

    plt.plot(x1_axis_data, y1_host_memory_percent_data, marker='x', markersize=4, color='g',
             linestyle='-', label='host_memory_percent_' + name1, alpha=0.8)
    plt.plot(x2_axis_data, y2_host_memory_percent_data, marker='x', markersize=4, color='g',
             linestyle='--', label='host_memory_percent_' + name2, alpha=0.8)

    plt.tight_layout()

    # 创建图例，并将其放置在图的右下角外部
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.04), ncol=3, borderaxespad=0., fontsize=12)
    plt.subplots_adjust(top=0.8)
    plt.xlabel('duration(s)')  # 数据库的单位得改一下
    plt.ylabel('utilization/precent(%)')  # y_label
    plt.title('CGCNN_200Epoch_Eval_BlackBoxResource')
    # NequIP_200Epoch_Train_BlackBoxResource

    plt.savefig(savePath + 'CGCNN_200Epoch_Eval_BlackBoxResource.jpg')
    plt.show()