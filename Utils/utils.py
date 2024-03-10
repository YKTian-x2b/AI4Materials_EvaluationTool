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
