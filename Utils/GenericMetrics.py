from nvitop import Device, GpuProcess, NA, colored, HostProcess
from thop import profile
import time

# Device.gpu_utilization()  Device.memory_utilization() Device.temperature()
# GpuProcess.gpu_memory_utilization()  gpu_sm_utilization()
# GpuProcess.elapsed_time() GpuProcess.elapsed_time_in_seconds()
# GpuProcess.cpu_percent() GpuProcess.memory_percent() GpuProcess.host_memory_percent()


# Device就是指GPU
def getDeviceInfo():
    devices = Device.all()
    for device in devices:
        # Dict[int, GpuProcess]
        processes = device.processes()
        sorted_pids = sorted(processes)

        print(device)
        print(f'  - Fan speed:       {device.fan_speed()}%')
        print(f'  - Temperature:     {device.temperature()}C')
        print(f'  - GPU utilization: {device.gpu_utilization()}%')
        print(f'  - Total memory:    {device.memory_total_human()}')
        print(f'  - Used memory:     {device.memory_used_human()}')
        print(f'  - Free memory:     {device.memory_free_human()}')
        print(f'  - Processes ({len(processes)}): {sorted_pids}')
        for pid in sorted_pids:
            print(f'    - {processes[pid]}')
        print('-' * 120)


def getCPUInfo():
    print(colored(time.strftime('%a %b %d %H:%M:%S %Y'), color='red', attrs=('bold',)))

    devices = Device.cuda.all()  # or `Device.all()` to use NVML ordinal instead
    separator = False
    for device in devices:
        processes = device.processes()

        print(colored(str(device), color='green', attrs=('bold',)))
        print(colored('  - Fan speed:       ', color='blue', attrs=('bold',)) + f'{device.fan_speed()}%')
        print(colored('  - Temperature:     ', color='blue', attrs=('bold',)) + f'{device.temperature()}C')
        print(colored('  - GPU utilization: ', color='blue', attrs=('bold',)) + f'{device.gpu_utilization()}%')
        print(colored('  - Total memory:    ', color='blue', attrs=('bold',)) + f'{device.memory_total_human()}')
        print(colored('  - Used memory:     ', color='blue', attrs=('bold',)) + f'{device.memory_used_human()}')
        print(colored('  - Free memory:     ', color='blue', attrs=('bold',)) + f'{device.memory_free_human()}')
        if len(processes) > 0:
            processes = GpuProcess.take_snapshots(processes.values(), failsafe=True)
            processes.sort(key=lambda process: (process.username, process.pid))

            print(colored(f'  - Processes ({len(processes)}):', color='blue', attrs=('bold',)))
            fmt = '    {pid:<5}  {username:<8} {cpu:>5}  {host_memory:>8} {time:>8}  {gpu_memory:>8}  {sm:>3}  {command:<}'.format
            print(colored(fmt(pid='PID', username='USERNAME',
                              cpu='CPU%', host_memory='HOST-MEM', time='TIME',
                              gpu_memory='GPU-MEM', sm='SM%',
                              command='COMMAND'),
                          attrs=('bold',)))
            for snapshot in processes:
                print(fmt(pid=snapshot.pid,
                          username=snapshot.username[:7] + (
                              '+' if len(snapshot.username) > 8 else snapshot.username[7:8]),
                          cpu=snapshot.cpu_percent, host_memory=snapshot.host_memory_human,
                          time=snapshot.running_time_human,
                          gpu_memory=(snapshot.gpu_memory_human if snapshot.gpu_memory_human is not NA else 'WDDM:N/A'),
                          sm=snapshot.gpu_sm_utilization,
                          command=snapshot.command))
        else:
            print(colored('  - No Running Processes', attrs=('bold',)))

        if separator:
            print('-' * 120)
        separator = True


def getProcessGPUInfo(HostPid):
    devices = Device.cuda.all()
    device = devices[0]
    processes = device.processes()
    if len(processes) > 0:
        processes = GpuProcess.take_snapshots(processes.values(), failsafe=True)
        for snapshot in processes:
            print(snapshot.gpu_memory_utilization)
            print(snapshot.gpu_sm_utilization)
            # The used GPU memory in human readable format
            print(snapshot.gpu_memory_human)
            if snapshot.pid == HostPid:
                print("snapshot.pid == HostPid")


def getProcessCPUInfo(HostPid):
    devices = Device.cuda.all()
    device = devices[0]
    processes = device.processes()
    if len(processes) > 0:
        for process in processes:
            process = HostProcess(process.value().pid)
            print(process.cpu_percent())
            print(process.memory_percent())
            if process.pid == HostPid:
                print("process.pid == HostPid")


def getProcessElapsedTime(HostPid):
    devices = Device.cuda.all()
    device = devices[0]
    processes = device.processes()
    if len(processes) > 0:
        processes = GpuProcess.take_snapshots(processes.values(), failsafe=True)
        for snapshot in processes:
            print(snapshot.elapsed_time)
            print(snapshot.elapsed_time_in_seconds)
            print(snapshot.running_time_human)
            if snapshot.pid == HostPid:
                print("snapshot.pid == HostPid")


def getFLOPSandParams(model, data_input):
    flops, params = profile(model, (data_input,))
    print('flops: %.2f M, params: %.2f M' % (flops/1000000.0, params/1000000.0))

