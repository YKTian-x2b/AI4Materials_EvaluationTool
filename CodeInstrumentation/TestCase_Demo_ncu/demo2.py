import torch
import torch.nn.functional as F
import nvtx

device = torch.device("cuda:0")


batch_size, input_features = 4, 128
output_features = 128


input_data = torch.randn(batch_size, input_features).to(device)


weight = torch.randn(input_features, output_features, requires_grad=False).to(device)

bias = torch.randn(output_features, requires_grad=False).to(device)

torch.cuda.synchronize(0)
linear_nvtx = nvtx.start_range(message="linear_nvtx", color="blue")
output_data = F.linear(input_data, weight, bias)
torch.cuda.synchronize(0)
nvtx.end_range(linear_nvtx)
