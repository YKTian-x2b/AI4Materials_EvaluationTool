import paddle
import paddle.nn.functional as F
import nvtx

paddle.device.set_device('gpu:0')


batch_size, input_features = 4, 128
output_features = 128


input_data = paddle.randn([batch_size, input_features])


weight = paddle.randn([input_features, output_features])

bias = paddle.randn([output_features])

paddle.device.cuda.synchronize(0)
linear_nvtx = nvtx.start_range(message="linear_nvtx", color="blue")
output_data = F.linear(input_data, weight, bias)
paddle.device.cuda.synchronize(0)
nvtx.end_range(linear_nvtx)
