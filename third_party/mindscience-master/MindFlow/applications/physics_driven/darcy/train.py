# ============================================================================
# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""train process"""
import argparse
import os
import time
import numpy as np

from mindspore import context, nn, ops, jit, set_seed, data_sink

from mindflow.utils import load_yaml_config, print_log, log_timer
from mindflow.cell import FCSequential

from src import create_training_dataset, create_test_dataset
from src import Darcy2D
from src import visual, calculate_l2_error

set_seed(123456)
np.random.seed(123456)


def parse_args():
    """Parse input args"""
    parser = argparse.ArgumentParser(description="darcy flow")
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Running in GRAPH_MODE OR PYNATIVE_MODE")
    parser.add_argument("--save_graphs", type=bool, default=False, choices=[True, False],
                        help="Whether to save intermediate compilation graphs")
    parser.add_argument("--save_graphs_path", type=str, default="./graphs")
    parser.add_argument("--device_target", type=str, default="GPU", choices=["GPU", "Ascend"],
                        help="The target device to run, support 'Ascend', 'GPU'")
    parser.add_argument("--device_id", type=int, default=0,
                        help="ID of the target device")
    parser.add_argument("--config_file_path", type=str,
                        default="./configs/darcy.yaml")
    input_args = parser.parse_args()
    return input_args


@log_timer
def train(input_args):
    """training process"""
    config = load_yaml_config(input_args.config_file_path)
    geom_name = "flow_region"
    # create train dataset
    flow_train_dataset = create_training_dataset(config, geom_name)
    train_data = flow_train_dataset.create_dataset(
        batch_size=config['data']['train']["batch_size"], shuffle=True, drop_remainder=True
    )
    # create test dataset
    test_input, test_label = create_test_dataset(config)

    # network model
    model = FCSequential(in_channels=config["model"]["in_channels"],
                         out_channels=config["model"]["out_channels"],
                         neurons=config["model"]["neurons"],
                         layers=config["model"]["layers"],
                         residual=config["model"]["residual"],
                         act=config["model"]["activation"],
                         weight_init=config["model"]["weight_init"])

    # define problem
    problem = Darcy2D(model)

    # optimizer
    params = model.trainable_params()
    optimizer = nn.Adam(params, learning_rate=config["optimizer"]["learning_rate"])
    # prepare loss scaler
    if use_ascend:
        from mindspore.amp import DynamicLossScaler, all_finite, auto_mixed_precision
        loss_scaler = DynamicLossScaler(1024, 2, 100)
        auto_mixed_precision(model, 'O3')
    else:
        loss_scaler = None

    def forward_fn(pde_data, bc_data):
        loss = problem.get_loss(pde_data, bc_data)
        if use_ascend:
            loss = loss_scaler.scale(loss)
        return loss

    grad_fn = ops.value_and_grad(
        forward_fn, None, optimizer.parameters, has_aux=False)

    @jit
    def train_step(pde_data, bc_data):
        loss, grads = grad_fn(pde_data, bc_data)
        if use_ascend:
            loss = loss_scaler.unscale(loss)
            is_finite = all_finite(grads)
            if is_finite:
                grads = loss_scaler.unscale(grads)
                loss = ops.depend(loss, optimizer(grads))
            loss_scaler.adjust(is_finite)
        else:
            loss = ops.depend(loss, optimizer(grads))
        return loss

    epochs = config["data"]["train"]["epochs"]
    steps_per_epochs = train_data.get_dataset_size()
    print_log(f"number of steps_per_epochs: {steps_per_epochs}")
    sink_process = data_sink(train_step, train_data, sink_size=1)
    model.set_train()

    for epoch in range(1, 1 + epochs):
        local_time_beg = time.time()
        model.set_train(True)
        for _ in range(steps_per_epochs):
            cur_loss = sink_process()
        local_time_end = time.time()
        epoch_seconds = local_time_end - local_time_beg
        step_seconds = (epoch_seconds/steps_per_epochs) * 1000
        print_log(f"epoch: {epoch} train loss: {cur_loss} "
                  f"epoch time: {epoch_seconds:5.3f}s step time: {step_seconds:5.3f}ms")
        model.set_train(False)
        if epoch % config["summary"]["eval_interval_epochs"] == 0:
            calculate_l2_error(model, test_input, test_label,
                               config["data"]["train"]["batch_size"])

    visual(model, config)


if __name__ == "__main__":
    print("pid:", os.getpid())
    args = parse_args()
    context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
                        save_graphs=args.save_graphs,
                        save_graphs_path=args.save_graphs_path,
                        device_target=args.device_target,
                        device_id=args.device_id)
    print(
        f"Running in {args.mode.upper()} mode, using device id: {args.device_id}.")
    use_ascend = (args.device_target == "Ascend")
    train(args)
