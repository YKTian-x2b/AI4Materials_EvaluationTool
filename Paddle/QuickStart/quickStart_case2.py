import numpy as np
import ppsci
from ppsci.autodiff import jacobian
from ppsci.utils import logger


# 我们仍然能保持模型的输入、输出不变，但优化目标变成u对x的偏导是cos(x)
# 且边界条件满足 u(-PI)+a=2
def sin_compute_func(data: dict):
    return np.sin(data["x"])


def cos_compute_func(data: dict):
    return np.cos(data["x"])


ppsci.utils.misc.set_random_seed(42)
OUTPUT_DIR = 'res/output_quick_start_case2'
logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

l_limit, r_limit = -np.pi, np.pi
x_domain = ppsci.geometry.Interval(l_limit, r_limit)

# Model 3层MLP 输入是x 预测输出u
model = ppsci.arch.MLP(("x",), ("u",), 3, 64)
#
ITERS_PER_EPOCH = 100
# 约束条件从约束“模型输出”，改为约束“模型输出对输入的一阶微分”
interior_constraint = ppsci.constraint.InteriorConstraint(
    output_expr={"du_dx": lambda out: jacobian(out["u"], out["x"])},
    label_dict={"du_dx": cos_compute_func},
    geom=x_domain,
    dataloader_cfg={
        "dataset": "NamedArrayDataset",
        "iters_per_epoch": ITERS_PER_EPOCH,
        "sampler": {
            "name": "BatchSampler",
            "shuffle": True,
        },
        "batch_size": 32
    },
    loss=ppsci.loss.MSELoss(),
)
#
bc_constraint = ppsci.constraint.BoundaryConstraint(
    {"u": lambda d: d["u"]},
    {"u": lambda d: sin_compute_func(d)+2},
    x_domain,
    dataloader_cfg={
        "dataset": "NamedArrayDataset",
        "iters_per_epoch": ITERS_PER_EPOCH,
        "sampler": {
            "name": "BatchSampler",
            "shuffle": True,
        },
        "batch_size": 1
    },
    loss=ppsci.loss.MSELoss(),
    criteria=lambda x: np.isclose(x, l_limit),
)

constraint = {interior_constraint.name: interior_constraint,
              bc_constraint.name: bc_constraint}

EPOCHS = 10
optimizer = ppsci.optimizer.Adam(2e-3)(model)

# 可视化
visual_input_dict = {
    "x": np.linspace(l_limit, r_limit, 1000, dtype="float32").reshape(1000,1)
}
visual_input_dict["u_ref"] = np.sin(visual_input_dict["x"]) + 2.0
visualizer = {
    "visualizer_u": ppsci.visualize.VisualizerScatter1D(
        visual_input_dict,
        ("x",),
        {"u_pred": lambda out: out["u"], "u_ref": lambda out: out["u_ref"]},
        prefix="u=sin(x)"
    )
}

solver = ppsci.solver.Solver(
    model, constraint, OUTPUT_DIR, optimizer, epochs=EPOCHS,
    iters_per_epoch=ITERS_PER_EPOCH, visualizer=visualizer
)
solver.train()

pred_u = solver.predict(visual_input_dict, return_numpy=True)["u"]
l2_rel = np.linalg.norm(pred_u - visual_input_dict["u_ref"]) / np.linalg.norm(
    visual_input_dict["u_ref"]
)
logger.info(f"l2_rel = {l2_rel:.5f}")
solver.visualize()
