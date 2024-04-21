import numpy as np
import ppsci
from ppsci.utils import logger


def sin_compute_func(data: dict):
    return np.sin(data["x"])


ppsci.utils.misc.set_random_seed(42)
OUTPUT_DIR = 'res/output_quick_start_case1'
logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

l_limit, r_limit = -np.pi, np.pi
x_domain = ppsci.geometry.Interval(l_limit, r_limit)

# Model 3层MLP 输入是x 预测输出u
model = ppsci.arch.MLP(("x",), ("u",), 3, 64)
#
ITERS_PER_EPOCH = 100
interior_constraint = ppsci.constraint.InteriorConstraint(
    output_expr={"u": lambda out: out["u"]},
    label_dict={"u": sin_compute_func},
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
constraint = {interior_constraint.name: interior_constraint}

EPOCHS = 10
optimizer = ppsci.optimizer.Adam(2e-3)(model)

# 可视化
visual_input_dict = {
    "x": np.linspace(l_limit, r_limit, 1000, dtype="float32").reshape(1000,1)
}
visual_input_dict["u_ref"] = np.sin(visual_input_dict["x"])
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
