import jax
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
from flax.metrics import tensorboard
from flax.training import train_state
import optax
from typing import List, Union

def doubao():
    def linear_model(params, x):
        W, b = params
        return jnp.dot(x, W)+b

    key = jax.random.PRNGKey(0)
    input_dim = 10
    output_dim = 5
    W = jax.random.normal(key, (input_dim, output_dim))
    b = jax.random.normal(key, (output_dim,))
    params = (W, b)

    batch_size = 3
    x = jax.random.normal(key, (batch_size, input_dim))

    output = linear_model(params, x)
    print(output)

    def mse_loss(params, x, y_true):
        y_pred = linear_model(params, x)
        return jnp.mean((y_pred - y_true)**2)

    lr = 0.01
    y_true = jax.random.normal(key, (batch_size, output_dim))
    grad_fn = jax.grad(mse_loss)
    grads = grad_fn(params, x, y_true)
    new_W = params[0] - lr * grads[0]
    new_b = params[1] - lr * grads[1]
    new_params = (new_W, new_b)


def quick_start():
    model = Model(arg_model)
    params = model.init(key)
    opt = Optimizer(arg_opt)
    state = opt.init(params)
    y = model.apply(params, x)
    loss = loss_f(y, target)

    def loss_func(params, x, target):
        y = model.apply(params, x)
        return loss_f(y, target)

    loss, grads = jax.value_and_grad(loss_func)(params, x, target)

    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)


class PRNGManager:
    def __init__(self, key: jnp.ndarray):
        self.initial_key = key
        self.current_key = None

    def __enter__(self) -> 'PRNGManager':
        self.current_key = self.initial_key
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # You can add any cleanup or handling here if needed
        pass

    def new_key(self) -> jnp.ndarray:
        keys = random.split(self.current_key, 2)
        self.current_key = keys[0]
        return keys[1]

    def new_n_keys(self, n: int) -> List[jnp.ndarray]:
        keys = random.split(self.current_key, n + 1)
        self.current_key = keys[0]
        return list(keys[1:])


def least_square():
    # Usage:
    key = random.PRNGKey(0)  # or any other initial seed

    # Set problem dimensions.
    n_samples = 20
    x_dim = 10
    y_dim = 5

    learning_rate = 0.003  # Gradient step size.

    with PRNGManager(key) as manager:
        x = random.normal(manager.new_key(), (n_samples, x_dim))  # Dummy input data
        reference_params = model.init(manager.new_key(), x)

        params = model.init(manager.new_key(), x)

        @jax.jit
        def loss_function(params, x, y):
            output = model.apply(params, x)
            return ((y - output) ** 2).sum()

        @jax.jit
        def update_params(params, learning_rate, grads):
            params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
            return params

        for i in range(100):
            x = random.normal(manager.new_key(), (n_samples, x_dim))  # training data
            y = model.apply(reference_params, x)
            loss_value, grads = jax.value_and_grad(loss_function)(params, x, y)
            params = update_params(params, learning_rate, grads)
            print(loss_value)


def linear():
    class Module(nn.Module):
        features: Tuple[int] = (16, 4)

        def setup(self):
            self.dense1 = Dense(self.features[0])
            self.dense2 = Dense(self.features[1])

        def __call__(self, x):
            return self.dense2(nn.relu(self.dense1(x)))


if __name__ == "__main__":
    least_square()
