from mindspore import Tensor

class msData():
    def __init__(self,
                 x,
                 z,
                 y,
                 tensor: Tensor):
        setattr(self, "x", x)
        setattr(self, "z", z)
        setattr(self, "y", y)
        setattr(self, "tensor", tensor)

    def __getitem__(self, item: str):
        return getattr(self, item)


if __name__ == "__main__":
    x = 3
    z = [1,2,3]
    y = (4,5,6)
    data = [1, 0, 1, 0]
    tensor = Tensor(data)
    dt = msData(x, z, y, tensor)
    print(dt["x"])
    print(dt["y"])
    print(dt["z"])
    print(dt["tensor"].shape)