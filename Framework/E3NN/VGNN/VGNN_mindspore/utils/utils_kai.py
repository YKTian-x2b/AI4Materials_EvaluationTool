import mindspore.dataset as ds
import numpy as np


def random_split(dataset, lengths):
    """
    Randomly split the dataset into non-overlapping new datasets of given lengths.

    Args:
        dataset (mindspore.dataset.Dataset): Input dataset to be split.
        lengths (sequence): A sequence of integers defining the lengths of the splits.

    Returns:
        list: List of new datasets.
    """
    total_length = dataset.get_dataset_size()

    # 检查lengths的总和是否等于总长度
    assert sum(lengths) == total_length, "Sum of input lengths does not equal the length of the input dataset!"

    # 生成随机索引
    indices = np.random.permutation(total_length)

    # 分割索引
    start = 0
    subsets = []
    for length in lengths:
        end = start + length
        subset_indices = indices[start:end]
        # 使用GeneratorDataset或者自定义Dataset类来根据索引创建子集
        # 这里假设我们有一个根据索引返回数据集的函数get_subset_dataset
        subset_dataset = get_subset_dataset(dataset, subset_indices)
        subsets.append(subset_dataset)
        start = end

    return subsets


# 假设函数，用于根据索引返回数据集的子集
def get_subset_dataset(dataset, indices):
    def generator():
        for idx in indices:
            yield dataset[idx]

    return ds.GeneratorDataset(generator)