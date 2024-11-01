sponge.function.GetVector
=============================

.. py:class:: sponge.function.GetVector(use_pbc: bool = None)

    获取有或者没有PBC box的向量。

    参数：
        - **use_pbc** (bool) - 计算向量时是否使用周期性边界条件。
          如果给出 ``None`` ，它将根据是否提供了 ``pbc_box`` 来决定是否使用周期性边界条件。默认值： ``None`` 。

    .. py:method:: calc_vector_default(initial: Tensor, terminal: Tensor, pbc_box: Tensor = None)

        获取向量。

        参数：
            - **initial** (Tensor) - 起始位置的坐标。张量的shape为 :math:`(B, ..., D)` 。
              B表示batchsize，例如，模拟中的步行者数量。D表示仿真系统的空间维度。通常为3。
              数据类型为float。
            - **terminal** (Tensor) - 终止位置的坐标。张量的shape为 :math:`(B, ..., D)`。数据类型为float。
            - **pbc_box** (Tensor) - 张量的shape为 :math:`(B, D)`。数据类型为float。默认值： ``None`` 。

    .. py:method:: calc_vector_nopbc(initial: Tensor, terminal: Tensor, pbc_box: Tensor = None)

        在没有周期性边界条件的情况下获取向量。

        参数：
            - **initial** (Tensor) - 起始位置的坐标。张量的shape为 :math:`(B, ..., D)` 。
              B表示batchsize，例如，模拟中的步行者数量。D表示仿真系统的空间维度。通常为3。
              数据类型为float。
            - **terminal** (Tensor) - 终止位置的坐标。张量的shape为 :math:`(B, ..., D)`。数据类型为float。
            - **pbc_box** (Tensor) - 张量的shape为 :math:`(B, D)`。数据类型为float。默认值： ``None`` 。

    .. py:method:: calc_vector_pbc(initial: Tensor, terminal: Tensor, pbc_box: Tensor = None)

        在有周期性边界条件的情况下获取向量。

        参数：
            - **initial** (Tensor) - 起始位置的坐标。张量的shape为 :math:`(B, ..., D)` 。
              B表示batchsize，例如，模拟中的步行者数量。D表示仿真系统的空间维度。通常为3。
              数据类型为float。
            - **terminal** (Tensor) - 终止位置的坐标。张量的shape为 :math:`(B, ..., D)`。数据类型为float。
            - **pbc_box** (Tensor) - 张量的shape为 :math:`(B, D)`。数据类型为float。默认值： ``None`` 。

    .. py:method:: set_pbc(use_pbc: bool)

        设定是否使用周期性边界条件。

        参数：
            - **use_pbc** (bool) - 是否使用周期性边界条件。默认值： ``None`` 。

    .. py:method:: use_pbc()
        :property:

        是否使用周期性边界条件。

        返回：
            bool。是否使用周期性边界条件。
