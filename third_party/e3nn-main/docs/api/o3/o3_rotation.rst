.. _Rotation functions:

Parametrization of Rotations
============================


Matrix Parametrization
----------------------

.. autofunction:: e3nn.o3.rand_matrix

.. autofunction:: e3nn.o3.matrix_x

.. autofunction:: e3nn.o3.matrix_y

.. autofunction:: e3nn.o3.matrix_z


Euler Angles Parametrization
----------------------------

.. autofunction:: e3nn.o3.identity_angles

.. autofunction:: e3nn.o3.rand_angles

.. autofunction:: e3nn.o3.compose_angles

.. autofunction:: e3nn.o3.inverse_angles


Quaternion Parametrization
--------------------------

.. autofunction:: e3nn.o3.identity_quaternion

.. autofunction:: e3nn.o3.rand_quaternion

.. autofunction:: e3nn.o3.compose_quaternion

.. autofunction:: e3nn.o3.inverse_quaternion


Axis-Angle Parametrization
--------------------------

.. autofunction:: e3nn.o3.rand_axis_angle

.. autofunction:: e3nn.o3.compose_axis_angle


Convertions
-----------

.. autofunction:: e3nn.o3.angles_to_matrix

.. autofunction:: e3nn.o3.matrix_to_angles

.. autofunction:: e3nn.o3.angles_to_quaternion

.. autofunction:: e3nn.o3.matrix_to_quaternion

.. autofunction:: e3nn.o3.axis_angle_to_quaternion

.. autofunction:: e3nn.o3.quaternion_to_axis_angle

.. autofunction:: e3nn.o3.matrix_to_axis_angle

.. autofunction:: e3nn.o3.angles_to_axis_angle

.. autofunction:: e3nn.o3.axis_angle_to_matrix

.. autofunction:: e3nn.o3.quaternion_to_matrix

.. autofunction:: e3nn.o3.quaternion_to_angles

.. autofunction:: e3nn.o3.axis_angle_to_angles


Convertions to point on the sphere
----------------------------------

.. autofunction:: e3nn.o3.angles_to_xyz

.. autofunction:: e3nn.o3.xyz_to_angles
