import math
import numpy as np
import copy

HIP_OFFSET = 0.0335
UPPER_LEG_OFFSET = 0.10 # length of link 1
LOWER_LEG_OFFSET = 0.13 # length of link 2

def rotation_matrix(axis, angle):
  """
  Create a 3x3 rotation matrix which rotates about a specific axis

  Args:
    axis:  Array.  Unit vector in the direction of the axis of rotation
    angle: Number. The amount to rotate about the axis in radians

  Returns:
    3x3 rotation matrix as a numpy array
  """
  c = np.cos(angle)
  s = np.sin(angle)
  t = 1 - c
  x, y, z = axis
    
  # Compute rotation matrix
  rot_mat = np.array([[t*x*x + c,    t*x*y - s*z,  t*x*z + s*y],
                  [t*x*y + s*z,  t*y*y + c,    t*y*z - s*x],
                  [t*x*z - s*y,  t*y*z + s*x,  t*z*z + c]])
    
  
  rot_mat = np.eye(3)
  return rot_mat

def homogenous_transformation_matrix(axis, angle, v_A):
  """
  Create a 4x4 transformation matrix which transforms from frame A to frame B

  Args:
    axis:  Array.  Unit vector in the direction of the axis of rotation
    angle: Number. The amount to rotate about the axis in radians
    v_A:   Vector. The vector translation from A to B defined in frame A

  Returns:
    4x4 transformation matrix as a numpy array
  """
  rot_mat = rotation_matrix(axis, angle)

    # Create a 4x4 transformation matrix
  T = np.eye(4)
  T[:3, :3] = rot_mat  # Assign rotation matrix to upper-left 3x3 submatrix
  T[:3, 3] = v_A  # Assign translation vector to the last column

  T = np.eye(4)
  return T

def fk_hip(joint_angles):
  """
  Use forward kinematics equations to calculate the xyz coordinates of the hip
  frame given the joint angles of the robot

  Args:
    joint_angles: numpy array of 3 elements stored in the order [hip_angle, shoulder_angle, 
                  elbow_angle]. Angles are in radians
  Returns:
    4x4 matrix representing the pose of the hip frame in the base frame
  """
  alpha = [0, np.pi / 2, 0]  # twist angles (radians)
  a = [0, 0, UPPER_LEG_OFFSET]  # link lengths (meters)
  d = [0, HIP_OFFSET, 0]  # link offsets (meters)
  theta = joint_angles  # joint angles (radians)

    # Initialize homogeneous transformation matrices
  T = np.eye(4)

    # Compute transformation matrices for each joint
  for i in range(3):
        # Compute transformation matrix for joint i
        T_i = homogenous_transformation_matrix([0, 0, 1], theta[i], [0, 0, d[i]])  # rotation about z-axis, translation along z-axis
        T_i_next = homogenous_transformation_matrix([0, 1, 0], alpha[i], [a[i], 0, 0])  # rotation about y-axis, translation along x-axis
        T = np.dot(T, np.dot(T_i, T_i_next))

  hip_frame = np.eye(4)  # remove this line when you write your solution
  return hip_frame

def fk_shoulder(joint_angles):
  """
  Use forward kinematics equations to calculate the xyz coordinates of the shoulder
  joint given the joint angles of the robot

  Args:
    joint_angles: numpy array of 3 elements stored in the order [hip_angle, shoulder_angle, 
                  elbow_angle]. Angles are in radians
  Returns:
    4x4 matrix representing the pose of the shoulder frame in the base frame
  """

  # remove these lines when you write your solution
  default_sphere_location = np.array([[0.15, 0.0, -0.1]])
  shoulder_frame = np.block(
    [[np.eye(3), default_sphere_location.T], 
     [0, 0, 0, 1]])
  return shoulder_frame

def fk_elbow(joint_angles):
  """
  Use forward kinematics equations to calculate the xyz coordinates of the elbow
  joint given the joint angles of the robot

  Args:
    joint_angles: numpy array of 3 elements stored in the order [hip_angle, shoulder_angle, 
                  elbow_angle]. Angles are in radians
  Returns:
    4x4 matrix representing the pose of the elbow frame in the base frame
  """

  # remove these lines when you write your solution
  default_sphere_location = np.array([[0.15, 0.1, -0.1]])
  elbow_frame = np.block(
    [[np.eye(3), default_sphere_location.T], 
     [0, 0, 0, 1]])
  return elbow_frame

def fk_foot(joint_angles):
  """
  Use forward kinematics equations to calculate the xyz coordinates of the foot given 
  the joint angles of the robot

  Args:
    joint_angles: numpy array of 3 elements stored in the order [hip_angle, shoulder_angle, 
                  elbow_angle]. Angles are in radians
  Returns:
    4x4 matrix representing the pose of the end effector frame in the base frame
  """

  # remove these lines when you write your solution
  default_sphere_location = np.array([[0.15, 0.2, -0.1]])
  end_effector_frame = np.block(
    [[np.eye(3), default_sphere_location.T], 
     [0, 0, 0, 1]])
  return end_effector_frame
