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
  cos = np.cos(angle)
  sin = np.sin(angle)
  x, y, z = axis

  # convert to rotation matrix
  rot_mat = np.array([[(1-cos)*x*x + cos,    (1-cos)*x*y - sin*z,  (1-cos)*x*z + sin*y],
                  [(1-cos)*x*y + sin*z,  (1-cos)*y*y + cos,    (1-cos)*y*z - sin*x],
                  [(1-cos)*x*z - sin*y,  (1-cos)*y*z + sin*x,  (1-cos)*z*z + cos]])
  
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
  # calculate rotation matrix
  rot_mat = rotation_matrix(axis, angle)

  T = np.eye(4)
  # place rotation matrix and translation vector
  T[:3, :3] = rot_mat 
  T[:3, 3] = v_A 
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
  # hip frame from axis is z-axis, only moves about z and no translation
  hip_frame = homogenous_transformation_matrix([0, 0, 1], joint_angles[0], [0, 0, 0])
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
  # get hip frame
  hip_frame = fk_hip(joint_angles)
  
  # get translation vector
  y = (-1) * HIP_OFFSET
  v_A = np.array([0,y,0])

  # create transformation matrix axis is y-axis
  shoulder_frame = homogenous_transformation_matrix([0, 1, 0], joint_angles[1], v_A)

  # to get with respect to base frame
  shoulder_frame = np.matmul(hip_frame, shoulder_frame)

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

  # calculate shoulder frame
  shoulder_frame = fk_shoulder(joint_angles)

  # translation vector for elbow
  v_A = np.array([0, 0, UPPER_LEG_OFFSET])

  # get elbow transformation matrix with axis about y-axis
  elbow_frame = homogenous_transformation_matrix([0, 1, 0], joint_angles[2], v_A)

  # get elbow with respect to base frame
  elbow_frame = np.matmul(shoulder_frame, elbow_frame)

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
  # get elbow frame
  elbow_frame = fk_elbow(joint_angles)

  # get translation vector
  v_A = np.array([0, 0, LOWER_LEG_OFFSET])

  # get transformation matrix, no movement on end_effector so 0 axis and angle
  foot_frame = homogenous_transformation_matrix([0, 0, 0], 0, v_A)

  # get with respect to base frame
  foot_frame = np.matmul(elbow_frame, foot_frame)

  return foot_frame
