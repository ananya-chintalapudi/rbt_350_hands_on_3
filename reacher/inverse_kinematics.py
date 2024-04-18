import math
import numpy as np
import copy
from reacher import forward_kinematics
import copy
HIP_OFFSET = 0.0335
UPPER_LEG_OFFSET = 0.10 # length of link 1
LOWER_LEG_OFFSET = 0.13 # length of link 2
TOLERANCE = 0.01 # tolerance for inverse kinematics
PERTURBATION = 0.0001 # perturbation for finite difference method
MAX_ITERATIONS = 10 #10

def ik_cost(end_effector_pos, guess):
    """Calculates the inverse kinematics cost.
    
    This function computes the inverse kinematics cost, which represents the Euclidean
    distance between the desired end-effector position and the end-effector position
    resulting from the provided 'guess' joint angles.

    Args:
        end_effector_pos (numpy.ndarray), (3,): The desired XYZ coordinates of the end-effector.
            A numpy array with 3 elements.
        guess (numpy.ndarray), (3,): A guess at the joint angles to achieve the desired end-effector
            position. A numpy array with 3 elements.

    Returns:
        float: The Euclidean distance between end_effector_pos and the calculated end-effector
        position based on the guess.
    """
    # Initialize cost to zero
    cost = 0.0

    # estimated end_effector_pos based on guess
    calculated_end_effector_pos = forward_kinematics.fk_foot(guess)
    
    # get position from calculated homogenous transformation matrix
    calculated_end_effector_pos = calculated_end_effector_pos[:3, 3]
    
    # calculate euclidean distance from guessed and desired
    cost = np.linalg.norm(end_effector_pos - calculated_end_effector_pos)
    
    return cost

def calculate_jacobian_FD(joint_angles, delta):
    """
    Calculate the Jacobian matrix using finite differences.

    This function computes the Jacobian matrix for a given set of joint angles using finite differences.

    Args:
        joint_angles (numpy.ndarray), (3,): The current joint angles. A numpy array with 3 elements.
        delta (float): The perturbation value used to approximate the partial derivatives.

    Returns:
        numpy.ndarray: The Jacobian matrix. A 3x3 numpy array representing the linear mapping
        between joint velocity and end-effector linear velocity.
    """
    J = np.zeros((3, 3))

    for i in range(3):
        joint_angle = copy.deepcopy(joint_angles)
        
        # change by delta
        joint_angle[i] += delta

        # original end_effector_pos
        original_end_effector_pos = forward_kinematics.fk_foot(joint_angles)[:3, 3]
        
        # calculate end_effector_pos from changed joint angle
        perturbed_end_effector_pos = forward_kinematics.fk_foot(joint_angle)[:3, 3]

        # calculate partial derivative
        partial_derivative = (perturbed_end_effector_pos - original_end_effector_pos) / delta
        
        # assign partial derivative
        J[:, i] = partial_derivative

    return J

def calculate_inverse_kinematics(end_effector_pos, guess):
    """
    Calculate the inverse kinematics solution using the Newton-Raphson method.

    This function iteratively refines a guess for joint angles to achieve a desired end-effector position.
    It uses the Newton-Raphson method along with a finite difference Jacobian to find the solution.

    Args:
        end_effector_pos (numpy.ndarray): The desired XYZ coordinates of the end-effector.
            A numpy array with 3 elements.
        guess (numpy.ndarray): The initial guess for joint angles. A numpy array with 3 elements.

    Returns:
        numpy.ndarray: The refined joint angles that achieve the desired end-effector position.
    """

    # Initialize previous cost to infinity
    previous_cost = np.inf
    # Initialize the current cost to 0.0
    cost = 0.0
    
    for iters in range(MAX_ITERATIONS):

        # calculate Jacobian based on guess
        J = calculate_jacobian_FD(guess, PERTURBATION)
        
        # calculate end effector position for guess
        calculated_end_effector_pos = forward_kinematics.fk_foot(guess)[:3, 3]
        
        # calculate residual 
        residual = end_effector_pos - calculated_end_effector_pos
        
        # moore-penrose pseudoinverse of jacobian
        step = np.linalg.pinv(J) @ residual
        
        # take newton step
        guess += step
        
        # calculate the cost based on the updated guess
        cost = ik_cost(end_effector_pos, guess)
        if abs(previous_cost - cost) < TOLERANCE:
            break
        previous_cost = cost
    
    return guess
