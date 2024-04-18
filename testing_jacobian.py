import numpy as np

def analytical_jacobian(theta1, theta2, L1, L2):
    J = np.array([
        [-L1 * np.sin(theta1) - L2 * np.sin(theta1 + theta2), -L2 * np.sin(theta1 + theta2)],
        [L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2), L2 * np.cos(theta1 + theta2)]
    ])
    return J

def calculate_jacobian_FD(joint_angles, delta, forward_kinematics):
    J = np.zeros((2, 2))

    for i in range(2):
        joint_angle = joint_angles.copy()
        
        # change by delta
        joint_angle[i] += delta

        # original end_effector_pos
        original_end_effector_pos = forward_kinematics(joint_angles)
        
        # calculate end_effector_pos from changed joint angle
        perturbed_end_effector_pos = forward_kinematics(joint_angle)

        # calculate partial derivative
        partial_derivative = (perturbed_end_effector_pos - original_end_effector_pos) / delta
        
        # assign partial derivative
        J[:, i] = partial_derivative

    return J

def forward_kinematics(joint_angles, L1=1.0, L2=0.5):
    x = L1 * np.cos(joint_angles[0]) + L2 * np.cos(joint_angles[0] + joint_angles[1])
    y = L1 * np.sin(joint_angles[0]) + L2 * np.sin(joint_angles[0] + joint_angles[1])
    return np.array([x, y])

# Example values
theta1 = np.pi / 4  # 45 degrees
theta2 = np.pi / 6  # 30 degrees
L1 = 1.0
L2 = 0.5
joint_angles = np.array([theta1, theta2])
delta = 0.0001

# Calculate analytical Jacobian
J_analytical = analytical_jacobian(theta1, theta2, L1, L2)

# Print analytical Jacobian
print("Analytical Jacobian Matrix:")
print(J_analytical)

# Calculate finite difference Jacobian
J_fd = calculate_jacobian_FD(joint_angles, delta, forward_kinematics)

# Print finite difference Jacobian
print("\nFinite Difference Jacobian Matrix:")
print(J_fd)
