import numpy as np

def wrap_angle(angle):
    """wraps an angle to [-pi, pi)"""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def get_relative_position(pos, goal_pos):
    """
    pos and goal_pos are [x, y, theta],
    returns [goal_rel_x, goal_rel_y, rel_theta]
    """
    return np.array([
        goal_pos[0] - pos[0],
        goal_pos[1] - pos[1],
        wrap_angle(goal_pos[2] - pos[2]),
    ])
    
