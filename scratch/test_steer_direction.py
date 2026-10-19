import numpy as np
import habitat_sim

# Habitat convention:
# Forward vector is [0, 0, -1]
# Right vector is [1, 0, 0]
# Up vector is [0, 1, 0]

forward_vec = np.array([0, 0, -1.0])
print(f"Initial forward vector: {forward_vec}")

# Positive steer: steer = +0.5 rad (~28 degrees)
steer = +0.5
rot = habitat_sim.utils.quat_from_angle_axis(steer, np.array([0, 1.0, 0]))
new_forward = habitat_sim.utils.quat_rotate_vector(rot, forward_vec)
print(f"After steer = +{steer} rad:")
print(f"new_forward: {new_forward}")
print(f"X coordinate: {new_forward[0]:.4f} (Positive X = RIGHT, Negative X = LEFT)")
if new_forward[0] > 0:
    print("=> Positive steer turns RIGHT!")
else:
    print("=> Positive steer turns LEFT!")

# Negative steer: steer = -0.5 rad
steer_neg = -0.5
rot_neg = habitat_sim.utils.quat_from_angle_axis(steer_neg, np.array([0, 1.0, 0]))
new_forward_neg = habitat_sim.utils.quat_rotate_vector(rot_neg, forward_vec)
print(f"\nAfter steer = {steer_neg} rad:")
print(f"new_forward: {new_forward_neg}")
print(f"X coordinate: {new_forward_neg[0]:.4f}")
if new_forward_neg[0] > 0:
    print("=> Negative steer turns RIGHT!")
else:
    print("=> Negative steer turns LEFT!")
