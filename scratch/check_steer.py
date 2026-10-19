import numpy as np
import quaternion

# In Habitat:
# Agent rotation is a quaternion
# Forward vector is [0, 0, -1]
# Right vector is [1, 0, 0]
# Left vector is [-1, 0, 0]

import habitat_sim

# Create a rotation around [0, 1, 0] with angle steer
steer = 0.5 # ~28.6 degrees
rot = habitat_sim.utils.quat_from_angle_axis(steer, np.array([0, 1.0, 0]))

# Let's rotate the forward vector [0, 0, -1]
fwd = np.array([0, 0, -1.0])
new_fwd = habitat_sim.utils.quat_rotate_vector(rot, fwd)

print(f"Original forward: {fwd}")
print(f"Steer: {steer} rad")
print(f"New forward: {new_fwd}")

# If new_fwd.x > 0, it turned towards +X (RIGHT in Habitat)
# If new_fwd.x < 0, it turned towards -X (LEFT in Habitat)
if new_fwd[0] > 0:
    print(f"RESULT: steer = +{steer} turns RIGHT! (X = +{new_fwd[0]:.3f})")
else:
    print(f"RESULT: steer = +{steer} turns LEFT! (X = {new_fwd[0]:.3f})")

steer_neg = -0.5
rot_neg = habitat_sim.utils.quat_from_angle_axis(steer_neg, np.array([0, 1.0, 0]))
new_fwd_neg = habitat_sim.utils.quat_rotate_vector(rot_neg, fwd)
if new_fwd_neg[0] > 0:
    print(f"RESULT: steer = {steer_neg} turns RIGHT! (X = +{new_fwd_neg[0]:.3f})")
else:
    print(f"RESULT: steer = {steer_neg} turns LEFT! (X = {new_fwd_neg[0]:.3f})")
