# Relative representation audit

The current legal Action7 is already an incremental controller-target action:

- translation: normalized delta relative to the current target, expressed in
  the fixed Aubo task/workspace axes;
- rotation: normalized local delta right-multiplied into the current target EE
  orientation;
- gripper: binary controller command.

Obs20 contains current TCP position/quaternion and wrench plus the explicit
`goal_position - tcp_position` vector and goal quaternion. Thus the task error
is directly represented without vision.

The paper's Franka implementation expresses the entire twist in the EE frame.
Changing Aubo translation axes would invalidate the semantics of all 11
accepted Action7 demonstrations and alter the validated contact controller.
Because the insertion/tool frame is fixed in this task, the existing normalized
incremental representation is the closest compatible Aubo realization and is
retained branch-locally. No action values are rewritten.

`PAPER_FIDELITY_RELATIVE_ACTION=YES` means compatible relative incremental
Action7, not bit-exact Franka body-frame control.
