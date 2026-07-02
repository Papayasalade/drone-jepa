"""State / action layout for SkyJEPA (Tier 1).

State x in R^18 = position(3) + velocity(3) + R flattened(9) + angular_vel(3).
Action a in R^4 = four individual rotor forces (Newtons).

We keep a single canonical ordering everywhere so encoders, the DKI and the
data pipeline never disagree about which slice is which.
"""

from __future__ import annotations

# ---- state layout (18-dim) -------------------------------------------------
STATE_DIM = 18
POS = slice(0, 3)      # world-frame position p  [m]
VEL = slice(3, 6)      # world-frame velocity v  [m/s]
ROT = slice(6, 15)     # rotation matrix R, row-major flatten (body->world)
OMEGA = slice(15, 18)  # body-frame angular velocity omega  [rad/s]

# ---- action layout (4-dim) -------------------------------------------------
ACTION_DIM = 4         # individual rotor forces f_i  [N]

# gravity (world frame, z-up)
GRAVITY = 9.81
