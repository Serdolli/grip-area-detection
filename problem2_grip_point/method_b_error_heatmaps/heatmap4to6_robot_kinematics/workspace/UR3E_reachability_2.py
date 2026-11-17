import pybullet as p
import numpy as np
import matplotlib.pyplot as plt
import os

PROJECT_ROOT = os.environ.get("GRIP_PROJECT_ROOT", os.path.abspath("project_root"))   # see README

# ===================================================================
#  SETTINGS
# ===================================================================
URDF_DIR  = os.path.join(PROJECT_ROOT, "..", r"urdf_files_dataset-main\urdf_files\ros-industrial\xacro_generated\universal_robots\ur_description\urdf")
URDF_FILE = "ur3e.urdf"

Z_HEIGHT     = 0.30    # metres above table - keep EE above floor
RANGE_LIMIT  = 0.65    # scan radius (UR3e max reach ~ 590 mm)
RESOLUTION   = 0.01    # 1 cm grid spacing
IK_TOLERANCE = 0.02    # accept IK solution if pos error < 2 cm

# -------------------------------------------------------------------
#  CONSTRAINT EXPLANATION
#
#  1. VACUUM GRIPPER = tool Z-axis must point straight DOWN into table
#     -> Roll=0, Pitch=pi, Yaw=0  ->  quaternion from [0, pi, 0]
#     This keeps the suction cup face flat on the workpiece surface.
#
#  2. ELBOW-UP configuration:
#     On a UR robot the joints are numbered:
#       J0 = base rotation
#       J1 = shoulder lift      <- negative = arm sweeps upward/back
#       J2 = elbow              <- POSITIVE  = elbow points UP
#       J3 = wrist 1
#       J4 = wrist 2            <- -pi/2 keeps gripper face down
#       J5 = wrist 3 (tool rot)
#
#     Elbow-up means J2 > 0  (elbow above the arm plane).
#     We enforce this in two ways:
#       a) All IK seeds start in an elbow-up configuration
#       b) After IK we REJECT any solution where J2 <= 0
# -------------------------------------------------------------------

# Gripper orientation: straight down (vacuum pick)
# RPY = [0, pi, 0]  ->  tool Z points -world Z
GRIPPER_ORN = None  # filled after pybullet starts (needs the function)

# IK seeds - ALL are elbow-up (J2 > 0) and wrist-down (J4 = -pi/2)
# Vary only J0 (base) to give the solver different starting angles
# so it doesn't get stuck in the singularity notch
IK_SEEDS_BASE = [0, np.pi/4, np.pi/2, 3*np.pi/4,
                 np.pi, -np.pi/4, -np.pi/2, -3*np.pi/4]

def make_seeds():
    """
    Each seed is [J0, J1, J2, J3, J4, J5].
    J1 = -pi/3 : shoulder tilted back a bit  (arm arches upward)
    J2 = +pi/2 : elbow pointing UP           <- the elbow-up constraint
    J3 = -pi/6 : wrist 1 slight bend to keep gripper vertical
    J4 = -pi/2 : wrist 2 pointing down       <- keeps suction cup flat
    J5 =  0    : tool rotation free
    """
    return [
        [b, -np.pi/3, np.pi/2, -np.pi/6, -np.pi/2, 0]
        for b in IK_SEEDS_BASE
    ]

IK_SEEDS = make_seeds()

# ===================================================================
#  PYBULLET INIT
# ===================================================================
client = p.connect(p.DIRECT)
p.setAdditionalSearchPath(URDF_DIR)

try:
    robot_id = p.loadURDF(URDF_FILE, [0, 0, 0], useFixedBase=True)
    print(f"[OK] Robot loaded (ID: {robot_id})")
except Exception as e:
    print(f"[X] Could not load URDF: {e}")
    exit()

GRIPPER_ORN = p.getQuaternionFromEuler([0, np.pi, 0])

# ===================================================================
#  JOINT DETECTION
# ===================================================================
n_joints = p.getNumJoints(robot_id)
movable_joints = [i for i in range(n_joints)
                  if p.getJointInfo(robot_id, i)[2] != p.JOINT_FIXED]

# Index of elbow joint inside the movable_joints list (0-based)
# UR robots: movable joint order is always  base, shoulder, elbow, wrist1, wrist2, wrist3
ELBOW_IDX_IN_MOVABLE = 2   # 3rd movable joint = elbow

# Auto-detect end-effector link
ee_index = n_joints - 1
for i in range(n_joints - 1, -1, -1):
    info = p.getJointInfo(robot_id, i)
    link_name = info[12].decode("utf-8") if isinstance(info[12], bytes) else info[12]
    if any(k in link_name.lower() for k in ("ee", "tool", "flange")):
        ee_index = i
        break

print(f"   Movable joints   : {movable_joints}")
print(f"   Elbow joint idx  : {movable_joints[ELBOW_IDX_IN_MOVABLE]}  "
      f"(movable[{ELBOW_IDX_IN_MOVABLE}])")
print(f"   End-effector     : link {ee_index}  "
      f"({p.getJointInfo(robot_id, ee_index)[12]})")
print(f"   Gripper orient.  : RPY=[0, pi, 0]  (straight down, vacuum)")
print(f"   Config           : elbow-UP enforced  (J2 > 0 required)\n")

# ===================================================================
#  MULTI-SEED IK WITH ELBOW-UP VALIDATOR
# ===================================================================
def best_ik_elbow_up(target_pos):
    """
    Try every seed. For each candidate solution:
      - measure position error
      - REJECT if elbow joint angle <= 0  (that's elbow-DOWN)
    Return the elbow-UP solution with the smallest position error.
    If no elbow-up solution exists, return (None, inf).
    """
    best_angles = None
    best_error  = float("inf")

    for seed in IK_SEEDS:
        # Warm-start the solver from this seed
        for idx, jidx in enumerate(movable_joints):
            if idx < len(seed):
                p.resetJointState(robot_id, jidx, seed[idx])

        angles = p.calculateInverseKinematics(
            robot_id,
            ee_index,
            target_pos,
            targetOrientation=GRIPPER_ORN,
            maxNumIterations=300,
            residualThreshold=1e-5,
        )

        # -- Elbow-up check -----------------------------------------
        if len(angles) <= ELBOW_IDX_IN_MOVABLE:
            continue  # solver returned too few angles (shouldn't happen)

        elbow_angle = angles[ELBOW_IDX_IN_MOVABLE]

        # UR convention: elbow-up means J2 (elbow) > 0
        # We add a small dead-band (> 0.05 rad ~ 3 deg) to avoid borderline cases
        if elbow_angle <= 0.05:
            continue   # <- REJECT: this is an elbow-down solution

        # -- Position accuracy check --------------------------------
        for idx, jidx in enumerate(movable_joints):
            if idx < len(angles):
                p.resetJointState(robot_id, jidx, angles[idx])

        actual = np.array(p.getLinkState(robot_id, ee_index)[4])
        err    = np.linalg.norm(actual - np.array(target_pos))

        if err < best_error:
            best_error  = err
            best_angles = angles

    return best_angles, best_error

# ===================================================================
#  GRID SCAN
# ===================================================================
x_coords = np.arange(-RANGE_LIMIT, RANGE_LIMIT + RESOLUTION, RESOLUTION)
y_coords = np.arange(-RANGE_LIMIT, RANGE_LIMIT + RESOLUTION, RESOLUTION)
grid = np.full((len(y_coords), len(x_coords)), np.nan)

print(f"Scanning {len(x_coords)}x{len(y_coords)} grid ...")
print(f"  Z = {Z_HEIGHT} m  |  {len(IK_SEEDS)} seeds  |  elbow-up + vacuum orientation\n")

reachable_radii = []
collision_radii = []

for i, y in enumerate(y_coords):
    if i % 15 == 0:
        print(f"  {100*i/len(y_coords):5.1f}% ...", end="\r")

    for j, x in enumerate(x_coords):
        dist = np.hypot(x, y)
        if dist > RANGE_LIMIT:
            continue

        angles, error = best_ik_elbow_up([x, y, Z_HEIGHT])

        if angles is not None and error < IK_TOLERANCE:
            # Apply best solution, then check self-collision
            for idx, jidx in enumerate(movable_joints):
                if idx < len(angles):
                    p.resetJointState(robot_id, jidx, angles[idx])

            contacts = p.getContactPoints(robot_id, robot_id)
            if len(contacts) == 0:
                grid[i, j] = 1.0           # [OK] reachable, elbow-up, no collision
                reachable_radii.append(dist)
            else:
                grid[i, j] = 0.5           # [WARNING] reachable but collision
                collision_radii.append(dist)
        else:
            grid[i, j] = 0.0               # [X] not reachable in elbow-up config

p.disconnect()
print("\nScan complete.          ")

# ===================================================================
#  STATISTICS
# ===================================================================
if reachable_radii:
    inner_r = np.min(reachable_radii)
    outer_r = np.max(reachable_radii)
    total   = len(reachable_radii) + len(collision_radii)
    pct     = 100 * len(reachable_radii) / max(total, 1)

    print("\n" + "="*52)
    print(f"  REACHABILITY SUMMARY  (Z={Z_HEIGHT}m, elbow-up, vacuum)")
    print("="*52)
    print(f"  Reachable (collision-free)   : {len(reachable_radii):>6}  pts")
    if collision_radii:
        print(f"  Collision                    : {len(collision_radii):>6}  pts")
    print(f"  Min reach radius             : {inner_r*100:>5.1f} cm  ({inner_r:.4f} m)")
    print(f"  Max reach radius             : {outer_r*100:>5.1f} cm  ({outer_r:.4f} m)")
    print(f"  Effective working annulus    : {inner_r*100:.1f} - {outer_r*100:.1f} cm")
    print("="*52)
else:
    print("[WARNING] No reachable points found.")
    inner_r = outer_r = None

# ===================================================================
#  VISUALISATION
# ===================================================================
fig, ax = plt.subplots(figsize=(10, 9))
extent  = [-RANGE_LIMIT, RANGE_LIMIT, -RANGE_LIMIT, RANGE_LIMIT]

img = ax.imshow(
    grid,
    extent=extent,
    origin="lower",
    cmap="RdYlGn",
    vmin=0, vmax=1,
    alpha=0.85,
    interpolation="nearest",
)

cbar = plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
cbar.set_ticks([0, 0.5, 1])
cbar.set_ticklabels(["Not reachable / elbow-down", "Collision", "Reachable"])

if inner_r and outer_r:
    ax.add_patch(plt.Circle((0,0), inner_r, color="dodgerblue",
                             fill=False, lw=2, ls="--",
                             label=f"Min reach: {inner_r*100:.1f} cm"))
    ax.add_patch(plt.Circle((0,0), outer_r, color="navy",
                             fill=False, lw=2, ls="-",
                             label=f"Max reach: {outer_r*100:.1f} cm"))

    ang = np.pi / 4   # annotate at 45 deg for readability
    ax.annotate(f"max {outer_r*100:.1f} cm",
                xy=(outer_r*np.cos(ang), outer_r*np.sin(ang)),
                xytext=(outer_r*np.cos(ang)+0.04, outer_r*np.sin(ang)+0.04),
                fontsize=9, color="navy",
                arrowprops=dict(arrowstyle="->", color="navy"))
    ax.annotate(f"min {inner_r*100:.1f} cm",
                xy=(inner_r*np.cos(ang), inner_r*np.sin(ang)),
                xytext=(inner_r*np.cos(ang)-0.14, inner_r*np.sin(ang)+0.06),
                fontsize=9, color="dodgerblue",
                arrowprops=dict(arrowstyle="->", color="dodgerblue"))

ax.scatter([0], [0], color="black", marker="+", s=200, zorder=5, label="Robot base")
ax.set_title(
    f"UR3e Reachability Map  -  Z = {Z_HEIGHT} m\n"
    f"Elbow-UP  |  Vacuum gripper (tool Z down)  |  Res: {RESOLUTION*100:.1f} cm",
    fontsize=13, fontweight="bold"
)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_xlim(-RANGE_LIMIT, RANGE_LIMIT)
ax.set_ylim(-RANGE_LIMIT, RANGE_LIMIT)
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig("ur3e_reachability_elbowup_vacuum.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved -> ur3e_reachability_elbowup_vacuum.png")