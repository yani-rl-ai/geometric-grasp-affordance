#!/usr/bin/env python3
"""
Shared model / constants / control code for the Fetch grasping simulation.

  - Sim            : MuJoCo physics + Cartesian (mocap-welded gripper) control, robot
                     placement P0..P3 (PLACEMENTS), per-trial reset (Sim.reset),
                     head-camera aiming (calibrate_head).
  - OBJECTS        : the five YCB objects (names; pre-tuned grasp constants are only
                     used by the legacy grasp_object() path, not by the paper pipeline).
  - grasp_object_via_candidate : executes a perception-derived grasp candidate
                     (the path evaluated in the paper, top-down entry).
  - get_intrinsics : pinhole intrinsics of the head camera.

Rendering/perception live in perception.py, YOLO in yolo_segmenter.py, the graph
in dbl_gng_paper.py, the geometric planner in grasp_planner.py.
"""
import os

import mujoco
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "scene.xml")

CAM_NAME = "head_camera_rgb"
IMG_W, IMG_H = 640, 480
DEPTH_MIN, DEPTH_MAX = 0.2, 2.5  # meters, for depth colormap normalization

TABLE_TOP_Z = 0.72
TABLE_CENTER_XY = np.array([1.05, 0.2641])
TRANSIT_Z = TABLE_TOP_Z + 0.35  # safe lateral-travel height, above tallest object
SITE_LOCAL_X_OFFSET = 0.02      # robot0:grip site offset along gripper_link local +X

GRIPPER_OPEN = 0.05
GRIPPER_CLOSED = 0.0

# Per-object grasp definition:
#   site_z     : world Z (meters) the grip site should be at while grasping.
#                The gripper fingers are short (~3.8 cm reach from the palm),
#                so for a top-down grasp this must sit close to the object's
#                TOP surface (fingertips wrap the upper rim while the palm
#                clears just above it) -- NOT at the object's mid-height,
#                otherwise the palm collides with the object before the
#                fingers ever get around it.
#   yaw90      : True -> rotate gripper 90 deg so fingers close along world X
#                (used for elongated objects whose short axis is along X)
#   grasp_half_width : half the object's cross-section along the finger
#                closing axis at the grasp height -- the finger close target
#                is derived from this so the fingers stop AT the object
#                surface (with a small squeeze) instead of slamming to a
#                fully-closed ctrl=0 and crushing/pushing the object.
#   place_xy   : where to place the object after picking it up (used by
#                check_headless.py's standalone place-elsewhere demo only --
#                the live GUI's actual "Place" button, sim_core.place_object,
#                returns the object to where it was actually picked up from
#                instead, see below)
OBJECTS = [
    # This whole row was originally at Y=-0.12 -- only ~6.6cm from the
    # table's front edge (Y=-0.186). Fine for the smaller/lighter mug,
    # banana, and tennis_ball, but check_headless.py showed cracker_box and
    # mustard_bottle (bigger footprints) landing at final Z=0.08 and 0.04 m
    # -- on the FLOOR, not the table -- after being placed there: netting
    # out their own half-width left only ~1-3cm of real clearance, not
    # enough margin for any settling/sliding after release. Moved the row
    # to Y=0.00, clearing the front edge by a real margin for every object
    # while staying clear of the object spawn cluster above it (checked all
    # 5x5 place-vs-spawn pairs stay well past their combined half-widths).
    dict(name="cracker_box",    site_z=0.928, yaw90=True,  grasp_half_width=0.036, place_xy=(0.86, 0.00)),
    dict(name="mustard_bottle", site_z=0.882, yaw90=False, grasp_half_width=0.029, place_xy=(0.97, 0.00)),
    dict(name="mug",            site_z=0.796, yaw90=False, grasp_half_width=0.046, place_xy=(1.08, 0.00)),
    dict(name="banana",         site_z=0.750, yaw90=True,  grasp_half_width=0.018, place_xy=(1.19, 0.00)),
    # tennis_ball needs its own, much gentler squeeze_margin (see close_ctrl_for):
    # the default 4mm forced overlap is fine for the other 4 (boxy/heavier)
    # objects but was enough, for this small (58g) rigid sphere, to build up
    # a contact-force spike that shot it sideways out of the gripper --
    # confirmed by stepping through the close motion: finger qpos overshot
    # PAST its own commanded target by ~4-5mm (the contact pushing the
    # fingers open harder than the position actuator could hold), right as
    # the ball started accelerating away.
    dict(name="tennis_ball",    site_z=0.760, yaw90=False, grasp_half_width=0.032, place_xy=(1.30, 0.00), squeeze_margin=0.001),
]

SQUEEZE_MARGIN = 0.004  # extra closure past the surface for a firm grip (m)
OBJECTS_BY_NAME = {o["name"]: o for o in OBJECTS}

# Perception-path-only squeeze_margin overrides, separate from OBJECTS'
# per-object squeeze_margin (which also feeds close_ctrl_for() for the
# pre-tuned grasp_object() path -- unchanged, still works, don't touch it).
#
# mustard_bottle's candidate site_z lands on its neck/cap (see the comment
# in grasp_object_via_candidate) rather than the wider body the pre-tuned
# path grabs, so candidate["width"] (measured across the WHOLE bottle, not
# just the cap) overstates the true local diameter at that height -- the
# default margin then leaves the fingers stopping short of the cap's
# actual (narrower) surface, never making contact. Empirically re-tuned
# with real (not reused-candidate) trials each time the camera's view of
# it changed -- the exact needed value has moved twice already as
# scene.xml/head_pan changed (0.012, then 0.013 was the bare minimum after
# a head_pan retune shifted the measured candidate width): 0.020 keeps a
# safety buffer clear of that kind of edge. This object is heavy enough
# (0.6kg) that the contact-force-spike/launch risk that ruled out a bigger
# margin for the tennis ball doesn't apply here. If scene.xml or the head
# camera's aim changes again, re-verify with real
# `run_single_grasp_trial.py --object mustard_bottle` runs, not a reused
# candidate in a loop -- the latter gives false confidence (see git history
# for how that happened once already).
# cracker_box: opposite problem and a NEGATIVE margin (i.e. extra opening,
# not extra squeeze). After the cracker_box scene.xml reposition + head_pan
# recalibration (see scene.xml/calibrate_head's comments), its measured
# candidate width dropped to ~0.052m from the ~0.072m true width -- a much
# bigger view-angle-driven swing than the usual ~0.015m GNG_WIDTH_BIAS
# undershoot, so the default margin over-squeezed (closing to ~0.022m
# against a true half-width of 0.036m, well past the real surface). -0.007
# backs that off to line up with the true surface instead. Same caveat as
# mustard_bottle above: this number is a function of the CURRENT camera
# angle to this object, not a fixed property of the box -- it already had
# to change once (from a positive default margin being fine, to needing
# this negative override) purely from a head_pan/position change, and will
# likely need re-tuning again if either moves further.
#
# FROZEN EMPTY for the pose-level evaluation: with the grip site 55 mm below
# the perceived top (GRASP_DEPTH_DEFAULT) every object uses its nominal
# margin (SQUEEZE_MARGIN, or the per-object value in OBJECTS), so there are
# no per-object execution overrides any more. The old values were
# {"mustard_bottle": 0.020, "cracker_box": -0.007}.
PERCEPTION_SQUEEZE_OVERRIDE = {}
if os.environ.get("LEGACY_OVERRIDES") == "1":   # ablation: the earlier hand-tuned per-object margins
    PERCEPTION_SQUEEZE_OVERRIDE.update({"mustard_bottle": 0.020, "cracker_box": -0.007})


def close_ctrl_for(obj):
    margin = obj.get("squeeze_margin", SQUEEZE_MARGIN)
    return max(0.0, obj["grasp_half_width"] - margin)


READY_QPOS = {
    # Verified collision-free at t=0 AND clear of the head camera's line of
    # sight (arm tucked up and panned to the robot's left, out of frame).
    # An earlier tuck pose (shoulder_pan=0) was collision-free but swung the
    # forearm directly in front of the head camera, filling the RGB/depth
    # feed with the arm itself -- see the render-based occlusion search that
    # picked this configuration (min camera depth 0.5 m vs 0.06 m before).
    "robot0:torso_lift_joint": 0.20,
    "robot0:shoulder_pan_joint": -1.4,
    "robot0:shoulder_lift_joint": -0.9,
    "robot0:upperarm_roll_joint": 0.0,
    "robot0:elbow_flex_joint": 1.7,
    "robot0:forearm_roll_joint": 0.0,
    "robot0:wrist_flex_joint": 1.1,
    "robot0:wrist_roll_joint": 0.0,
    "robot0:r_gripper_finger_joint": GRIPPER_OPEN,
    "robot0:l_gripper_finger_joint": GRIPPER_OPEN,
}

# --------------------------------------------------------------------------
# Rotation matrices for a top-down grasp approach (gripper local +X points
# straight down, i.e. world -Z). Two variants differ by 90 deg about the
# approach axis so the finger-opening axis (local Y) can align with whichever
# world axis is the *short* axis of the target object's footprint.
# --------------------------------------------------------------------------
R_TOPDOWN_0 = np.array([0, 0, 1, 0, 1, 0, -1, 0, 0], dtype=np.float64)   # local Y -> world Y
R_TOPDOWN_90 = np.array([0, 1, 0, 0, 0, -1, -1, 0, 0], dtype=np.float64)  # local Y -> world X


def mat2quat(mat9):
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat9)
    return quat


QUAT_TOPDOWN_0 = mat2quat(R_TOPDOWN_0)
QUAT_TOPDOWN_90 = mat2quat(R_TOPDOWN_90)

# --------------------------------------------------------------------------
# Side-approach orientation: gripper reaches in HORIZONTALLY (local +X ->
# world +X, the same direction the arm already extends into the workspace
# from the robot's base) instead of straight down, with the finger-closing
# axis VERTICAL (local Y -> world Z) instead of horizontal -- for gripping
# an object around its HEIGHT (e.g. hugging a mug's body from the front)
# rather than its horizontal footprint. Matches grasp_planner's "side"
# candidate (close_axis=2/Z); see grasp_object_via_candidate's side branch.
# --------------------------------------------------------------------------
R_SIDE_X = np.array([1, 0, 0, 0, 0, -1, 0, 1, 0], dtype=np.float64)  # local Y -> world Z
QUAT_SIDE_X = mat2quat(R_SIDE_X)
SIDE_STANDOFF = 0.18  # meters, horizontal clearance before the final approach


def slerp(q0, q1, t):
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    theta0 = np.arccos(dot)
    theta = theta0 * t
    q2 = q1 - q0 * dot
    q2 = q2 / np.linalg.norm(q2)
    return q0 * np.cos(theta) + q2 * np.sin(theta)


def site_target_to_mocap(xy, site_z, quat):
    """Convert a desired world position for the robot0:grip site into the
    robot0:mocap target (mocap is welded to gripper_link, whose origin is
    offset from the site by SITE_LOCAL_X_OFFSET along local +X)."""
    r = np.zeros(9)
    mujoco.mju_quat2Mat(r, quat)
    r = r.reshape(3, 3)
    offset_world = r @ np.array([SITE_LOCAL_X_OFFSET, 0, 0])
    site_pos = np.array([xy[0], xy[1], site_z])
    return site_pos - offset_world


def calibrate_head(model, data):
    """Search head_pan / head_tilt so the RGB-D camera looks at the
    centroid of the 5 spawned objects (not the table's fixed geometric
    center) -- the two aren't the same point, since the objects are laid
    out closer to the robot and skewed in Y, and aiming at the table's own
    center risked framing them off-balance (some objects closer to the
    edge of the camera's field of view than others). Using the objects'
    own centroid instead means the aim self-corrects whenever their
    scene.xml positions change, instead of needing a fresh manual pan/tilt
    retune each time. Uses forward kinematics only (no simulation
    stepping), and object spawn positions (model.body_pos) rather than
    current xpos, since calibration runs once at startup before any
    physics settling or manual repositioning."""
    obj_xy = np.array([model.body(name).pos[:2] for name in OBJECTS_BY_NAME])
    centroid_xy = obj_xy.mean(axis=0)
    target = np.array([centroid_xy[0], centroid_xy[1], TABLE_TOP_Z])
    pan_range = np.linspace(-0.3, 0.3, 25)
    tilt_range = np.linspace(-0.2, 1.2, 60)
    best = None
    for pan in pan_range:
        data.joint("robot0:head_pan_joint").qpos[0] = pan
        for tilt in tilt_range:
            data.joint("robot0:head_tilt_joint").qpos[0] = tilt
            mujoco.mj_forward(model, data)
            cam_id = model.camera(CAM_NAME).id
            cam_pos = data.cam_xpos[cam_id]
            cam_mat = data.cam_xmat[cam_id].reshape(3, 3)
            forward = -cam_mat[:, 2]
            to_target = target - cam_pos
            to_target /= np.linalg.norm(to_target)
            ang = np.arccos(np.clip(np.dot(forward, to_target), -1, 1))
            if best is None or ang < best[0]:
                best = (ang, pan, tilt)
    ang, pan, tilt = best
    data.joint("robot0:head_pan_joint").qpos[0] = pan
    data.joint("robot0:head_tilt_joint").qpos[0] = tilt
    mujoco.mj_forward(model, data)
    print(f"[calib] head_pan={pan:.3f} rad, head_tilt={tilt:.3f} rad, "
          f"aim error={np.degrees(ang):.2f} deg", flush=True)
    return pan, tilt


# Robot placements used by the pose-level evaluation (paper Sec. V-B). The Fetch
# base is put on different sides of the table, always yaw-facing the object
# centroid, while the table and the objects stay exactly where scene.xml puts
# them (so the axis-aligned world-frame extents used by grasp_planner stay valid).
#   P0  front (the original placement; the ONLY placement used for calibration)
#   P1  back (opposite side of the table)
#   P2  +y side
#   P3  front-diagonal, 20 deg from P0
# They were chosen from 36 candidates by scenario_search.py (collision-free base,
# all five objects segmented by YOLO26s-seg at confidence 0.05).
PLACEMENTS = {
    "P0": dict(xy=(0.2869, 0.2641), yaw=0.0),
    "P1": dict(xy=(1.632, 0.296), yaw=float(np.pi)),
    "P2": dict(xy=(1.054, 0.985), yaw=float(np.radians(260))),
    "P3": dict(xy=(0.274, 0.057), yaw=float(np.radians(20))),
}
PLACEMENT_ORDER = ["P0", "P1", "P2", "P3"]


def apply_scenario(model, placement):
    """Moves/rotates robot0:base_link in the (not yet instantiated) model.
    placement: a key of PLACEMENTS ("P0".."P3") or a dict(xy=(x, y), yaw=rad)."""
    spec = PLACEMENTS[placement] if isinstance(placement, str) else placement
    bid = model.body("robot0:base_link").id
    model.body_pos[bid][:2] = spec["xy"]
    model.body_quat[bid] = [np.cos(spec["yaw"] / 2), 0, 0, np.sin(spec["yaw"] / 2)]


FINGER_KP_DEFAULT = 30000.0   # scene.xml value; GRASP_FINGER_KP overrides (finger servo stiffness -> grip force)
GRASP_DEPTH_DEFAULT = 0.055   # grip site below the perceived top (m), calibrated on P0 (plateau 50-65 mm); GRASP_DEPTH_M overrides (the earlier benchmarks used 0.005)


class Sim:
    """Physics + Cartesian teleop control. No rendering/camera code here --
    that lives in perception.py (offscreen head camera) and run_live_demo.py."""

    def __init__(self, scenario=None):
        self.scenario = scenario or os.environ.get("GRASP_SCENARIO", "P0")  # "P0".."P3" or dict(xy, yaw)
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        apply_scenario(self.model, self.scenario)
        bid = self.model.body("robot0:base_link").id
        self._base_id = bid
        self._base_xy0 = self.model.body_pos[bid][:2].copy()
        self._base_yaw0 = 2.0 * np.arctan2(self.model.body_quat[bid][3], self.model.body_quat[bid][0])
        kp = float(os.environ.get("GRASP_FINGER_KP", FINGER_KP_DEFAULT))
        for aname in ("act_r_gripper_finger", "act_l_gripper_finger"):
            aid = self.model.actuator(aname).id
            self.model.actuator_gainprm[aid, 0] = kp      # position servo: force = kp * (ctrl - q)
            self.model.actuator_biasprm[aid, 1] = -kp
        self.data = mujoco.MjData(self.model)
        self.mocap_id = self.model.body("robot0:mocap").mocapid[0]
        self.step_hook = None  # optional callback(self), called after every mj_step
        self._init_state()

    def reset(self, dx=0.0, dy=0.0, dyaw=0.0):
        """Fresh simulation state (same effect as a new process: objects back
        at their spawn poses, zero velocities, arm at READY_QPOS), with the
        robot base shifted by (dx, dy) m and rotated by dyaw rad relative to
        the scenario's nominal placement. The head is re-aimed afterwards."""
        m = self.model
        m.body_pos[self._base_id][:2] = self._base_xy0 + np.array([dx, dy])
        yaw = self._base_yaw0 + dyaw
        m.body_quat[self._base_id] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_resetData(m, self.data)
        self._init_state()

    def _init_state(self):
        for name, val in READY_QPOS.items():
            self.data.joint(name).qpos[0] = val
        mujoco.mj_forward(self.model, self.data)

        gripper_pos = self.data.body("robot0:gripper_link").xpos.copy()
        gripper_quat = self.data.body("robot0:gripper_link").xquat.copy()
        self.data.mocap_pos[self.mocap_id] = gripper_pos
        self.data.mocap_quat[self.mocap_id] = gripper_quat
        self.cur_pos = gripper_pos.copy()
        self.cur_quat = gripper_quat.copy()
        self.home_pos = gripper_pos.copy()   # ready-pose gripper pose, reused
        self.home_quat = gripper_quat.copy()  # whenever we need to return "home"

        self.data.ctrl[self.model.actuator("act_r_gripper_finger").id] = GRIPPER_OPEN
        self.data.ctrl[self.model.actuator("act_l_gripper_finger").id] = GRIPPER_OPEN
        self.data.ctrl[self.model.actuator("act_torso_lift").id] = READY_QPOS["robot0:torso_lift_joint"]

        pan, tilt = calibrate_head(self.model, self.data)
        self.data.ctrl[self.model.actuator("act_head_pan").id] = pan
        self.data.ctrl[self.model.actuator("act_head_tilt").id] = tilt

    def step(self, n=1):
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
            if self.step_hook is not None:
                self.step_hook(self)

    def move_mocap(self, target_pos, target_quat, duration_s):
        n = max(1, int(duration_s / self.model.opt.timestep))
        start_pos, start_quat = self.cur_pos.copy(), self.cur_quat.copy()
        for i in range(1, n + 1):
            t = i / n
            t_smooth = t * t * (3 - 2 * t)  # smoothstep
            pos = start_pos + t_smooth * (target_pos - start_pos)
            quat = slerp(start_quat, target_quat, t_smooth)
            self.data.mocap_pos[self.mocap_id] = pos
            self.data.mocap_quat[self.mocap_id] = quat
            self.step(1)
        self.cur_pos, self.cur_quat = target_pos.copy(), target_quat.copy()

    def move_xy_at_transit(self, xy, quat, duration_s=1.2):
        """Move to (xy, TRANSIT_Z), safely regardless of the CURRENT height.

        If already near transit height, this is a single lateral move (the
        original, simple behavior). If not (e.g. called right after
        return_home(), which ends at the low ready-pose height), it first
        rises straight up at the current x/y before moving laterally --
        combining a large z change with an x/y change in one interpolated
        move risks cutting diagonally low across the table, which is
        exactly the bug that once knocked the cracker box onto the floor
        (an early demo sequence did this). grasp_object()/place_object()
        below can be called from a variety of starting heights, in any
        order, so the guard belongs here instead, not at each call site.
        """
        cur_xy = self.cur_xy(self.cur_quat)
        rise_target = site_target_to_mocap(cur_xy, TRANSIT_Z, self.cur_quat)
        if abs(self.cur_pos[2] - rise_target[2]) > 0.02:
            self.move_mocap(rise_target, self.cur_quat, max(0.4, duration_s * 0.4))
        target = site_target_to_mocap(xy, TRANSIT_Z, quat)
        self.move_mocap(target, quat, duration_s)

    def move_z_at_xy(self, site_z, quat, duration_s=0.8):
        """Move straight up/down to a given grip-site height, keeping the
        current x/y (so this never needs to cross over another object)."""
        target = site_target_to_mocap(self.cur_xy(quat), site_z, quat)
        self.move_mocap(target, quat, duration_s)

    def cur_xy(self, quat):
        r = np.zeros(9)
        mujoco.mju_quat2Mat(r, quat)
        offset_world = r.reshape(3, 3) @ np.array([SITE_LOCAL_X_OFFSET, 0, 0])
        site_pos = self.cur_pos + offset_world
        return site_pos[:2]

    def set_gripper(self, value, duration_s):
        self.data.ctrl[self.model.actuator("act_r_gripper_finger").id] = value
        self.data.ctrl[self.model.actuator("act_l_gripper_finger").id] = value
        self.step(max(1, int(duration_s / self.model.opt.timestep)))

    def object_pos(self, name):
        return self.data.body(name).xpos.copy()

    def return_home(self):
        """Back to the ready pose, routed through the safe transit height so
        this never cuts low across the table."""
        self.move_xy_at_transit(self.home_pos[:2], QUAT_TOPDOWN_0, duration_s=1.2)
        self.move_mocap(self.home_pos, self.home_quat, 1.2)


# ---------------------------------------------------------------------------
# LEGACY pre-tuned grasp / place actions, driven by a body name. They use the
# hand-tuned per-object site_z / yaw90 / grasp_half_width of OBJECTS and do NOT
# use perception. The paper pipeline uses grasp_object_via_candidate() below
# (perception-derived position, width and depth); these two functions are kept
# only as a reference baseline.
# ---------------------------------------------------------------------------
def grasp_object(sim, body_name):
    obj = OBJECTS_BY_NAME[body_name]
    quat = QUAT_TOPDOWN_90 if obj["yaw90"] else QUAT_TOPDOWN_0
    obj_xy = sim.object_pos(body_name)[:2].copy()

    sim.set_gripper(GRIPPER_OPEN, 0.3)
    sim.move_xy_at_transit(obj_xy, quat, duration_s=1.0)
    sim.move_z_at_xy(obj["site_z"] + 0.15, quat, duration_s=0.6)
    sim.move_z_at_xy(obj["site_z"], quat, duration_s=0.6)
    sim.set_gripper(close_ctrl_for(obj), 0.6)
    sim.move_z_at_xy(TRANSIT_Z, quat, duration_s=0.8)

    lifted_z = sim.object_pos(body_name)[2]
    success = lifted_z > TABLE_TOP_Z + 0.05
    return dict(body=body_name, quat=quat.tolist(), obj_xy=obj_xy.tolist(), success=bool(success))


def place_object(sim, held):
    body_name = held["body"]
    quat = np.array(held["quat"])
    obj_xy = np.array(held["obj_xy"])
    # held["site_z"] if present (set by grasp_object_via_candidate below, so
    # the object goes back down at the height it was actually grasped at)
    # else fall back to the pre-tuned constant grasp_object() used.
    site_z = held.get("site_z", OBJECTS_BY_NAME[body_name]["site_z"])

    sim.move_xy_at_transit(obj_xy, quat, duration_s=0.8)
    sim.move_z_at_xy(site_z, quat, duration_s=0.8)
    sim.set_gripper(GRIPPER_OPEN, 0.4)
    sim.move_z_at_xy(TRANSIT_Z, quat, duration_s=0.6)
    sim.return_home()


# ---------------------------------------------------------------------------
# Perception-driven grasp: same motion pattern as grasp_object() above, but
# every number (approach orientation, grasp height, closing width) is read
# from a LIVE grasp_planner.compute_grasp_candidate() result (built from
# that trial's own YOLO detection + DBL-GNG fit -- see gng_overlay.py /
# grasp_planner.py / benchmark_grasp_perception.py) instead of the
# pre-tuned OBJECTS table. This is what "grasp using what DBL-GNG/YOLO
# actually measured this frame" means, as opposed to grasp_object()'s
# "perception picks WHICH object, proven constants decide HOW".
# ---------------------------------------------------------------------------
def grasp_object_via_candidate(sim, body_name, candidate):
    if candidate is None or not candidate.get("feasible"):
        return dict(body=body_name, success=False, reason="no feasible grasp candidate")
    if candidate["approach"] not in ("top", "side"):
        return dict(body=body_name, success=False, reason=f"unsupported approach '{candidate['approach']}'")

    obj = OBJECTS_BY_NAME.get(body_name, {})
    default_margin = obj.get("squeeze_margin", SQUEEZE_MARGIN)
    margin = PERCEPTION_SQUEEZE_OVERRIDE.get(body_name, default_margin)
    close_ctrl = max(0.0, float(candidate["width"]) / 2.0 - margin)
    # Horizontal grasp position: from the perceived candidate (centre of the
    # GNG node bounds) by default; GRASP_XY_SOURCE=truth uses the simulator's
    # ground-truth body position instead (oracle ablation only).
    if os.environ.get("GRASP_XY_SOURCE", "perception") == "truth":
        obj_xy = sim.object_pos(body_name)[:2].copy()
    else:
        obj_xy = np.asarray(candidate["center"][:2], dtype=float).copy()

    if candidate["approach"] == "side":
        # Horizontal approach, closing around the object's HEIGHT instead
        # of its footprint. See QUAT_SIDE_X's comment for the orientation
        # convention.
        quat = QUAT_SIDE_X
        standoff_xy = obj_xy - np.array([SIDE_STANDOFF, 0.0])

        # Grasp height: the object's own observed vertical midpoint, EXCEPT
        # a fully-open gripper's lower finger extends GRIPPER_OPEN below
        # whatever height it's centered at -- for an object resting
        # directly on the table, centering on the true mid-height only
        # clears the table if the object is taller than 2x GRIPPER_OPEN
        # (10 cm). Traced directly on the mug (7.6-8.1 cm tall) via
        # contact-pair inspection: `table_top <-> l_gripper_finger_link`
        # with the finger link's approach-phase geometry -- the object's
        # own mid-height put the open lower finger ~1 cm BELOW the table,
        # dragging on it and destabilizing the whole approach well before
        # the fingers ever got near the object. Raising the grasp height to
        # guarantee table clearance for the fully-open gripper (clamped to
        # the object's own observed top, so it doesn't aim above the object
        # entirely) trades a perfectly-centered grip for one that's still
        # within the object's height but closer to its top -- the same
        # trade the top-down branch already makes for short-fingered
        # reach, just on a different axis.
        # Using a FULLY open gripper (GRIPPER_OPEN=0.05) before insertion
        # turned out to need more table clearance than a short tabletop
        # object like the mug (7.6-8.1cm tall) can offer at all: even
        # centered as high as physically sensible, the open lower finger
        # (and then the palm body) kept contacting the table. Since
        # close_ctrl is already close to the maximum this object's own
        # height allows, there's little room for "open wide, then close" --
        # so approach with the gripper barely more open than its final
        # target instead of fully open, which shrinks the needed vertical
        # clearance to almost nothing.
        side_open_ctrl = close_ctrl + 0.015
        mid_z = float(candidate["mins"][2] + candidate["maxs"][2]) / 2.0
        min_clear_z = TABLE_TOP_Z + side_open_ctrl + 0.005
        grasp_z = min(max(mid_z, min_clear_z), float(candidate["maxs"][2]) - 0.005)

        # Insert to just past the object's NEAR surface along the approach
        # axis -- mirroring the top-down branch's own site_z = top - 5mm
        # convention (graze the surface, don't drive into the object's
        # solid volume). Two deeper insertion depths were tried and traced
        # first: all the way to the object's center (obj_xy directly) and
        # 85% of the way there -- both still shoved the mug forward by
        # several cm during this move, even with the gripper barely open,
        # because the gripper's own body occupies real volume and MuJoCo
        # pushes the rigid mug out of the way rather than letting hardware
        # phase through it once any part of the approach path overlaps the
        # object's own volume.
        half_width_x = float(candidate["maxs"][0] - candidate["mins"][0]) / 2.0
        insert_xy = np.array([obj_xy[0] - half_width_x + 0.005, obj_xy[1]])

        sim.set_gripper(side_open_ctrl, 0.3)
        sim.move_xy_at_transit(standoff_xy, quat, duration_s=1.0)
        sim.move_z_at_xy(grasp_z, quat, duration_s=0.6)
        sim.move_mocap(site_target_to_mocap(insert_xy, grasp_z, quat), quat, duration_s=0.6)
        sim.set_gripper(close_ctrl, 0.6)
        sim.move_z_at_xy(TRANSIT_Z, quat, duration_s=0.8)

        lifted_z = sim.object_pos(body_name)[2]
        success = lifted_z > TABLE_TOP_Z + 0.05
        return dict(body=body_name, quat=quat.tolist(), obj_xy=obj_xy.tolist(),
                    site_z=grasp_z, close_ctrl=close_ctrl, success=bool(success))

    close_axis = candidate["close_axis"]  # 0 (x) -> QUAT_TOPDOWN_90, 1 (y) -> QUAT_TOPDOWN_0
    quat = QUAT_TOPDOWN_90 if close_axis == 0 else QUAT_TOPDOWN_0

    # Grasp near the measured top (same reasoning as the hand-tuned OBJECTS
    # site_z values: this gripper's fingers are short, so it must grasp
    # close to the object's top surface or the palm hits the object first).
    #
    # A height-search alternative was tried here (walk down from the top
    # until the local cross-section matches the planned closing width, to
    # handle a tapered object like a bottle's neck/cap) and it DID fix
    # mustard_bottle -- but it also broke cracker_box, which was already
    # working: with only ~30 GNG nodes spread over the whole object, the
    # handful nearest the very top edge of ANY object (tapered or not)
    # reads as artificially narrow just from sampling sparsity, not real
    # shape -- cracker_box's uniform, genuinely-untapered profile got
    # pulled down by the same search logic as mustard's real taper, with
    # no reliable way found to tell the two cases apart from this few
    # points. Reverted in favor of the simple, previously-validated
    # top-of-object height; mustard_bottle's neck/cap grasp remains a
    # known limitation (see grasp_planner.py's module docstring).
    depth = float(os.environ.get("GRASP_DEPTH_M", GRASP_DEPTH_DEFAULT))
    site_z = max(float(candidate["maxs"][2]) - depth, TABLE_TOP_Z + 0.02)

    # Per-object squeeze_margin override (see OBJECTS) still applies here --
    # it's about this gripper's contact dynamics with that object's mass/
    # rigidity (e.g. the tennis ball), not about where the width number
    # came from.
    #
    # Units: candidate["width"] is a FULL extent (grasp_planner computes it
    # as maxs-mins), but the finger actuators' ctrl target is a HALF-gap
    # (distance from the object's centerline to where each finger should
    # stop -- see close_ctrl_for() above, which is why OBJECTS' hand-tuned
    # grasp_half_width values are already halved). Skipping the /2 here
    # commanded the fingers to stay open at ~2x the true half-width, i.e.
    # never actually closing on the object -- confirmed as the cause of a
    # 0/75 perception-driven benchmark run across every object, including
    # simple box/bottle/mug/ball shapes that aren't the curved-banana edge
    # case this candidate math has a separate, known limitation with.
    # (margin/close_ctrl already computed above, shared with the side branch)

    sim.set_gripper(GRIPPER_OPEN, 0.3)
    sim.move_xy_at_transit(obj_xy, quat, duration_s=1.0)
    sim.move_z_at_xy(site_z + 0.15, quat, duration_s=0.6)
    sim.move_z_at_xy(site_z, quat, duration_s=0.6)
    sim.set_gripper(close_ctrl, 0.6)
    sim.move_z_at_xy(TRANSIT_Z, quat, duration_s=0.8)

    lifted_z = sim.object_pos(body_name)[2]
    success = lifted_z > TABLE_TOP_Z + 0.05
    return dict(body=body_name, quat=quat.tolist(), obj_xy=obj_xy.tolist(),
                site_z=site_z, close_ctrl=close_ctrl, success=bool(success))


def get_intrinsics(model, cam_name, width, height):
    fovy = model.camera(cam_name).fovy[0]
    fy = height / (2 * np.tan(np.radians(fovy) / 2))
    fx = fy
    cx, cy = width / 2.0, height / 2.0
    return fx, fy, cx, cy
