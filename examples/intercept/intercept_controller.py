# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# See the LICENSE file at the repository root for full terms.

"""Run an exported Intercept policy on a Crazyflie via **cflib** (CTBR).

This controller talks to the drone directly through ``cflib`` -- no ROS 2 /
Crazyswarm2 layer. It loads the TorchScript policy + ``metadata.json`` produced
by [export_policy.py](export_policy.py) (so it needs only ``torch``, ``numpy``
and ``cflib``), reconstructs the exact Intercept observation from live drone
state (read via ``cflib`` logging), evaluates the deterministic policy, decodes
the raw action into a collective-thrust + body-rate (CTBR) command, and streams
it to the drone with ``Commander.send_setpoint``.

Command fidelity
----------------
The policy was trained on a CTBR interface (collective thrust + body rates).
``cflib``'s ``send_setpoint(roll, pitch, yawrate, thrust)`` interprets the
roll/pitch fields as **body rates** (deg/s) only when the firmware roll/pitch
stabilization mode is set to RATE. This controller therefore switches
``flightmode.stabModeRoll`` / ``flightmode.stabModePitch`` to RATE (0) on
startup and restores ANGLE (1) on shutdown, giving a faithful CTBR interface
(the same one the low-level cflib example uses).

State feedback
--------------
Live state is streamed back with ``cflib`` log blocks:

* position + world-frame velocity: ``stateEstimate.{x,y,z,vx,vy,vz}``
* orientation quaternion: ``stateEstimate.{qw,qx,qy,qz}``
* body-frame angular velocity (only when the policy needs it): ``gyro.{x,y,z}``
Motion capture (optional)
-------------------------
With ``--mocap`` the controller opens an OptiTrack/NatNet stream, transforms
each rigid-body pose into the ROS FLU convention (see
``intercept_common.transform_mocap_pose``) and forwards it to the matching
drone's on-board estimator via ``extpos.send_extpose``. The fused estimate is
then read back through the ``stateEstimate.*`` log blocks as usual, so the
policy always consumes the EKF output. Poses can additionally be published on a
ROS 2 ``/tf`` tree with ``--publish-tf`` (``rclpy`` is imported lazily only
then).
Example
-------
::

    python deploy/intercept_controller.py \\
        --uri udp://127.0.0.1:19850 \\
        --artifact-dir deploy/artifacts/intercept_ppo \\
        --evader-source scripted \\
        --log-commands
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

# Hide the known cflib warning when firmware still uses legacy hover packet type.
warnings.filterwarnings(
    "ignore",
    message=r"Using legacy TYPE_HOVER_LEGACY\. Please update your crazyflie-firmware\.",
    category=DeprecationWarning,
    module=r"cflib\.crazyflie\.commander",
)

# Make the sibling ``intercept_common`` importable regardless of CWD.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import intercept_common as ic  # noqa: E402

# OptiTrack / NatNet streaming client (sibling module). ``_THIS_DIR`` is already
# on ``sys.path``; kept optional so the controller runs without a mocap setup.
try:
    from NatNetClient import NatNetClient  # noqa: E402
except Exception:  # pragma: no cover - mocap is an optional feature
    NatNetClient = None


# Firmware stabilization modes for send_setpoint's roll/pitch fields.
STAB_MODE_RATE = 0
STAB_MODE_ANGLE = 1


# ---------------------------------------------------------------------------
# Small state container
# ---------------------------------------------------------------------------
@dataclass
class DroneState:
    """World-frame drone state with an Isaac-style ``(w, x, y, z)`` quaternion."""

    pos: np.ndarray                 # (3,)
    quat_wxyz: np.ndarray           # (4,)
    lin_vel: np.ndarray             # (3,) world frame
    ang_vel: np.ndarray             # (3,) world frame (rad/s)
    stamp: float                    # seconds (wall clock)


def _quat_to_rotation_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert a ``(w, x, y, z)`` quaternion to a 3x3 rotation matrix."""
    w, x, y, z = quat_wxyz
    tx, ty, tz = 2.0 * x, 2.0 * y, 2.0 * z
    twx, twy, twz = tx * w, ty * w, tz * w
    txx, txy, txz = tx * x, ty * x, tz * x
    tyy, tyz, tzz = ty * y, tz * y, tz * z
    return np.array([
        [1 - (tyy + tzz), txy - twz, txz + twy],
        [txy + twz, 1 - (txx + tzz), tyz - twx],
        [txz - twy, tyz + twx, 1 - (txx + tyy)],
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Thread-safe state buffers, populated from cflib log callbacks
# ---------------------------------------------------------------------------
class StateBuffer:
    """Accumulates the latest pursuer state from several cflib log blocks.

    Log callbacks run on a background thread, so all reads/writes go through a
    lock and consumers take an immutable :class:`DroneState` snapshot.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pos = np.zeros(3)
        self._vel = np.zeros(3)
        self._quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
        self._ang_vel_body_rad = np.zeros(3)
        self._pos_stamp = 0.0
        self._att_stamp = 0.0

    def update_pos_vel(self, x, y, z, vx, vy, vz) -> None:
        with self._lock:
            self._pos = np.array([x, y, z], dtype=np.float64)
            self._vel = np.array([vx, vy, vz], dtype=np.float64)
            self._pos_stamp = time.time()

    def update_quat(self, qw, qx, qy, qz) -> None:
        with self._lock:
            self._quat_wxyz = np.array([qw, qx, qy, qz], dtype=np.float64)
            self._att_stamp = time.time()

    def update_gyro_deg(self, gx, gy, gz) -> None:
        with self._lock:
            self._ang_vel_body_rad = np.radians(
                np.array([gx, gy, gz], dtype=np.float64))

    def snapshot(self) -> Optional[DroneState]:
        """Return the latest state, or ``None`` if no pose has arrived yet."""
        with self._lock:
            if self._pos_stamp == 0.0 or self._att_stamp == 0.0:
                return None
            rot = _quat_to_rotation_matrix(self._quat_wxyz)
            ang_vel_world = rot @ self._ang_vel_body_rad
            return DroneState(
                pos=self._pos.copy(),
                quat_wxyz=self._quat_wxyz.copy(),
                lin_vel=self._vel.copy(),
                ang_vel=ang_vel_world,
                stamp=min(self._pos_stamp, self._att_stamp),
            )


class PositionBuffer:
    """Latest position (+ finite-difference velocity) for the evader drone."""

    def __init__(self, vel_lpf: float = 0.4) -> None:
        self._lock = threading.Lock()
        self._pos = np.zeros(3)
        self._vel = np.zeros(3)
        self._stamp = 0.0
        self._vel_lpf = vel_lpf

    def update(self, x, y, z) -> None:
        with self._lock:
            now = time.time()
            pos = np.array([x, y, z], dtype=np.float64)
            if self._stamp > 0.0:
                dt = now - self._stamp
                if dt > 1e-4:
                    raw = (pos - self._pos) / dt
                    self._vel = (self._vel_lpf * raw
                                 + (1.0 - self._vel_lpf) * self._vel)
            self._pos = pos
            self._stamp = now

    def snapshot(self) -> Optional[DroneState]:
        with self._lock:
            if self._stamp == 0.0:
                return None
            return DroneState(
                pos=self._pos.copy(),
                quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                lin_vel=self._vel.copy(),
                ang_vel=np.zeros(3),
                stamp=self._stamp,
            )


# ---------------------------------------------------------------------------
# Motion-capture (OptiTrack / NatNet) integration
# ---------------------------------------------------------------------------
def _detect_local_ip_for_server(server_ip: str) -> str:
    """Return the local interface IP that routes to ``server_ip``.

    Uses a connect-less UDP socket so no packets are sent; the kernel simply
    resolves the outbound interface. Falls back to ``0.0.0.0`` (bind-any) if the
    route cannot be determined.
    """
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((server_ip, 1))
        return sock.getsockname()[0]
    except OSError:
        return "0.0.0.0"
    finally:
        sock.close()


class MocapTfPublisher:
    """Optional ROS 2 TF publisher for mocap poses.

    ``rclpy`` is imported lazily so the controller keeps its pure-``cflib``
    dependency footprint unless TF publishing is explicitly requested.
    """

    def __init__(self, world_frame: str = "world",
                 node_name: str = "intercept_mocap_tf") -> None:
        import rclpy
        from tf2_msgs.msg import TFMessage
        from geometry_msgs.msg import TransformStamped

        self._rclpy = rclpy
        self._TFMessage = TFMessage
        self._TransformStamped = TransformStamped
        self._world_frame = world_frame
        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init()
        self._node = rclpy.create_node(node_name)
        self._pub = self._node.create_publisher(TFMessage, "tf", 10)

    def publish(self, child_frame_id: str, pose: "ic.MocapPose") -> None:
        transform = self._TransformStamped()
        transform.header.stamp = self._node.get_clock().now().to_msg()
        transform.header.frame_id = self._world_frame
        transform.child_frame_id = child_frame_id
        px, py, pz = pose.position
        transform.transform.translation.x = float(px)
        transform.transform.translation.y = float(py)
        transform.transform.translation.z = float(pz)
        qx, qy, qz, qw = pose.quat_xyzw
        transform.transform.rotation.x = float(qx)
        transform.transform.rotation.y = float(qy)
        transform.transform.rotation.z = float(qz)
        transform.transform.rotation.w = float(qw)
        self._pub.publish(self._TFMessage(transforms=[transform]))

    def close(self) -> None:
        try:
            self._node.destroy_node()
        except Exception:  # pragma: no cover - best-effort teardown
            pass
        if self._owns_rclpy:
            try:
                self._rclpy.shutdown()
            except Exception:  # pragma: no cover - best-effort teardown
                pass


class MocapReceiver:
    """Stream OptiTrack rigid-body poses and forward them to Crazyflie EKFs.

    The receiver owns a background NatNet thread (started by :meth:`start`).
    Each rigid-body frame is transformed into the ROS FLU convention via
    :func:`intercept_common.transform_mocap_pose` and pushed to the Crazyflie
    registered under the matching streaming id through ``extpos.send_extpose``.
    Poses are optionally re-published on a ROS 2 TF tree. Frames for
    unregistered ids are ignored.
    """

    def __init__(self, cfg: "ic.MocapConfig",
                 tf_publisher: Optional[MocapTfPublisher] = None) -> None:
        if NatNetClient is None:
            raise RuntimeError(
                "NatNetClient.py could not be imported; place it next to "
                "intercept_controller.py to use --mocap.")
        self._cfg = cfg
        self._tf_publisher = tf_publisher
        self._targets: dict = {}
        self._lock = threading.Lock()
        self._client = None

    def register(self, rigid_body_id: int, cf: Crazyflie,
                 frame_id: Optional[str] = None) -> None:
        """Route frames for ``rigid_body_id`` to ``cf`` (and TF ``frame_id``)."""
        rb_id = int(rigid_body_id)
        with self._lock:
            self._targets[rb_id] = (cf, frame_id or f"cf_{rb_id}")

    def start(self) -> None:
        if NatNetClient is None:  # pragma: no cover - guarded in __init__ too
            raise RuntimeError("NatNetClient.py is not importable.")
        client = NatNetClient()
        client.serverIPAddress = self._cfg.server_ip
        client.localIPAddress = (
            self._cfg.local_ip
            or _detect_local_ip_for_server(self._cfg.server_ip))
        client.multicastAddress = self._cfg.multicast_address
        client.commandPort = self._cfg.command_port
        client.dataPort = self._cfg.data_port
        client.rigidBodyListener = self._on_rigid_body
        self._client = client
        client.run()

    def _on_rigid_body(self, rigid_body_id, position, rotation,
                       tracking_valid) -> None:
        with self._lock:
            target = self._targets.get(int(rigid_body_id))
        if target is None:
            return
        cf, frame_id = target

        pose = ic.transform_mocap_pose(
            self._cfg, int(rigid_body_id), position, rotation, tracking_valid)
        if not pose.tracking_valid:
            return

        px, py, pz = pose.position
        qx, qy, qz, qw = pose.quat_xyzw
        try:
            cf.extpos.send_extpose(px, py, pz, qx, qy, qz, qw)
        except Exception:  # pragma: no cover - link may be tearing down
            return

        if self._tf_publisher is not None:
            self._tf_publisher.publish(frame_id, pose)

    def stop(self) -> None:
        """Best-effort teardown of the NatNet sockets (threads are daemons)."""
        client = self._client
        self._client = None
        if client is None:
            return
        for sock_attr in ("dataSocket", "commandSocket"):
            sock = getattr(client, sock_attr, None)
            if sock is not None:
                try:
                    sock.close()
                except Exception:  # pragma: no cover - best-effort teardown
                    pass


# ---------------------------------------------------------------------------
# The controller
# ---------------------------------------------------------------------------
class InterceptController:
    """Closed-loop cflib policy controller for the Intercept task."""

    def __init__(self, args: argparse.Namespace) -> None:
        # -- load the exported policy ---------------------------------------
        self.artifact_dir = os.path.abspath(os.path.expanduser(args.artifact_dir))
        ts_path, meta_path = ic.artifact_paths(self.artifact_dir)
        if not (os.path.isfile(ts_path) and os.path.isfile(meta_path)):
            raise FileNotFoundError(
                f"Missing artifact(s) under {self.artifact_dir}: expected "
                f"{ic.POLICY_TS_FILENAME} and {ic.METADATA_FILENAME}. "
                f"Run export_policy.py first."
            )
        self.metadata = ic.load_metadata(meta_path)
        self.policy = torch.jit.load(ts_path, map_location="cpu").eval()
        print(f"[intercept] Loaded {self.metadata.algo} policy "
              f"(obs_dim={self.metadata.obs.obs_dim}) from {ts_path}")

        # -- parameters ------------------------------------------------------
        self.uri = args.uri
        self.evader_uri = args.evader_uri
        self.evader_source = args.evader_source          # scripted|cf
        self.rw_cache = args.rw_cache
        self.max_thrust_pwm = float(args.max_thrust_pwm)
        # Per-axis sign flips for the body rates [roll, pitch, yaw] to reconcile
        # the training body-frame convention with the firmware's.
        self.rate_sign = np.array(args.rate_sign, dtype=np.float64)
        self.log_commands = bool(args.log_commands)
        self.state_timeout = float(args.state_timeout)
        self.min_altitude = float(args.min_altitude)     # safety cutoff
        # Control-loop rate. 0 (default) uses the policy's training dt
        # (metadata.ctbr.dt); set >0 to force a fixed rate.
        self.control_rate_hz = float(args.control_rate_hz)
        self.control_dt = (
            1.0 / self.control_rate_hz if self.control_rate_hz > 0.0
            else self.metadata.ctbr.dt
        )
        # Optional open-loop takeoff before handing control to the policy.
        self.takeoff = bool(args.takeoff)
        self.takeoff_thrust = int(args.takeoff_thrust)
        self.takeoff_duration = float(args.takeoff_duration)
        self.takeoff_hover_z = float(args.takeoff_hover_z)

        # Scripted evader parameters (used when evader_source == "scripted").
        self.evader_speed = float(args.evader_speed)
        self.evader_start = np.array(args.evader_start, dtype=np.float64)
        self.evader_dir = np.array(args.evader_dir, dtype=np.float64)

        # -- motion capture (OptiTrack / NatNet) -----------------------------
        self.mocap_enabled = bool(args.mocap)
        self.pursuer_rigid_body_id = int(args.pursuer_rigid_body_id)
        self.evader_rigid_body_id = (
            int(args.evader_rigid_body_id)
            if args.evader_rigid_body_id is not None else None
        )
        self.publish_tf = bool(args.publish_tf)
        self.mocap_world_frame = str(args.mocap_world_frame)
        self.mocap_cfg = ic.MocapConfig(
            server_ip=args.mocap_server_ip,
            local_ip=(args.mocap_local_ip or None),
            multicast_address=args.mocap_multicast,
            command_port=int(args.mocap_command_port),
            data_port=int(args.mocap_data_port),
            body_to_flu_quat_xyzw=tuple(float(v) for v in args.mocap_body_flu_quat),
        )

        # -- runtime state ---------------------------------------------------
        self._need_rot_speed = self.metadata.obs.use_rot_speed
        self._pursuer = StateBuffer()
        self._evader = PositionBuffer()
        self._mocap_receiver: Optional[MocapReceiver] = None
        self._mocap_tf_publisher: Optional[MocapTfPublisher] = None
        # Optional trajectory logging for post-flight visualization.
        self.save_trajectory = bool(args.save_trajectory)
        self.trajectory_dir = os.path.abspath(
            os.path.expanduser(args.trajectory_dir)
        )
        self.trajectory_prefix = str(args.trajectory_prefix)
        self._pursuer_traj_fp = None
        self._evader_traj_fp = None
        self._start_time = time.time()
        self._active = True

    def _setup_trajectory_logging(self) -> None:
        if not self.save_trajectory:
            return
        os.makedirs(self.trajectory_dir, exist_ok=True)
        pursuer_path = os.path.join(
            self.trajectory_dir, f"{self.trajectory_prefix}_pursuer.csv"
        )
        evader_path = os.path.join(
            self.trajectory_dir, f"{self.trajectory_prefix}_evader.csv"
        )
        self._pursuer_traj_fp = open(pursuer_path, "w", encoding="utf-8")
        self._evader_traj_fp = open(evader_path, "w", encoding="utf-8")
        self._pursuer_traj_fp.write("t_rel,state_stamp,x,y,z,vx,vy,vz\n")
        self._evader_traj_fp.write("t_rel,state_stamp,x,y,z,vx,vy,vz\n")
        print(f"[intercept] Saving pursuer trajectory to {pursuer_path}")
        print(f"[intercept] Saving evader trajectory to {evader_path}")

    def _log_trajectory(self, pursuer: DroneState, evader: DroneState) -> None:
        if self._pursuer_traj_fp is None or self._evader_traj_fp is None:
            return
        t_rel = time.time() - self._start_time
        self._pursuer_traj_fp.write(
            f"{t_rel:.6f},{pursuer.stamp:.6f},"
            f"{pursuer.pos[0]:.6f},{pursuer.pos[1]:.6f},{pursuer.pos[2]:.6f},"
            f"{pursuer.lin_vel[0]:.6f},{pursuer.lin_vel[1]:.6f},"
            f"{pursuer.lin_vel[2]:.6f}\n"
        )
        self._evader_traj_fp.write(
            f"{t_rel:.6f},{evader.stamp:.6f},"
            f"{evader.pos[0]:.6f},{evader.pos[1]:.6f},{evader.pos[2]:.6f},"
            f"{evader.lin_vel[0]:.6f},{evader.lin_vel[1]:.6f},"
            f"{evader.lin_vel[2]:.6f}\n"
        )

    def _close_trajectory_logging(self) -> None:
        for fp in (self._pursuer_traj_fp, self._evader_traj_fp):
            if fp is None:
                continue
            try:
                fp.flush()
                fp.close()
            except Exception:  # pragma: no cover - best-effort teardown
                pass
        self._pursuer_traj_fp = None
        self._evader_traj_fp = None

    # -- motion capture ------------------------------------------------------
    def _start_mocap(self, pursuer_cf: Crazyflie,
                     evader_cf: Optional[Crazyflie] = None) -> None:
        """Open the OptiTrack stream and route poses to the drone EKF(s)."""
        if not self.mocap_enabled:
            return

        tf_publisher = None
        if self.publish_tf:
            tf_publisher = MocapTfPublisher(world_frame=self.mocap_world_frame)

        receiver = MocapReceiver(self.mocap_cfg, tf_publisher=tf_publisher)
        receiver.register(self.pursuer_rigid_body_id, pursuer_cf)
        registered = f"pursuer id={self.pursuer_rigid_body_id}"
        if evader_cf is not None and self.evader_rigid_body_id is not None:
            receiver.register(self.evader_rigid_body_id, evader_cf)
            registered += f", evader id={self.evader_rigid_body_id}"

        receiver.start()
        self._mocap_receiver = receiver
        self._mocap_tf_publisher = tf_publisher
        print(f"[intercept] Mocap streaming from {self.mocap_cfg.server_ip} "
              f"({registered})"
              + ("; publishing /tf" if tf_publisher is not None else "") + ".")

    def _stop_mocap(self) -> None:
        if self._mocap_receiver is not None:
            self._mocap_receiver.stop()
            self._mocap_receiver = None
        if self._mocap_tf_publisher is not None:
            self._mocap_tf_publisher.close()
            self._mocap_tf_publisher = None

    # -- cflib logging setup -------------------------------------------------
    def _setup_pursuer_logging(self, cf: Crazyflie) -> None:
        """Register the log blocks that feed the pursuer state buffer."""
        pos_log = LogConfig(name="pos_vel", period_in_ms=10)
        for var in ("stateEstimate.x", "stateEstimate.y", "stateEstimate.z",
                    "stateEstimate.vx", "stateEstimate.vy", "stateEstimate.vz"):
            pos_log.add_variable(var, "float")
        pos_log.data_received_cb.add_callback(self._pos_vel_cb)

        att_log = LogConfig(name="quat", period_in_ms=10)
        for var in ("stateEstimate.qw", "stateEstimate.qx",
                    "stateEstimate.qy", "stateEstimate.qz"):
            att_log.add_variable(var, "float")
        att_log.data_received_cb.add_callback(self._quat_cb)

        cf.log.add_config(pos_log)
        cf.log.add_config(att_log)
        pos_log.start()
        att_log.start()

        if self._need_rot_speed:
            gyro_log = LogConfig(name="gyro", period_in_ms=10)
            for var in ("gyro.x", "gyro.y", "gyro.z"):
                gyro_log.add_variable(var, "float")
            gyro_log.data_received_cb.add_callback(self._gyro_cb)
            cf.log.add_config(gyro_log)
            gyro_log.start()

    def _setup_evader_logging(self, cf: Crazyflie) -> None:
        pos_log = LogConfig(name="evader_pos", period_in_ms=20)
        for var in ("stateEstimate.x", "stateEstimate.y", "stateEstimate.z"):
            pos_log.add_variable(var, "float")
        pos_log.data_received_cb.add_callback(self._evader_pos_cb)
        cf.log.add_config(pos_log)
        pos_log.start()

    # -- log callbacks -------------------------------------------------------
    def _pos_vel_cb(self, timestamp, data, logconf) -> None:
        self._pursuer.update_pos_vel(
            data["stateEstimate.x"], data["stateEstimate.y"], data["stateEstimate.z"],
            data["stateEstimate.vx"], data["stateEstimate.vy"], data["stateEstimate.vz"])

    def _quat_cb(self, timestamp, data, logconf) -> None:
        self._pursuer.update_quat(
            data["stateEstimate.qw"], data["stateEstimate.qx"],
            data["stateEstimate.qy"], data["stateEstimate.qz"])

    def _gyro_cb(self, timestamp, data, logconf) -> None:
        self._pursuer.update_gyro_deg(
            data["gyro.x"], data["gyro.y"], data["gyro.z"])

    def _evader_pos_cb(self, timestamp, data, logconf) -> None:
        self._evader.update(
            data["stateEstimate.x"], data["stateEstimate.y"], data["stateEstimate.z"])

    # -- flight-mode helpers -------------------------------------------------
    @staticmethod
    def _set_rate_mode(cf: Crazyflie) -> None:
        cf.param.set_value("flightmode.stabModeRoll", STAB_MODE_RATE)
        cf.param.set_value("flightmode.stabModePitch", STAB_MODE_RATE)

    @staticmethod
    def _restore_angle_mode(cf: Crazyflie) -> None:
        cf.param.set_value("flightmode.stabModeRoll", STAB_MODE_ANGLE)
        cf.param.set_value("flightmode.stabModePitch", STAB_MODE_ANGLE)

    # -- scripted evader -----------------------------------------------------
    def _scripted_evader_state(self) -> DroneState:
        t = time.time() - self._start_time
        direction = self.evader_dir / (np.linalg.norm(self.evader_dir) + 1e-6)
        pos = self.evader_start + direction * (self.evader_speed * t)
        vel = direction * self.evader_speed
        return DroneState(pos=pos, quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                          lin_vel=vel, ang_vel=np.zeros(3), stamp=t)

    # -- observation ---------------------------------------------------------
    def _build_observation(self, pursuer: DroneState, evader: DroneState) -> torch.Tensor:
        cfg = self.metadata.obs
        obs = ic.build_observation(
            cfg,
            pursuer_pos=torch.as_tensor(pursuer.pos, dtype=torch.float32),
            pursuer_quat_wxyz=torch.as_tensor(pursuer.quat_wxyz, dtype=torch.float32),
            pursuer_lin_vel_world=torch.as_tensor(pursuer.lin_vel, dtype=torch.float32),
            evader_pos=torch.as_tensor(evader.pos, dtype=torch.float32),
            pursuer_ang_vel_world=torch.as_tensor(pursuer.ang_vel, dtype=torch.float32),
            evader_lin_vel_world=torch.as_tensor(evader.lin_vel, dtype=torch.float32),
        )
        return obs.reshape(1, cfg.obs_dim)

    # -- command -------------------------------------------------------------
    def _send_command(self, cf: Crazyflie, command: ic.CTBRCommand) -> None:
        rates = command.body_rate_deg.detach().cpu().numpy().reshape(-1)  # deg/s
        sign = self.rate_sign
        roll_rate = float(sign[0]) * float(rates[0])
        pitch_rate = float(sign[1]) * float(rates[1])
        yaw_rate = float(sign[2]) * float(rates[2])
        thrust = int(np.clip(
            float(command.thrust_pwm.detach().cpu().item()),
            0.0, self.max_thrust_pwm))
        cf.commander.send_setpoint(roll_rate, pitch_rate, yaw_rate, thrust)
        # cf.commander.send_setpoint(0.0, 0.0, 0.0, thrust)  # --- IGNORE ---

    # -- main flight loop ----------------------------------------------------
    def _fly(self, cf: Crazyflie, evader_cf: Optional[Crazyflie] = None) -> None:
        # Modern firmware / CrazySim SITL require an explicit arm request.
        cf.supervisor.send_arming_request(True)
        time.sleep(1.0)

        # The first setpoint must be a zero-thrust one to unlock the commander.
        cf.commander.send_setpoint(0.0, 0.0, 0.0, 0)
        time.sleep(0.1)
        self._set_rate_mode(cf)
        time.sleep(0.1)

        # Wait for the first state packets before running the policy.
        print("[intercept] Waiting for state feedback...")
        t0 = time.time()
        while self._pursuer.snapshot() is None:
            cf.commander.send_setpoint(0.0, 0.0, 0.0, 0)
            if time.time() - t0 > 5.0:
                raise TimeoutError("No state feedback received within 5 s.")
            time.sleep(0.05)

        if self.takeoff:
            print(f"[intercept] Open-loop takeoff (hover): z={self.takeoff_hover_z:.2f} m "
                  f"for {self.takeoff_duration:.1f}s")
            if evader_cf is not None:
                evader_cf.supervisor.send_arming_request(True)
                time.sleep(1.0)

                # The first setpoint must be a zero-thrust one to unlock the commander.
                evader_cf.commander.send_setpoint(0.0, 0.0, 0.0, 0)
                time.sleep(0.1)
                self._set_rate_mode(evader_cf)
                time.sleep(0.1)

            steps = max(1, int(self.takeoff_duration / self.control_dt))
            for _ in range(steps):
                cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, self.takeoff_hover_z)
                if evader_cf is not None:
                    evader_cf.commander.send_hover_setpoint(
                        0.0, 0.0, 0.0, self.takeoff_hover_z)
                time.sleep(self.control_dt)

        print(f"[intercept] Running policy at {1.0 / self.control_dt:.1f} Hz, "
              f"evader_source='{self.evader_source}'. Press Ctrl+C to stop.")
        next_t = time.time()
        while self._active:
            self._control_step(cf)
            if evader_cf is not None:
                evader_cf.commander.send_hover_setpoint(
                    0.0, 0.0, 0.0, self.takeoff_hover_z)
            next_t += self.control_dt
            sleep = next_t - time.time()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.time()

    def _control_step(self, cf: Crazyflie) -> None:
        pursuer = self._pursuer.snapshot()
        if pursuer is None:
            return

        now = time.time()
        if now - pursuer.stamp > self.state_timeout:
            print("[intercept] Pursuer state timed out; stopping for safety.")
            self._active = False
            return

        # Safety: mirror the training "misbehave" altitude floor.
        if pursuer.pos[2] < self.min_altitude:
            print(f"[intercept] Pursuer below min altitude "
                  f"({pursuer.pos[2]:.2f} m); stopping.")
            self._active = False
            return

        if self.evader_source == "scripted":
            evader = self._scripted_evader_state()
        else:
            evader = self._evader.snapshot()
            if evader is None:
                return

        self._log_trajectory(pursuer, evader)

        obs = self._build_observation(pursuer, evader)
        with torch.no_grad():
            raw_action = self.policy(obs)
        command = ic.decode_action_to_ctbr(raw_action, self.metadata.ctbr)
        self._send_command(cf, command)

        if self.log_commands:
            rates = command.body_rate_deg.detach().cpu().numpy().reshape(-1)
            dist = float(np.linalg.norm(evader.pos - pursuer.pos))
            print(f"[intercept] alt={pursuer.pos[2]:.2f}m dist={dist:.2f}m "
                  f"rates(deg/s)=[{rates[0]:+.0f},{rates[1]:+.0f},{rates[2]:+.0f}] "
                  f"thrust_pwm={float(command.thrust_pwm.item()):.0f} "
                  f"(ratio={float(command.thrust_ratio.item()):.2f})")

    # -- lifecycle -----------------------------------------------------------
    def run(self) -> None:
        cflib.crtp.init_drivers()
        cf = Crazyflie(rw_cache=self.rw_cache)
        self._setup_trajectory_logging()

        evader_scf = None
        evader_cf = None
        try:
            with SyncCrazyflie(self.uri, cf=cf) as scf:
                self._setup_pursuer_logging(scf.cf)

                if self.evader_source == "cf":
                    if not self.evader_uri:
                        raise ValueError(
                            "--evader-uri is required when --evader-source=cf.")
                    evader_cf = Crazyflie(rw_cache=self.rw_cache)
                    evader_scf = SyncCrazyflie(
                        self.evader_uri, cf=evader_cf)
                    evader_scf.open_link()
                    self._setup_evader_logging(evader_scf.cf)

                self._start_mocap(scf.cf, evader_cf)

                try:
                    self._fly(scf.cf, evader_cf=evader_cf)
                except KeyboardInterrupt:
                    print("\n[intercept] Stopping.")
                finally:
                    self._shutdown(scf.cf)
        finally:
            self._stop_mocap()
            if evader_scf is not None:
                try:
                    evader_scf.close_link()
                except Exception:  # pragma: no cover - best-effort teardown
                    pass
            self._close_trajectory_logging()

    def _shutdown(self, cf: Crazyflie) -> None:
        self._active = False
        try:
            cf.commander.send_stop_setpoint()
            cf.commander.send_notify_setpoint_stop()
        except Exception:  # pragma: no cover - best-effort teardown
            pass
        try:
            self._restore_angle_mode(cf)
        except Exception:  # pragma: no cover - best-effort teardown
            pass
        print("[intercept] Sent stop command and restored angle mode.")


# ---------------------------------------------------------------------------
# Configuration (YAML)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG_PATH = os.path.join(_THIS_DIR, "intercept_config.yaml")


def load_config(path: str) -> argparse.Namespace:
    """Load the controller configuration from a YAML file.

    The nested YAML schema (see ``intercept_config.yaml``) is flattened into the
    attribute names consumed by :class:`InterceptController`. Missing keys fall
    back to documented defaults so partial configs remain valid.
    """
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "PyYAML is required to read the config file. "
            "Install it with: pip install pyyaml"
        ) from exc

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root of {path} must be a mapping.")

    def section(name: str) -> dict:
        value = raw.get(name, {}) or {}
        if not isinstance(value, dict):
            raise ValueError(f"Config section '{name}' must be a mapping.")
        return value

    takeoff = section("takeoff")
    evader = section("evader")
    trajectory = section("trajectory")
    mocap = section("mocap")

    ns = argparse.Namespace(
        # pursuer / connection
        uri=raw.get("uri", "udp://127.0.0.1:19850"),
        rw_cache=raw.get("rw_cache", "./cache"),
        artifact_dir=raw.get("artifact_dir"),
        # control loop
        control_rate_hz=float(raw.get("control_rate_hz", 0.0)),
        max_thrust_pwm=float(raw.get("max_thrust_pwm", 65535.0)),
        rate_sign=list(raw.get("rate_sign", [1.0, 1.0, 1.0])),
        state_timeout=float(raw.get("state_timeout", 0.5)),
        min_altitude=float(raw.get("min_altitude", 0.15)),
        log_commands=bool(raw.get("log_commands", False)),
        # takeoff
        takeoff=bool(takeoff.get("enabled", False)),
        takeoff_thrust=int(takeoff.get("thrust", 50000)),
        takeoff_hover_z=float(takeoff.get("hover_z", 1.0)),
        takeoff_duration=float(takeoff.get("duration", 3.0)),
        # evader
        evader_source=str(evader.get("source", "cf")),
        evader_uri=evader.get("uri", "udp://127.0.0.1:19851"),
        evader_speed=float(evader.get("speed", 3.0)),
        evader_start=list(evader.get("start", [3.0, 0.0, 1.6])),
        evader_dir=list(evader.get("dir", [1.0, 0.0, 0.0])),
        # trajectory logging
        save_trajectory=bool(trajectory.get("save", False)),
        trajectory_dir=trajectory.get("dir", "./trajectory_logs"),
        trajectory_prefix=trajectory.get("prefix", "intercept"),
        # motion capture
        mocap=bool(mocap.get("enabled", False)),
        mocap_server_ip=mocap.get("server_ip", "127.0.0.1"),
        mocap_local_ip=(mocap.get("local_ip") or ""),
        mocap_multicast=mocap.get("multicast", "239.255.42.99"),
        mocap_command_port=int(mocap.get("command_port", 1510)),
        mocap_data_port=int(mocap.get("data_port", 1511)),
        pursuer_rigid_body_id=int(mocap.get("pursuer_rigid_body_id", 31)),
        evader_rigid_body_id=mocap.get("evader_rigid_body_id", None),
        mocap_body_flu_quat=list(
            mocap.get("body_flu_quat", ic.DEFAULT_MOCAP_BODY_TO_FLU_QUAT_XYZW)),
        publish_tf=bool(mocap.get("publish_tf", False)),
        mocap_world_frame=mocap.get("world_frame", "world"),
    )

    # -- validation ----------------------------------------------------------
    if not ns.artifact_dir:
        raise ValueError("Config must set 'artifact_dir'.")
    if ns.evader_source not in ("scripted", "cf"):
        raise ValueError(
            f"evader.source must be 'scripted' or 'cf', got '{ns.evader_source}'.")
    for name, seq, length in (("rate_sign", ns.rate_sign, 3),
                              ("evader.start", ns.evader_start, 3),
                              ("evader.dir", ns.evader_dir, 3),
                              ("mocap.body_flu_quat", ns.mocap_body_flu_quat, 4)):
        if len(seq) != length:
            raise ValueError(f"'{name}' must have {length} elements, got {len(seq)}.")
    return ns


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Run an exported Intercept policy on a Crazyflie via cflib. "
                    "All settings are read from a YAML config file.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                        help="Path to the YAML configuration file "
                             f"(default: {DEFAULT_CONFIG_PATH}).")
    cli = parser.parse_args(argv)
    config = load_config(cli.config)
    controller = InterceptController(config)
    controller.run()


if __name__ == "__main__":
    main()
