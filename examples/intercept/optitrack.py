import time
import logging
import socket
from threading import Thread, Event
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
# Assuming you have downloaded NatNetClient.py in the same folder
from NatNetClient import NatNetClient

import rclpy
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped

# Change these to match your exact setup
DRONE_1_URI = 'radio://0/80/2M/E7E7E7E701'
DRONE_2_URI = 'radio://0/80/2M/E7E7E7E702'

# Track your Motive Rigid Body IDs here
CF1_RIGID_BODY_ID = 31
CF2_RIGID_BODY_ID = 32
MOTIVE_SERVER_IP = "192.168.0.210"
# Set to None to auto-detect the outbound local interface used to reach Motive.
CLIENT_LOCAL_IP = None

# Global dictionary to store the objects
cfs = {}

logging.basicConfig(level=logging.ERROR)
tf_publisher = None  # Global variable for the TF publisher
tf_publisher_node = None

# Fixed body-frame correction from Motive rigid body axes to ROS FLU body axes.
# Motive body axes reported by user: X=left, Y=up, Z=forward.
# Target ROS body axes: X=forward, Y=left, Z=up.
# Quaternion format throughout this file is (x, y, z, w).
MOTIVE_BODY_TO_ROS_FLU_Q = (0.0, 0.0, -0.70710678, 0.70710678)


def _quat_mul(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def _quat_normalize(q):
    x, y, z, w = q
    n = (x * x + y * y + z * z + w * w) ** 0.5
    if n == 0.0:
        return (0.0, 0.0, 0.0, 1.0)
    return (x / n, y / n, z / n, w / n)


def _make_attitude_log_callback(rigid_body_id):
    """Return a log callback that prints the Crazyflie EKF-estimated attitude."""
    def _cb(timestamp, data, logconf):
        roll = data.get('stabilizer.roll', float('nan'))
        pitch = data.get('stabilizer.pitch', float('nan'))
        yaw = data.get('stabilizer.yaw', float('nan'))
        print(
            f"[EKF cf_{rigid_body_id}] roll={roll:+.1f} pitch={pitch:+.1f} yaw={yaw:+.1f}"
        )
    return _cb


def wait_for_connection(cf, uri, timeout=10.0):
    """Open the link and block until the Crazyflie is fully connected."""
    connected_event = Event()
    failed_event = Event()

    def _connected(_uri):
        connected_event.set()

    def _failed(_uri, msg):
        print(f"Connection to {_uri} failed: {msg}")
        failed_event.set()

    cf.connected.add_callback(_connected)
    cf.connection_failed.add_callback(_failed)

    cf.open_link(uri)

    while not connected_event.is_set() and not failed_event.is_set():
        if not connected_event.wait(timeout):
            break

    return connected_event.is_set()


def start_attitude_logging(cf, rigid_body_id):
    """Set up EKF attitude logging on a connected Crazyflie."""
    log_conf = LogConfig(name=f'Attitude_{rigid_body_id}', period_in_ms=200)
    log_conf.add_variable('stabilizer.roll', 'float')
    log_conf.add_variable('stabilizer.pitch', 'float')
    log_conf.add_variable('stabilizer.yaw', 'float')
    cf.log.add_config(log_conf)
    log_conf.data_received_cb.add_callback(_make_attitude_log_callback(rigid_body_id))
    log_conf.start()
    return log_conf


def _detect_local_ip_for_server(server_ip: str) -> str:
    """Resolve the local interface IP that routes traffic to the NatNet server."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((server_ip, 1510))
        return sock.getsockname()[0]
    except OSError:
        return "0.0.0.0"
    finally:
        sock.close()


def receive_rigid_body_frame(new_id, position, rotation, tracking_valid):
    """
    Callback function from NatNet client.
    position: [x, y, z] from OptiTrack
    rotation: [qx, qy, qz, qw] quaternion from OptiTrack
    tracking_valid: True if Motive marks the rigid body tracking as valid
    """
    # Rotate Motive body orientation into ROS FLU body convention.
    rotation_ros = _quat_normalize(_quat_mul(rotation, MOTIVE_BODY_TO_ROS_FLU_Q))

    # Send data to Drone 1
    if new_id == CF1_RIGID_BODY_ID and CF1_RIGID_BODY_ID in cfs:
        cf = cfs[CF1_RIGID_BODY_ID]
        # send_extpose sends both position and orientation to stop yaw drift
        cf.extpos.send_extpose(position[0], position[1], position[2],
                               rotation_ros[0], rotation_ros[1], rotation_ros[2], rotation_ros[3])

    # Send data to Drone 2
    elif new_id == CF2_RIGID_BODY_ID and CF2_RIGID_BODY_ID in cfs:
        cf = cfs[CF2_RIGID_BODY_ID]
        cf.extpos.send_extpose(position[0], position[1], position[2],
                               rotation_ros[0], rotation_ros[1], rotation_ros[2], rotation_ros[3])

    # print(
    #     f"Received Rigid Body {new_id}: Position {position}, "
    #     f"Rotation {rotation}, TrackingValid={tracking_valid}"
    # )
    if tf_publisher is None or tf_publisher_node is None:
        return

    transform = TransformStamped()
    transform.header.stamp = tf_publisher_node.get_clock().now().to_msg()
    transform.header.frame_id = 'world'
    transform.child_frame_id = f'cf_{new_id}'
    transform.transform.translation.x = float(position[0])
    transform.transform.translation.y = float(position[1])
    transform.transform.translation.z = float(position[2])
    transform.transform.rotation.x = float(rotation_ros[0])
    transform.transform.rotation.y = float(rotation_ros[1])
    transform.transform.rotation.z = float(rotation_ros[2])
    transform.transform.rotation.w = float(rotation_ros[3])

    tf_msg = TFMessage(transforms=[transform])
    tf_publisher.publish(tf_msg)


def main():
    # Initialize the low-level communication drivers
    cflib.crtp.init_drivers()

    # Create Crazyflie instances
    cf1 = Crazyflie(rw_cache='./cache')
    cf2 = Crazyflie(rw_cache='./cache')

    # Setup ROS 2 TF publishers
    rclpy.init()
    global tf_publisher, tf_publisher_node
    tf_publisher_node = rclpy.create_node('crazyflie_tf_publisher')
    tf_publisher = tf_publisher_node.create_publisher(TFMessage, 'tf', 10)

    print("Connecting to Crazyflies...")
    if not wait_for_connection(cf1, DRONE_1_URI):
        print("cf1 did not connect; aborting.")
        rclpy.shutdown()
        return

    if not wait_for_connection(cf2, DRONE_2_URI):
        print("cf2 did not connect; aborting.")
        cf1.close_link()  # Close drone 1 if drone 2 fails
        rclpy.shutdown()
        return

    # Store them in our global dict so the mocap thread can use them
    cfs[CF1_RIGID_BODY_ID] = cf1
    cfs[CF2_RIGID_BODY_ID] = cf2

    # Give them a few seconds to fully connect and warm up parameters
    time.sleep(3)

    # Temporary: log EKF-estimated attitude to verify orientation is physically correct.
    start_attitude_logging(cf1, CF1_RIGID_BODY_ID)
    # start_attitude_logging(cf2, CF2_RIGID_BODY_ID)

    # Setup NatNet Client to listen to Motive
    streaming_client = NatNetClient()
    # NatNetClient uses camelCase listener names.
    streaming_client.serverIPAddress = MOTIVE_SERVER_IP
    streaming_client.localIPAddress = (
        CLIENT_LOCAL_IP if CLIENT_LOCAL_IP is not None
        else _detect_local_ip_for_server(MOTIVE_SERVER_IP)
    )
    streaming_client.rigidBodyListener = receive_rigid_body_frame
    print(
        "NatNet config: "
        f"server={streaming_client.serverIPAddress}, "
        f"local={streaming_client.localIPAddress}, "
        f"multicast={streaming_client.multicastAddress}"
    )

    print("Starting OptiTrack stream loop...")
    streaming_client.run()  # This typically starts a background thread

    try:
        print("System running. Press Ctrl+C to stop.")
        while True:
            # You can place your high-level autonomous takeoff/flight commands here
            # Example: cf1.commander.send_position_setpoint(0, 0, 0.5, 0)
            time.sleep(0.1)
            rclpy.spin_once(tf_publisher_node, timeout_sec=0.1)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        print("Closing links...")
        cf1.close_link()
        cf2.close_link()
        tf_publisher_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
