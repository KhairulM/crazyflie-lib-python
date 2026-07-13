#!/usr/bin/env python3
"""Publish saved intercept trajectories as ROS 2 Path topics for RViz.

Example:
    python publish_trajectories_rviz.py \
        --pursuer-csv trajectory_logs/intercept_pursuer.csv \
        --evader-csv trajectory_logs/intercept_evader.csv
"""

from __future__ import annotations

import argparse
import csv
from typing import List

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy


class TrajectoryPublisher(Node):
    """Publishes pursuer and evader trajectories as nav_msgs/Path."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("intercept_trajectory_publisher")
        self._frame_id = args.frame_id
        self._pursuer_path = self._load_csv_as_path(args.pursuer_csv)
        self._evader_path = self._load_csv_as_path(args.evader_csv)

        # Transient local makes RViz receive the last published path on connect.
        qos = QoSProfile(depth=1)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        qos.reliability = ReliabilityPolicy.RELIABLE

        self._pursuer_pub = self.create_publisher(Path, args.pursuer_topic, qos)
        self._evader_pub = self.create_publisher(Path, args.evader_topic, qos)

        self._timer = self.create_timer(args.publish_period, self._publish)
        self.get_logger().info(
            "Publishing trajectories: "
            f"pursuer poses={len(self._pursuer_path.poses)} on {args.pursuer_topic}, "
            f"evader poses={len(self._evader_path.poses)} on {args.evader_topic}, "
            f"frame={self._frame_id}"
        )

    def _load_csv_as_path(self, csv_path: str) -> Path:
        poses: List[PoseStamped] = []
        with open(csv_path, "r", encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp)
            required = {"x", "y", "z"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(
                    f"CSV {csv_path} is missing required columns {sorted(required)}"
                )
            for row in reader:
                pose = PoseStamped()
                pose.header.frame_id = self._frame_id
                pose.pose.position.x = float(row["x"])
                pose.pose.position.y = float(row["y"])
                pose.pose.position.z = float(row["z"])
                pose.pose.orientation.w = 1.0
                poses.append(pose)

        path = Path()
        path.header.frame_id = self._frame_id
        path.poses = poses
        return path

    def _publish(self) -> None:
        stamp = self.get_clock().now().to_msg()

        self._pursuer_path.header.stamp = stamp
        for pose in self._pursuer_path.poses:
            pose.header.stamp = stamp

        self._evader_path.header.stamp = stamp
        for pose in self._evader_path.poses:
            pose.header.stamp = stamp

        self._pursuer_pub.publish(self._pursuer_path)
        self._evader_pub.publish(self._evader_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish intercept trajectory CSV files as ROS 2 Path topics."
    )
    parser.add_argument(
        "--pursuer-csv",
        required=True,
        help="Path to the pursuer trajectory CSV file.",
    )
    parser.add_argument(
        "--evader-csv",
        required=True,
        help="Path to the evader trajectory CSV file.",
    )
    parser.add_argument(
        "--frame-id",
        default="map",
        help="Frame id for published Path messages.",
    )
    parser.add_argument(
        "--pursuer-topic",
        default="/intercept/pursuer_path",
        help="ROS 2 topic for the pursuer Path.",
    )
    parser.add_argument(
        "--evader-topic",
        default="/intercept/evader_path",
        help="ROS 2 topic for the evader Path.",
    )
    parser.add_argument(
        "--publish-period",
        type=float,
        default=0.5,
        help="Publish period in seconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rclpy.init()
    node = TrajectoryPublisher(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
