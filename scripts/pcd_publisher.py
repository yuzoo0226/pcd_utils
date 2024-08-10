#!/usr/bin/env python3
# -*-encoding:UTF-8-*-

import rospy
import numpy as np
import open3d as o3d
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2


class PcdPublisher():
    def __init__(self) -> None:
        # ファイルパスの取得
        self.pcd_path = "temp.pcd"
        rospy.loginfo(self.pcd_path)

        # ファイルの読み込み
        self.pcd = o3d.io.read_point_cloud(self.pcd_path)
        colors = self.pcd.colors
        o3d.visualization.draw_geometries([self.pcd])

        self.points = np.asarray(self.pcd.points)  # rosで送信しやすい形に変更

        # ros interface
        # パブリッシャの作成
        self.pcd_pub = rospy.Publisher('/pointcloud', PointCloud2, queue_size=10)
        rospy.loginfo("inital settings complete")

    def run(self):
        """実行関数
        """
        # メッセージの作成
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"

        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
        ]

        self.cloud_msg = pc2.create_cloud(header, fields, self.points)
        self.pcd_pub.publish(self.cloud_msg)


if __name__ == "__main__":
    # ノードの初期化
    rospy.init_node("pointcloud_publisher", anonymous=True)
    cls = PcdPublisher()

    rate = rospy.Rate(0.1)  # ループの周期を設定
    while not rospy.is_shutdown():
        cls.run()
        rate.sleep
