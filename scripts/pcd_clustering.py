#!/usr/bin/env python3
# -*- encoding: UTF-8 -*-

import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans


class PCDClustering:
    """
    A class to handle point cloud creation and clustering for a cuboid.
    """

    def __init__(self) -> None:
        """
        Initialize the PCDClustering object.
        """
        pass

    def create_box_point_cloud(self, width: float, height: float, depth: float, density: float = 0.1) -> o3d.geometry.PointCloud:
        """
        Create a point cloud representing a cuboid with the given dimensions.

        Parameters:
        width (float): The width of the cuboid.
        height (float): The height of the cuboid.
        depth (float): The depth of the cuboid.
        density (float): The density of the points on each face of the cuboid.

        Returns:
        o3d.geometry.PointCloud: The generated point cloud object.
        """
        # 直方体の頂点を定義
        box = o3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=depth)
        box.translate([-width / 2, -height / 2, -depth / 2])  # 原点を中心に移動

        # 各面ごとに点をサンプリング
        points = []
        for i in np.arange(-width/2, width/2, density):
            for j in np.arange(-height/2, height/2, density):
                points.append([i, j, -depth/2])  # 前面
                points.append([i, j, depth/2])   # 背面

        for i in np.arange(-width/2, width/2, density):
            for k in np.arange(-depth/2, depth/2, density):
                points.append([i, -height/2, k])  # 下側
                points.append([i, height/2, k])   # 上側

        for j in np.arange(-height/2, height/2, density):
            for k in np.arange(-depth/2, depth/2, density):
                points.append([-width/2, j, k])  # 左側
                points.append([width/2, j, k])   # 右側

        # 点群を作成
        points = np.array(points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # 点群の色をすべて黒に設定
        black_color = np.array([[0, 0, 0] for _ in range(len(points))])
        pcd.colors = o3d.utility.Vector3dVector(black_color)

        o3d.visualization.draw_geometries([pcd])

        return pcd

    def cluster(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Perform clustering on the point cloud based on surface normals.

        Parameters:
        pcd (o3d.geometry.PointCloud): The point cloud to cluster.

        Returns:
        o3d.geometry.PointCloud: The point cloud with colors assigned to clusters.
        """
        # 法線の推定
        pcd.estimate_normals()

        # 法線ベクトルを取得
        normals = np.asarray(pcd.normals)

        # K-meansを用いて法線方向によるクラスタリング
        kmeans = KMeans(n_clusters=3)  # 直方体には6つの面があるので、クラスター数を6に設定
        labels = kmeans.fit_predict(normals)

        # クラスタごとに異なる色を割り当てる
        colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]])
        pcd.colors = o3d.utility.Vector3dVector(colors[labels])

        # クラスタリング結果の可視化
        o3d.visualization.draw_geometries([pcd])

        return pcd


if __name__ == "__main__":
    cls = PCDClustering()

    width, height, depth = 2.0, 1.0, 1.5
    density = 0.05
    pcd = cls.create_box_point_cloud(width, height, depth, density=density)

    cls.cluster(pcd)