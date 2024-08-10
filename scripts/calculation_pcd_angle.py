import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA


class PointCloudAnalyzer(object):
    """
    A class for analyzing point cloud data, including calculating bounding box dimensions
    and performing principal component analysis (PCA) to determine orientation.

    Attributes:
    -----------
    pcd : o3d.geometry.PointCloud
        The point cloud to be analyzed.

    Methods:
    --------
    calc_rectangular() -> np.ndarray:
        Approximates the point cloud with a rectangular bounding box and visualizes it.

    calc_pca() -> float:
        Performs PCA on the point cloud to calculate the orientation angle with respect to the x-axis.
    """

    def __init__(self, pcd: o3d.geometry.PointCloud):
        """
        Initializes the PointCloudAnalyzer with a point cloud.

        Parameters:
        -----------
        pcd : o3d.geometry.PointCloud
            The point cloud to be analyzed.
        """
        self.pcd = pcd

    def calc_rectangular(self) -> np.ndarray:
        """
        Approximates the point cloud with a rectangular bounding box and visualizes it.

        Returns:
        --------
        np.ndarray
            The coordinates of the bottom face of the bounding box.
        """
        bbox = self.pcd.get_axis_aligned_bounding_box()
        dimensions = bbox.get_extent()
        ratio = dimensions[0] / dimensions[1]

        box_points = np.asarray(bbox.get_box_points())
        bottom_face = box_points[[0, 1, 2, 3], :]

        vertices = np.asarray(bbox.get_box_points())

        o3d.visualization.draw_geometries([
            self.pcd, bbox,
            o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(vertices),
                                 lines=o3d.utility.Vector2iVector([[0, 1], [0, 3], [3, 6], [6, 1]]))
        ])

        return bottom_face

    def calc_pca(self) -> float:
        """
        Performs PCA on the point cloud to calculate the orientation angle with respect to the x-axis.

        Returns:
        --------
        float
            The angle between the first principal component and the x-axis.
        """
        points = np.asarray(self.pcd.points)
        pca = PCA(n_components=1)
        pca.fit(points)

        def unit_vector(vector: np.ndarray) -> np.ndarray:
            return vector / np.linalg.norm(vector)

        def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
            v1_u = unit_vector(v1)
            v2_u = unit_vector(v2)
            return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

        angle = angle_between(np.array([1, 0, 0]), np.array([pca.components_[0, 0], pca.components_[0, 1], 0]))
        if pca.components_[0, 1] < 0:
            angle *= -1.0

        return angle


if __name__ == "__main__":
    path = "/home/hma/yuga_ws/pcd_ws/G-PCD/stimuli/D01/bunny.ply"
    pcd = o3d.io.read_point_cloud(path)

    points = np.ascontiguousarray(pcd.points)
    points[:, 2] = 0

    analyzer = PointCloudAnalyzer(pcd)
    bottom_face = analyzer.calc_rectangular()
    angle = analyzer.calc_pca()

    pca = PCA(n_components=1)
    projected_points = pca.fit_transform(points)
    first_eigenvector = pca.components_[0]
    grasp_angle = np.arccos(np.dot(first_eigenvector, [1, 0, 0]) / np.linalg.norm(first_eigenvector)) + 1.57
    if grasp_angle > 3.14:
        grasp_angle -= 3.14

    print(angle)
    print(grasp_angle)

    pointcloud1 = o3d.geometry.PointCloud()
    pointcloud1.points = o3d.utility.Vector3dVector(points)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pointcloud1)
    vis.run()
    vis.destroy_window()
