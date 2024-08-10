#!/usr/bin/env python3
# -*-encoding:UTF-8-*-

import time
import argparse
import open3d as o3d


class VisualizePcd(object):
    def __init__(self) -> None:
        pass

    def run(self, pcd_path: str) -> None:
        pcd = o3d.io.read_point_cloud(pcd_path)
        o3d.visualization.draw_geometries([pcd], window_name="upsampled_pcd")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Increase pointcloud from .pcd file")
    parser.add_argument("--input-pcd", "-i", required=True, help="対象とするファイル")
    args = parser.parse_args()
    pcd_path = args.input_pcd
    cls = VisualizePcd()
    cls.run(pcd_path)
