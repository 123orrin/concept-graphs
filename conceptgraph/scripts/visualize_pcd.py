# Visualize point cloud data in 3d

import argparse
import os
import open3d as o3d
from pathlib import Path


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default=None)
    
    return parser

def visualize_pcds(pcd_paths):
    pcds = []
    # Visualize point clouds
    for pcd_path in pcd_paths:
        pcds.append(o3d.io.read_point_cloud(str(pcd_path)))
        
    o3d.visualization.draw_geometries(pcds)
    

if __name__ == "__main__":
    # microwave = True
    microwave = False

    parser = get_parser()
    args = parser.parse_args()

    result_path = args.result_path

    potential_path = os.path.realpath(result_path)
    if potential_path != result_path:
        print(f"Resolved symlink for result_path: {result_path} -> \n{potential_path}")
        result_path = potential_path

    scene_path = Path(os.path.realpath(result_path)).parents[2]
    if microwave:
        global_pcd_dir = scene_path / "microwave_pcd"
    else:
        global_pcd_dir = scene_path / "global_pcd"

    # Get all global pcd paths
    global_pcd_paths = sorted(Path(global_pcd_dir).rglob("*.ply"))

    # Load point cloud data
    visualize_pcds(global_pcd_paths)
