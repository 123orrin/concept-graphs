import os
from plyfile import PlyData
from collections import defaultdict

def analyze_ply_folders(root_dir):
    """
    Analyze PLY files in all subdirectories of root_dir.
    Returns statistics about point clouds in each folder.
    """
    folder_stats = defaultdict(lambda: {
        'num_files': 0,
        'total_points': 0,
        'min_points': float('inf'),
        'max_points': 0,
        'files': []
    })
    
    # Walk through all subdirectories
    for dirpath, dirnames, filenames in os.walk(root_dir):
        ply_files = [f for f in filenames if f.lower().endswith('.ply')]
        
        if not ply_files:
            continue
            
        folder_name = os.path.basename(dirpath)
        
        # Process each PLY file in the current directory
        for ply_file in ply_files:
            file_path = os.path.join(dirpath, ply_file)
            try:
                ply_data = PlyData.read(file_path)
                num_points = len(ply_data['vertex'])
                
                # Update statistics
                stats = folder_stats[folder_name]
                stats['num_files'] += 1
                stats['total_points'] += num_points
                stats['min_points'] = min(stats['min_points'], num_points)
                stats['max_points'] = max(stats['max_points'], num_points)
                stats['files'].append(ply_file)
                
            except Exception as e:
                print(f"Error reading {file_path}: {str(e)}")
    
    return folder_stats

def print_summary(stats):
    """
    Print a formatted summary of the point cloud analysis.
    """
    print("\nPoint Cloud Analysis Summary")
    print("=" * 50)
    
    for folder, data in stats.items():
        if data['num_files'] > 0:  # Only print folders that contain PLY files
            print(f"\nFolder: {folder}")
            print("-" * 30)
            print(f"Number of point clouds: {data['num_files']}")
            avg_points = data['total_points'] / data['num_files']
            print(f"Average points per cloud: {avg_points:.2f}")
            print(f"Minimum points in a cloud: {data['min_points']:}")
            print(f"Maximum points in a cloud: {data['max_points']:}")
            # print(f"Files: {', '.join(data['files'])}")

def main():
    # Get the root directory from user input
    root_dir = input("Enter the root directory path: ")
    
    if not os.path.exists(root_dir):
        print(f"Error: Directory '{root_dir}' does not exist.")
        return
    
    print(f"\nAnalyzing PLY files in {root_dir} and its subdirectories...")
    stats = analyze_ply_folders(root_dir)
    print_summary(stats)

if __name__ == "__main__":
    main()