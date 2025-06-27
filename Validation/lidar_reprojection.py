import os
import cv2
import numpy as np
from Calibration.camera_intrinsics.camera import CameraModel

# --- 1. Parse sensor_calibration.txt and create CameraModel objects ---
def parse_sensor_calibration(calib_path):
    cameras = {}
    extrinsics = {}
    lidar_extrinsics = {}
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('Name:'):
            name = line.split(':', 1)[1].strip()
            # Read intrinsics (if present)
            intrinsics = {}
            is_lidar = name.upper() in ['LIDAR', 'VELODYNE', 'CEPTON']
            while i < len(lines) and not lines[i].strip().startswith('Extrinsics'):
                l = lines[i].strip()
                if ':' in l:
                    k, v = l.split(':', 1)
                    intrinsics[k.strip()] = v.strip()
                i += 1
            # Camera matrix and distortion (only for cameras)
            if not is_lidar:
                fx = float(intrinsics.get('Focal_x', 0))
                fy = float(intrinsics.get('Focal_y', 0))
                cx = float(intrinsics.get('COD_x', 0))
                cy = float(intrinsics.get('COD_y', 0))
                camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
                dist_coeffs = [float(intrinsics.get(f'Dist_{j+1}', 0)) for j in range(4)]
                cameras[name] = CameraModel(camera_matrix, dist_coeffs)
            # Read extrinsics
            extr = {'X': 0, 'Y': 0, 'Z': 0, 'R': np.eye(3, dtype=np.float32)}
            while i < len(lines) and lines[i].strip() != '':
                l = lines[i].strip()
                if ':' in l:
                    k, v = l.split(':', 1)
                    k = k.strip()
                    v = v.strip()
                    if k in ['X', 'Y', 'Z']:
                        try:
                            extr[k] = float(v)
                        except ValueError:
                            extr[k] = 0
                    elif k.startswith('R_'):
                        idx = k[2:]
                        row, col = int(idx[0]), int(idx[1])
                        if 'R' not in extr or not isinstance(extr['R'], np.ndarray):
                            extr['R'] = np.eye(3, dtype=np.float32)
                        try:
                            extr['R'][row, col] = float(v)
                        except ValueError:
                            extr['R'][row, col] = 0
                i += 1
            if is_lidar:
                lidar_extrinsics[name] = extr
            else:
                extrinsics[name] = extr
        i += 1
    return cameras, extrinsics, lidar_extrinsics

def apply_extrinsics(points, extr):
    # points: Nx3, extr: dict with X, Y, Z, R (3x3)
    R = extr['R']
    t = np.array([extr['X'], extr['Y'], extr['Z']], dtype=np.float32).reshape(1, 3)
    points_out = (R @ points.T).T + t
    return points_out

# --- 2. Load video and PCD folder ---
def load_pcd_xyz(pcd_path):
    # Simple loader for ASCII XYZ PCD files
    points = []
    with open(pcd_path, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#') and not line[0].isalpha():
                vals = line.strip().split()
                if len(vals) >= 3:
                    points.append([float(vals[0]), float(vals[1]), float(vals[2])])
    return np.array(points, dtype=np.float32)

def get_sorted_pcd_files(pcd_folder):
    files = [f for f in os.listdir(pcd_folder) if f.lower().endswith('.pcd')]
    files.sort()
    return [os.path.join(pcd_folder, f) for f in files]

# --- 3. Main visualization loop ---
def main(calib_path, video_path, pcd_folder, camera_name, lidar_name):
    cameras, extrinsics, lidar_extrinsics = parse_sensor_calibration(calib_path)
    if camera_name not in cameras:
        print(f'Camera {camera_name} not found in calibration file.')
        return
    if lidar_name not in lidar_extrinsics:
        print(f'LIDAR {lidar_name} not found in calibration file.')
        return
    camera = cameras[camera_name]
    cam_extr = extrinsics[camera_name]
    lidar_extr = lidar_extrinsics[lidar_name]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'Failed to open video: {video_path}')
        return
    pcd_files = get_sorted_pcd_files(pcd_folder)
    if not pcd_files:
        print(f'No PCD files found in {pcd_folder}')
        return

    # Trackbar state
    start_frame = [0]
    def on_trackbar(val):
        start_frame[0] = val

    cv2.namedWindow('PCD Sync')
    cv2.createTrackbar('PCD Start Frame', 'PCD Sync', 0, max(0, len(pcd_files)-1), on_trackbar)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pcd_idx = frame_idx - start_frame[0]
        if 0 <= pcd_idx < len(pcd_files):
            points = load_pcd_xyz(pcd_files[pcd_idx])
            # Apply LIDAR and camera extrinsics
            points_world = apply_extrinsics(points, lidar_extr)
            points_cam = apply_extrinsics(points_world, cam_extr)
            # Project LIDAR points to image
            img_pts = camera.project_point(points_cam)
            for pt in img_pts:
                x, y = int(pt[0]), int(pt[1])
                if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        cv2.imshow('LIDAR Reprojection', frame)
        cv2.imshow('PCD Sync', np.zeros((100, 400, 3), dtype=np.uint8))  # Dummy window for trackbar
        key = cv2.waitKey(30)
        if key == 27:  # ESC
            break
        frame_idx += 1
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Set your parameters here
    calib_path = '../../sensor_calibration.txt' 
    video_path = 'video.mp4'  # Change to your video file
    pcd_folder = 'pcds'       # Change to your PCD folder
    camera_name = 'FLIR 8.9MP'  # Change to your camera
    lidar_name = 'Velodyne'      # Change to your LIDAR
    main(calib_path, video_path, pcd_folder, camera_name, lidar_name)
