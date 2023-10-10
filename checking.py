import open3d as o3d
import numpy as np
from Depth_to_PCD import pcd_block, exr2numpy
from pyrr import Quaternion

path_exr = 'Initial Orientation Tests/Depth1155.exr'
path_blen = 'Initial Orientation Tests/1155.npz'
translation = []
obj_q = []
depth = exr2numpy(path_exr, 500)
pcd = pcd_block(depth, True)
R = pcd.get_rotation_matrix_from_xyz((np.pi, 0, 0))
pcd = pcd.rotate(R, center=[0, 0, 0])
pcd.paint_uniform_color([0, 0, 1])
gg = pcd.get_center()
print("Blue Center: {}".format(gg))
data_b = np.load(path_blen)
print(data_b.files)
translations_b = data_b['location']
for obj in translations_b:
    obj_name = obj['name']
    obj_loc = obj[1]
    if obj_name == 'ISS_final_data':
        print(obj_name)
        translations_b = obj_loc
        print(translations_b)

obj_poses_b = data_b['object_poses']
print('\tObject poses:')
for obj in obj_poses_b:
    obj_name = obj['name']
    obj_p_b = obj['pose']
    if obj_name == 'ISS_final_data':
        print(obj_name)
        obj_q_b = obj_p_b
        print(obj_q_b)
        t = -obj_q_b[2]
        obj_q_b[2] = obj_q_b[3]
        obj_q_b[3] = t

RSS = o3d.io.read_point_cloud("ISS_final.pcd")
gg = RSS.get_center()
T = RSS.get_rotation_matrix_from_quaternion(obj_q_b)
RSS.translate([translations_b[0],
               translations_b[1],
               translations_b[2]], relative=True)
RSS.rotate(T, center=translations_b)
# RSS.translate([0,0,0], relative=False)
RSS.paint_uniform_color([1, 0, 0])
rr = RSS.get_center()
print(f'total Points : {len(RSS.points)}')
RSS = RSS.voxel_down_sample(voxel_size=0.9)

TSS = o3d.io.read_point_cloud("ISS_final.pcd")
TSS = TSS.translate([translations_b[0],
                     translations_b[1],
                     translations_b[2]], relative=True)
TSS.rotate(T)
TSS.paint_uniform_color([0, 1, 0])
TSS = TSS.voxel_down_sample(voxel_size=1)

import point_cloud_utils as pcu

dists = pcd.compute_point_cloud_distance(RSS)
dists = np.asarray(dists)
print(f"distance between Point_clouds = {dists.mean()}")
n = len(TSS.points)
from extensions.emd .emd import earth_mover_distance
EMD = earth_mover_distance().cuda()
np1 = np.asarray(TSS.points)
np2 = np.asarray(RSS.points)
cdl = pcu.chamfer_distance(np1, np2)


def normalize_pc(points):
    centroid = np.mean(points, axis=0)
    points -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(points) ** 2, axis=-1)))
    points /= furthest_distance

    return points

import torch
print(torch.cuda.is_available())
print(torch.__version__)
from knn_cuda import KNN
from extensions.chamfer_dist import ChamferDistanceL1

T1 = torch.from_numpy(np1).type(torch.float32).cuda()
T2 = torch.from_numpy(np2).type(torch.float32).cuda()
T1 = T1.unsqueeze(0)
T2 = T2.unsqueeze(0)
l1 = EMD(T1, T2)
cld2 = ChamferDistanceL1().cuda()
jack = cld2(T1,T2)
print(f"Observed : {len(pcd.points)}, Complete : {len(RSS.points)}")
print(f"Loss : {torch.mean(l1)/ len(pcd.points)}, CDL: {cdl,jack}")
frame_coord_1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20.0, origin=([0, 0, 0]))
frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20.0, origin=([0, 0, 0]))
f1 = frame.get_center()
f3 = frame_coord_1.get_center()
frame_coord_1.rotate(T)
f2 = frame_coord_1.get_center()
list_pcd = [RSS, pcd, frame, frame_coord_1]
# pcd, frame, frame_coord_1]
o3d.visualization.draw_geometries(list_pcd)
knn = KNN(k=10, transpose_mode=True)
ref = torch.rand(32, 1000, 5).cuda()
query = torch.rand(32, 50, 5).cuda()
dist, indx = knn(T1, T2)
print(torch.min(dist), torch.max(dist))
print(ref.shape, T1.shape, T2.shape)

