import OpenEXR
import Imath
import array
import numpy as np
import open3d as o3d


def exr2numpy(exr, maxvalue=1.):
    """ converts 1-channel exr-data to 2D numpy arrays """
    file = OpenEXR.InputFile(exr)

    # Compute the size
    dw = file.header()['dataWindow']
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # Read the three color channels as 32-bit floats
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    (R) = [array.array('f', file.channel(Chan, FLOAT)).tolist() for Chan in ("R") ]

    # create numpy 2D-array
    img = np.zeros((sz[1],sz[0],3), np.float64)

    # normalize
    data = np.array(R)
    data[data > maxvalue] = -1
    img = np.array(data).reshape(img.shape[0],-1)

    return img

# depth_data = exr2numpy("non_centre test/ISS_NC.exr/Image0125.exr", maxvalue=1000, normalize=False)
# print(depth_data.shape)
# fig = plt.figure()
# plt.imshow(depth_data)
# plt.colorbar()
# plt.show()
# image = plt.imread("non_centre test/ISS_NC.png/Image0125.png")
# fig1 = plt.figure()
# plt.imshow(image)
# plt.show()

# print(depth_data.shape, image.shape)

def pcd_block (depth, downsample=False):
    pcd = []
    cx = 1920/2
    cy = 1080/2
    fx = 1920
    fy = 1920
    height, width = depth.shape
    for i in range(height):
        for j in range(width):
            z = depth[i][j]
            x = (j - cx) * z / fx
            y = (i - cy) * z / fy
            pcd.append([x, y, z])

    pc_EXT = o3d.geometry.PointCloud()  # create point cloud object
    pc_EXT.points = o3d.utility.Vector3dVector(pcd)
    if downsample :
        pc_EXT = pc_EXT.voxel_down_sample(voxel_size=1)
    return pc_EXT

# Visualize:
# C = pcd_o3d.get_rotation_matrix_from_xyz((1 * np.pi, 0, 0))
# pcd = pcd_o3d.rotate(C , center=(0,0,0))
# downpcd = pcd_o3d.voxel_down_sample(voxel_size=0.5)
# print(downpcd)
# frame = o3d.geometry.TriangleMesh().create_coordinate_frame()



