import torch
print(torch.cuda.is_available())
print(torch.__version__)
from knn_cuda import KNN
from extensions.chamfer_dist import ChamferDistanceL2
from extensions.cubic_feature_sampling import CubicFeatureSampling
from extensions.gridding import Gridding, GriddingReverse
cld2 = ChamferDistanceL2().cuda()
G = Gridding().cuda()
GR = GriddingReverse().cuda()
knn = KNN(k=10).cuda()
cfs = CubicFeatureSampling().cuda()
from extensions.emd.emd import earth_mover_distance
emd = earth_mover_distance().cuda()
print(knn, cld2, emd, cfs, G, GR)