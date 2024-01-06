import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.io as sio
from PIL import Image
from eightpoint import eightpoint
from epipolarCorrespondence import epipolarCorrespondence
from essentialMatrix import essentialMatrix
from camera2 import camera2
from triangulate import triangulate
from displayEpipolarF import displayEpipolarF
from epipolarMatchGUI import epipolarMatchGUI
from mpl_toolkits.mplot3d import Axes3D

# 1
# Load images and points
img1 = cv2.imread('../data/im1.png')
img2 = cv2.imread('../data/im2.png')
pts = np.load('../data/someCorresp.npy', allow_pickle=True).tolist()
pts1 = pts['pts1']
pts2 = pts['pts2']
# M = pts['M']
M = 640
print(M)

np.set_printoptions(precision=6, suppress=True)

# 2
# Run eightpoint to compute the fundamental matrix F
F = eightpoint(pts1, pts2, M);
# print(F.shape)
print(F)

# # 3.1.1
# displayEpipolarF(img1, img2, F);

# 3.1.2
# epipolarMatchGUI(img1, img2, F)

# 3
# Load the points in image 1 from templeCoords.npy
pts_temple = np.load('../data/templeCoords.npy', allow_pickle=True).tolist()
pts1_temple = pts_temple['pts1']
# Run epipolarCorrespondence to get the corresponding points in image 2
pts2_temple = epipolarCorrespondence(img1, img2, F, pts1_temple)

# 4
# Load intrinsics.npy
K = np.load('../data/intrinsics.npy', allow_pickle=True).tolist()
K1 = K['K1']
K2 = K['K2']
# Compute the essential matrix E
E = essentialMatrix(F,K1,K2) 
print(E)

# 5
# Compute the first camera projection matrix P1 and the four candidates for P2
P1 = np.dot(K1, np.hstack((np.eye(3), np.zeros((3, 1)))))
P2s = camera2(E)

# 6
# Run your triangulate function using the four sets of camera matrix candidates, 
# the points from templeCoords.npy and their computed correspondences.
best_P2 = None
best_pts3d = None
for i in range(P2s.shape[2]):
    P2_try = np.dot(K2, P2s[:, :, i])
    pts3d_try  = triangulate(P1, pts1_temple, P2_try, pts2_temple)

    # 7
    # Figure out the correct P2 and the corresponding 3D points.
    # find the one that all z cooridinate > 0
    if np.all(pts3d_try[:, 2] > 0):
        best_pts3d = pts3d_try
        best_P2 = P2_try
        break

# get R|t from KR|t
R1 = np.linalg.inv(K1) @ P1[:, :3]
t1 = np.linalg.inv(K1) @ P1[:, 3]
R2 = np.linalg.inv(K2) @ best_P2[:, :3]
t2 = np.linalg.inv(K2) @ best_P2[:, 3]

# 8
# Plot these point correspondences on screen.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(best_pts3d[:, 0], best_pts3d[:, 1], best_pts3d[:, 2], )
ax.set_box_aspect([1,1,1])
plt.show()

# # write your code here
# R1, t1 = np.eye(3), np.zeros((3, 1))
# R2, t2 = np.eye(3), np.zeros((3, 1))

# save extrinsic parameters for dense reconstruction
np.save('../results/extrinsics', {'R1': R1, 't1': t1, 'R2': R2, 't2': t2})
