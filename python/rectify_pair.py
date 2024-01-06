import numpy as np

def rectify_pair(K1, K2, R1, R2, t1, t2):
    """
    takes left and right camera paramters (K, R, T) and returns left
    and right rectification matrices (M1, M2) and updated camera parameters. You
    can test your function using the provided script testRectify.py
    """
    # YOUR CODE HERE

    M1, M2, K1n, K2n, R1n, R2n, t1n, t2n = None, None, None, None, None, None, None, None

    # 1. Compute the optical center c1 and c2 of each camera by c = −(KR)^{−1}(Kt).
    c1 = -(np.linalg.inv(K1 @ R1)) @ (K1 @ t1)
    c2 = -(np.linalg.inv(K2 @ R2)) @ (K2 @ t2)

    # 2. Compute the new rotation matrix  where r1 , r2 , r3 ∈ R3×1 are orthonormal vectors
    # that represent x-, y-, and z-axes of the camera reference frame, respectively.
    # The new x-axis (r1) is parallel to the baseline: r1 = (c1 − c2)/∥c1 − c2∥.
    r1 = (c1 - c2)/np.linalg.norm(c1 - c2)
    # The new y-axis (r2) is orthogonal to x and to any arbitrary unit vector,
    # which we set to be the z unit vector of the old left matrix: 
    # r2 is the cross product of R1(3, :) and r1.
    r2 = np.cross(R1[2, :], r1).T
    r2 /= np.linalg.norm(r2)
    # The new z-axis (r3) is orthogonal to x and y: r3 is the cross product of r2 and r1.
    r3 = np.cross(r2, r1)
    r3 /= np.linalg.norm(r3)
    R_new = np.column_stack((r1, r2, r3)).T

    # Compute the new intrinsic parameter. We can use an arbitrary one.
    # In our test code, we just let = K2.
    K_new = K2

    # Compute the new translation:  t1n=−Rnewc1,  t2n=−Rnewc2.
    t1_new = -R_new.dot(c1)
    t2_new = -R_new.dot(c2)

    # Finally, the rectification matrix of the first camera and second camera
    M1 = (K_new.dot(R_new)).dot(np.linalg.inv(K1.dot(R1)))
    M2 = (K_new.dot(R_new)).dot(np.linalg.inv(K2.dot(R2)))

    K1n, K2n, R1n, R2n, t1n, t2n = K_new, K_new, R_new, R_new, t1_new, t2_new

    return M1, M2, K1n, K2n, R1n, R2n, t1n, t2n
