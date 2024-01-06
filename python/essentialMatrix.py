import numpy as np

def essentialMatrix(F, K1, K2):
    """
    Args:
        F:  Fundamental Matrix
        K1: Camera Matrix 1
        K2: Camera Matrix 2   
    Returns:
        E:  Essential Matrix  
    """
    # Compute the essential matrix using the formula E = K2.T * F * K1
    E = K2.T @ F @ K1
    # E =  np.dot(K2.T, np.dot(F, K1))
    return E
