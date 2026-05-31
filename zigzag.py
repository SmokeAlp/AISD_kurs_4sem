import numpy as np

def zigzag_8x8(matrix):
    if matrix.shape != (8,8):
        raise ValueError("Матрица должна быть 8x8")
    order = [(0,0),(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),
             (2,1),(3,0),(4,0),(3,1),(2,2),(1,3),(0,4),(0,5),
             (1,4),(2,3),(3,2),(4,1),(5,0),(6,0),(5,1),(4,2),
             (3,3),(2,4),(1,5),(0,6),(0,7),(1,6),(2,5),(3,4),
             (4,3),(5,2),(6,1),(7,0),(7,1),(6,2),(5,3),(4,4),
             (3,5),(2,6),(1,7),(2,7),(3,6),(4,5),(5,4),(6,3),
             (7,2),(7,3),(6,4),(5,5),(4,6),(3,7),(4,7),(5,6),
             (6,5),(7,4),(7,5),(6,6),(5,7),(6,7),(7,6),(7,7)]
    return [matrix[i,j] for i,j in order]

def inverse_zigzag_8x8(seq):
    mat = np.zeros((8,8))
    order = [(0,0),(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),
             (2,1),(3,0),(4,0),(3,1),(2,2),(1,3),(0,4),(0,5),
             (1,4),(2,3),(3,2),(4,1),(5,0),(6,0),(5,1),(4,2),
             (3,3),(2,4),(1,5),(0,6),(0,7),(1,6),(2,5),(3,4),
             (4,3),(5,2),(6,1),(7,0),(7,1),(6,2),(5,3),(4,4),
             (3,5),(2,6),(1,7),(2,7),(3,6),(4,5),(5,4),(6,3),
             (7,2),(7,3),(6,4),(5,5),(4,6),(3,7),(4,7),(5,6),
             (6,5),(7,4),(7,5),(6,6),(5,7),(6,7),(7,6),(7,7)]
    for idx, (i,j) in enumerate(order):
        mat[i,j] = seq[idx]
    return mat

def zigzag_generic(matrix):
    n, m = matrix.shape
    result = []
    for s in range(n + m - 1):
        if s % 2 == 0:
            i = min(s, n-1)
            j = s - i
            while i >= 0 and j < m:
                result.append(matrix[i, j])
                i -= 1
                j += 1
        else:
            j = min(s, m-1)
            i = s - j
            while j >= 0 and i < n:
                result.append(matrix[i, j])
                i += 1
                j -= 1
    return result

def inverse_zigzag_generic(seq, shape):
    n, m = shape
    mat = np.zeros((n,m))
    idx = 0
    for s in range(n + m - 1):
        if s % 2 == 0:
            i = min(s, n-1)
            j = s - i
            while i >= 0 and j < m:
                if idx < len(seq):
                    mat[i,j] = seq[idx]
                    idx += 1
                i -= 1
                j += 1
        else:
            j = min(s, m-1)
            i = s - j
            while j >= 0 and i < n:
                if idx < len(seq):
                    mat[i,j] = seq[idx]
                    idx += 1
                i += 1
                j -= 1
    return mat