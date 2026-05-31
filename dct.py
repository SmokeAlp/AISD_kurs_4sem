import numpy as np

def create_dct_matrix(N: int) -> np.ndarray:
    C = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            if j == 0:
                C[i, j] = 1 / np.sqrt(N)
            else:
                C[i, j] = np.sqrt(2 / N) * np.cos((2 * i + 1) * j * np.pi / (2 * N))
    return C

def dct_2d(block, C=None):
    h, w = block.shape
    C_h = create_dct_matrix(h)
    C_w = create_dct_matrix(w)
    return C_h @ block @ C_w.T

def idct_2d(dct_block, C=None):
    h, w = dct_block.shape
    C_h = create_dct_matrix(h)
    C_w = create_dct_matrix(w)
    return C_h.T @ dct_block @ C_w

def split_into_blocks(image, block_size=8):
    h, w = image.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    if pad_h > 0 or pad_w > 0:
        padded = np.empty((h+pad_h, w+pad_w), dtype=image.dtype)
        padded[:h, :w] = image
        if pad_w > 0:
            for row_start in range(0, h, block_size):
                row_end = min(row_start + block_size, h)
                block_col = image[row_start:row_end, -block_size+pad_w:]
                mean_val = np.mean(block_col)
                padded[row_start:row_end, w:] = mean_val
        if pad_h > 0:
            for col_start in range(0, w, block_size):
                col_end = min(col_start + block_size, w)
                block_row = image[-block_size+pad_h:, col_start:col_end]
                mean_val = np.mean(block_row)
                padded[h:, col_start:col_end] = mean_val
        if pad_h > 0 and pad_w > 0:
            block_corner = image[-block_size+pad_h:, -block_size+pad_w:]
            mean_val = np.mean(block_corner)
            padded[h:, w:] = mean_val
    else:
        padded = image
    blocks = []
    for r in range(0, padded.shape[0], block_size):
        for c in range(0, padded.shape[1], block_size):
            block = padded[r:r+block_size, c:c+block_size]
            blocks.append(block)
    rows_blocks = padded.shape[0] // block_size
    cols_blocks = padded.shape[1] // block_size
    return blocks, rows_blocks, cols_blocks, (image.shape[0], image.shape[1])

def merge_blocks(blocks, rows_blocks, cols_blocks, block_size=8, original_shape=None):
    H = rows_blocks * block_size
    W = cols_blocks * block_size
    img = np.empty((H, W), dtype=np.float64)
    idx = 0
    for r in range(rows_blocks):
        for c in range(cols_blocks):
            img[r*block_size:(r+1)*block_size, c*block_size:(c+1)*block_size] = blocks[idx]
            idx += 1
    if original_shape is not None:
        img = img[:original_shape[0], :original_shape[1]]
    return img