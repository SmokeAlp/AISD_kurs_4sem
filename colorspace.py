import numpy as np

def rgb_to_ycbcr(img):
    R = img[:,:,0].astype(np.float64)
    G = img[:,:,1].astype(np.float64)
    B = img[:,:,2].astype(np.float64)
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128.0
    Cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 128.0
    img_ycbcr = np.empty_like(img, dtype=np.uint8)
    img_ycbcr[:,:,0] = np.clip(Y, 0, 255).astype(np.uint8)
    img_ycbcr[:,:,1] = np.clip(Cb, 0, 255).astype(np.uint8)
    img_ycbcr[:,:,2] = np.clip(Cr, 0, 255).astype(np.uint8)
    return img_ycbcr

def ycbcr_to_rgb(img):
    Y = img[:,:,0].astype(np.float64)
    Cb = img[:,:,1].astype(np.float64) - 128.0
    Cr = img[:,:,2].astype(np.float64) - 128.0
    R = Y + 1.402 * Cr
    G = Y - 0.344136 * Cb - 0.714136 * Cr
    B = Y + 1.772 * Cb
    img_rgb = np.empty_like(img, dtype=np.uint8)
    img_rgb[:,:,0] = np.clip(R, 0, 255).astype(np.uint8)
    img_rgb[:,:,1] = np.clip(G, 0, 255).astype(np.uint8)
    img_rgb[:,:,2] = np.clip(B, 0, 255).astype(np.uint8)
    return img_rgb