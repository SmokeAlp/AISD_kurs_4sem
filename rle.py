import numpy as np

def compute_size_and_bits(value):
    if value == 0:
        raise ValueError("value не должен быть 0")
    abs_val = abs(value)
    size = int(np.ceil(np.log2(abs_val + 1)))
    if value > 0:
        bits = value
    else:
        bits = value - 1
    mask = (1 << size) - 1
    bits &= mask
    return size, bits

def decode_coefficient(size, bits):
    if size == 0:
        return 0
    if (bits >> (size-1)) & 1:
        value = bits
    else:
        value = bits - (1 << size) + 1
    return value

def dpcm_encode(dc_coeffs):
    if not dc_coeffs:
        return []
    res = [dc_coeffs[0]]
    for i in range(1, len(dc_coeffs)):
        res.append(dc_coeffs[i] - dc_coeffs[i-1])
    return res

def dpcm_decode(dc_diffs):
    if not dc_diffs:
        return []
    dc = [dc_diffs[0]]
    for i in range(1, len(dc_diffs)):
        dc.append(dc[-1] + dc_diffs[i])
    return dc

def rle_encode(ac_coeffs):
    runs = []
    zeros = 0
    for coeff in ac_coeffs:
        if coeff == 0:
            zeros += 1
            if zeros == 16:
                runs.append((15, 0))
                zeros = 0
        else:
            while zeros >= 16:
                runs.append((15, 0))
                zeros -= 16
            if zeros == 15:
                size = compute_size_and_bits(coeff)[0]
                if size > 1:
                    runs.append((15, 0))
                    zeros = 0
            runs.append((zeros, coeff))
            zeros = 0
    runs.append((0, 0))
    return runs

def rle_decode(runs, total_count=63):
    coeffs = []
    for run, value in runs:
        if run == 0 and value == 0:
            coeffs.extend([0] * (total_count - len(coeffs)))
            break
        coeffs.extend([0] * run)
        coeffs.append(value)
    if len(coeffs) < total_count:
        coeffs.extend([0] * (total_count - len(coeffs)))
    return coeffs[:total_count]