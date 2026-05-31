import struct
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from colorspace import rgb_to_ycbcr, ycbcr_to_rgb
from dct import dct_2d, idct_2d, split_into_blocks, merge_blocks
from quantization import (QT_LUMINANCE, QT_CHROMINANCE, scale_quantization_table, quantize, dequantize)
from zigzag import zigzag_8x8, inverse_zigzag_8x8
from huffman import (DC_LUM_CODES, DC_CHROM_CODES, AC_LUM_CODES, AC_CHROM_CODES, huffman_encode, huffman_decode)
from rle import (compute_size_and_bits, decode_coefficient, dpcm_encode, dpcm_decode, rle_encode, rle_decode)
from bitstream import BitWriter, BitReader

MAGIC = b'JCMP'

def write_compressed_file(filename, width, height, components, q_tables, huff_tables_ids):
    with open(filename, 'wb') as f:
        f.write(MAGIC)
        f.write(struct.pack('<HH', width, height))
        f.write(struct.pack('B', len(components)))
        for qt_id, dc_id, ac_id in huff_tables_ids:
            f.write(struct.pack('BBB', qt_id, dc_id, ac_id))
        f.write(struct.pack('B', len(q_tables)))
        for qt_id, table in q_tables.items():
            f.write(struct.pack('B', qt_id))
            q_zigzag = zigzag_8x8(table)
            f.write(struct.pack('64B', *[int(x) for x in q_zigzag]))
        for data in components:
            f.write(struct.pack('<I', len(data)))
            f.write(data)

def read_compressed_file(filename):
    with open(filename, 'rb') as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError("Неверный формат файла")
        width, height = struct.unpack('<HH', f.read(4))
        num_comp = struct.unpack('B', f.read(1))[0]
        huff_ids = [struct.unpack('BBB', f.read(3)) for _ in range(num_comp)]
        num_qtables = struct.unpack('B', f.read(1))[0]
        q_tables = {}
        for _ in range(num_qtables):
            qt_id = struct.unpack('B', f.read(1))[0]
            zigzag_vals = struct.unpack('64B', f.read(64))
            q_tables[qt_id] = inverse_zigzag_8x8(zigzag_vals)
        comp_data = []
        for _ in range(num_comp):
            length = struct.unpack('<I', f.read(4))[0]
            data = f.read(length)
            comp_data.append(data)
    return width, height, num_comp, huff_ids, q_tables, comp_data

def compress_channel(channel_2d, q_table, dc_huff, ac_huff):
    blocks, rows_blocks, cols_blocks, orig = split_into_blocks(channel_2d, 8)
    dc_list = []
    ac_runs_list = []
    for block in blocks:
        coeffs = dct_2d(block.astype(np.float64) - 128)
        quant = quantize(coeffs, q_table)
        zz = zigzag_8x8(quant)
        dc = zz[0]
        ac = zz[1:]
        dc_list.append(dc)
        ac_runs_list.append(rle_encode(ac))
    dc_diffs = dpcm_encode(dc_list)
    writer = BitWriter()

    for diff in dc_diffs:
        if diff == 0:
            size = 0
            bits = 0
        else:
            size, bits = compute_size_and_bits(diff)
        code, codelen = huffman_encode(size, dc_huff)
        writer.write_bits(code, codelen)
        if size > 0:
            writer.write_bits(bits, size)

    for runs in ac_runs_list:
        i = 0
        while i < len(runs):
            run, value = runs[i]
            if run == 0 and value == 0:
                code, codelen = huffman_encode(0x00, ac_huff)
                writer.write_bits(code, codelen)
                break
            if run == 15 and value == 0:
                code, codelen = huffman_encode(0xF0, ac_huff)
                writer.write_bits(code, codelen)
                i += 1
                continue
            size, bits = compute_size_and_bits(value)
            symbol = (run << 4) | size
            code, codelen = huffman_encode(symbol, ac_huff)
            writer.write_bits(code, codelen)
            writer.write_bits(bits, size)
            i += 1
    return writer.to_bytes()

def decompress_channel(data, rows_blocks, cols_blocks, q_table, dc_huff, ac_huff):
    reader = BitReader(data)
    blocks = []
    dc_diffs = []

    for i in range(rows_blocks * cols_blocks):
        cat = huffman_decode(reader, dc_huff)
        if cat == 0:
            diff = 0
        else:
            bits = reader.read_bits(cat)
            diff = decode_coefficient(cat, bits)
        dc_diffs.append(diff)
    dc_coeffs = dpcm_decode(dc_diffs)

    for blk_idx in range(rows_blocks * cols_blocks):
        ac_rle_pairs = []
        while True:
            symbol = huffman_decode(reader, ac_huff)
            if symbol == 0x00:  # EOB
                ac_rle_pairs.append((0, 0))
                break
            run = (symbol >> 4) & 0x0F
            size = symbol & 0x0F
            if run == 15 and size == 0:  # ZRL
                ac_rle_pairs.append((15, 0))
                continue
            bits = reader.read_bits(size)
            value = decode_coefficient(size, bits)
            ac_rle_pairs.append((run, value))
        ac_coeffs = rle_decode(ac_rle_pairs)
        coeffs = [dc_coeffs[blk_idx]] + ac_coeffs[:63]
        quant = inverse_zigzag_8x8(coeffs)
        deq = dequantize(quant.astype(np.float64), q_table)
        block = idct_2d(deq.astype(np.float64)) + 128
        block = np.clip(block, 0, 255)
        blocks.append(block)

    channel = merge_blocks(blocks, rows_blocks, cols_blocks, 8)
    return channel

class JPEGCompressor:
    def __init__(self, quality=50):
        self.quality = quality
        self.qt_lum = scale_quantization_table(QT_LUMINANCE, quality)
        self.qt_chrom = scale_quantization_table(QT_CHROMINANCE, quality)
        self.dc_lum_huff = DC_LUM_CODES
        self.dc_chrom_huff = DC_CHROM_CODES
        self.ac_lum_huff = AC_LUM_CODES
        self.ac_chrom_huff = AC_CHROM_CODES

    def compress_raw(self, raw_path, output_path):
        with open(raw_path, 'rb') as f:
            header = f.read(7)
            magic = header[0]
            img_type = header[1]
            colorspace_id = header[2]
            width = header[3] | (header[4] << 8)
            height = header[5] | (header[6] << 8)
            img_data = f.read()
        if img_type == 0x03:
            img = np.frombuffer(img_data, dtype=np.uint8).reshape((height, width, 3))
            ycbcr = rgb_to_ycbcr(img)
            channels = [ycbcr[:,:,0], ycbcr[:,:,1], ycbcr[:,:,2]]
            q_tables = [self.qt_lum, self.qt_chrom, self.qt_chrom]
            dc_huffs = [self.dc_lum_huff, self.dc_chrom_huff, self.dc_chrom_huff]
            ac_huffs = [self.ac_lum_huff, self.ac_chrom_huff, self.ac_chrom_huff]
            huff_ids = [(0,0,0), (1,1,1), (1,1,1)]
            q_table_dict = {0: self.qt_lum, 1: self.qt_chrom}
        elif img_type in (0x01, 0x02):
            img = np.frombuffer(img_data, dtype=np.uint8).reshape((height, width))
            channels = [img]
            q_tables = [self.qt_lum]
            dc_huffs = [self.dc_lum_huff]
            ac_huffs = [self.ac_lum_huff]
            huff_ids = [(0,0,0)]
            q_table_dict = {0: self.qt_lum}
        else:
            raise ValueError(f"Неподдерживаемый тип изображения: {img_type}")

        compressed = []
        for ch, qt, dc_h, ac_h in zip(channels, q_tables, dc_huffs, ac_huffs):
            comp_bytes = compress_channel(ch, qt, dc_h, ac_h)
            compressed.append(comp_bytes)
        write_compressed_file(output_path, width, height, compressed, q_table_dict, huff_ids)

    def decompress_to_image(self, jcpg_path):
        width, height, num_comp, huff_ids, q_tables, comp_data = read_compressed_file(jcpg_path)
        rows_blocks = (height + 7) // 8
        cols_blocks = (width + 7) // 8
        channels = []
        for idx, (qt_id, dc_id, ac_id) in enumerate(huff_ids):
            q_table = q_tables[qt_id]
            if dc_id == 0:
                dc_huff = DC_LUM_CODES
            else:
                dc_huff = DC_CHROM_CODES
            if ac_id == 0:
                ac_huff = AC_LUM_CODES
            else:
                ac_huff = AC_CHROM_CODES
            channel = decompress_channel(comp_data[idx], rows_blocks, cols_blocks, q_table, dc_huff, ac_huff)
            channels.append(channel)
        if num_comp == 1:
            img_out = np.clip(channels[0][:height, :width], 0, 255).astype(np.uint8)
            return img_out
        else:
            y = channels[0][:height, :width]
            cb = channels[1][:height, :width]
            cr = channels[2][:height, :width]
            ycbcr = np.stack((y, cb, cr), axis=-1)
            rgb = ycbcr_to_rgb(ycbcr)
            return np.clip(rgb, 0, 255).astype(np.uint8)

def run_tests():
    test_dir = Path("Тестовые данные")
    out_dir = Path("Декомпрессия")
    raw_files = [
        "RAW_lena.raw",
        "RAW_color.raw",
        "RAW_gray.raw",
        "RAW_bw_nodith.raw",
        "RAW_bw_dith.raw"
    ]
    qualities = range(10, 91, 10)
    for raw_file in raw_files:
        raw_path = test_dir / raw_file
        if not raw_path.exists():
            continue
        print(f"Обработка {raw_path.name}")
        sizes = []
        for q in qualities:
            compressor = JPEGCompressor(quality=q)
            jcpg_path = out_dir / f"{raw_path.stem}_q{q}.jcpg"
            compressor.compress_raw(raw_path, jcpg_path)
            size = jcpg_path.stat().st_size
            sizes.append(size)
            img = compressor.decompress_to_image(jcpg_path)
            if img.ndim == 2:
                pil_img = Image.fromarray(img, 'L')
            else:
                pil_img = Image.fromarray(img, 'RGB')
            pil_img.save(out_dir / f"{raw_path.stem}_decompressed_{q}.jpg")
        plt.figure()
        plt.plot(qualities, sizes, marker='o')
        plt.title(f"{raw_file} – зависимость размера от качества")
        plt.xlabel("Quality")
        plt.ylabel("Размер файла (байт)")
        plt.grid(True)
        plt.savefig(out_dir / f"{raw_path.stem}_graph.jpg")
        plt.close()
    print("Тестирование завершено.")

run_tests()