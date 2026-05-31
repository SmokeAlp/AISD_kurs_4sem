import struct
import time

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
from canonical_huffman import CanonicalHuffman

MAGIC = b'JPSH'
MAGIC_CANONICAL = b'JPCH'


def write_compressed_file(filename, width, height, components, q_tables, huff_tables_ids,
                          dc_tables, ac_tables, use_canonical=False):
    with open(filename, 'wb') as f:
        magic = MAGIC_CANONICAL if use_canonical else MAGIC
        f.write(magic)
        f.write(struct.pack('<HH', width, height))
        f.write(struct.pack('B', len(components)))
        for comp_idx, (qt_id, dc_id, ac_id) in enumerate(huff_tables_ids):
            f.write(struct.pack('BBB', qt_id, dc_id, ac_id))
        f.write(struct.pack('B', len(q_tables)))
        for qt_id, table in q_tables.items():
            f.write(struct.pack('B', qt_id))
            q_zigzag = zigzag_8x8(table)
            f.write(struct.pack('64B', *[int(x) for x in q_zigzag]))
        f.write(struct.pack('B', len(dc_tables)))
        for table_id, table in dc_tables.items():
            f.write(struct.pack('B', table_id))
            entries = [(sym, code, length) for sym, (code, length) in table.items()]
            f.write(struct.pack('<H', len(entries)))
            for sym, code, length in entries:
                f.write(struct.pack('B', sym))
                f.write(struct.pack('B', length))
                if length > 0:
                    code_bytes = code.to_bytes((length + 7) // 8, 'big')
                    f.write(code_bytes)

        f.write(struct.pack('B', len(ac_tables)))
        for table_id, table in ac_tables.items():
            f.write(struct.pack('B', table_id))
            entries = [(sym, code, length) for sym, (code, length) in table.items()]
            f.write(struct.pack('<H', len(entries)))
            for sym, code, length in entries:
                f.write(struct.pack('B', sym))
                f.write(struct.pack('B', length))
                if length > 0:
                    code_bytes = code.to_bytes((length + 7) // 8, 'big')
                    f.write(code_bytes)
        for data in components:
            f.write(struct.pack('<I', len(data)))
            f.write(data)

def read_compressed_file(filename):
    with open(filename, 'rb') as f:
        magic = f.read(4)
        if magic not in (MAGIC, MAGIC_CANONICAL):
            raise ValueError("Неверный формат файла")

        use_canonical = (magic == MAGIC_CANONICAL)
        width, height = struct.unpack('<HH', f.read(4))
        num_comp = struct.unpack('B', f.read(1))[0]
        huff_ids = [struct.unpack('BBB', f.read(3)) for _ in range(num_comp)]
        num_qtables = struct.unpack('B', f.read(1))[0]
        q_tables = {}
        for _ in range(num_qtables):
            qt_id = struct.unpack('B', f.read(1))[0]
            zigzag_vals = struct.unpack('64B', f.read(64))
            q_tables[qt_id] = inverse_zigzag_8x8(zigzag_vals)
        num_dc_tables = struct.unpack('B', f.read(1))[0]
        dc_tables = {}
        for _ in range(num_dc_tables):
            table_id = struct.unpack('B', f.read(1))[0]
            num_entries = struct.unpack('<H', f.read(2))[0]
            table = {}
            for _ in range(num_entries):
                sym = struct.unpack('B', f.read(1))[0]
                length = struct.unpack('B', f.read(1))[0]
                if length > 0:
                    code_bytes = f.read((length + 7) // 8)
                    code = int.from_bytes(code_bytes, 'big')
                else:
                    code = 0
                table[sym] = (code, length)
            dc_tables[table_id] = table
        num_ac_tables = struct.unpack('B', f.read(1))[0]
        ac_tables = {}
        for _ in range(num_ac_tables):
            table_id = struct.unpack('B', f.read(1))[0]
            num_entries = struct.unpack('<H', f.read(2))[0]
            table = {}
            for _ in range(num_entries):
                sym = struct.unpack('B', f.read(1))[0]
                length = struct.unpack('B', f.read(1))[0]
                if length > 0:
                    code_bytes = f.read((length + 7) // 8)
                    code = int.from_bytes(code_bytes, 'big')
                else:
                    code = 0
                table[sym] = (code, length)
            ac_tables[table_id] = table
        comp_data = []
        for _ in range(num_comp):
            length = struct.unpack('<I', f.read(4))[0]
            data = f.read(length)
            comp_data.append(data)

    return width, height, num_comp, huff_ids, q_tables, dc_tables, ac_tables, comp_data, use_canonical

def compress_channel_with_custom_huffman(channel_2d, q_table, dc_huff, ac_huff):
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
            if value != 0:
                size, bits = compute_size_and_bits(value)
                symbol = (run << 4) | size
                code, codelen = huffman_encode(symbol, ac_huff)
                writer.write_bits(code, codelen)
                writer.write_bits(bits, size)
            i += 1
    return writer.to_bytes()

def decompress_channel_with_custom_huffman(data, rows_blocks, cols_blocks, q_table,
                                           dc_huff, ac_huff):
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
            if symbol == 0x00:
                ac_rle_pairs.append((0, 0))
                break
            run = (symbol >> 4) & 0x0F
            size = symbol & 0x0F
            if run == 15 and size == 0:
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

def build_canonical_huffman_from_channel_data(channel_2d, q_table):
    blocks, rows_blocks, cols_blocks, orig = split_into_blocks(channel_2d, 8)
    dc_list = []
    all_ac_coeffs = []

    for block in blocks:
        coeffs = dct_2d(block.astype(np.float64) - 128)
        quant = quantize(coeffs, q_table)
        zz = zigzag_8x8(quant)
        dc = zz[0]
        ac = list(zz[1:])
        dc_list.append(dc)
        all_ac_coeffs.append(ac)
    dc_diffs = dpcm_encode(dc_list)
    dc_sizes = []
    for diff in dc_diffs:
        if diff == 0:
            dc_sizes.append(0)
        else:
            size, _ = compute_size_and_bits(diff)
            dc_sizes.append(size)
    ac_symbols = []
    for ac_coeffs in all_ac_coeffs:
        runs = rle_encode(ac_coeffs)
        i = 0
        while i < len(runs):
            run, value = runs[i]
            if run == 0 and value == 0:  # EOB
                ac_symbols.append(0x00)
                break
            if run == 15 and value == 0:  # ZRL
                ac_symbols.append(0xF0)
                i += 1
                continue
            if value != 0:
                size, _ = compute_size_and_bits(value)
                symbol = (run << 4) | size
                ac_symbols.append(symbol)
            i += 1
    dc_encode, dc_decode, dc_lengths = CanonicalHuffman.build_from_data(dc_sizes)
    ac_encode, ac_decode, ac_lengths = CanonicalHuffman.build_from_data(ac_symbols)

    return dc_encode, ac_encode

class JPEGCompressor:
    def __init__(self, quality=50):
        self.quality = quality
        self.qt_lum = scale_quantization_table(QT_LUMINANCE, quality)
        self.qt_chrom = scale_quantization_table(QT_CHROMINANCE, quality)
        self.dc_lum_huff = DC_LUM_CODES
        self.dc_chrom_huff = DC_CHROM_CODES
        self.ac_lum_huff = AC_LUM_CODES
        self.ac_chrom_huff = AC_CHROM_CODES

    def compress_raw(self, raw_path, output_path, use_canonical=False):
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
            channels = [ycbcr[:, :, 0], ycbcr[:, :, 1], ycbcr[:, :, 2]]
            q_tables = [self.qt_lum, self.qt_chrom, self.qt_chrom]
            q_table_dict = {0: self.qt_lum, 1: self.qt_chrom}

            if use_canonical:
                dc_tables = {}
                ac_tables = {}
                huff_ids = []
                for idx, (ch, qt) in enumerate(zip(channels, q_tables)):
                    dc_huff, ac_huff = build_canonical_huffman_from_channel_data(ch, qt)
                    dc_tables[idx] = dc_huff
                    ac_tables[idx] = ac_huff
                    qt_id = 0 if idx == 0 else 1
                    huff_ids.append((qt_id, idx, idx))
            else:
                dc_tables = {0: self.dc_lum_huff, 1: self.dc_chrom_huff}
                ac_tables = {0: self.ac_lum_huff, 1: self.ac_chrom_huff}
                huff_ids = [(0, 0, 0), (1, 1, 1), (1, 1, 1)]

        elif img_type in (0x01, 0x02):
            img = np.frombuffer(img_data, dtype=np.uint8).reshape((height, width))
            channels = [img]
            q_tables = [self.qt_lum]
            q_table_dict = {0: self.qt_lum}

            if use_canonical:
                dc_tables = {}
                ac_tables = {}
                dc_huff, ac_huff = build_canonical_huffman_from_channel_data(img, self.qt_lum)
                dc_tables[0] = dc_huff
                ac_tables[0] = ac_huff
                huff_ids = [(0, 0, 0)]
            else:
                dc_tables = {0: self.dc_lum_huff}
                ac_tables = {0: self.ac_lum_huff}
                huff_ids = [(0, 0, 0)]
        else:
            raise ValueError(f"Неподдерживаемый тип изображения: {img_type}")

        compressed = []
        for idx, (ch, qt) in enumerate(zip(channels, q_tables)):
            if use_canonical:
                dc_huff = dc_tables[idx]
                ac_huff = ac_tables[idx]
            else:
                if img_type == 0x03:
                    if idx == 0:
                        dc_huff = self.dc_lum_huff
                        ac_huff = self.ac_lum_huff
                    else:
                        dc_huff = self.dc_chrom_huff
                        ac_huff = self.ac_chrom_huff
                else:
                    dc_huff = self.dc_lum_huff
                    ac_huff = self.ac_lum_huff

            comp_bytes = compress_channel_with_custom_huffman(ch, qt, dc_huff, ac_huff)
            compressed.append(comp_bytes)

        write_compressed_file(output_path, width, height, compressed, q_table_dict,
                              huff_ids, dc_tables, ac_tables, use_canonical)

    def decompress_to_image(self, jcpg_path):
        (width, height, num_comp, huff_ids, q_tables,
         dc_tables, ac_tables, comp_data, use_canonical) = read_compressed_file(jcpg_path)

        rows_blocks = (height + 7) // 8
        cols_blocks = (width + 7) // 8

        channels = []
        for idx, (qt_id, dc_id, ac_id) in enumerate(huff_ids):
            q_table = q_tables[qt_id]
            dc_huff = dc_tables[dc_id]
            ac_huff = ac_tables[ac_id]
            channel = decompress_channel_with_custom_huffman(
                comp_data[idx], rows_blocks, cols_blocks, q_table, dc_huff, ac_huff
            )
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

    modes = [('static', False), ('canonical', True)]
    sizes_static = []
    for mode_name, use_canonical in modes:
        start_time = time.time()
        print(f"\n---Тестирование в режиме: {mode_name}")
        for i, raw_file in enumerate(raw_files):
            raw_path = test_dir / raw_file
            if not raw_path.exists():
                print(f"Файл {raw_path} не найден, пропускаем")
                continue

            print(f"\nОбработка {raw_path.name}")
            sizes = []

            for q in qualities:
                compressor = JPEGCompressor(quality=q)
                mode_dir = out_dir / mode_name
                jcpg_path = mode_dir / f"{raw_path.stem}_q{q}.jcpg"

                compressor.compress_raw(raw_path, jcpg_path, use_canonical=use_canonical)
                size = jcpg_path.stat().st_size
                sizes.append(size)
                print(f"->Quality {q}: {size:,} байт")

                img = compressor.decompress_to_image(jcpg_path)
                if img.ndim == 2:
                    pil_img = Image.fromarray(img, 'L')
                else:
                    pil_img = Image.fromarray(img, 'RGB')
                pil_img.save(mode_dir / f"{raw_path.stem}_decompressed_{q}.jpg")
            if use_canonical:
                plt.figure()
                plt.plot(qualities, sizes_static[i], marker='o', label='Статические')
                plt.plot(qualities, sizes, marker='o', label='Канонические')
                plt.title(f"{raw_file} – зависимость размера от качества")
                plt.xlabel("Quality")
                plt.ylabel("Размер файла (байт)")
                plt.grid(True)
                plt.legend()
                plt.savefig(out_dir / f"{raw_path.stem}_graph.jpg")
                plt.close()
            else:
                sizes_static.append(sizes)
        end_time = time.time()
        print(f"Время, затраченное на обработку изображений в режиме {mode_name}: {end_time - start_time}")

run_tests()