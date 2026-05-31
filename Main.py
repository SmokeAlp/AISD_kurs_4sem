import numpy as np
from PIL import Image

def convert(input_path, output_path, type, colorspace="RGB"):
    with Image.open(input_path) as img:
        width, height = img.size

        if type == 'bw_nodith':
            converted = img.convert('1', dither=Image.Dither.NONE).convert('L')
            img_type = 0x01
            colorspace_id = 0x00
        elif type == 'bw_dith':
            converted = img.convert('1').convert('L')
            img_type = 0x01
            colorspace_id = 0x00
        elif type == 'gray':
            converted = img.convert('L')
            img_type = 0x02
            colorspace_id = 0x00
        elif type == 'color':
            converted = img.convert('RGB')
            img_type = 0x03
            if colorspace == 'RGB':
                colorspace_id = 0x01
            elif colorspace == 'YCbCr':
                print("Не та конвертация")
                return

        pixel_data = converted.tobytes()
        header = bytearray(7)
        header[0] = 0xAB
        header[1] = img_type
        header[2] = colorspace_id
        header[3] = width & 0xFF
        header[4] = (width >> 8) & 0xFF
        header[5] = height & 0xFF
        header[6] = (height >> 8) & 0xFF

        with open(output_path, 'wb') as f:
            f.write(header)
            f.write(pixel_data)


def read_raw_file(raw_file, info=0):
    with open(raw_file, 'rb') as f:
        header = f.read(7)

        magic = header[0]
        img_type = header[1]
        colorspace_id = header[2]
        width = header[3] | (header[4] << 8)
        height = header[5] | (header[6] << 8)
        pixel_data = f.read()

        if info:
            type_names = {0x01: "Ч/б",
                          0x02: "Оттенки серого",
                          0x03: "Цветное"}
            colorspace_names = {
                0x00: "None",
                0x01: "RGB",
                0x02: "YCbCr"}
            print(f"\nФайл: {raw_file}")
            print(f"  Magic: 0x{magic:02X}")
            print(f"  Тип: {type_names.get(img_type)}")
            print(f"  Цветовое пространство: {colorspace_names.get(colorspace_id)}")
            print(f"  Размер данных: {len(pixel_data):,} байт")
            print(f"  Размер изображения: {width}x{height} байт")

def get_pixel_data(raw_file):
    with open(raw_file, 'rb') as f:
        header = f.read(7)
        pixel_data = f.read()
        return pixel_data

def raw_to_jpg(raw_file, output_jpg):
    with open(raw_file, 'rb') as f:
        header = f.read(7)
        width = header[3] | (header[4] << 8)
        heigth = header[5] | (header[6] << 8)
        img_type = header[1]
        data = f.read()
    if img_type == 0x03:
        img = Image.frombytes('RGB', (width, heigth), data)
    elif img_type == 0x02:
        img = Image.frombytes('L', (width, heigth), data)
    else:
        img = Image.frombytes('L', (width, heigth), data)
    img.save(output_jpg, 'JPEG')

convert("Тестовые данные/color.jpg", "Тестовые данные/RAW_color.raw", "color")
convert("Тестовые данные/color.jpg", "Тестовые данные/RAW_gray.raw", "gray")
convert("Тестовые данные/color.jpg", "Тестовые данные/RAW_bw_nodith.raw", "bw_nodith")
convert("Тестовые данные/color.jpg", "Тестовые данные/RAW_bw_dith.raw", "bw_dith")
convert("Тестовые данные/lena.png", "Тестовые данные/RAW_lena.raw", "color")

# read_raw_file("Тестовые данные/RAW_color.raw", 1)
# read_raw_file("Тестовые данные/RAW_gray.raw",1)
# read_raw_file("Тестовые данные/RAW_bw_nodith.raw", 1)
# read_raw_file("Тестовые данные/RAW_bw_dith.raw",1)
# read_raw_file("Тестовые данные/RAW_lena.raw", 1)