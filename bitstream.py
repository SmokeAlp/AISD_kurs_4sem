class BitWriter:
    def __init__(self):
        self.buffer = bytearray()
        self.current_byte = 0
        self.num_bits = 0

    def write_bit(self, bit):
        self.current_byte = (self.current_byte << 1) | (bit & 1)
        self.num_bits += 1
        if self.num_bits == 8:
            self.buffer.append(self.current_byte)
            self.current_byte = 0
            self.num_bits = 0

    def write_bits(self, value, num_bits):
        for i in range(num_bits-1, -1, -1):
            self.write_bit((value >> i) & 1)

    def flush(self):
        if self.num_bits > 0:
            self.current_byte <<= (8 - self.num_bits)
            self.buffer.append(self.current_byte)
            self.current_byte = 0
            self.num_bits = 0

    def to_bytes(self):
        self.flush()
        return bytes(self.buffer)

class BitReader:
    def __init__(self, data):
        self.data = data
        self.byte_idx = 0
        self.bit_idx = 0

    def read_bit(self):
        if self.byte_idx >= len(self.data):
            raise EOFError("Нет данных для чтения")
        byte = self.data[self.byte_idx]
        bit = (byte >> (7 - self.bit_idx)) & 1
        self.bit_idx += 1
        if self.bit_idx == 8:
            self.byte_idx += 1
            self.bit_idx = 0
        return bit

    def read_bits(self, num_bits):
        value = 0
        for _ in range(num_bits):
            value = (value << 1) | self.read_bit()
        return value