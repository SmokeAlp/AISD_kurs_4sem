from collections import defaultdict


class HuffmanNode:
    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right

class HuffmanCoder:
    @staticmethod
    def build_huffman_tree(frequencies):
        nodes = []
        for symbol, freq in frequencies.items():
            nodes.append(HuffmanNode(symbol, freq))
        nodes.sort(key=lambda x: x.freq)

        while len(nodes) > 1:
            left = nodes.pop(0)
            right = nodes.pop(0)
            parent = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
            nodes.append(parent)
            nodes.sort(key=lambda x: x.freq)
        return nodes[0] if nodes else None

    @staticmethod
    def get_code_lengths(node, depth=0, lengths=None):
        if lengths is None:
            lengths = {}
        if node.symbol is not None:
            lengths[node.symbol] = depth
        else:
            if node.left:
                HuffmanCoder.get_code_lengths(node.left, depth + 1, lengths)
            if node.right:
                HuffmanCoder.get_code_lengths(node.right, depth + 1, lengths)
        return lengths

    @staticmethod
    def build_canonical_codes(code_lengths):
        symbols_by_length = defaultdict(list)
        for symbol, length in code_lengths.items():
            symbols_by_length[length].append(symbol)
        for length in symbols_by_length:
            symbols_by_length[length].sort()

        canonical_codes = {}
        code = 0
        prev_length = 0

        for length in sorted(symbols_by_length.keys()):
            code <<= (length - prev_length)
            for symbol in symbols_by_length[length]:
                canonical_codes[symbol] = code
                code += 1
            prev_length = length

        return canonical_codes, code_lengths

    @staticmethod
    def encode_with_frequencies(data, frequencies):
        if not data:
            return b'', {}, {}

        tree = HuffmanCoder.build_huffman_tree(frequencies)
        code_lengths = HuffmanCoder.get_code_lengths(tree)
        canonical_codes, _ = HuffmanCoder.build_canonical_codes(code_lengths)
        encode_table = {}
        for symbol, code in canonical_codes.items():
            encode_table[symbol] = (code, code_lengths[symbol])

        return encode_table, code_lengths

    @staticmethod
    def build_decode_table(code_lengths):
        symbols_by_length = defaultdict(list)
        for symbol, length in code_lengths.items():
            symbols_by_length[length].append(symbol)
        for length in symbols_by_length:
            symbols_by_length[length].sort()

        decode_table = {}
        code = 0
        prev_length = 0
        for length in sorted(symbols_by_length.keys()):
            code <<= (length - prev_length)
            for symbol in symbols_by_length[length]:
                decode_table[(code, length)] = symbol
                code += 1
            prev_length = length
        return decode_table

class CanonicalHuffman:
    @staticmethod
    def build_from_data(data):
        if not data:
            return {}, {}

        frequencies = {}
        for val in data:
            frequencies[val] = frequencies.get(val, 0) + 1

        coder = HuffmanCoder()
        encode_table, code_lengths = coder.encode_with_frequencies(data, frequencies)

        decode_table = HuffmanCoder.build_decode_table(code_lengths)

        return encode_table, decode_table, code_lengths