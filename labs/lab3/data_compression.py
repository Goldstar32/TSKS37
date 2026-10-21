#!/usr/bin/env python3
"""
Compress data using Huffman coding.
First argument passed is path to file that is used to create huffman codec.
Second argument passed is path to file that is encoded/decoded.
"""


import sys
from dahuffman import HuffmanCodec
from hamming import hamming_encode, hamming_decode
import numpy as np
import random
from matplotlib import pyplot as plt


def read_file(filename, max=-1):
    with open(filename, "r") as f:
        return f.readlines(max)

def bytes_to_bits(data):
    """Convert bytes into bits, most-significant bit first."""
    return [
        (byte >> shift) & 1
        for byte in data
        for shift in range(7, -1, -1)
    ]

def ber(original_bits, recieved_bits):
    errors = 0
    for [a,b] in zip(original_bits, recieved_bits):
        if a != b:
            errors += 1
    return errors/len(original_bits)

def ser(original_symbols, received_symbols):
    errors = 0
    max_len = max(len(original_symbols), len(received_symbols))
    for i in range(max_len):
        orig_char = original_symbols[i] if i < len(original_symbols) else None
        recv_char = received_symbols[i] if i < len(received_symbols) else None
        if orig_char != recv_char:
            errors += 1
    return errors / max_len if max_len > 0 else 0

def bits_to_bytes(bits):
    """Pack bits into bytes, padding the final byte with zeros if needed."""
    result = bytearray()

    for i in range(0, len(bits), 8):
        chunk = bits[i:i + 8]
        value = 0
        for bit in chunk:
            value = (value << 1) | bit
        value <<= 8 - len(chunk)
        result.append(value)

    return bytes(result)

def transmit(bits, snr_db, skip_print = False):
    x = 2 * np.asarray(bits, dtype=np.float64) - 1

    snr_linear = 10 ** (snr_db / 10)

    noise_std = np.sqrt(1 / snr_linear)
    noise = noise_std * np.random.randn(len(x))
    y = x + noise

    if not skip_print: print(f"SNR: {snr_db:.1f} dB")
    return (y > 0).astype(np.uint8)

def monte_carlo(lines, codec: HuffmanCodec, snrs=range(-2, 11, 2), trials=100):
    snrs = list(snrs)

    coded_ber = []
    uncoded_ber = []
    coded_ser = []
    uncoded_ser = []

    for snr in snrs:
        coded_bit_errors = 0
        uncoded_bit_errors = 0
        coded_symbol_errors = 0
        uncoded_symbol_errors = 0
        total_info_bits = 0
        total_symbols = 0

        for line in lines:
            huffman_bytes = codec.encode(line)
            info_bits = np.asarray(bytes_to_bits(huffman_bytes), dtype=np.uint8)
            coded_bits, padding = hamming_encode(info_bits)
            coded_bits = np.asarray(coded_bits, dtype=np.uint8)

            for _ in range(trials):
                # Hamming-coded transmission
                coded_rx = transmit(coded_bits, snr, skip_print=True)
                decoded_bits = hamming_decode(coded_rx, data_padding=padding)
                decoded_bits = np.asarray(decoded_bits[:len(info_bits)], dtype=np.uint8)

                coded_bit_errors += np.count_nonzero(decoded_bits != info_bits)

                try:
                    decoded_text = codec.decode(bits_to_bytes(decoded_bits))
                    coded_symbol_errors += ser(line, decoded_text) * len(line)
                except Exception:
                    coded_symbol_errors += len(line)

                # Uncoded transmission
                uncoded_rx = transmit(info_bits, snr, skip_print=True)
                uncoded_bit_errors += np.count_nonzero(uncoded_rx != info_bits)

                try:
                    uncoded_text = codec.decode(bits_to_bytes(uncoded_rx))
                    uncoded_symbol_errors += ser(line, uncoded_text) * len(line)
                except Exception:
                    uncoded_symbol_errors += len(line)

                total_info_bits += len(info_bits)
                total_symbols += len(line)

        coded_ber.append(coded_bit_errors / total_info_bits if total_info_bits else 0)
        uncoded_ber.append(uncoded_bit_errors / total_info_bits if total_info_bits else 0)
        coded_ser.append(coded_symbol_errors / total_symbols if total_symbols else 0)
        uncoded_ser.append(uncoded_symbol_errors / total_symbols if total_symbols else 0)

    plt.figure()
    plt.title("BER")
    plt.semilogy(snrs, coded_ber, "o-", label="Coded")
    plt.semilogy(snrs, uncoded_ber, "s-", label="Uncoded")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.title("SER")
    plt.semilogy(snrs, coded_ser, "o-", label="Coded")
    plt.semilogy(snrs, uncoded_ser, "s-", label="Uncoded")
    plt.xlabel("SNR (dB)")
    plt.ylabel("SER")
    plt.grid(True)
    plt.legend()

    plt.show()



if __name__ == "__main__":
    training_text = None
    sample_lines = None
    sample_text = ""
    if len(sys.argv)>2:
        training_file_name = sys.argv[1]
        sample_file_name = sys.argv[2]
        training_text = "".join(read_file(training_file_name))
        sample_lines = read_file(sample_file_name, 200)
        sample_text += sample_lines[0]

    # Huffman
    codec = HuffmanCodec.from_data(training_text)
    huffman_encoded = codec.encode(sample_text)
    test_decoding = codec.decode(huffman_encoded)
    uncompressed_size = len(sample_text.encode("utf-8"))
    compressed_size = len(huffman_encoded)
    print(f"Original size: {uncompressed_size} bytes")
    print(f"Huffman size: {compressed_size} bytes")
    print(f"Compression ratio: {uncompressed_size/compressed_size}")
    print(f"Decoding successful: {test_decoding == sample_text}")

    # Hamming
    unencoded_bits = bytes_to_bits(huffman_encoded)
    encoded_bits, padding = hamming_encode(unencoded_bits)

    original_length = len(unencoded_bits)
    padded_length = original_length + padding
    encoded_length = len(encoded_bits)

    ratio = original_length / encoded_length if encoded_length else 0

    print(f"Before hamming coding: {len(unencoded_bits)} bits")
    print(f"After hamming coding: {len(encoded_bits)} bits")
    print(f"Information-to-total ratio: {ratio:.4f}")
    print(f"Theoretical rate k/n: {4/7:.4f}")

    # Transmission
    snr_db = random.randint(-2, 10)
    
    rx_bits = transmit(encoded_bits, snr_db)
    decoded_bits = hamming_decode(rx_bits, data_padding=padding)
    decoded_text = codec.decode(bits_to_bytes(decoded_bits))

    SER = ser(sample_text, decoded_text)
    BER = ber(unencoded_bits, decoded_bits)

    print(f"SER: {SER}")
    print(f"BER: {BER}")
    print(f"SAMPLE:\n{sample_text[:50]}")
    print(f"DECODED:\n{decoded_text[:50]}")

    # Try without hamming
    print("\n\n\nWithout Hamming encoding")

    rx_bits = transmit(unencoded_bits, snr_db)
    decoded_text = codec.decode(bits_to_bytes(rx_bits))

    SER = ser(sample_text, decoded_text)
    BER = ber(unencoded_bits, rx_bits)

    print(f"SER: {SER}")
    print(f"BER: {BER}")
    print(f"SAMPLE:\n{sample_text[:50]}")
    print(f"DECODED:\n{decoded_text[:50]}")

    monte_carlo(sample_lines, codec)
