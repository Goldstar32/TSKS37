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


def read_file(filename):
    with open(filename, "r") as f:
        return f.read()

def bytes_to_bits(data):
    """Convert bytes into bits, most-significant bit first."""
    return [
        (byte >> shift) & 1
        for byte in data
        for shift in range(7, -1, -1)
    ]

def ser(original_symbols, recieved_symbols):
    errors = 0
    for [a,b] in zip(original_symbols, recieved_symbols):
        if a is not b:
            errors += 1
    return errors/len(original_symbols)

def ber(original_bits, recieved_bits):
    return ser(original_bits, recieved_bits)

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

def monte_carlo(snrs = range(-2, 11, 2), trials = 100):
    snrs = list(snrs)

    coded_ber, uncoded_ber = [], []
    coded_ser, uncoded_ser = [], []

    for snr in snrs:
        cb = ub = cs = us = 0
        total_info = total_coded = 0

        print(f"Testing")

        for i in range(trials):
            if i % 100 == 0:
                print(f"Calculating... {i}/{trials}")

            # Coded transmission
            coded_rx = transmit(encoded_bits, snr, True)
            decoded = hamming_decode(coded_rx, data_padding=padding)

            cb += np.count_nonzero(np.array(decoded) != unencoded_bits)
            cs += np.count_nonzero(coded_rx != encoded_bits)
            total_info += len(unencoded_bits)
            total_coded += len(encoded_bits)

            # Uncoded transmission
            uncoded_rx = transmit(unencoded_bits, snr, True)
            ub += np.count_nonzero(uncoded_rx != unencoded_bits)
            us += np.count_nonzero(uncoded_rx != unencoded_bits)

        coded_ber.append(cb / total_info)
        uncoded_ber.append(ub / total_info)
        coded_ser.append(cs / total_coded)
        uncoded_ser.append(us / total_info)

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
    sample_text = None
    if len(sys.argv)>2:
        training_file_name = sys.argv[1]
        sample_file_name = sys.argv[2]
        training_text = read_file(training_file_name)
        sample_text = read_file(sample_file_name)[:200]

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

    monte_carlo()