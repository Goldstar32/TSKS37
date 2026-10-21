import numpy as np


# Generator matrix G (4 x 7)
G = np.array([
    [1, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1],
], dtype=np.uint8)

H = np.array([
    [1, 1, 0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0, 1, 0],
    [0, 1, 1, 1, 0, 0, 1],
], dtype=np.uint8)


def hamming_encode(bits):
    """
    Encode a bit sequence using the systematic (7, 4) Hamming code.
    """
    # Convert input to a list of integer bits.
    if isinstance(bits, str):
        if any(bit not in "01" for bit in bits):
            raise ValueError("Input string must contain only '0' and '1'.")
        bits = [int(bit) for bit in bits]
    else:
        bits = [int(bit) for bit in bits]
        if any(bit not in (0, 1) for bit in bits):
            raise ValueError("Input must contain only 0 and 1.", bits)

    original_length = len(bits)

    # Pad with zeros to make the length divisible by 4.
    padding = (-original_length) % 4
    bits.extend([0] * padding)

    # Encode each 4-bit block using c = uG (mod 2).
    encoded_bits = []

    for i in range(0, len(bits), 4):
        u = np.array(bits[i:i + 4], dtype=np.uint8)

        # Matrix multiplication over GF(2): addition is XOR.
        c = (u @ G) % 2

        encoded_bits.extend(c.astype(int).tolist())

    return encoded_bits, padding

def hamming_decode(received_bits, data_padding=0):
    """
    Decode systematic (7, 4) Hamming codewords.
    """
    bits = [int(bit) for bit in received_bits]

    if any(bit not in (0, 1) for bit in bits):
        raise ValueError("Input must contain only 0 and 1.")
    if len(bits) % 7 != 0:
        raise ValueError("Received bit sequence length must be divisible by 7.")
    if data_padding not in (0, 1, 2, 3):
        raise ValueError("data_padding must be between 0 and 3.")

    decoded_bits = []

    for i in range(0, len(bits), 7):
        r = np.array(bits[i:i + 7], dtype=np.uint8)

        # Syndrome s = H @ r.T (mod 2)
        syndrome = (H @ r) % 2

        if np.any(syndrome):
            error_position = next(
                (j for j in range(7) if np.array_equal(H[:, j], syndrome)),
                None
            )

            if error_position is not None: r[error_position] ^= 1

        # First 4 bits are the data bits.
        decoded_bits.extend(r[:4].astype(int).tolist())

    if data_padding:
        decoded_bits = decoded_bits[:-data_padding]

    return decoded_bits
