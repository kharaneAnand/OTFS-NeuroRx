import sys
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------
# Allow Python to find the cloned OTFS toolbox
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OTFS_ROOT = PROJECT_ROOT / "Phy_Mod_OTFS"
sys.path.insert(0, str(OTFS_ROOT))

from OTFS import OTFS
from OTFSResGrid import OTFSResGrid


def main():
    # ---------------------------------------------------------
    # Basic OTFS configuration
    # ---------------------------------------------------------
    M = 16                  # Delay bins / subcarriers
    N = 8                   # Doppler bins / time slots
    SNR_dB = 10

    # QPSK
    M_mod = 4
    bits_per_symbol = 2

    # ---------------------------------------------------------
    # Generate random bits
    # ---------------------------------------------------------
    num_bits = M * N * bits_per_symbol

    bits = torch.randint(
        0,
        2,
        (num_bits,),
        dtype=torch.int64
    )

    # ---------------------------------------------------------
    # QPSK mapping
    # ---------------------------------------------------------
    bit_pairs = bits.reshape(-1, 2)

    real = 1 - 2 * bit_pairs[:, 0]
    imag = 1 - 2 * bit_pairs[:, 1]

    symbols = (
        real.to(torch.float32)
        + 1j * imag.to(torch.float32)
    ) / np.sqrt(2)

    symbols = symbols.to(torch.complex64)

    # ---------------------------------------------------------
    # Create OTFS resource grid
    # ---------------------------------------------------------
    rg = OTFSResGrid(M, N)

    # Rectangular pulse shaping
    rg.setPulse2Recta()

    # Map QPSK symbols
    rg.map(symbols)

    # ---------------------------------------------------------
    # OTFS modulation
    # ---------------------------------------------------------
    otfs = OTFS()

    otfs.modulate(rg)

    # ---------------------------------------------------------
    # OTFS multipath channel
    # ---------------------------------------------------------
    p = 3
    lmax = 2
    kmax = 1

    otfs.setChannel(
        p,
        lmax,
        kmax
    )

    # ---------------------------------------------------------
    # Channel + AWGN
    # ---------------------------------------------------------
    noise_power = 10 ** (-SNR_dB / 10)

    otfs.passChannel(noise_power)

    # ---------------------------------------------------------
    # OTFS demodulation
    # ---------------------------------------------------------
    rg_rx = otfs.demodulate()

    # ---------------------------------------------------------
    # Recover received data
    # demap() returns:
    # y, channel gains, delay indices, Doppler indices
    # ---------------------------------------------------------
    y, _, _, _ = rg_rx.demap()

    y = np.asarray(y).reshape(-1)

    # ---------------------------------------------------------
    # QPSK detection
    # ---------------------------------------------------------
    detected_bits = np.zeros(
        y.size * 2,
        dtype=np.int64
    )

    detected_bits[0::2] = (
        y.real < 0
    ).astype(np.int64)

    detected_bits[1::2] = (
        y.imag < 0
    ).astype(np.int64)

    detected_bits = detected_bits[:num_bits]

    # ---------------------------------------------------------
    # BER
    # ---------------------------------------------------------
    bits_np = bits.numpy()

    ber = np.mean(
        bits_np != detected_bits
    )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------
    print("----------------------------------------")
    print("OTFS BASIC SANITY TEST")
    print("----------------------------------------")
    print(f"Grid size       : {M} x {N}")
    print(f"Modulation      : QPSK")
    print(f"SNR             : {SNR_dB} dB")
    print(f"Channel paths   : {p}")
    print(f"Maximum delay   : {lmax}")
    print(f"Maximum Doppler : {kmax}")
    print(f"Number of bits  : {num_bits}")
    print(f"BER             : {ber:.6f}")
    print("----------------------------------------")


if __name__ == "__main__":
    main()