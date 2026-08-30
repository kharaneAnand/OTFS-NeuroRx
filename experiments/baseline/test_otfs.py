import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
OTFS_ROOT = PROJECT_ROOT / "Phy_Mod_OTFS"

sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(OTFS_ROOT))

from config import load_config
from OTFS import OTFS
from OTFSResGrid import OTFSResGrid


def qpsk_detect(symbols):
    symbols = np.asarray(symbols).reshape(-1)

    detected_bits = np.empty(
        symbols.size * 2,
        dtype=np.int64,
    )

    detected_bits[0::2] = (
        symbols.real < 0
    ).astype(np.int64)

    detected_bits[1::2] = (
        symbols.imag < 0
    ).astype(np.int64)

    return detected_bits


def main():
    config_path = (
        PROJECT_ROOT
        / "configs"
        / "experiment_v1.yaml"
    )

    config = load_config(config_path)

    M = config.otfs.M
    N = config.otfs.N

    snr_db = config.baseline.data_snr_db
    velocity_kmh = config.channel.velocity_kmh[0]

    num_paths = config.channel.num_paths
    max_delay = config.channel.max_delay

    carrier_frequency_ghz = (
        config.otfs.carrier_frequency_ghz
    )

    subcarrier_spacing_khz = (
        config.otfs.subcarrier_spacing_khz
    )

    seed = config.reproducibility.seed

    rng = np.random.default_rng(seed)

    bits_per_symbol = 2
    num_symbols = M * N
    num_bits = num_symbols * bits_per_symbol

    bits = rng.integers(
        0,
        2,
        size=num_bits,
    )

    bit_pairs = bits.reshape(-1, 2)

    real = 1 - 2 * bit_pairs[:, 0]
    imag = 1 - 2 * bit_pairs[:, 1]

    symbols = (
        real.astype(np.float32)
        + 1j * imag.astype(np.float32)
    ) / np.sqrt(2.0)

    symbols = symbols.astype(np.complex64)

    speed_mps = velocity_kmh / 3.6
    light_speed = 299792458.0

    doppler_khz = (
        speed_mps
        / light_speed
        * carrier_frequency_ghz
        * 1e6
    )

    kmax = (
        doppler_khz
        / (subcarrier_spacing_khz / N)
    )

    rg = OTFSResGrid(
        M,
        N,
    )

    rg.setPulse2Recta()

    rg.map(
        torch.from_numpy(symbols)
    )

    otfs = OTFS(
        fc=carrier_frequency_ghz,
        fq_sp=subcarrier_spacing_khz,
    )

    otfs.modulate(rg)

    otfs.setChannel(
        num_paths,
        max_delay,
        kmax,
        force_frac=True,
    )

    noise_power = (
        10.0 ** (-snr_db / 10.0)
    )

    otfs.passChannel(
        noise_power
    )

    his, lis, kis = otfs.getCSI(
        sort_by_delay_doppler=True
    )

    rg_rx = otfs.demodulate()

    y, _, _, _ = rg_rx.demap()

    y = np.asarray(y).reshape(-1)

    detected_bits = qpsk_detect(y)

    detected_bits = detected_bits[:num_bits]

    ber = np.mean(
        bits != detected_bits
    )

    print("-" * 50)
    print("OTFS BASIC SANITY TEST")
    print("-" * 50)
    print(
        f"Grid size       : {M} x {N}"
    )
    print(
        f"Modulation      : "
        f"{config.modulation.scheme}"
    )
    print(
        f"SNR             : {snr_db} dB"
    )
    print(
        f"Velocity        : "
        f"{velocity_kmh} km/h"
    )
    print(
        f"Channel paths   : "
        f"{num_paths}"
    )
    print(
        f"Maximum delay   : "
        f"{max_delay}"
    )
    print(
        f"Maximum Doppler : "
        f"{kmax:.4f}"
    )
    print(
        f"Number of bits  : "
        f"{num_bits}"
    )
    print(
        f"BER             : "
        f"{ber:.6f}"
    )
    print("-" * 50)


if __name__ == "__main__":
    main()