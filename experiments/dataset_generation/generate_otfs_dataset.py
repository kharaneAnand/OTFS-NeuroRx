import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Project paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OTFS_ROOT = PROJECT_ROOT / "Phy_Mod_OTFS"

sys.path.insert(0, str(OTFS_ROOT))

from OTFS import OTFS
from OTFSResGrid import OTFSResGrid


# Configuration

M = 32
N = 16

SNR_VALUES_DB = [10, 15, 20]
VELOCITIES_KMH = [30, 120, 500]

NUM_FRAMES_PER_CONDITION = 100

NUM_PATHS = 4
MAX_DELAY = 4

FC_GHZ = 4.0
FREQ_SP_KHZ = 15.0

PL_LEN = 1
PK_LEN = 1

PILOT_SNR_DB = 25

SEED = 42

DATASET_ROOT = PROJECT_ROOT / "datasets" / "otfs"
RAW_DIR = DATASET_ROOT / "raw"
METADATA_DIR = DATASET_ROOT / "metadata"


# Utility

def calculate_doppler(velocity_kmh):
    """
    Calculate normalized maximum Doppler index.
    """

    speed_mps = velocity_kmh / 3.6
    light_speed = 299792458.0

    doppler_khz = (
        speed_mps
        / light_speed
        * FC_GHZ
        * 1e6
    )

    return doppler_khz / (FREQ_SP_KHZ / N)


def qpsk_symbols(num_symbols, rng):
    """
    Generate unit-power QPSK symbols.
    """

    constellation = np.array(
        [
            -1 - 1j,
            -1 + 1j,
             1 - 1j,
             1 + 1j,
        ],
        dtype=np.complex64,
    ) / np.sqrt(2.0)

    indices = rng.integers(
        0,
        4,
        size=num_symbols,
    )

    return constellation[indices]


def get_num_paths(kmax):
    """
    Select a valid path count for the OTFS toolbox.
    """

    available_limit = int(
        MAX_DELAY
        * (2 * np.ceil(kmax) + 1)
    )

    return max(
        1,
        min(NUM_PATHS, available_limit),
    )


# Dataset generation

def generate_dataset():

    rng = np.random.default_rng(SEED)

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = []
    sample_id = 0

    guard_delay_neg = MAX_DELAY
    guard_delay_pos = MAX_DELAY

    num_data_symbols = (
        M * N
        - (
            PL_LEN
            + guard_delay_neg
            + guard_delay_pos
        ) * N
    )

    total_samples = (
        len(SNR_VALUES_DB)
        * len(VELOCITIES_KMH)
        * NUM_FRAMES_PER_CONDITION
    )

    print("=" * 70)
    print("OTFS DATASET GENERATION — V1")
    print("=" * 70)

    print(f"Grid size              : {M} x {N}")
    print(f"SNR values             : {SNR_VALUES_DB}")
    print(f"Velocities             : {VELOCITIES_KMH} km/h")
    print(f"Frames / condition     : {NUM_FRAMES_PER_CONDITION}")
    print(f"Channel paths          : {NUM_PATHS}")
    print(f"Maximum delay          : {MAX_DELAY}")
    print(f"Pilot SNR              : {PILOT_SNR_DB} dB")
    print(f"Data symbols / frame   : {num_data_symbols}")
    print(f"Total samples          : {total_samples}")
    print()

    for velocity in VELOCITIES_KMH:

        kmax = calculate_doppler(velocity)
        num_paths = get_num_paths(kmax)

        print("-" * 70)
        print(
            f"Velocity = {velocity} km/h | "
            f"kmax = {kmax:.4f} | "
            f"paths = {num_paths}"
        )
        print("-" * 70)

        for snr_db in SNR_VALUES_DB:

            print(f"SNR = {snr_db:>2} dB")

            noise_power = 10.0 ** (-snr_db / 10.0)

            pilot_power = (
                noise_power
                * 10.0 ** (PILOT_SNR_DB / 10.0)
            )

            for frame in range(NUM_FRAMES_PER_CONDITION):

                # Generate QPSK data

                tx_data = qpsk_symbols(
                    num_data_symbols,
                    rng,
                )

                # Create resource grid

                rg = OTFSResGrid(
                    M,
                    N,
                )

                rg.setPulse2Recta()

                rg.setPilot2Center(
                    PL_LEN,
                    PK_LEN,
                )

                rg.setGuard(
                    guard_delay_neg,
                    guard_delay_pos,
                    guard_doppl_full=True,
                )

                rg.map(
                    tx_data,
                    pilots_pow=pilot_power,
                )

                # OTFS transmitter

                otfs = OTFS(
                    fc=FC_GHZ,
                    fq_sp=FREQ_SP_KHZ,
                )

                otfs.modulate(rg)

                # Multipath channel

                otfs.setChannel(
                    num_paths,
                    MAX_DELAY,
                    kmax,
                    force_frac=True,
                )

                # Channel + AWGN

                otfs.passChannel(
                    noise_power,
                )

                # Channel information

                his, lis, kis = otfs.getCSI(
                    sort_by_delay_doppler=True,
                )

                # OTFS receiver

                rg_rx = otfs.demodulate()

                # Received DD grid

                rx_dd = rg_rx.getContent()

                # Channel representation

                h_dd = otfs.getChannel(
                    his,
                    lis,
                    kis,
                )

                # Save sample

                sample_file = (
                    RAW_DIR
                    / f"sample_{sample_id:06d}.npz"
                )

                np.savez_compressed(
                    sample_file,
                    rx_dd=rx_dd,
                    tx_dd=tx_data,
                    h_dd=h_dd,
                    his=his,
                    lis=lis,
                    kis=kis,
                )

                metadata.append(
                    {
                        "sample_id": sample_id,
                        "file": sample_file.name,
                        "snr_db": snr_db,
                        "velocity_kmh": velocity,
                        "kmax": kmax,
                        "num_paths": num_paths,
                        "max_delay": MAX_DELAY,
                        "pilot_snr_db": PILOT_SNR_DB,
                        "pilot_power": pilot_power,
                        "noise_power": noise_power,
                        "M": M,
                        "N": N,
                    }
                )

                sample_id += 1

                if (frame + 1) % 25 == 0:

                    print(
                        f"  completed "
                        f"{frame + 1}/"
                        f"{NUM_FRAMES_PER_CONDITION}"
                    )

    # Save metadata

    metadata_df = pd.DataFrame(
        metadata
    )

    metadata_file = (
        METADATA_DIR
        / "dataset_v1_metadata.csv"
    )

    metadata_df.to_csv(
        metadata_file,
        index=False,
    )

    # Save configuration

    config_file = (
        METADATA_DIR
        / "dataset_v1_config.txt"
    )

    with open(
        config_file,
        "w",
        encoding="utf-8",
    ) as f:

        f.write("OTFS DATASET V1\n")
        f.write("================\n")
        f.write(f"M = {M}\n")
        f.write(f"N = {N}\n")
        f.write(f"SNR = {SNR_VALUES_DB}\n")
        f.write(f"Velocity = {VELOCITIES_KMH}\n")
        f.write(
            f"Frames per condition = "
            f"{NUM_FRAMES_PER_CONDITION}\n"
        )
        f.write(f"Paths = {NUM_PATHS}\n")
        f.write(f"Maximum delay = {MAX_DELAY}\n")
        f.write(f"Pilot SNR = {PILOT_SNR_DB} dB\n")
        f.write(
            f"Carrier frequency = "
            f"{FC_GHZ} GHz\n"
        )
        f.write(
            f"Subcarrier spacing = "
            f"{FREQ_SP_KHZ} kHz\n"
        )
        f.write("Modulation = QPSK\n")
        f.write(f"Random seed = {SEED}\n")

    # Summary

    print()
    print("=" * 70)
    print("DATASET GENERATION COMPLETED")
    print("=" * 70)

    print(
        f"Samples generated : {sample_id}"
    )

    print(
        f"Raw data          : {RAW_DIR}"
    )

    print(
        f"Metadata           : {metadata_file}"
    )

    print(
        f"Configuration      : {config_file}"
    )

    print("=" * 70)


# Entry point

if __name__ == "__main__":
    generate_dataset()