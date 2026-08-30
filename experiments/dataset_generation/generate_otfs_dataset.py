import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OTFS_ROOT = PROJECT_ROOT / "Phy_Mod_OTFS"

sys.path.insert(0, str(OTFS_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from OTFS import OTFS
from OTFSResGrid import OTFSResGrid
from config import load_config


CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment_v1.yaml"


def calculate_doppler(
    velocity_kmh: float,
    carrier_frequency_ghz: float,
    subcarrier_spacing_khz: float,
    num_doppler_bins: int,
) -> float:
    speed_mps = velocity_kmh / 3.6
    light_speed = 299792458.0

    doppler_khz = (
        speed_mps
        / light_speed
        * carrier_frequency_ghz
        * 1e6
    )

    return float(
        doppler_khz
        / (subcarrier_spacing_khz / num_doppler_bins)
    )


def qpsk_symbols(
    num_symbols: int,
    rng: np.random.Generator,
) -> np.ndarray:
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
        len(constellation),
        size=num_symbols,
    )

    return constellation[indices]


def calculate_num_data_symbols(config) -> int:
    M = config.otfs.M
    N = config.otfs.N

    pilot_delay_length = (
        config.pilot.pilot_delay_length
    )

    max_delay = config.channel.max_delay

    num_data_symbols = (
        M * N
        - (
            pilot_delay_length
            + max_delay
            + max_delay
        ) * N
    )

    if num_data_symbols <= 0:
        raise ValueError(
            "Calculated number of data symbols must be positive."
        )

    return int(num_data_symbols)


def get_num_paths(
    config,
    kmax: float,
) -> int:
    max_delay = config.channel.max_delay
    configured_paths = config.channel.num_paths

    available_limit = int(
        max_delay
        * (2 * np.ceil(kmax) + 1)
    )

    return max(
        1,
        min(
            configured_paths,
            available_limit,
        ),
    )


def validate_dataset_contract(
    config,
    rx_dd: np.ndarray,
    tx_dd: np.ndarray,
    h_dd: np.ndarray,
) -> None:
    expected_grid_shape = (
        config.representation.expected_grid_shape
    )

    expected_rx_shape = (
        expected_grid_shape.rows,
        expected_grid_shape.cols,
    )

    expected_data_symbols = (
        config.representation.expected_data_symbols
    )

    expected_channel_shape = (
        config.representation.expected_channel_shape
    )

    expected_h_shape = (
        expected_channel_shape.rows,
        expected_channel_shape.cols,
    )

    if rx_dd.shape != expected_rx_shape:
        raise ValueError(
            f"Invalid rx_dd shape: {rx_dd.shape}; "
            f"expected {expected_rx_shape}."
        )

    if tx_dd.shape != (expected_data_symbols,):
        raise ValueError(
            f"Invalid tx_dd shape: {tx_dd.shape}; "
            f"expected {(expected_data_symbols,)}."
        )

    if h_dd.shape != expected_h_shape:
        raise ValueError(
            f"Invalid h_dd shape: {h_dd.shape}; "
            f"expected {expected_h_shape}."
        )


def generate_dataset(config) -> None:
    rng = np.random.default_rng(
        config.reproducibility.seed
    )

    dataset_root = (
        PROJECT_ROOT / config.dataset.root
    )

    raw_dir = (
        dataset_root / config.dataset.raw_dir
    )

    metadata_dir = (
        dataset_root / config.dataset.metadata_dir
    )

    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    M = config.otfs.M
    N = config.otfs.N

    snr_values_db = config.channel.snr_db
    velocities_kmh = config.channel.velocity_kmh

    frames_per_condition = (
        config.dataset.frames_per_condition
    )

    configured_paths = config.channel.num_paths
    max_delay = config.channel.max_delay

    carrier_frequency_ghz = (
        config.otfs.carrier_frequency_ghz
    )

    subcarrier_spacing_khz = (
        config.otfs.subcarrier_spacing_khz
    )

    pilot_snr_db = (
        config.pilot.pilot_snr_db
    )

    pilot_delay_length = (
        config.pilot.pilot_delay_length
    )

    pilot_doppler_length = (
        config.pilot.pilot_doppler_length
    )

    force_fractional_doppler = (
        config.channel.force_fractional_doppler
    )

    num_data_symbols = (
        calculate_num_data_symbols(config)
    )

    total_samples = (
        len(snr_values_db)
        * len(velocities_kmh)
        * frames_per_condition
    )

    expected_samples = (
        config.dataset.expected_samples
    )

    if total_samples != expected_samples:
        raise ValueError(
            f"Dataset configuration expects "
            f"{expected_samples} samples, but the "
            f"configured sweep produces {total_samples}."
        )

    metadata = []
    sample_id = 0

    print("=" * 70)
    print("OTFS DATASET GENERATION — V1")
    print("=" * 70)

    print(
        f"Grid size              : {M} x {N}"
    )
    print(
        f"SNR values             : {snr_values_db}"
    )
    print(
        f"Velocities             : {velocities_kmh} km/h"
    )
    print(
        f"Frames / condition     : {frames_per_condition}"
    )
    print(
        f"Channel paths          : {configured_paths}"
    )
    print(
        f"Maximum delay          : {max_delay}"
    )
    print(
        f"Pilot SNR              : {pilot_snr_db} dB"
    )
    print(
        f"Data symbols / frame   : {num_data_symbols}"
    )
    print(
        f"Total samples          : {total_samples}"
    )
    print()

    for velocity in velocities_kmh:

        kmax = calculate_doppler(
            velocity,
            carrier_frequency_ghz,
            subcarrier_spacing_khz,
            N,
        )

        num_paths = get_num_paths(
            config,
            kmax,
        )

        print("-" * 70)
        print(
            f"Velocity = {velocity} km/h | "
            f"kmax = {kmax:.4f} | "
            f"paths = {num_paths}"
        )
        print("-" * 70)

        for snr_db in snr_values_db:

            print(
                f"SNR = {snr_db:>2} dB"
            )

            noise_power = (
                10.0 ** (-snr_db / 10.0)
            )

            pilot_power = (
                noise_power
                * 10.0 ** (
                    pilot_snr_db / 10.0
                )
            )

            for frame in range(
                frames_per_condition
            ):

                tx_data = qpsk_symbols(
                    num_data_symbols,
                    rng,
                )

                rg = OTFSResGrid(
                    M,
                    N,
                )

                rg.setPulse2Recta()

                rg.setPilot2Center(
                    pilot_delay_length,
                    pilot_doppler_length,
                )

                rg.setGuard(
                    max_delay,
                    max_delay,
                    guard_doppl_full=True,
                )

                rg.map(
                    tx_data,
                    pilots_pow=pilot_power,
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
                    force_frac=force_fractional_doppler,
                )

                otfs.passChannel(
                    noise_power,
                )

                his, lis, kis = otfs.getCSI(
                    sort_by_delay_doppler=True,
                )

                rg_rx = otfs.demodulate()

                rx_dd = np.asarray(
                    rg_rx.getContent()
                )

                h_dd = np.asarray(
                    otfs.getChannel(
                        his,
                        lis,
                        kis,
                    )
                )

                validate_dataset_contract(
                    config,
                    rx_dd,
                    tx_data,
                    h_dd,
                )

                sample_file = (
                    raw_dir
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
                        "max_delay": max_delay,
                        "pilot_snr_db": pilot_snr_db,
                        "pilot_power": pilot_power,
                        "noise_power": noise_power,
                        "M": M,
                        "N": N,
                    }
                )

                sample_id += 1

                if (
                    (frame + 1)
                    % 25
                    == 0
                ):
                    print(
                        f"  completed "
                        f"{frame + 1}/"
                        f"{frames_per_condition}"
                    )

    metadata_df = pd.DataFrame(
        metadata
    )

    metadata_file = (
        metadata_dir
        / "dataset_v1_metadata.csv"
    )

    metadata_df.to_csv(
        metadata_file,
        index=False,
    )

    config_file = (
        metadata_dir
        / "dataset_v1_config.txt"
    )

    with config_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "OTFS DATASET V1\n"
        )
        file.write(
            "================\n"
        )
        file.write(
            f"M = {M}\n"
        )
        file.write(
            f"N = {N}\n"
        )
        file.write(
            f"SNR = {snr_values_db}\n"
        )
        file.write(
            f"Velocity = {velocities_kmh}\n"
        )
        file.write(
            f"Frames per condition = "
            f"{frames_per_condition}\n"
        )
        file.write(
            f"Paths = {configured_paths}\n"
        )
        file.write(
            f"Maximum delay = {max_delay}\n"
        )
        file.write(
            f"Pilot SNR = {pilot_snr_db} dB\n"
        )
        file.write(
            f"Carrier frequency = "
            f"{carrier_frequency_ghz} GHz\n"
        )
        file.write(
            f"Subcarrier spacing = "
            f"{subcarrier_spacing_khz} kHz\n"
        )
        file.write(
            f"Modulation = "
            f"{config.modulation.scheme}\n"
        )
        file.write(
            f"Random seed = "
            f"{config.reproducibility.seed}\n"
        )

    if sample_id != expected_samples:
        raise RuntimeError(
            f"Generated {sample_id} samples; "
            f"expected {expected_samples}."
        )

    print()
    print("=" * 70)
    print("DATASET GENERATION COMPLETED")
    print("=" * 70)

    print(
        f"Samples generated : {sample_id}"
    )

    print(
        f"Raw data          : {raw_dir}"
    )

    print(
        f"Metadata           : {metadata_file}"
    )

    print(
        f"Configuration      : {config_file}"
    )

    print("=" * 70)


def main() -> None:
    config = load_config(
        CONFIG_PATH
    )

    generate_dataset(
        config
    )


if __name__ == "__main__":
    main()