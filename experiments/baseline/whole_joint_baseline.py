import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OTFS_ROOT = PROJECT_ROOT / "Phy_Mod_OTFS"

sys.path.insert(0, str(OTFS_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from OTFS import OTFS
from OTFSResGrid import OTFSResGrid
from config import load_config


CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment_v1.yaml"


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


def calculate_num_data_symbols(config) -> int:
    M = config.otfs.M
    N = config.otfs.N

    pilot_delay_length = config.pilot.pilot_delay_length
    max_delay = config.channel.max_delay

    return int(
        M * N
        - (
            pilot_delay_length
            + max_delay
            + max_delay
        ) * N
    )


def run_experiment(
    config,
    velocity_kmh: float,
    pilot_snr_db: float,
    rng: np.random.Generator,
) -> float:
    M = config.otfs.M
    N = config.otfs.N

    carrier_frequency_ghz = (
        config.otfs.carrier_frequency_ghz
    )

    subcarrier_spacing_khz = (
        config.otfs.subcarrier_spacing_khz
    )

    num_paths = config.channel.num_paths
    max_delay = config.channel.max_delay
    force_fractional_doppler = (
        config.channel.force_fractional_doppler
    )

    pilot_delay_length = (
        config.pilot.pilot_delay_length
    )

    pilot_doppler_length = (
        config.pilot.pilot_doppler_length
    )

    data_snr_db = config.baseline.data_snr_db

    num_frames = config.baseline.num_frames

    kmax = calculate_doppler(
        velocity_kmh,
        carrier_frequency_ghz,
        subcarrier_spacing_khz,
        N,
    )

    noise_power = 10.0 ** (
        -data_snr_db / 10.0
    )

    pilot_power = (
        noise_power
        * 10.0 ** (pilot_snr_db / 10.0)
    )

    num_data_symbols = calculate_num_data_symbols(
        config
    )

    nmse_values = []

    for _ in range(num_frames):

        x_dd = qpsk_symbols(
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
            x_dd,
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
            noise_power
        )

        his, lis, kis = otfs.getCSI(
            sort_by_delay_doppler=True
        )

        rg_rx = otfs.demodulate()

        threshold = (
            3.0
            * np.sqrt(noise_power)
        )

        (
            _,
            his_est,
            lis_est,
            kis_est,
        ) = rg_rx.demap(
            isData=False,
            threshold=threshold,
        )

        h_dd = np.asarray(
            otfs.getChannel(
                his,
                lis,
                kis,
                data_only=False,
            )
        )

        true_channel_power = np.mean(
            np.abs(h_dd) ** 2
        )

        if true_channel_power <= np.finfo(float).eps:
            continue

        if (
            his_est is None
            or lis_est is None
            or kis_est is None
        ):
            nmse_linear = 1.0

        elif len(np.atleast_1d(his_est)) == 0:
            nmse_linear = 1.0

        else:
            h_dd_est = np.asarray(
                otfs.getChannel(
                    his_est,
                    lis_est,
                    kis_est,
                    data_only=False,
                )
            )

            error_power = np.mean(
                np.abs(
                    h_dd - h_dd_est
                ) ** 2
            )

            nmse_linear = (
                error_power
                / true_channel_power
            )

        nmse_db = 10.0 * np.log10(
            max(
                nmse_linear,
                np.finfo(float).eps,
            )
        )

        nmse_values.append(
            nmse_db
        )

    if not nmse_values:
        return float("nan")

    return float(
        np.mean(nmse_values)
    )


def save_results(
    output_file: Path,
    pilot_snr_values,
    velocities,
    results,
) -> None:
    columns = [
        "pilot_snr_db"
    ]

    columns.extend(
        f"nmse_{velocity}_kmh_db"
        for velocity in velocities
    )

    rows = []

    for index, pilot_snr in enumerate(
        pilot_snr_values
    ):
        row = {
            "pilot_snr_db": pilot_snr
        }

        for velocity in velocities:
            row[
                f"nmse_{velocity}_kmh_db"
            ] = results[velocity][index]

        rows.append(row)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    import pandas as pd

    pd.DataFrame(rows).to_csv(
        output_file,
        index=False,
    )


def save_plot(
    output_file: Path,
    pilot_snr_values,
    velocities,
    results,
) -> None:
    plt.figure(
        figsize=(8, 5)
    )

    for velocity in velocities:
        plt.plot(
            pilot_snr_values,
            results[velocity],
            marker="o",
            label=f"{velocity} km/h",
        )

    plt.xlabel(
        "Pilot SNR (dB)"
    )

    plt.ylabel(
        "Channel NMSE (dB)"
    )

    plt.title(
        "OTFS Whole-Joint Channel Estimation Baseline"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        output_file,
        dpi=300,
    )

    plt.close()


def main() -> None:
    config = load_config(
        CONFIG_PATH
    )

    rng = np.random.default_rng(
        config.reproducibility.seed
    )

    M = config.otfs.M
    N = config.otfs.N

    velocities = config.channel.velocity_kmh

    pilot_snr_values = (
        config.baseline.pilot_snr_db
    )

    output_dir = (
        PROJECT_ROOT
        / config.baseline.output_dir
    )

    results_file = (
        output_dir
        / config.baseline.results_file
    )

    plot_file = (
        output_dir
        / config.baseline.plot_file
    )

    print("=" * 70)
    print(
        "OTFS WHOLE-JOINT BASELINE"
    )
    print("=" * 70)

    print(
        f"M                  : {M}"
    )

    print(
        f"N                  : {N}"
    )

    print(
        f"Carrier frequency  : "
        f"{config.otfs.carrier_frequency_ghz} GHz"
    )

    print(
        f"Subcarrier spacing : "
        f"{config.otfs.subcarrier_spacing_khz} kHz"
    )

    print(
        f"Channel paths      : "
        f"{config.channel.num_paths}"
    )

    print(
        f"Maximum delay      : "
        f"{config.channel.max_delay}"
    )

    print(
        f"Data SNR           : "
        f"{config.baseline.data_snr_db} dB"
    )

    print(
        f"Frames             : "
        f"{config.baseline.num_frames}"
    )

    print()

    print(
        "Doppler configuration"
    )

    print("-" * 70)

    for velocity in velocities:
        kmax = calculate_doppler(
            velocity,
            config.otfs.carrier_frequency_ghz,
            config.otfs.subcarrier_spacing_khz,
            N,
        )

        print(
            f"{velocity:>4} km/h"
            f" -> kmax = {kmax:.4f}"
            f" -> paths = "
            f"{config.channel.num_paths}"
        )

    print()

    results = {}

    for velocity in velocities:

        kmax = calculate_doppler(
            velocity,
            config.otfs.carrier_frequency_ghz,
            config.otfs.subcarrier_spacing_khz,
            N,
        )

        print("=" * 70)

        print(
            f"Velocity = {velocity} km/h"
        )

        print(
            f"kmax     = {kmax:.4f}"
        )

        print(
            f"Paths    = "
            f"{config.channel.num_paths}"
        )

        print("=" * 70)

        velocity_results = []

        for pilot_snr in pilot_snr_values:

            print(
                f"Pilot SNR = "
                f"{pilot_snr:>2} dB ... ",
                end="",
                flush=True,
            )

            nmse = run_experiment(
                config,
                velocity,
                pilot_snr,
                rng,
            )

            velocity_results.append(
                nmse
            )

            print(
                f"NMSE = "
                f"{nmse:.4f} dB"
            )

        results[velocity] = np.asarray(
            velocity_results
        )

    print()

    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    header = "Pilot SNR"

    for velocity in velocities:
        header += (
            f" | {velocity:>12} km/h"
        )

    print(header)
    print("-" * len(header))

    for index, pilot_snr in enumerate(
        pilot_snr_values
    ):
        row = f"{pilot_snr:>9} dB"

        for velocity in velocities:
            row += (
                f" | "
                f"{results[velocity][index]:>14.4f}"
            )

        print(row)

    save_results(
        results_file,
        pilot_snr_values,
        velocities,
        results,
    )

    save_plot(
        plot_file,
        pilot_snr_values,
        velocities,
        results,
    )

    print()

    print("=" * 70)
    print(
        "BASELINE EXPERIMENT COMPLETED"
    )
    print("=" * 70)

    print(
        f"CSV saved to:\n{results_file}"
    )

    print(
        f"Plot saved to:\n{plot_file}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()