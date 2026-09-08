"""Repeat channel-estimation NMSE across seeds for mobility comparison."""

from __future__ import annotations

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
OUTPUT_PATH = PROJECT_ROOT / "experiments" / "baseline" / "mobility_seed_check.csv"
SEEDS = (42, 43, 44, 45, 46)
FRAMES_PER_CONDITION = 20


def qpsk_symbols(num_symbols: int, rng: np.random.Generator) -> np.ndarray:
    constellation = np.array(
        [-1 - 1j, -1 + 1j, 1 - 1j, 1 + 1j],
        dtype=np.complex64,
    ) / np.sqrt(2.0)
    return constellation[rng.integers(0, len(constellation), size=num_symbols)]


def calculate_doppler(velocity_kmh: float, config) -> float:
    speed_mps = velocity_kmh / 3.6
    doppler_khz = (
        speed_mps
        / 299792458.0
        * config.otfs.carrier_frequency_ghz
        * 1e6
    )
    return float(
        doppler_khz
        / (config.otfs.subcarrier_spacing_khz / config.otfs.N)
    )


def condition_nmse(
    config,
    snr_db: float,
    velocity_kmh: float,
    rng: np.random.Generator,
) -> float:
    noise_power = 10.0 ** (-snr_db / 10.0)
    pilot_power = noise_power * 10.0 ** (
        config.pilot.pilot_snr_db / 10.0
    )
    kmax = calculate_doppler(velocity_kmh, config)
    num_symbols = config.representation.expected_data_symbols
    values = []

    for _ in range(FRAMES_PER_CONDITION):
        rg = OTFSResGrid(config.otfs.M, config.otfs.N)
        rg.setPulse2Recta()
        rg.setPilot2Center(
            config.pilot.pilot_delay_length,
            config.pilot.pilot_doppler_length,
        )
        rg.setGuard(
            config.channel.max_delay,
            config.channel.max_delay,
            guard_doppl_full=True,
        )
        rg.map(
            qpsk_symbols(num_symbols, rng),
            pilots_pow=pilot_power,
        )

        otfs = OTFS(
            fc=config.otfs.carrier_frequency_ghz,
            fq_sp=config.otfs.subcarrier_spacing_khz,
        )
        otfs.modulate(rg)
        otfs.setChannel(
            config.channel.num_paths,
            config.channel.max_delay,
            kmax,
            force_frac=config.channel.force_fractional_doppler,
            rng=rng,
        )
        otfs.passChannel(noise_power, rng=rng)

        his, lis, kis = otfs.getCSI(sort_by_delay_doppler=True)
        rg_rx = otfs.demodulate()
        threshold = 3.0 * np.sqrt(noise_power)
        _, his_hat, lis_hat, kis_hat = rg_rx.demap(
            isData=False,
            threshold=threshold,
        )
        h_true = np.asarray(
            otfs.getChannel(his, lis, kis)
        )

        if his_hat is None or len(np.atleast_1d(his_hat)) == 0:
            h_hat = np.zeros_like(h_true)
        else:
            h_hat = np.asarray(
                otfs.getChannel(his_hat, lis_hat, kis_hat)
            )

        signal_power = np.sum(np.abs(h_true) ** 2)
        error_power = np.sum(np.abs(h_hat - h_true) ** 2)
        values.append(error_power / signal_power)

    return float(10.0 * np.log10(np.mean(values)))


def main() -> None:
    config = load_config(CONFIG_PATH)
    rows = []

    for seed in SEEDS:
        for snr_db in config.channel.snr_db:
            for velocity_kmh in config.channel.velocity_kmh:
                nmse_db = condition_nmse(
                    config,
                    snr_db,
                    velocity_kmh,
                    np.random.default_rng(
                        np.random.SeedSequence(
                            [seed, int(snr_db), int(velocity_kmh)]
                        )
                    ),
                )
                rows.append(
                    {
                        "seed": seed,
                        "snr_db": snr_db,
                        "velocity_kmh": velocity_kmh,
                        "pilot_snr_db": config.pilot.pilot_snr_db,
                        "h_hat_nmse_db": nmse_db,
                    }
                )

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_PATH, index=False)
    summary = (
        results.groupby(["snr_db", "velocity_kmh"])["h_hat_nmse_db"]
        .agg(["mean", "std"])
        .reset_index()
    )
    print(summary.to_string(index=False))

    mobility_summary = (
        results.groupby("velocity_kmh")["h_hat_nmse_db"]
        .agg(["mean", "std"])
        .reset_index()
    )
    print()
    print("Mobility summary across SNR conditions")
    print(mobility_summary.to_string(index=False))
    print(f"Mobility seed-check results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()