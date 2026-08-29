import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OTFS_ROOT = PROJECT_ROOT / "Phy_Mod_OTFS"

sys.path.insert(0, str(OTFS_ROOT))

from OTFS import OTFS
from OTFSResGrid import OTFSResGrid


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

# OTFS frame
M = 32
N = 16

# Carrier / waveform
FC_GHZ = 4.0
FREQ_SP_KHZ = 15.0

# Channel
#
# IMPORTANT:
# The repository's random tap-generation routine can produce
# fewer usable taps than the theoretical maximum for some
# fractional-Doppler configurations.
#
# Four paths is therefore used as a stable common configuration
# across all three velocities.
#
NUM_PATHS = 4
MAX_DELAY = 4

# Velocities
VELOCITIES_KMH = [
    30,
    120,
    500,
]

# Data SNR
DATA_SNR_DB = 10

# Pilot SNR sweep
PILOT_SNR_DB = np.arange(
    25,
    51,
    5,
)

# Pilot
PL_LEN = 1
PK_LEN = 1

# Guard
GUARD_DELAY_NEG = MAX_DELAY
GUARD_DELAY_POS = MAX_DELAY

# Monte Carlo
#
# Validation value.
# Do NOT change to 1e6 until the implementation is validated.
NUM_FRAMES = 100

# Reproducibility
RANDOM_SEED = 42


# ============================================================
# QPSK
# ============================================================

def qpsk_symbols(num_symbols):
    """
    Generate unit-average-power QPSK symbols.
    """

    indices = np.random.randint(
        0,
        4,
        size=num_symbols,
    )

    constellation = np.array(
        [
            -1 - 1j,
            -1 + 1j,
             1 - 1j,
             1 + 1j,
        ],
        dtype=np.complex64,
    )

    constellation /= np.sqrt(2.0)

    return constellation[indices]


# ============================================================
# DOPPLER
# ============================================================

def calculate_doppler(velocity_kmh):
    """
    Calculate the normalized maximum Doppler index.

    Repository MATLAB configuration:

        fd = vs/3.6/physconst('LightSpeed')*fc*1e6
        kmax = fd/(freq_sp/N)

    """

    speed_mps = (
        velocity_kmh / 3.6
    )

    light_speed = 299792458.0

    doppler_khz = (
        speed_mps
        / light_speed
        * FC_GHZ
        * 1e6
    )

    kmax = (
        doppler_khz
        / (FREQ_SP_KHZ / N)
    )

    return float(kmax)


# ============================================================
# NUMBER OF DATA SYMBOLS
# ============================================================

def calculate_num_data_symbols():
    """
    MATLAB-style data-symbol calculation.

        N_syms_perfram =
            N*M-(pl_len + gdn_len + gdp_len)*N
    """

    return int(
        M * N
        - (
            PL_LEN
            + GUARD_DELAY_NEG
            + GUARD_DELAY_POS
        ) * N
    )


# ============================================================
# SINGLE EXPERIMENT
# ============================================================

def run_experiment(
    velocity_kmh,
    pilot_snr_db,
):
    """
    Run the channel-estimation experiment for:

        velocity_kmh
        pilot_snr_db

    Returns:
        Mean channel NMSE in dB.
    """

    # --------------------------------------------------------
    # Doppler
    # --------------------------------------------------------

    kmax = calculate_doppler(
        velocity_kmh
    )

    # --------------------------------------------------------
    # Noise power
    # --------------------------------------------------------

    noise_power = (
        10.0
        ** (
            -DATA_SNR_DB
            / 10.0
        )
    )

    # --------------------------------------------------------
    # Pilot power
    # --------------------------------------------------------

    pilot_power = (
        noise_power
        * 10.0
        ** (
            pilot_snr_db
            / 10.0
        )
    )

    # --------------------------------------------------------
    # Number of data symbols
    # --------------------------------------------------------

    num_data_symbols = (
        calculate_num_data_symbols()
    )

    nmse_values = []

    # --------------------------------------------------------
    # Monte Carlo frames
    # --------------------------------------------------------

    for frame_idx in range(
        NUM_FRAMES
    ):

        # ----------------------------------------------------
        # Generate QPSK data
        # ----------------------------------------------------

        x_dd = qpsk_symbols(
            num_data_symbols
        )

        # ----------------------------------------------------
        # Resource grid
        # ----------------------------------------------------

        rg = OTFSResGrid(
            M,
            N,
        )

        # Rectangular pulse
        rg.setPulse2Recta()

        # Center pilot
        rg.setPilot2Center(
            PL_LEN,
            PK_LEN,
        )

        # Delay guard
        rg.setGuard(
            GUARD_DELAY_NEG,
            GUARD_DELAY_POS,
            guard_doppl_full=True,
        )

        # Map data + pilot
        rg.map(
            x_dd,
            pilots_pow=pilot_power,
        )

        # ----------------------------------------------------
        # OTFS transmitter
        # ----------------------------------------------------

        otfs = OTFS(
            fc=FC_GHZ,
            fq_sp=FREQ_SP_KHZ,
        )

        otfs.modulate(
            rg
        )

        # ----------------------------------------------------
        # Channel
        #
        # Keep fractional kmax.
        # This is important for the high-speed OTFS case.
        # ----------------------------------------------------

        otfs.setChannel(
            NUM_PATHS,
            MAX_DELAY,
            kmax,
        )

        # ----------------------------------------------------
        # Channel + AWGN
        # ----------------------------------------------------

        otfs.passChannel(
            noise_power
        )

        # ----------------------------------------------------
        # Perfect CSI
        # ----------------------------------------------------

        his, lis, kis = otfs.getCSI(
            sort_by_delay_doppler=True
        )

        # ----------------------------------------------------
        # Demodulation
        # ----------------------------------------------------

        rg_rx = otfs.demodulate()

        # ----------------------------------------------------
        # Pilot-based channel estimation
        # ----------------------------------------------------

        threshold = (
            3.0
            * np.sqrt(
                noise_power
            )
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

        # ----------------------------------------------------
        # True channel matrix
        # ----------------------------------------------------

        h_dd = otfs.getChannel(
            his,
            lis,
            kis,
            data_only=False,
        )

        # ----------------------------------------------------
        # Validate true channel
        # ----------------------------------------------------

        h_dd = np.asarray(
            h_dd
        )

        true_channel_power = np.mean(
            np.abs(h_dd) ** 2
        )

        if (
            true_channel_power
            <= np.finfo(float).eps
        ):
            continue

        # ----------------------------------------------------
        # Estimated channel
        # ----------------------------------------------------

        if (
            his_est is None
            or lis_est is None
            or kis_est is None
        ):

            # No estimated channel.
            # NMSE = 1 -> 0 dB.
            nmse_linear = 1.0

        else:

            his_est_arr = np.atleast_1d(
                his_est
            )

            if len(
                his_est_arr
            ) == 0:

                nmse_linear = 1.0

            else:

                h_dd_est = otfs.getChannel(
                    his_est,
                    lis_est,
                    kis_est,
                    data_only=False,
                )

                h_dd_est = np.asarray(
                    h_dd_est
                )

                # ------------------------------------------------
                # NMSE
                # ------------------------------------------------

                error_power = np.mean(
                    np.abs(
                        h_dd
                        - h_dd_est
                    ) ** 2
                )

                nmse_linear = (
                    error_power
                    / true_channel_power
                )

        # ----------------------------------------------------
        # Convert NMSE to dB
        # ----------------------------------------------------

        nmse_db = (
            10.0
            * np.log10(
                max(
                    nmse_linear,
                    np.finfo(float).eps,
                )
            )
        )

        nmse_values.append(
            nmse_db
        )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if len(
        nmse_values
    ) == 0:

        return float(
            "nan"
        )

    return float(
        np.mean(
            nmse_values
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    np.random.seed(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

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
        f"{FC_GHZ} GHz"
    )

    print(
        f"Subcarrier spacing : "
        f"{FREQ_SP_KHZ} kHz"
    )

    print(
        f"Channel paths      : "
        f"{NUM_PATHS}"
    )

    print(
        f"Maximum delay      : "
        f"{MAX_DELAY}"
    )

    print(
        f"Data SNR           : "
        f"{DATA_SNR_DB} dB"
    )

    print(
        f"Frames             : "
        f"{NUM_FRAMES}"
    )

    print()

    # --------------------------------------------------------
    # Doppler configuration
    # --------------------------------------------------------

    print(
        "Doppler configuration"
    )

    print(
        "-" * 70
    )

    for velocity in VELOCITIES_KMH:

        kmax = calculate_doppler(
            velocity
        )

        print(
            f"{velocity:>4} km/h"
            f" -> kmax = {kmax:.4f}"
            f" -> paths = {NUM_PATHS}"
        )

    print()

    # --------------------------------------------------------
    # Results dictionary
    # --------------------------------------------------------

    results = {}

    # --------------------------------------------------------
    # Velocity sweep
    # --------------------------------------------------------

    for velocity in VELOCITIES_KMH:

        kmax = calculate_doppler(
            velocity
        )

        print("=" * 70)

        print(
            f"Velocity = "
            f"{velocity} km/h"
        )

        print(
            f"kmax     = "
            f"{kmax:.4f}"
        )

        print(
            f"Paths    = "
            f"{NUM_PATHS}"
        )

        print("=" * 70)

        velocity_results = []

        # ----------------------------------------------------
        # Pilot SNR sweep
        # ----------------------------------------------------

        for pilot_snr in PILOT_SNR_DB:

            print(
                f"Pilot SNR = "
                f"{pilot_snr:>2} dB ... ",
                end="",
                flush=True,
            )

            nmse = run_experiment(
                velocity,
                pilot_snr,
            )

            velocity_results.append(
                nmse
            )

            print(
                f"NMSE = "
                f"{nmse:.4f} dB"
            )

        results[
            velocity
        ] = np.array(
            velocity_results
        )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print()

    print("=" * 70)
    print(
        "FINAL RESULTS"
    )
    print("=" * 70)

    header = (
        "Pilot SNR"
    )

    for velocity in VELOCITIES_KMH:

        header += (
            f" | "
            f"{velocity:>12} km/h"
        )

    print(header)

    print(
        "-" * len(header)
    )

    for idx, pilot_snr in enumerate(
        PILOT_SNR_DB
    ):

        row = (
            f"{pilot_snr:>9} dB"
        )

        for velocity in VELOCITIES_KMH:

            row += (
                f" | "
                f"{results[velocity][idx]:>14.4f}"
            )

        print(row)

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    output_dir = (
        Path(__file__).resolve().parent
    )

    # ========================================================
    # SAVE CSV
    # ========================================================

    results_file = (
        output_dir
        / "whole_joint_baseline_results.csv"
    )

    with open(
        results_file,
        "w",
        encoding="utf-8",
    ) as f:

        columns = [
            "pilot_snr_db"
        ]

        for velocity in VELOCITIES_KMH:

            columns.append(
                f"nmse_{velocity}_kmh_db"
            )

        f.write(
            ",".join(
                columns
            )
            + "\n"
        )

        for idx, pilot_snr in enumerate(
            PILOT_SNR_DB
        ):

            values = [
                str(
                    pilot_snr
                )
            ]

            for velocity in VELOCITIES_KMH:

                values.append(
                    str(
                        results[
                            velocity
                        ][idx]
                    )
                )

            f.write(
                ",".join(
                    values
                )
                + "\n"
            )

    # ========================================================
    # PLOT
    # ========================================================

    plt.figure(
        figsize=(8, 5)
    )

    for velocity in VELOCITIES_KMH:

        plt.plot(
            PILOT_SNR_DB,
            results[velocity],
            marker="o",
            label=(
                f"{velocity} km/h"
            ),
        )

    plt.xlabel(
        "Pilot SNR (dB)"
    )

    plt.ylabel(
        "Channel NMSE (dB)"
    )

    plt.title(
        "OTFS Whole-Joint Channel "
        "Estimation Baseline"
    )

    plt.grid(
        True
    )

    plt.legend()

    plt.tight_layout()

    plot_file = (
        output_dir
        / "whole_joint_baseline.png"
    )

    plt.savefig(
        plot_file,
        dpi=300,
    )

    plt.close()

    # ========================================================
    # COMPLETION
    # ========================================================

    print()

    print(
        "============================================================"
    )

    print(
        "BASELINE EXPERIMENT COMPLETED"
    )

    print(
        "============================================================"
    )

    print()

    print(
        f"CSV saved to:"
    )

    print(
        results_file
    )

    print()

    print(
        f"Plot saved to:"
    )

    print(
        plot_file
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()