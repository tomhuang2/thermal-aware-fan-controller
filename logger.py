"""
cpu_temp_logger.py
------------------
Logs CPU temperatures using LibreHardwareMonitor via PyLibreHardwareMonitor.

Setup:
    pip install PyLibreHardwareMonitor

    Run as Administrator — LHM needs elevated privileges to read CPU sensors.

Usage:
    python cpu_temp_logger.py                  # logs to cpu_temps.csv
    python cpu_temp_logger.py --list           # print available CPU sensors and exit
    python cpu_temp_logger.py --interval 5     # poll every 5 seconds (default: 2)
    python cpu_temp_logger.py --out my_log.csv
"""

import csv
import time
import argparse
from datetime import datetime

try:
    from PyLibreHardwareMonitor import Computer
except ImportError:
    raise SystemExit(
        "PyLibreHardwareMonitor not found.\n"
        "Install it with:  pip install PyLibreHardwareMonitor\n"
        "Then re-run this script as Administrator."
    )


# ── Sensor helpers ────────────────────────────────────────────────────────────

def get_cpu_temps(computer):
    """
    Returns a list of (label, value) tuples for all CPU temperature sensors.
    computer.cpu is auto-refreshed on each access.
    """
    results = []
    for sensor in computer.cpu:
        if sensor["SensorType"] == "Temperature":
            label = f"{sensor['HardwareName']} / {sensor['Name']}"
            results.append((label, sensor["Value"]))
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Log CPU temperatures via LibreHardwareMonitor")
    parser.add_argument("--list",     action="store_true",     help="Print available sensors and exit")
    parser.add_argument("--interval", type=float, default=2.0, help="Poll interval in seconds (default: 2)")
    parser.add_argument("--out",      type=str, default="cpu_temps.csv", help="Output CSV file (default: cpu_temps.csv)")
    args = parser.parse_args()

    computer = Computer()

    # First read to discover sensors and build column headers
    sensors = get_cpu_temps(computer)

    if not sensors:
        raise SystemExit(
            "No CPU temperature sensors found.\n"
            "Make sure you're running as Administrator."
        )

    if args.list:
        print(f"{'Sensor':<60} {'Value':>8}")
        print("-" * 70)
        for label, value in sensors:
            val_str = f"{value:.1f} °C" if value is not None else "N/A"
            print(f"{label:<60} {val_str:>8}")
        return

    columns = [label for label, _ in sensors]

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp"] + columns)

        print(f"Logging {len(sensors)} CPU temperature sensors to {args.out}")
        print("Ctrl+C to stop\n")

        try:
            while True:
                readings = get_cpu_temps(computer)
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                values = [v for _, v in readings]
                row = [ts] + [f"{v:.1f}" if v is not None else "" for v in values]
                writer.writerow(row)
                f.flush()

                # Console preview
                preview = "  |  ".join(
                    f"{label.split('/')[-1].strip()}: {v:.1f}°C"
                    for label, v in readings
                    if v is not None
                )
                print(f"[{ts}]  {preview}", flush=True)

                time.sleep(args.interval)

        except KeyboardInterrupt:
            print(f"\nStopped. Data saved to {args.out}")


if __name__ == "__main__":
    main()