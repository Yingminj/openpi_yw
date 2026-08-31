#!/usr/bin/env python3
from __future__ import annotations

import re
import statistics
from pathlib import Path


LOG_DIR = Path("/ssd/hhw/openpi-hzh/logs/zadai_pytorch_batch_benchmark")
INTERVAL_STEPS = 20


def main() -> None:
    print("batch\tstatus\tlast_step\tsteps_per_s\tsamples_per_s\tpeak_reserved_gb")
    for batch in (1, 2, 4, 8, 12, 16):
        path = LOG_DIR / f"batch_{batch}.log"
        text = path.read_text(errors="replace") if path.exists() else ""
        measurements = [
            (int(step), float(elapsed))
            for step, elapsed in re.findall(
                r"step=(\d+) loss=[^\r\n]*? time=([0-9.]+)s",
                text,
            )
            if int(step) > 0 and float(elapsed) > 0
        ]
        stable = measurements[-5:]
        rates = [INTERVAL_STEPS / elapsed for _, elapsed in stable]
        steps_per_second = statistics.median(rates) if rates else 0.0
        samples_per_second = steps_per_second * batch * 4
        peaks = [float(value) for value in re.findall(r"peak_reserved: ([0-9.]+)GB", text)]

        if re.search(r"out of memory|OutOfMemoryError", text, re.IGNORECASE):
            status = "OOM"
        elif measurements:
            status = "OK"
        else:
            status = "FAILED"

        last_step = measurements[-1][0] if measurements else 0
        peak_reserved = max(peaks) if peaks else 0.0
        print(
            f"{batch}\t{status}\t{last_step}\t{steps_per_second:.3f}\t"
            f"{samples_per_second:.2f}\t{peak_reserved:.2f}"
        )


if __name__ == "__main__":
    main()
