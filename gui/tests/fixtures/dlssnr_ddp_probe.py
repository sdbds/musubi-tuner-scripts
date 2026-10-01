"""CPU-only launcher probe; no models, datasets, or training outputs."""

import json
import os
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as distributed


def main():
    distributed.init_process_group(
        "gloo", init_method="env://?use_libuv=False" if os.name == "nt" else "env://", timeout=timedelta(seconds=45)
    )
    try:
        total = torch.tensor(float(distributed.get_rank() + 1))
        distributed.all_reduce(total)
        target = Path(sys.argv[1]) / f"rank-{distributed.get_rank()}.json"
        target.write_text(
            json.dumps({"rank": distributed.get_rank(), "world_size": distributed.get_world_size(), "sum": total.item()}),
            encoding="utf-8",
        )
    finally:
        distributed.destroy_process_group()


if __name__ == "__main__":
    main()
