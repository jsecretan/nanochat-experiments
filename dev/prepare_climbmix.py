"""
Prepare the ClimbMix dataset (OptimalScale/ClimbMix) as sharded parquets split by cluster_id.

- Loads from HuggingFace (downloads full dataset by default; use --streaming to stream).
- Writes one directory per cluster_id: <output_dir>/<cluster_id>/shard_00000.parquet, ...
- Each parquet has a single "text" column (same as base_data for dataloader compatibility).
- Shard size and row group size match repackage_data_reference.py.

Use for two-phase training: point phase 1 to one cluster dir and phase 2 to another, e.g.:
  --phase2-data-dir /path/to/base_data/climbmix/1

Run from project root:
  python -m dev.prepare_climbmix --output-dir /path/to/base_data/climbmix
  python -m dev.prepare_climbmix --cluster-ids 0,1 --chars-per-shard 100000000
"""

import os
import argparse
import time
from collections import defaultdict
from typing import Dict, List, Optional

from datasets import load_dataset
import pyarrow as pa
import pyarrow.parquet as pq

from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# Defaults (aligned with repackage_data_reference.py)
DEFAULT_CHARS_PER_SHARD = 250_000_000
DEFAULT_ROW_GROUP_SIZE = 1024
CLIMBMIX_REPO = "OptimalScale/ClimbMix"
CLIMBMIX_SPLIT = "train"


def prepare_climbmix(
    output_dir: str,
    cluster_ids: Optional[List[int]] = None,
    chars_per_shard: int = DEFAULT_CHARS_PER_SHARD,
    row_group_size: int = DEFAULT_ROW_GROUP_SIZE,
    streaming: bool = False,
    max_docs: Optional[int] = None,
):
    """
    Load ClimbMix from HuggingFace, split by cluster_id, write sharded parquets per cluster.

    output_dir: base directory; each cluster gets output_dir/<cluster_id>/shard_XXXXX.parquet
    cluster_ids: if set, only these clusters are written; otherwise all clusters are written
    max_docs: if set, stop after this many docs (for testing)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Buffers per cluster: list of text strings and total char count
    buffers: Dict[int, List[str]] = defaultdict(list)
    buffer_chars: Dict[int, int] = defaultdict(int)
    shard_index: Dict[int, int] = defaultdict(int)

    def flush_cluster(cid: int, force: bool = False):
        """Write one or more shards from cluster cid's buffer if we have enough data."""
        nonlocal buffers, buffer_chars, shard_index
        docs = buffers[cid]
        if not docs:
            return
        # Take docs until we have at least chars_per_shard, then trim to multiple of row_group_size
        shard_char_count = 0
        take = 0
        for i, t in enumerate(docs):
            shard_char_count += len(t)
            take = i + 1
            if shard_char_count >= chars_per_shard:
                break
        if not force and (shard_char_count < chars_per_shard or take < row_group_size):
            return
        # Trim to multiple of row_group_size so we don't write a tiny last row group
        n = (take // row_group_size) * row_group_size
        if n == 0 and force:
            n = take  # last shard: write all, pad below
        if n == 0:
            return
        write_docs = docs[:n]
        buffers[cid] = docs[n:]
        buffer_chars[cid] = sum(len(x) for x in buffers[cid])
        # Last shard can have a partial row group (no padding with empty strings)
        _write_shard(cid, write_docs, shard_index[cid], output_dir, row_group_size)
        shard_index[cid] += 1

    def _write_shard(cid: int, docs: list[str], idx: int, out_dir: str, rg_size: int):
        cluster_dir = os.path.join(out_dir, str(cid))
        os.makedirs(cluster_dir, exist_ok=True)
        path = os.path.join(cluster_dir, f"shard_{idx:05d}.parquet")
        table = pa.Table.from_pydict({"text": docs})
        pq.write_table(
            table,
            path,
            row_group_size=rg_size,
            use_dictionary=False,
            compression="zstd",
            compression_level=3,
            write_statistics=False,
        )

    load_kwargs = {
        "path": CLIMBMIX_REPO,
        "split": CLIMBMIX_SPLIT,
        "streaming": streaming,
    }
    print(f"Loading ClimbMix (streaming={streaming})...")
    ds = load_dataset(**load_kwargs)

    total_docs = 0
    total_chars = 0
    t0 = time.time()
    last_log = 0

    for example in ds:
        if max_docs is not None and total_docs >= max_docs:
            break
        text = example.get("text")
        cluster_id = example.get("cluster_id")
        if text is None or cluster_id is None:
            continue
        if cluster_ids is not None and cluster_id not in cluster_ids:
            continue
        text = str(text) if not isinstance(text, str) else text
        buffers[cluster_id].append(text)
        buffer_chars[cluster_id] = buffer_chars.get(cluster_id, 0) + len(text)
        total_docs += 1
        total_chars += len(text)

        # Flush any cluster that has enough data
        for cid in list(buffers.keys()):
            if buffer_chars[cid] >= chars_per_shard and len(buffers[cid]) >= row_group_size:
                flush_cluster(cid)

        if total_docs - last_log >= 100_000:
            elapsed = time.time() - t0
            rate = total_docs / elapsed if elapsed > 0 else 0
            print(f"Processed {total_docs:,} docs, {total_chars / 1e9:.2f}B chars, {rate:.0f} docs/s")
            last_log = total_docs

    # Final flush for all clusters (repeat until each buffer is empty)
    for cid in list(buffers.keys()):
        while buffers[cid]:
            flush_cluster(cid, force=True)

    elapsed = time.time() - t0
    print(f"Done. Total docs: {total_docs:,}, total chars: {total_chars / 1e9:.2f}B, time: {elapsed / 60:.1f}m")
    print(f"Clusters written: {list(shard_index.keys())}")
    for cid in sorted(shard_index.keys()):
        print(f"  cluster {cid}: {shard_index[cid]} shards -> {os.path.join(output_dir, str(cid))}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare ClimbMix dataset as cluster-split sharded parquets for two-phase training."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Output base directory (default: <base_dir>/base_data/climbmix)",
    )
    parser.add_argument(
        "--cluster-ids",
        type=str,
        default=None,
        help="Comma-separated cluster IDs to include (default: all clusters)",
    )
    parser.add_argument(
        "--chars-per-shard",
        type=int,
        default=DEFAULT_CHARS_PER_SHARD,
        help=f"Target characters per shard (default: {DEFAULT_CHARS_PER_SHARD})",
    )
    parser.add_argument(
        "--row-group-size",
        type=int,
        default=DEFAULT_ROW_GROUP_SIZE,
        help=f"Parquet row group size (default: {DEFAULT_ROW_GROUP_SIZE})",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Stream from HuggingFace instead of downloading (faster start but can fail on network issues)",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Stop after this many documents (for testing)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        base_dir = get_base_dir()
        output_dir = os.path.join(base_dir, "base_data", "climbmix")

    cluster_ids = None
    if args.cluster_ids is not None:
        cluster_ids = [int(x.strip()) for x in args.cluster_ids.split(",")]
        print(f"Filtering to cluster IDs: {cluster_ids}")

    prepare_climbmix(
        output_dir=output_dir,
        cluster_ids=cluster_ids,
        chars_per_shard=args.chars_per_shard,
        row_group_size=args.row_group_size,
        streaming=args.streaming,
        max_docs=args.max_docs,
    )


if __name__ == "__main__":
    main()
