"""One-shot vast.ai runner for the FertiLuna chart-vision pipeline.

End-to-end:
  1. Sanity-check local git state (current HEAD must be pushed to origin).
  2. Pick a GPU offer matching --gpu / --min-cuda / --min-disk.
  3. Create a vast.ai instance with a PyTorch+CUDA Docker image.
  4. Wait for SSH, then run scripts/vast_remote.sh with SSH agent forwarding
     so the box can clone our private repo.
  5. rsync model/artifacts/ back from the box.
  6. Destroy the instance (unless --no-destroy).

Auth:
  - vast.ai:   `vastai set api-key <key>` once, locally.
  - GitHub:    your local ssh-agent must hold the key used for `git push`.
               We forward the agent into the box (`ssh -A`); no key is copied.

Quick usage (matches the "from-scratch" recipe in the README):

  python -m scripts.vast_run run \\
      --gpu RTX_4090 --train-n 200000 --val-n 20000 \\
      --width 3.0 --epochs 40 --batch-size 128 --version v1

Other commands:
  python -m scripts.vast_run search --gpu RTX_4090
  python -m scripts.vast_run destroy --instance 1234567
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "model"
REMOTE_SH = Path(__file__).parent / "vast_remote.sh"

DEFAULT_IMAGE = "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel"
DEFAULT_DISK_GB = 80  # 200k uint8 charts at 224x384x3 ~= 31 GB; + val + artifacts


# ---------------------------------------------------------------------------
# shelling out
# ---------------------------------------------------------------------------

def _which(cmd: str) -> str:
    p = shutil.which(cmd)
    if not p:
        sys.exit(
            f"error: `{cmd}` not found on PATH. "
            f"Install with: pip install --user vastai  "
            f"(then `vastai set api-key <key>`)"
        )
    return p


def _vastai_json(args: list[str]) -> Any:
    """Run `vastai ... --raw` and parse JSON."""
    vastai = _which("vastai")
    out = subprocess.run(
        [vastai, *args, "--raw"], check=True, text=True, capture_output=True
    )
    return json.loads(out.stdout)


# ---------------------------------------------------------------------------
# local git preflight
# ---------------------------------------------------------------------------

def _git(args: list[str]) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def preflight(ref: str | None, allow_dirty: bool) -> tuple[str, str]:
    """Return (repo_ssh_url, ref_to_check_out).

    Fail fast if HEAD is dirty or unpushed — the remote clones from origin and
    would silently train on stale code otherwise.
    """
    remote_url = _git(["remote", "get-url", "origin"])
    target_ref = ref or _git(["rev-parse", "HEAD"])

    dirty = _git(["status", "--porcelain"])
    if dirty:
        print(
            "warning: working tree has uncommitted changes:\n"
            f"{dirty}\n"
            "These will NOT be on the box. Commit + push first (or pass "
            "--allow-dirty to ignore).",
            file=sys.stderr,
        )
        if not allow_dirty:
            sys.exit(2)

    # Make sure target_ref is reachable from some remote branch on origin.
    subprocess.run(["git", "fetch", "--quiet", "origin"], cwd=REPO_ROOT, check=False)
    r = subprocess.run(
        ["git", "branch", "-r", "--contains", target_ref],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if r.returncode != 0 or not r.stdout.strip():
        sys.exit(
            f"error: ref {target_ref!r} is not on origin. "
            f"Push first (e.g. `git push origin HEAD`) or pass --ref <sha>."
        )
    return remote_url, target_ref


# ---------------------------------------------------------------------------
# vast.ai operations
# ---------------------------------------------------------------------------

def search_offers(gpu: str, num_gpus: int, min_cuda: str, min_disk: int,
                  min_ram: int, max_dph: float | None,
                  limit: int = 10) -> list[dict]:
    q = (
        f"gpu_name={gpu} "
        f"num_gpus={num_gpus} "
        f"cuda_vers>={min_cuda} "
        f"disk_space>={min_disk} "
        f"cpu_ram>={min_ram} "
        f"reliability>0.95 "
        f"inet_down>100 "
        f"rentable=true verified=true"
    )
    if max_dph is not None:
        q += f" dph_total<{max_dph}"
    offers = _vastai_json(["search", "offers", q, "-o", "dph_total+"])
    return offers[:limit]


def create_instance(offer_id: int, image: str, disk_gb: int) -> int:
    """Create + start an instance. Returns the new contract/instance id."""
    res = _vastai_json([
        "create", "instance", str(offer_id),
        "--image", image,
        "--disk", str(disk_gb),
        "--ssh",
    ])
    iid = res.get("new_contract") or res.get("instance_id") or res.get("id")
    if not iid:
        sys.exit(f"unexpected vastai create response: {res}")
    print(f"[vast] created instance {iid}")
    return int(iid)


def show_instance(iid: int) -> dict:
    return _vastai_json(["show", "instance", str(iid)])


def wait_for_running(iid: int, timeout_s: int = 900) -> dict:
    """Poll until instance is actually_running and SSH is reachable."""
    deadline = time.monotonic() + timeout_s
    last_status = None
    while time.monotonic() < deadline:
        info = show_instance(iid)
        status = info.get("actual_status") or info.get("status")
        if status != last_status:
            print(f"[vast] status: {status}")
            last_status = status
        if status == "running" and info.get("ssh_host") and info.get("ssh_port"):
            # status=running doesn't yet guarantee sshd is up — probe it.
            if _probe_ssh(info["ssh_host"], int(info["ssh_port"])):
                return info
        time.sleep(10)
    sys.exit(f"timed out waiting for instance {iid} to be SSH-ready")


def _probe_ssh(host: str, port: int) -> bool:
    """Single `ssh -o BatchMode=yes ... true` — succeeds when sshd accepts us."""
    r = subprocess.run(
        [
            "ssh",
            "-p", str(port),
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=8",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR",
            f"root@{host}",
            "true",
        ],
        capture_output=True, text=True,
    )
    return r.returncode == 0


def destroy_instance(iid: int) -> None:
    vastai = _which("vastai")
    subprocess.run([vastai, "destroy", "instance", str(iid)], check=False)
    print(f"[vast] destroyed instance {iid}")


# ---------------------------------------------------------------------------
# remote execution
# ---------------------------------------------------------------------------

def _ssh_base(host: str, port: int, *, agent_forward: bool = False) -> list[str]:
    args = [
        "ssh",
        "-p", str(port),
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        "-o", "ServerAliveInterval=30",
    ]
    if agent_forward:
        args.append("-A")
    args.append(f"root@{host}")
    return args


def run_remote(host: str, port: int, env: dict[str, str]) -> int:
    """Stream the remote pipeline. Returns the remote exit code."""
    # Upload the bash script then exec it with the env injected. Streaming
    # output via inherited stdio gives us live training logs locally.
    print(f"[remote] uploading {REMOTE_SH.name}")
    scp = subprocess.run(
        [
            "scp", "-P", str(port),
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR",
            str(REMOTE_SH),
            f"root@{host}:/tmp/vast_remote.sh",
        ],
        check=False,
    )
    if scp.returncode != 0:
        return scp.returncode

    env_kvs = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
    remote_cmd = f"chmod +x /tmp/vast_remote.sh && {env_kvs} bash /tmp/vast_remote.sh"
    print(f"[remote] exec: {remote_cmd}")
    return subprocess.run(
        _ssh_base(host, port, agent_forward=True) + [remote_cmd]
    ).returncode


def rsync_artifacts(host: str, port: int, also_data: bool) -> None:
    """Pull artifacts/ (and optionally data/) back into model/."""
    local_artifacts = MODEL_DIR / "artifacts"
    local_artifacts.mkdir(parents=True, exist_ok=True)
    ssh = (
        f"ssh -p {port} "
        f"-o StrictHostKeyChecking=accept-new "
        f"-o UserKnownHostsFile=/dev/null "
        f"-o LogLevel=ERROR"
    )
    print("[fetch] rsync artifacts/")
    subprocess.run(
        [
            "rsync", "-avz", "--progress",
            "-e", ssh,
            f"root@{host}:/workspace/repo/model/artifacts/",
            f"{local_artifacts}/",
        ],
        check=True,
    )
    if also_data:
        local_data = MODEL_DIR / "data"
        local_data.mkdir(parents=True, exist_ok=True)
        print("[fetch] rsync data/ (large — opt-in via --fetch-data)")
        subprocess.run(
            [
                "rsync", "-avz", "--progress",
                "-e", ssh,
                f"root@{host}:/workspace/repo/model/data/",
                f"{local_data}/",
            ],
            check=True,
        )


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------

def cmd_search(args: argparse.Namespace) -> int:
    offers = search_offers(
        gpu=args.gpu, num_gpus=args.num_gpus,
        min_cuda=args.min_cuda, min_disk=args.min_disk,
        min_ram=args.min_ram, max_dph=args.max_dph,
        limit=args.limit,
    )
    if not offers:
        print("no offers match.", file=sys.stderr)
        return 1
    print(f"{'id':>10}  {'gpu':<22}  {'n':>2}  {'cpu':>4}  {'ram':>5}  {'disk':>5}  {'$/h':>7}")
    for o in offers:
        print(
            f"{o['id']:>10}  {o.get('gpu_name', '?'):<22}  "
            f"{o.get('num_gpus', 1):>2}  "
            f"{int(o.get('cpu_cores_effective', o.get('cpu_cores', 0))):>4}  "
            f"{int(o.get('cpu_ram', 0)) // 1024:>4}G  "
            f"{int(o.get('disk_space', 0)):>4}G  "
            f"{o.get('dph_total', 0):>6.3f}"
        )
    return 0


def cmd_destroy(args: argparse.Namespace) -> int:
    destroy_instance(args.instance)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    repo_url, ref = preflight(args.ref, args.allow_dirty)
    print(f"[git] repo={repo_url}  ref={ref}")

    if args.instance:
        info = show_instance(args.instance)
        if info.get("actual_status") != "running":
            info = wait_for_running(args.instance)
        iid = args.instance
        we_created = False
    else:
        offers = search_offers(
            gpu=args.gpu, num_gpus=args.num_gpus,
            min_cuda=args.min_cuda, min_disk=args.min_disk,
            min_ram=args.min_ram, max_dph=args.max_dph,
            limit=5,
        )
        if not offers:
            sys.exit("no GPU offers match. Try `vast_run search` to inspect.")
        offer = offers[0]
        print(
            f"[vast] picked offer {offer['id']}: "
            f"{offer.get('gpu_name')} x{offer.get('num_gpus', 1)} "
            f"@ ${offer.get('dph_total', 0):.3f}/h "
            f"(cpu={offer.get('cpu_cores_effective', offer.get('cpu_cores', '?'))})"
        )
        iid = create_instance(offer["id"], args.image, args.disk)
        we_created = True
        info = wait_for_running(iid)

    host = info["ssh_host"]
    port = int(info["ssh_port"])
    print(f"[ssh] root@{host}:{port}")

    env = {
        "REPO_URL": repo_url,
        "REPO_REF": ref,
        "TRAIN_N": str(args.train_n),
        "VAL_N": str(args.val_n),
        "TRAIN_SEED": str(args.train_seed),
        "VAL_SEED": str(args.val_seed),
        "STYLE": args.style,
        "WIDTH": str(args.width),
        "EPOCHS": str(args.epochs),
        "BATCH_SIZE": str(args.batch_size),
        "VERSION": args.version,
    }
    if args.workers is not None:
        env["WORKERS"] = str(args.workers)

    exit_code = run_remote(host, port, env)

    if exit_code == 0:
        try:
            rsync_artifacts(host, port, also_data=args.fetch_data)
        except subprocess.CalledProcessError as e:
            print(f"[fetch] rsync failed: {e}", file=sys.stderr)
            exit_code = exit_code or 3
    else:
        print(f"[remote] pipeline failed with exit {exit_code}", file=sys.stderr)

    if we_created and not args.no_destroy:
        if exit_code != 0 and args.keep_on_error:
            print(
                f"[vast] keeping instance {iid} alive for debugging "
                f"(remote failed). Destroy with: vast_run destroy --instance {iid}",
                file=sys.stderr,
            )
        else:
            destroy_instance(iid)
    elif we_created:
        print(
            f"[vast] leaving instance {iid} running (--no-destroy). "
            f"Remember to destroy it when done."
        )

    return exit_code


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------

def _add_search_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--gpu", default="RTX_4090",
                   help="Exact vast.ai gpu_name token (e.g. RTX_4090, RTX_3090, A100_PCIE).")
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--min-cuda", default="12.1")
    p.add_argument("--min-disk", type=int, default=DEFAULT_DISK_GB,
                   help="Minimum host disk available (GB).")
    p.add_argument("--min-ram", type=int, default=32 * 1024,
                   help="Minimum host RAM in MB (vast.ai reports MB).")
    p.add_argument("--max-dph", type=float, default=None,
                   help="Max $/hour (total). Omit for any price.")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="vast_run",
        description="Provision a vast.ai GPU box, run the FertiLuna chart-vision "
                    "dataset + training pipeline, pull artifacts back, destroy.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # search
    sp = sub.add_parser("search", help="List matching GPU offers.")
    _add_search_flags(sp)
    sp.add_argument("--limit", type=int, default=10)
    sp.set_defaults(func=cmd_search)

    # destroy
    sp = sub.add_parser("destroy", help="Destroy an instance by id.")
    sp.add_argument("--instance", type=int, required=True)
    sp.set_defaults(func=cmd_destroy)

    # run
    sp = sub.add_parser("run", help="End-to-end pipeline run on a fresh box.")
    _add_search_flags(sp)
    sp.add_argument("--instance", type=int, default=None,
                    help="Reuse an existing instance instead of creating one.")
    sp.add_argument("--image", default=DEFAULT_IMAGE)
    sp.add_argument("--disk", type=int, default=DEFAULT_DISK_GB)
    sp.add_argument("--ref", default=None,
                    help="Git ref to check out on the box (default: current HEAD SHA).")
    sp.add_argument("--allow-dirty", action="store_true",
                    help="Don't fail if the local tree has uncommitted changes.")
    # dataset
    sp.add_argument("--train-n", type=int, default=200_000)
    sp.add_argument("--val-n", type=int, default=20_000)
    sp.add_argument("--train-seed", type=int, default=1)
    sp.add_argument("--val-seed", type=int, default=99)
    sp.add_argument("--style", choices=["generic", "premom", "blend"], default="blend")
    # training
    sp.add_argument("--width", type=float, default=3.0)
    sp.add_argument("--epochs", type=int, default=40)
    sp.add_argument("--batch-size", type=int, default=128)
    sp.add_argument("--version", default="v1")
    sp.add_argument("--workers", type=int, default=None,
                    help="Override remote --workers (default: nproc on the box).")
    # fetch / lifecycle
    sp.add_argument("--fetch-data", action="store_true",
                    help="Also rsync data/ (large — multi-GB .npy memmaps).")
    sp.add_argument("--no-destroy", action="store_true",
                    help="Leave the box running after success.")
    sp.add_argument("--keep-on-error", action="store_true",
                    help="Leave the box running if the remote pipeline fails.")
    sp.set_defaults(func=cmd_run)

    return ap


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
