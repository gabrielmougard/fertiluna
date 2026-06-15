"""Download + convert PaddleOCR PP-OCRv3 English recognition model to ONNX.

We only need the RECOGNITION model (not detection or classification): the
fertiluna_vision_cv pipeline already locates axis-tick labels and table cells
itself, so PaddleOCR is asked to recognize TEXT INSIDE A KNOWN BBOX, not to
find text. That cuts the browser bundle to a single ~10 MB ONNX.

Run:
    cd model
    uv pip install paddleocr paddle2onnx onnxruntime
    python -m scripts.build_paddleocr_onnx --out ../public/models

Emits:
    ../public/models/paddle-ocr-rec-v1.onnx          (~10 MB)
    ../public/models/paddle-ocr-dict-v1.txt          (en_dict.txt)
    ../public/models/paddle-ocr-manifest-v1.json     (versioned manifest)

The manifest matches the same shape your other models use (cycle / vision):
sha256, byte size, version. The browser-side `paddleOcrManifest.ts` reads it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path


PADDLE_REC_URL = (
    "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/"
    "en_PP-OCRv3_rec_infer.tar"
)
# en_dict.txt is shipped with PaddleOCR; mirror it from the repo so users
# don't need to clone the whole thing.
EN_DICT_URL = (
    "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/"
    "release/2.7/ppocr/utils/en_dict.txt"
)

VERSION = "v1"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    print(f"[paddleocr] downloading {url}")
    import ssl
    # macOS Python often ships without the system root store; fall back to
    # certifi when the default SSL context can't verify.
    try:
        urllib.request.urlretrieve(url, dest)
    except urllib.error.URLError:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url, context=ctx) as r, open(dest, "wb") as f:
            while True:
                chunk = r.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk)


def _extract_tar(tar_path: Path, into: Path) -> Path:
    print(f"[paddleocr] extracting {tar_path}")
    into.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tf:
        tf.extractall(into)
    # The tar unpacks to a single subdirectory; return it.
    children = [c for c in into.iterdir() if c.is_dir()]
    if not children:
        raise RuntimeError(f"No inner directory in {tar_path}")
    return children[0]


def _convert(paddle_model_dir: Path, out_onnx: Path) -> None:
    """Run paddle2onnx on the unpacked Paddle inference model.

    paddle2onnx 2.x ships its entrypoint as the `paddle2onnx` console script
    only (no `-m paddle2onnx` package main). We call the binary directly
    from the same venv as the running interpreter.
    """
    paddle2onnx_bin = Path(sys.executable).parent / "paddle2onnx"
    cmd = [
        str(paddle2onnx_bin),
        "--model_dir", str(paddle_model_dir),
        "--model_filename", "inference.pdmodel",
        "--params_filename", "inference.pdiparams",
        "--save_file", str(out_onnx),
        "--opset_version", "11",
        "--enable_onnx_checker", "True",
    ]
    print("[paddleocr] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True,
                   help="Output dir for the ONNX + dict + manifest "
                        "(typically ../public/models)")
    p.add_argument("--work", type=Path, default=Path("/tmp/paddleocr-build"),
                   help="Temp workdir for downloads + extraction")
    p.add_argument("--version", default=VERSION,
                   help="Manifest version tag (default: %(default)s)")
    args = p.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    work = args.work
    work.mkdir(parents=True, exist_ok=True)

    # 1. Download Paddle inference model archive
    tar = work / "en_PP-OCRv3_rec_infer.tar"
    if not tar.exists():
        _download(PADDLE_REC_URL, tar)
    inference_dir = _extract_tar(tar, work / "rec")

    # 2. Convert to ONNX
    onnx_path = out / f"paddle-ocr-rec-{args.version}.onnx"
    _convert(inference_dir, onnx_path)

    # 3. Download char dictionary
    dict_path = out / f"paddle-ocr-dict-{args.version}.txt"
    _download(EN_DICT_URL, dict_path)

    # 4. Emit manifest
    manifest = {
        "version": args.version,
        "task": "ocr-recognition",
        "model": {
            "name": "en_PP-OCRv3_rec",
            "input": {
                "shape": "(N, 3, 48, W)",
                "height": 48,
                "channel_order": "RGB",
                "normalize": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
                "scale": "1/255",
            },
            "output": {
                "shape": "(N, T, num_classes)",
                "decoder": "CTC",
            },
        },
        "files": {
            "model": {
                "path": onnx_path.name,
                "sha256": _sha256(onnx_path),
                "bytes": onnx_path.stat().st_size,
            },
            "dict": {
                "path": dict_path.name,
                "sha256": _sha256(dict_path),
                "bytes": dict_path.stat().st_size,
            },
        },
        "source": {
            "model_url": PADDLE_REC_URL,
            "dict_url": EN_DICT_URL,
            "opset": 11,
        },
    }
    manifest_path = out / f"paddle-ocr-manifest-{args.version}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[paddleocr] manifest -> {manifest_path}")
    print(f"[paddleocr] model    -> {onnx_path}  "
          f"({onnx_path.stat().st_size // 1024} KB)")
    print(f"[paddleocr] dict     -> {dict_path}  "
          f"({dict_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
