"""Wan2.1-T2V-1.3B: clean->noise inversion, x0 estimate decoded at each sigma. -i: a video or a folder."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent          # <project>/PhysTrace
ROOT = HERE.parent                              # <project>/
sys.path.insert(0, str(ROOT))
from demo.core.gt import GT_CSV  # noqa: E402
from demo.core.pairs import load_pairs, main_videos_dir  # noqa: E402
DM = Path(os.environ.get("DIFFUSION_MODELS", ROOT / "DiffusionModels"))


def _pick(env: str, *candidates) -> str:
    if os.environ.get(env):
        return os.environ[env]
    for c in candidates:
        if Path(c).exists():
            return str(c)
    return str(candidates[-1])


WAN_ROOT = _pick("WAN_ROOT", DM / "Wan2.1")                                   # official repo
CKPT_DIR = _pick("WAN_CKPT", DM / "weights" / "wan" / "Wan2.1-T2V-1.3B",
                 Path(WAN_ROOT) / "Wan2.1-T2V-1.3B")
sys.path.insert(0, WAN_ROOT)


CHUNK = 81          # frames per forward; the Wan VAE needs 4k+1
K_STEPS = 100       # inversion steps (sigma resolution = 1/K)


# ---------------------------------------------------------------- video io
def read_video(path: str, size: int, stride: int, fps_out: float = 0.0) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames, i, j = [], 0, 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        keep = 0
        if fps_out > 0:
            while round(j * src_fps / fps_out) == i:      # a frame may repeat when upsampling
                keep += 1
                j += 1
        elif i % stride == 0:
            keep = 1
        if keep:
            rgb = np.asarray(Image.fromarray(bgr[:, :, ::-1]).resize((size, size), Image.BICUBIC))
            frames.extend([rgb] * keep)
        i += 1
    cap.release()
    if not frames:
        raise IOError(f"no frames decoded from {path}")
    return np.stack(frames), (fps_out if fps_out > 0 else src_fps / stride)


def write_frames(dir_path: Path, video: np.ndarray, every: int, workers: int = 8) -> int:
    dir_path.mkdir(parents=True, exist_ok=True)
    idx = list(range(0, len(video), every))
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda i: Image.fromarray(video[i]).save(dir_path / f"f{i:05d}.png", compress_level=3), idx))
    return len(idx)


def write_video(path: Path, video: np.ndarray, fps: float) -> None:
    import imageio
    part = path.with_suffix(".part.mp4")
    imageio.mimwrite(str(part), video, fps=fps, quality=8, macro_block_size=1)
    os.replace(part, path)


# ---------------------------------------------------------------- chunking with cross-fade
def chunk_plan(n_frames: int, chunk: int, overlap: int) -> list[tuple[int, int]]:
    if not 0 <= overlap < chunk:
        raise ValueError(f"overlap must be in [0, {chunk})")
    step, plan, s = chunk - overlap, [], 0
    while s + chunk <= n_frames:
        plan.append((s, s + chunk))
        s += step
    if not plan or plan[-1][1] < n_frames:
        plan.append((max(n_frames - chunk, 0), n_frames))
    return plan


def blend_weights(plan: list[tuple[int, int]], overlap: int) -> list[np.ndarray]:
    out = []
    for i, (a, b) in enumerate(plan):
        w = np.ones(b - a, dtype=np.float32)
        if overlap > 0:
            ramp = (np.arange(overlap, dtype=np.float32) + 1) / (overlap + 1)
            if i > 0:
                w[:overlap] = np.minimum(w[:overlap], ramp)
            if i < len(plan) - 1:
                w[-overlap:] = np.minimum(w[-overlap:], ramp[::-1])
        out.append(w)
    return out


class Assembler:

    def __init__(self, n_frames: int, size: int, tags: list[str], plan, weights):
        self.acc = {t: np.zeros((n_frames, size, size, 3), np.float32) for t in tags}
        self.wsum = {t: np.zeros(n_frames, np.float32) for t in tags}
        self.plan, self.weights = plan, weights

    def add(self, tag: str, ci: int, frames: np.ndarray) -> None:
        (a, b), w = self.plan[ci], self.weights[ci]
        n = b - a
        self.acc[tag][a:b] += frames[:n].astype(np.float32) * w[:n, None, None, None]
        self.wsum[tag][a:b] += w[:n]

    def video(self, tag: str) -> np.ndarray:
        assert (self.wsum[tag] > 0).all(), f"{tag}: {(self.wsum[tag] == 0).sum()} frames never decoded"
        return np.clip(self.acc[tag] / self.wsum[tag][:, None, None, None] + 0.5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- joint (MultiDiffusion) mode
def latent_len(n_pixel: int, mult: int = 4) -> int:
    return 1 + (n_pixel - 1) // mult


def pad_to(video: np.ndarray, n_min: int, mult: int = 4) -> tuple[np.ndarray, int]:
    t = max(len(video), n_min)
    t += (-(t - 1)) % mult
    n_pad = t - len(video)
    if n_pad:
        video = np.concatenate([video, np.repeat(video[-1:], n_pad, axis=0)])
    return video, n_pad


# ---------------------------------------------------------------- inputs / outputs
def list_videos(src: Path, names: str, shard: int, nshards: int, gt_csv: Path) -> list[Path]:
    if not src.is_dir():
        videos = [src]
    elif (src.parent / "metadata.csv").exists():
        videos = [src / f"{p[k]}.mp4" for p in load_pairs(src, gt_csv=gt_csv) for k in ("good_name", "bad_name")]
    else:
        videos = sorted(src.glob("*.mp4"))
    if names:
        want = {n.strip() for n in names.split(",") if n.strip()}
        videos = [v for v in videos if v.stem in want]
        if want - {v.stem for v in videos}:
            raise SystemExit(f"not in {src}: {sorted(want - {v.stem for v in videos})}")
    return videos[shard::nshards]


def prepare_layout(dirs: dict[str, Path], meta_csv: Path | None) -> None:
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
        if meta_csv and not (d.parent / "metadata.csv").exists():
            shutil.copyfile(meta_csv, d.parent / "metadata.csv")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--video", required=True, help="input video file, a folder of .mp4 files, or an IntPhys2 root holding Main/Videos "
                                                     "(as score/run_test.py --intphys2-dir)")
    ap.add_argument("-o", "--out", default="", help="output directory (default: ./<video or folder name>_wan)")
    ap.add_argument("--names", default="", help="folder input: only these video names (comma-separated, no .mp4)")
    ap.add_argument("--shard", type=int, default=0, help="folder input: take videos shard, shard+nshards, ...")
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--force", action="store_true", help="redo videos whose outputs already exist")
    ap.add_argument("--gt-csv", type=Path, default=GT_CSV,
                    help="IntPhys2 input: generate the pairs this GT CSV lists (default: %(default)s)")
    ap.add_argument("--sigmas", default="0.5,0.6,0.8,0.9,0.99", help="comma-separated sigmas (= flow time) to decode")
    ap.add_argument("--every", type=int, default=0, help="instead of --sigmas, decode every N-th of the K steps")
    ap.add_argument("--size", type=int, default=256, help="square resolution fed to the model")
    ap.add_argument("--stride", type=int, default=8, help="use every N-th source frame (8 = 7.5 fps from 60 fps)")
    ap.add_argument("--fps", type=float, default=0.0, help="resample to this frame rate instead of --stride "
                                                           "(same duration; 0 = off)")
    ap.add_argument("--chunk", type=int, default=CHUNK, help="frames per forward pass (must be 4k+1)")
    ap.add_argument("--chunked", action="store_true",
                    help="invert each chunk independently and cross-fade (default: joint/MultiDiffusion)")
    ap.add_argument("--latent_overlap", type=int, default=0,
                    help="joint mode: latent frames shared by consecutive windows (1 = 4 pixel frames)")
    ap.add_argument("--overlap", type=int, default=10, help="--chunked: pixel frames shared by consecutive chunks")
    ap.add_argument("--steps", type=int, default=K_STEPS, help="inversion steps K")
    ap.add_argument("--prompt", default="", help="text condition (default: empty, as in the IHoP runs)")
    ap.add_argument("--save_frames", type=int, default=0, metavar="N",
                    help="also save every N-th frame as PNG (0 = mp4 only; 1 = every frame)")
    ap.add_argument("--no_video", action="store_true", help="skip the per-sigma mp4")
    ap.add_argument("--device_id", type=int, default=0)
    args = ap.parse_args()

    if (args.chunk - 1) % 4:
        raise SystemExit(f"--chunk must be 4k+1 for the Wan VAE, got {args.chunk}")
    K = args.steps
    steps = {k: k / K for k in range(0, K, args.every)} if args.every else \
            {round(s * K): s for s in (float(x) for x in args.sigmas.split(","))}
    tags = {k: f"sigma_{s:.2f}" for k, s in steps.items()}
    src = Path(args.video).expanduser().resolve()
    if (src / "Main").is_dir():
        src = main_videos_dir(src)
    if not src.exists():
        raise SystemExit(f"no such file or folder: {src}")
    folder = src.is_dir()
    out_root = Path(args.out or f"{src.stem}_wan").resolve()      # keep writes anchored here even if the CWD changes later
    meta_csv = src.parent / "metadata.csv" if folder and (src.parent / "metadata.csv").exists() else None
    all_tags = ["sigma_0.00_INPUT"] + list(tags.values())
    vid_dir = {t: out_root / t / "Main" / "Videos" if meta_csv else out_root / t for t in all_tags}
    output_of = lambda v, t: out_root / t / "frames" / v.stem if args.no_video else vid_dir[t] / f"{v.stem}.mp4"
    is_done = lambda v: all(output_of(v, t).exists() for t in all_tags)

    videos = list_videos(src, args.names, args.shard, args.nshards, args.gt_csv.expanduser().resolve())
    if not videos:
        raise SystemExit(f"no videos under {src}: pass an IntPhys2 root holding Main/Videos, a folder of .mp4 files, or a file")
    todo = [v for v in videos if args.force or not is_done(v)]
    print(f"{len(videos)} video(s), {len(videos) - len(todo)} already done -> {out_root}", flush=True)
    if not todo:
        return
    prepare_layout(vid_dir, meta_csv)

    from wan.configs import WAN_CONFIGS
    from wan.text2video import WanT2V
    print(f"repo {WAN_ROOT}\nckpt {CKPT_DIR}", flush=True)

    import wan.modules.attention as A, wan.modules.model as M, wan.modules.clip as C
    if not (A.FLASH_ATTN_2_AVAILABLE or A.FLASH_ATTN_3_AVAILABLE):
        def _sdpa(*a, version=None, **k):
            return A.attention(*a, fa_version=version, **k)
        M.flash_attention = C.flash_attention = _sdpa
        print("flash_attn not installed -> torch SDPA fallback", flush=True)

    device = torch.device(f"cuda:{args.device_id}")
    torch.cuda.set_device(device)
    config = WAN_CONFIGS["t2v-1.3B"]
    print("loading Wan2.1-T2V-1.3B (T5 + VAE + DiT)...", flush=True)
    pipe = WanT2V(config=config, checkpoint_dir=CKPT_DIR, device_id=args.device_id, t5_cpu=False)
    with torch.no_grad():
        pipe.text_encoder.model.to(device)
        context = pipe.text_encoder([args.prompt], device)
        pipe.text_encoder.model.cpu()
        torch.cuda.empty_cache()

    dt = 1.0 / K
    ph, pw = config.patch_size[1], config.patch_size[2]

    def to_uint8(x):
        return ((x.clamp(-1, 1) + 1) / 2 * 255).round().byte().permute(1, 2, 3, 0).cpu().numpy()

    def to_model(frames):
        v = torch.from_numpy(np.ascontiguousarray(frames)).permute(3, 0, 1, 2).float()
        return v.div_(255.0).sub_(0.5).div_(0.5).to(device)                       # [C,T,H,W] in [-1,1]

    def invert(video_path: Path) -> None:
        name = video_path.stem
        t0 = time.time()
        video_np, fps = read_video(str(video_path), args.size, args.stride, args.fps)
        keep = len(video_np)
        win_lat = latent_len(args.chunk)
        if args.chunked:
            video_np, n_pad = pad_to(video_np, args.chunk, 1)
            plan = chunk_plan(len(video_np), args.chunk, args.overlap)
            weights = blend_weights(plan, args.overlap)
            shape = f"{len(plan)} chunks of {args.chunk}, overlap {args.overlap}"
        else:
            video_np, n_pad = pad_to(video_np, args.chunk, 4)
            plan = chunk_plan(latent_len(len(video_np)), win_lat, args.latent_overlap)
            weights = blend_weights(plan, args.latent_overlap)
            shape = (f"joint: {latent_len(len(video_np))} latent frames, {len(plan)} windows of {win_lat}, "
                     f"overlap {args.latent_overlap}")
        print(f"{video_path.name}: {keep} frames @ {fps:.2f} fps, {args.size}px"
              f"{f' (+{n_pad} pad)' if n_pad else ''} -> {shape}; decoding sigmas {sorted(steps.values())}", flush=True)

        def emit(tag: str, video: np.ndarray) -> None:
            video = video[:keep]
            n_png = write_frames(out_root / tag / "frames" / name, video, args.save_frames) if args.save_frames else 0
            if not args.no_video:
                write_video(vid_dir[tag] / f"{name}.mp4", video, fps)
            print(f"  {tag}: {len(video)} frames, {n_png} png", flush=True)

        emit("sigma_0.00_INPUT", video_np)

        if args.chunked:
            asm = Assembler(len(video_np), args.size, list(tags.values()), plan, weights)
            for ci, (a, b) in enumerate(plan):
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=config.param_dtype):
                    z = pipe.vae.encode([to_model(video_np[a:b])])[0]
                    seq_len = z.shape[1] * (z.shape[2] // ph) * (z.shape[3] // pw)
                    x = z.clone()
                    for k in range(K):
                        sigma = k * dt
                        t = torch.tensor([sigma * config.num_train_timesteps], device=device, dtype=torch.float32)
                        v = pipe.model([x], t=t, context=context, seq_len=seq_len)[0]  # velocity
                        if k in steps:
                            asm.add(tags[k], ci, to_uint8(pipe.vae.decode([(x - sigma * v).float()])[0]))
                        x = x + dt * v                                                 # one step clean -> noise
                print(f"  chunk {ci+1}/{len(plan)} [{a}:{b}) at {time.time()-t0:.0f}s", flush=True)
            for k in sorted(tags):
                emit(tags[k], asm.video(tags[k]))
        else:
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=config.param_dtype):
                X = pipe.vae.encode([to_model(video_np)])[0].float()                   # [C,T_lat,H,W]
            assert X.shape[1] == latent_len(len(video_np)), (X.shape, len(video_np))
            w_lat = [torch.from_numpy(w).to(device) for w in weights]
            seq_len = win_lat * (X.shape[2] // ph) * (X.shape[3] // pw)
            for k in range(K):
                sigma = k * dt
                t = torch.tensor([sigma * config.num_train_timesteps], device=device, dtype=torch.float32)
                V = torch.zeros_like(X)
                wsum = torch.zeros(X.shape[1], device=device)
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=config.param_dtype):
                    for (a, b), w in zip(plan, w_lat):
                        v = pipe.model([X[:, a:b].contiguous()], t=t, context=context, seq_len=seq_len)[0]
                        V[:, a:b] += v.float() * w[None, :, None, None]
                        wsum[a:b] += w
                assert (wsum > 0).all(), "some latent frame is covered by no window"
                V /= wsum[None, :, None, None]
                if k in steps:
                    with torch.no_grad(), torch.amp.autocast("cuda", dtype=config.param_dtype):
                        emit(tags[k], to_uint8(pipe.vae.decode([X - sigma * V])[0]))
                    print(f"  sigma {sigma:.2f} decoded at {time.time()-t0:.0f}s", flush=True)
                X = X + dt * V                                                         # one step clean -> noise

        print(f"done in {time.time()-t0:.0f}s", flush=True)

    (out_root / "meta.json").write_text(json.dumps({
        "model": "Wan2.1-T2V-1.3B", "input": str(src), "prompt": args.prompt, "size": args.size, "stride": args.stride,
        "fps_arg": args.fps, "mode": "chunked" if args.chunked else "joint", "chunk": args.chunk,
        "overlap": args.overlap if args.chunked else args.latent_overlap,
        "k_steps": K, "sigmas": {tags[k]: s for k, s in steps.items()},
    }, indent=2))
    for i, v in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {v}", flush=True)
        invert(v)


if __name__ == "__main__":
    main()
