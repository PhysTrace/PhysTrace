#!/usr/bin/env python3
"""CogVideoX-5B: responsibility-guided P/I generation (Method A) with the model's OFFICIAL sampler.
-i: an IntPhys2 root (Main/Videos) or a folder of .mp4 files; pairs (P / I clips, violation onset) come from --gt-csv.

Sampler: CogVideoXDPMScheduler, 50 steps, trailing spacing, zero-SNR v-prediction (official 5B recipe).
Windows of 49 frames; from the onset on they are regenerated toward the side (P / I)
the model's own x0 agrees with (r_P, --lambdas); see generate_pair.
Outputs: <out>/<cid>/resp_l<lambda>_s<sigma'>_p<prefix>_seed<n>.mp4 + meta.json (r_P trace, choice);
existing files are skipped. GPU: CUDA_VISIBLE_DEVICES.

  CUDA_VISIBLE_DEVICES=0 python3 PhysTrace/gen_cogvideox.py \
      -i /Path/To/IntPhys2 \
      -o results/gen_cogvideox \
      --sigmas 0.5,0.8,1.0 \
      --lambdas 0.5,1.0 \
      --seeds 2 \
      --prefix I
  (--names 236-1,sol-1-2 for a subset of the pairs; --size 256 by default)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent          # <project>/PhysTrace
ROOT = HERE.parent                              # <project>/
sys.path.insert(0, str(ROOT))
from demo.core.gt import GT_CSV  # noqa: E402
from demo.core.pairs import main_videos_dir  # noqa: E402
DM = Path(os.environ.get("DIFFUSION_MODELS", ROOT / "DiffusionModels"))


def _pick(env: str, *candidates) -> str:
    if os.environ.get(env):
        return os.environ[env]
    for c in candidates:
        if Path(c).exists():
            return str(c)
    return str(candidates[-1])


CKPT_DIR = _pick("COGVIDEOX_CKPT", DM / "weights" / "CogVideoX-5b")                     # hf download zai-org/CogVideoX-5b
PROMPT_CACHE = Path(_pick("PROMPT_CACHE", DM / "prompt_cache"))                     # <name>_empty.pt: the empty-prompt embedding, built once

MODEL_KEY, MODEL_NAME = "cogvideox", "CogVideoX-5B"
CHUNK = 49                 # frames per model window
SUB_BATCH = 4               # windows per model forward (--sub-batch)
K = 100                           # flow grid used to map sigma' to a DDPM timestep of equal SNR
PRE = {"continuity": "", "immutability": "imm-", "permanence": "perm-", "solidity": "sol-"}


# ---------------------------------------------------------------- video io (as PhysTrace/prob_wan.py)
def read_video(path: str, size: int) -> tuple[np.ndarray, float]:
    """[T,H,W,3] uint8 RGB, every frame resized to size x size, and the source fps."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(np.asarray(Image.fromarray(bgr[:, :, ::-1]).resize((size, size), Image.BICUBIC)))
    cap.release()
    if not frames:
        raise IOError(f"no frames decoded from {path}")
    return np.stack(frames), src_fps


def write_video(path: Path, video: np.ndarray, fps: float) -> None:
    import imageio
    part = path.with_suffix(".part.mp4")
    imageio.mimwrite(str(part), video, fps=fps, quality=8, macro_block_size=1)
    os.replace(part, path)


def latent_len(n_pixel: int, mult: int = 4) -> int:
    return 1 + (n_pixel - 1) // mult


# ---------------------------------------------------------------- pairs (from the GT csv)
def cid_of(row: dict) -> str:
    return f"{PRE[row['attribute']]}{int(row['scene'])}-{int(row['variant'])}"


def load_pairs(video_dir: Path, gt_csv: Path, names: str) -> list[dict]:
    """The pairs of `gt_csv` whose two clips are under `video_dir`: {cid, attribute, category, onset, end_frame, good, bad}."""
    available = {p.stem for p in video_dir.glob("*.mp4")}
    with open(gt_csv, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    pairs = [dict(cid=cid_of(r), attribute=r["attribute"], category=r["category"], onset=int(r["onset_frame"]), end_frame=int(r["end_frame"]),
                  good=str(video_dir / f"{r['possible_video']}.mp4"), bad=str(video_dir / f"{r['impossible_video']}.mp4"))
             for r in rows if r["onset_frame"] and r["possible_video"] in available and r["impossible_video"] in available]
    if missing := len(rows) - len(pairs):
        print(f"[pairs] {missing} of the {len(rows)} pairs in {Path(gt_csv).name} have no complete pair under {video_dir}", file=sys.stderr)
    if names:
        want = {n.strip() for n in names.split(",") if n.strip()}
        pairs = [p for p in pairs if p["cid"] in want or Path(p["bad"]).stem in want]
        if want - {p["cid"] for p in pairs} - {Path(p["bad"]).stem for p in pairs}:
            raise SystemExit(f"not under {video_dir}: {sorted(want - {p['cid'] for p in pairs})}")
    return pairs


# ---------------------------------------------------------------- windows, masks, assembly
def window_plan(n: int, chunk: int) -> list[tuple[int, int]]:
    """Non-overlapping [(a, b)] of length `chunk` covering [0, n); the last one may run past n (padding)."""
    return [(a, a + chunk) for a in range(0, n, chunk)]


def take_window(video: np.ndarray, a: int, b: int) -> np.ndarray:
    w = video[a:b]
    if len(w) < b - a:                                       # last window: repeat the final frame
        w = np.concatenate([w, np.repeat(w[-1:], b - a - len(w), axis=0)])
    return w


def latent_frames(l: int, n_lat: int, chunk: int) -> tuple[int, int]:
    """Pixel frames [lo, hi] (inclusive, window-local) covered by latent frame l (4x causal VAEs: 0 | 4l-3..4l)."""
    if n_lat == chunk:
        return l, l
    return (0, 0) if l == 0 else (4 * l - 3, 4 * l)


def mix_mask(plan, onset: int, n_lat: int, chunk: int, t_axis: int, like: torch.Tensor) -> torch.Tensor:
    """Float mask shaped like `like`: latent frames from the one holding the onset to the window end."""
    m = torch.zeros(len(plan), n_lat, dtype=torch.bool)
    for w, (a, b) in enumerate(plan):
        local = onset - a
        if local < b - a:
            first = 0 if local <= 0 else next(l for l in range(n_lat) if latent_frames(l, n_lat, chunk)[1] >= local)
            m[w, first:] = True
    shape = [1] * like.dim(); shape[0] = len(plan); shape[t_axis] = n_lat
    return m.view(shape).to(like.device).expand_as(like).float()


def assemble(vc: np.ndarray, plan, todo, parts, onset: int, n: int) -> np.ndarray:
    """Full-length video: the context clip with the generated windows pasted in; frames before the onset stay context."""
    full = vc.copy()
    for w, d in zip(todo, parts):
        a, b = plan[w]; full[a:min(b, n)] = d[:min(b, n) - a]
    full[:onset] = vc[:onset]
    return full


def batched(fn, x: torch.Tensor, sub: int) -> torch.Tensor:
    return torch.cat([fn(x[i:i + sub]) for i in range(0, x.shape[0], sub)], 0)


# ---------------------------------------------------------------- back end (inlined from full60/multi_full2.py)
BF = torch.bfloat16


def cached_embeds(name: str, build):
    """The empty-prompt embedding of `name`: built once (text encoder loaded, then dropped) and kept in PROMPT_CACHE."""
    f = PROMPT_CACHE / f"{name}_empty.pt"
    if f.exists():
        return torch.load(f, map_location="cpu")
    print(f"encoding the empty prompt for {name} (one-off, cached in {PROMPT_CACHE})...", flush=True)
    emb = build(); PROMPT_CACHE.mkdir(parents=True, exist_ok=True); torch.save(emb, f); torch.cuda.empty_cache(); return emb


def to_model(wins_uint8: np.ndarray, device, dtype=BF) -> torch.Tensor:
    """[B,T,H,W,3] uint8 -> [B,3,T,H,W] in [-1, 1]."""
    return torch.from_numpy(np.ascontiguousarray(wins_uint8)).permute(0, 4, 1, 2, 3).to(device, dtype).div_(127.5).sub_(1.0)


def to_uint8(x: torch.Tensor) -> list[np.ndarray]:
    """[B,3,T,H,W] in [-1, 1] -> list of [T,H,W,3] uint8."""
    return [((u.float().clamp(-1, 1) + 1) / 2 * 255).round().byte().permute(1, 2, 3, 0).cpu().numpy() for u in x]


def snr_timesteps(abar: torch.Tensor) -> list[int]:
    """t_of_k: flow time k/K -> the DDPM timestep of equal SNR (abar/(1-abar) = ((1-s)/s)^2); 0 at s = 0, the last at s = 1."""
    log_snr = torch.log(abar / (1 - abar).clamp_min(1e-12))
    return [0] + [int(torch.argmin((log_snr - math.log(((1 - k / K) / (k / K)) ** 2)).abs())) for k in range(1, K)] + [len(abar) - 1]


class Backend:
    """CogVideoX-5B (diffusers): 3D causal VAE 4x8x8 (z 16, scaled by the VAE scaling factor), v-prediction DDPM with zero
    terminal SNR, T5-XXL text encoder (the empty prompt, encoded once and cached)."""
    chunk = CHUNK

    def __init__(self, device):
        from diffusers import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel, CogVideoXDDIMScheduler
        self.device = device; ck = CKPT_DIR
        self.transformer = CogVideoXTransformer3DModel.from_pretrained(ck, subfolder="transformer", torch_dtype=BF).to(device).eval()
        self.vae = AutoencoderKLCogVideoX.from_pretrained(ck, subfolder="vae", torch_dtype=BF).to(device).eval()
        self.scale = float(self.vae.config.scaling_factor)
        self.abar = CogVideoXDDIMScheduler.from_pretrained(ck, subfolder="scheduler").alphas_cumprod.double().to(device)   # [1000]
        self.t_of_k = snr_timesteps(self.abar)

        def build():
            from transformers import AutoTokenizer, T5EncoderModel
            tok = AutoTokenizer.from_pretrained(ck, subfolder="tokenizer")
            enc = T5EncoderModel.from_pretrained(ck, subfolder="text_encoder", torch_dtype=BF).to(device).eval()
            ids = tok([""], padding="max_length", max_length=226, truncation=True, add_special_tokens=True, return_tensors="pt").input_ids
            with torch.no_grad():
                h = enc(ids.to(device))[0]
            del enc; return h.cpu()
        self.ctx = cached_embeds("cogvideox", build).to(device, BF)
        self.rope = None

    def rotary(self, T_lat: int, h: int, w: int):
        if self.rope is None:
            from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
            from diffusers.models.embeddings import get_3d_rotary_pos_embed
            cfg = self.transformer.config; p = cfg.patch_size; grid = (h // p, w // p)
            crops = get_resize_crop_region_for_grid(grid, cfg.sample_width // p, cfg.sample_height // p)
            self.rope = get_3d_rotary_pos_embed(embed_dim=cfg.attention_head_dim, crops_coords=crops, grid_size=grid,
                                                temporal_size=T_lat, device=self.device)
        return self.rope

    @torch.no_grad()
    def encode(self, wins):
        z = self.vae.encode(to_model(wins, self.device)).latent_dist.mode().float()   # [B,C,T,h,w]
        return z.permute(0, 2, 1, 3, 4) * self.scale                                  # [B,T,C,h,w], the DiT layout

    @torch.no_grad()
    def decode(self, z):
        return to_uint8(self.vae.decode((z / self.scale).permute(0, 2, 1, 3, 4).to(BF)).sample)

    @torch.no_grad()
    def v_pred(self, x, t):
        B = x.shape[0]
        tt = torch.full((B,), int(t), device=self.device, dtype=torch.long)
        return self.transformer(hidden_states=x.to(BF), encoder_hidden_states=self.ctx.expand(B, -1, -1), timestep=tt,
                                image_rotary_emb=self.rotary(x.shape[1], x.shape[3], x.shape[4]), return_dict=False)[0].float()


# ---------------------------------------------------------------- the model: VAE + official sampler
class Base:
    """encode(wins) -> z [B, ...]; decode(z) -> list of [T,H,W,3] uint8; levels() -> the official schedule (descending);
    start(s) -> first level index with noise <= sigma' s; noised(z, i, eps) -> forward process at level i;
    init(z, s, i0, eps) -> the starting sample; x0(x, i) -> clean estimate; step(x, x0, i, noise) -> the official update;
    n_ctx() -> context is re-noised after step i while i + 1 < n_ctx; finish(x, guide) -> final sample."""
    t_axis = 2
    chunk = CHUNK
    desc = ""

    def __init__(self, sub: int):
        self.device = torch.device("cuda:0"); torch.cuda.set_device(self.device); self.sub = sub      # pick the GPU with CUDA_VISIBLE_DEVICES

    def reset(self, gen): pass
    def init(self, z, s, i0, eps): return self.noised(z, i0, eps)
    def n_ctx(self): return len(self.levels())
    def finish(self, x, guide): return x


class Model(Base):
    t_axis = 1
    STEPS = 50

    def __init__(self, sub):
        super().__init__(sub)
        from diffusers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler
        self.M = Backend(self.device); self.chunk = self.M.chunk
        cfg = CogVideoXDDIMScheduler.load_config(CKPT_DIR, subfolder="scheduler")
        self.sch = CogVideoXDPMScheduler.from_config(cfg); self.sch.set_timesteps(self.STEPS, device=self.device)
        self.ts = [int(t) for t in self.sch.timesteps]; self.abar = self.sch.alphas_cumprod.double().to(self.device)
        self.desc = f"CogVideoXDPMScheduler {self.STEPS} steps, timestep_spacing trailing, zero-SNR v-prediction (official 5B recipe)"

    @torch.no_grad()
    def encode(self, wins): return torch.cat([self.M.encode(np.stack(wins[i:i + 1])) for i in range(len(wins))], 0)

    @torch.no_grad()
    def decode(self, z): return [d for i in range(len(z)) for d in self.M.decode(z[i:i + 1])]

    def t_star(self, s): return 999 if s >= 1 else int(self.M.t_of_k[round(s * K)])          # same-SNR DDPM timestep
    def levels(self): return self.ts
    def start(self, s): t = self.t_star(s); return next(i for i, u in enumerate(self.ts) if u <= t)
    def ab(self, i): a = float(self.abar[self.ts[i]]); return a ** 0.5, (1 - a) ** 0.5
    def noised(self, z, i, eps): A, B = self.ab(i); return A * z + B * eps
    def reset(self, gen): self.old = None; self.gen = gen

    def x0(self, x, i):
        A, B = self.ab(i); t = self.ts[i]
        return A * x - B * batched(lambda u: self.M.v_pred(u, t), x, self.sub)

    def step(self, x, x0, i, noise):
        A, B = self.ab(i); t = self.ts[i]
        v = (A * x - x0) / max(B, 1e-12)                                      # v consistent with the guided x0
        tb = torch.tensor(self.ts[i - 1], device=self.device) if self.old is not None else None
        x, self.old = self.sch.step(v, self.old, torch.tensor(t, device=self.device), tb, x, generator=self.gen, return_dict=False)
        return x


# ---------------------------------------------------------------- one pair
# Method A (responsibility guidance), applied inside the official sampler:
#   frames before the onset come from the --prefix clip (context, RePaint: re-noised to every level); the windows from the
#   onset on are regenerated. Their masked latents (onset..end) start as (z_P + z_I) / 2 noised to sigma' with the model's
#   own forward process (sigma' 1.0 = pure noise; the run starts at the first official level whose noise <= sigma').
#   Each step, the x0 estimate (empty prompt, no CFG) is compared with z_P / z_I inside the mask:
#     d_k = sum_mask (x0 - z_k)^2,  r_P = sigmoid((d_I - d_P) / (tau * dPI)),  dPI = sum_mask (z_P - z_I)^2,
#     x0 <- x0 + lambda (r_P z_P + (1 - r_P) z_I - x0)      (lambda 1.0 = overwrite with the chosen side);
#   after `commit` of the steps r_P is hardened to argmax (frozen unless --no-freeze) and the official step is rebuilt
#   from the guided x0. The last r_P is the sample's "choice" (P / I).
def generate_pair(M: Model, job: dict, a) -> dict:
    """All (sigma', lambda, seed) outputs of one pair; returns the meta dict (outputs with r_P trace and choice)."""
    cid = job["cid"]; out = Path(job["out"]); t0 = time.time()
    names = [(s, lam, sd, f"resp_l{lam:g}_s{s:.2f}_p{a.prefix}_seed{sd}.mp4") for s in a.sigmas for lam in a.lambdas for sd in range(a.seeds)]
    out.mkdir(parents=True, exist_ok=True)
    vi, fps = read_video(job["bad"], a.size); vp, _ = read_video(job["good"], a.size)
    n = min(len(vi), len(vp)); vi, vp = vi[:n], vp[:n]; vc = vi if a.prefix == "I" else vp
    end = job["end_frame"] if job["category"] == "SphereFallingDown" else n - 1       # GT window end (SphereFallingDown: labelled end)
    plan = window_plan(n, M.chunk); todo = [w for w, (lo, hi) in enumerate(plan) if job["onset"] < hi]
    wins_i = [take_window(vi, *plan[w]) for w in todo]; wins_p = [take_window(vp, *plan[w]) for w in todo]
    choices = {}
    with torch.no_grad():
        zi, zp = M.encode(wins_i), M.encode(wins_p); zc = zi if a.prefix == "I" else zp
        n_lat = zi.shape[M.t_axis]
        m = mix_mask([plan[w] for w in todo], job["onset"], n_lat, M.chunk, M.t_axis, zi); keep = m == 0
        z_mix = torch.where(keep, zc, (zp + zi) / 2); dPI = float(((zp - zi) ** 2 * m).sum())
        print(f"  {cid}: windows {todo}, masked latents {int(m.sum())}, dPI {dPI:.1f}", flush=True)
        for s, lam, sd, fname in names:
            if (out / fname).exists() and not a.force:
                continue
            gen = torch.Generator(device=M.device).manual_seed(42 + 1000 * sd + round(s * 100))
            noise = lambda: torch.randn(zi.shape, generator=gen, device=M.device, dtype=zi.dtype)
            i0 = M.start(s); L = M.levels(); N = len(L) - i0; M.reset(gen); trace = []

            def guide(x0, hard):
                if a.freeze and hard and trace:
                    rP = trace[-1]
                else:
                    dP = float(((x0 - zp) ** 2 * m).sum()); dI = float(((x0 - zi) ** 2 * m).sum())
                    rP = 1 / (1 + np.exp(-(dI - dP) / (a.tau * dPI)))
                if hard:
                    rP = float(rP > 0.5)
                trace.append(round(float(rP), 3))
                target = rP * zp + (1 - rP) * zi
                return torch.where(keep, x0, x0 + lam * (target - x0))

            x = M.init(z_mix, s, i0, noise())
            x = torch.where(keep, M.noised(zc, i0, noise()), x); n_ctx = M.n_ctx()
            for k, i in enumerate(range(i0, len(L))):
                x0 = guide(M.x0(x, i), k >= a.commit * N)
                x = M.step(x, x0, i, noise)
                if i + 1 < n_ctx:
                    x = torch.where(keep, M.noised(zc, i + 1, noise()), x)
            x = M.finish(x, lambda x0: guide(x0, True))
            x = torch.where(keep, zc, x)
            write_video(out / fname, assemble(vc, plan, todo, M.decode(x), job["onset"], n), fps)
            choices[fname] = {"choice": "P" if trace[-1] > 0.5 else "I", "rP_trace": trace, "start_level": float(L[i0]), "steps": N}
            print(f"  {cid} s={s:.2f} lam={lam:g} seed{sd} done at {time.time()-t0:.0f}s ({N} steps) choice {choices[fname]['choice']}", flush=True)
    prev = json.load(open(out / "meta.json")) if (out / "meta.json").exists() else {}
    meta = {"model": MODEL_NAME, "method": "resp", "sampler": "official", "schedule": M.desc, "code": Path(__file__).name,
            "cid": cid, "attribute": job["attribute"], "category": job["category"], "good": job["good"], "bad": job["bad"],
            "onset_frame": job["onset"], "end_frame": end, "frames": n, "fps": fps, "size": a.size, "chunk": M.chunk,
            "windows": [list(p) for p in plan], "generated_windows": todo, "sigmas": ",".join(f"{s:g}" for s in a.sigmas),
            "seeds": list(range(a.seeds)), "lambdas": a.lambdas, "tau": a.tau, "commit": a.commit, "freeze": a.freeze,
            "prefix": a.prefix, "prompt": "", "cfg": None, "outputs": {**prev.get("outputs", {}), **choices},
            "seconds": round(time.time() - t0, 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--video", required=True, help="an IntPhys2 root holding Main/Videos, or a folder of .mp4 files")
    ap.add_argument("-o", "--out", default="", help=f"output directory (default: {HERE / 'out_gen' / MODEL_KEY})")
    ap.add_argument("--names", default="", help="only these pairs: cids (236-1, sol-1-2) or Impossible video names, comma-separated")
    ap.add_argument("--force", action="store_true", help="redo outputs that already exist")
    ap.add_argument("--gt-csv", type=Path, default=Path(GT_CSV), help="the pairs: videos, attribute, scene, onset (default: %(default)s)")
    ap.add_argument("--sigmas", default="0.5,0.8,1.0", help="sigma' start levels (flow sigma; 1.0 = pure noise)")
    ap.add_argument("--lambdas", default="0.5,1.0", help="guidance strengths")
    ap.add_argument("--seeds", type=int, default=2, help="seeds 0..N-1 (noise seed = 42 + 1000 seed + 100 sigma')")
    ap.add_argument("--prefix", choices=("I", "P"), default="I", help="clip providing the frames before the onset (context)")
    ap.add_argument("--tau", type=float, default=0.1, help="responsibility temperature")
    ap.add_argument("--commit", type=float, default=0.99, help="fraction of the steps after which r_P is hardened to argmax")
    ap.add_argument("--no-freeze", dest="freeze", action="store_false", help="re-evaluate the hardened choice every step (default: frozen)")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--sub-batch", type=int, default=SUB_BATCH, help="windows per model forward")
    a = ap.parse_args()
    a.sigmas = sorted({float(v) for v in a.sigmas.split(",")}); a.lambdas = [float(v) for v in a.lambdas.split(",")]

    src = Path(a.video).expanduser().resolve()
    if (src / "Main").is_dir():
        src = main_videos_dir(src)
    if not src.is_dir():
        raise SystemExit(f"not a folder: {src} (pass an IntPhys2 root holding Main/Videos, or a folder of .mp4 files)")
    out_root = Path(a.out or HERE / "out_gen" / MODEL_KEY).expanduser().resolve()
    pairs = load_pairs(src, a.gt_csv.expanduser().resolve(), a.names)
    if not pairs:
        raise SystemExit(f"no complete pairs of {a.gt_csv} under {src}")
    for p in pairs:
        p["out"] = str(out_root / p["cid"])
    is_done = lambda p: all((Path(p["out"]) / f"resp_l{lam:g}_s{s:.2f}_p{a.prefix}_seed{sd}.mp4").exists()
                            for s in a.sigmas for lam in a.lambdas for sd in range(a.seeds))
    todo = [p for p in pairs if a.force or not is_done(p)]
    print(f"{MODEL_KEY}: {len(pairs)} pair(s), {len(pairs) - len(todo)} already done -> {out_root}", flush=True)
    if not todo:
        print("ALL JOBS DONE", flush=True); return
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "run_meta.json").write_text(json.dumps({
        "model": MODEL_NAME, "code": Path(__file__).name, "input": str(src), "gt_csv": str(a.gt_csv), "size": a.size, "chunk": CHUNK,
        "sigmas": a.sigmas, "lambdas": a.lambdas, "seeds": a.seeds, "prefix": a.prefix, "tau": a.tau, "commit": a.commit,
        "freeze": a.freeze, "sampler": "official"}, indent=2))

    tl = time.time(); print(f"loading {MODEL_NAME}...", flush=True)
    M = Model(a.sub_batch)
    print(f"loaded in {time.time()-tl:.0f}s, {torch.cuda.memory_allocated(M.device)/2**30:.1f} GiB, chunk {M.chunk}, "
          f"sub-batch {a.sub_batch}; sampler: {M.desc}", flush=True)
    for i, p in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {p['cid']}", flush=True); t0 = time.time()
        generate_pair(M, p, a)
        print(f"JOB DONE {p['cid']} {time.time()-t0:.0f}s", flush=True)
    print("ALL JOBS DONE", flush=True)


if __name__ == "__main__":
    main()
