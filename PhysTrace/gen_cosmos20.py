#!/usr/bin/env python3
"""Cosmos-Predict2-2B (Video2World): responsibility-guided P/I generation (Method A) with the model's OFFICIAL sampler.
-i: an IntPhys2 root (Main/Videos) or a folder of .mp4 files; pairs (P / I clips, violation onset) come from --gt-csv.

Sampler: RectifiedFlow AB2, Karras 35 steps (official Cosmos-Predict2 video2world pipeline).
Windows of 93 frames; from the onset on they are regenerated toward the side (P / I)
the model's own x0 agrees with (r_P, --lambdas); see generate_pair.
Outputs: <out>/<cid>/resp_l<lambda>_s<sigma'>_p<prefix>_seed<n>.mp4 + meta.json (r_P trace, choice);
existing files are skipped. GPU: CUDA_VISIBLE_DEVICES.
With an IntPhys2 root as -i, each output is also copied into <out>/sets/<output name>/Main (Videos + metadata.csv;
the generated clip in the Impossible slot, the original Possible clip, meta.json with the choice), the layout
score/run_test.py reads: --intphys2-dir <out>/sets/resp_l0.5_s1.00_pI_seed0 (see gen_sets.py).

  CUDA_VISIBLE_DEVICES=0 python3 PhysTrace/gen_cosmos20.py \
      -i /Path/To/IntPhys2 \
      -o results/gen_cosmos20 \
      --sigmas 1.0 \
      --lambdas 0.5 \
      --seeds 0,1,3,4,5 \
      --sigma-hard 1 \
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
GT_CSV_FULL = GT_CSV.with_name("intphys2_gt_full.csv")   # all 499 labelled pairs (the 7 without an onset are skipped)
from demo.core.pairs import main_videos_dir  # noqa: E402
from PhysTrace.gen_sets import export_sets  # noqa: E402
DM = Path(os.environ.get("DIFFUSION_MODELS", ROOT / "DiffusionModels"))


def _pick(env: str, *candidates) -> str:
    if os.environ.get(env):
        return os.environ[env]
    for c in candidates:
        if Path(c).exists():
            return str(c)
    return str(candidates[-1])


COSMOS20_ROOT = _pick("COSMOS20_ROOT", DM / "cosmos-predict2")                 # official repo
# weights: `hf download nvidia/Cosmos-Predict2-2B-Video2World --revision f50c09f5d8ab133a90cac3f4886a6471e9ba3f18
#   --include model-480p-16fps.pt config.json model_index.json "scheduler/*" "text_encoder/*" "tokenizer/*" "vae/*"
#   --local-dir DiffusionModels/weights/Cosmos-Predict2-2B-Video2World` 
COSMOS20_WEIGHTS = Path(os.environ.get("COSMOS20_WEIGHTS") or next(
    (p for p in (DM / "weights" / "Cosmos-Predict2-2B-Video2World", DM / "weights" / "cosmos" / "nvidia" / "Cosmos-Predict2-2B-Video2World")
     if (p / "model-480p-16fps.pt").exists()), DM / "weights" / "Cosmos-Predict2-2B-Video2World"))


def load_cosmos20(device):
    """The official Cosmos-Predict2-2B Video2World pipeline (480p, 16 fps config; T5 text encoder, guardrail off), with
    every file taken from COSMOS20_WEIGHTS: DiT model-480p-16fps.pt, video tokenizer tokenizer/tokenizer.pth, T5 text_encoder/
    (its tokenizer files sit in tokenizer/ of the HF repo, not beside the encoder)."""
    W = COSMOS20_WEIGHTS
    for f in ("model-480p-16fps.pt", "tokenizer/tokenizer.pth", "text_encoder/config.json", "tokenizer/spiece.model"):
        if not (W / f).exists():
            raise FileNotFoundError(f"Cosmos-Predict2 weights: {W / f} missing (set COSMOS20_WEIGHTS or see README)")
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    if not hasattr(PreTrainedTokenizerBase, "batch_encode_plus"):          # newer transformers dropped it; the repo still calls it
        PreTrainedTokenizerBase.batch_encode_plus = lambda self, *a, **kw: self(*a, **kw)
    os.environ.setdefault("COSMOS_PREDICT2_ARGS", f"--checkpoints {W.parent}")   # read at import time; every path is set below
    sys.path.insert(0, COSMOS20_ROOT)
    import importlib.util, types
    if importlib.util.find_spec("flash_attn") is None:                      # only the (unused) Qwen2.5-VL module needs it
        stub = types.ModuleType("imaginaire.networks.qwen2_5_vl")
        for n in ("Qwen2_5_VisionTransformerPretrainedModel", "Qwen2_5_VLModel", "get_rope_index"):
            setattr(stub, n, None)
        sys.modules["imaginaire.networks.qwen2_5_vl"] = stub
        print("flash_attn not installed -> qwen2_5_vl stubbed", flush=True)
    from cosmos_predict2.configs.base.config_video2world import get_cosmos_predict2_video2world_pipeline
    from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
    import imaginaire.auxiliary.text_encoder as TE
    if not (W / "text_encoder" / "spiece.model").exists():                 # T5 tokenizer from tokenizer/ (no copy into text_encoder/ needed)
        fast = TE.T5TokenizerFast
        class _T5Tok(fast):
            @classmethod
            def from_pretrained(cls, path, *a, **kw):
                return fast.from_pretrained(str(W / "tokenizer") if Path(path) == W / "text_encoder" else path, *a, **kw)
        TE.T5TokenizerFast = _T5Tok
    cfg = get_cosmos_predict2_video2world_pipeline(model_size="2B", resolution="480", fps=16)
    cfg.tokenizer.vae_pth = str(W / "tokenizer" / "tokenizer.pth")
    cfg.text_encoder.t5.ckpt_path = str(W / "text_encoder")
    cfg.guardrail_config.enabled = False
    print(f"loading Cosmos-Predict2-2B-Video2World (repo {COSMOS20_ROOT}, weights {W})...", flush=True)
    return Video2WorldPipeline.from_config(cfg, dit_path=str(W / "model-480p-16fps.pt"), use_text_encoder=True,
                                           device=str(device), torch_dtype=torch.bfloat16)


MODEL_KEY, MODEL_NAME = "cosmos20", "Cosmos-Predict2.0-2B"
CHUNK = 93                 # frames per model window
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
    unlabelled = sum(1 for r in rows if not r["onset_frame"])
    if missing := len(rows) - unlabelled - len(pairs):
        print(f"[pairs] {missing} of the {len(rows) - unlabelled} labelled pairs in {Path(gt_csv).name} have no complete pair under {video_dir}", file=sys.stderr)
    if unlabelled:
        print(f"[pairs] {unlabelled} of the {len(rows)} pairs in {Path(gt_csv).name} have no onset and are skipped", file=sys.stderr)
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
    STEPS = 35

    def __init__(self, sub):
        super().__init__(sub)
        self.pipe = load_cosmos20(self.device); self.size = None                   # set per --size at encode time
        self.sch = self.pipe.scheduler; self.sch.set_timesteps(self.STEPS, device=self.device)
        self.sg = [float(v) for v in self.sch.sigmas]; self.ts = list(self.sch.timesteps); self.x0_fn = None
        self.desc = f"RectifiedFlow AB2, Karras {self.STEPS} steps (official Cosmos-Predict2 pipeline)"

    @torch.no_grad()
    def encode(self, wins):
        if self.size is None:
            self.size = wins[0].shape[1]
        if self.x0_fn is None:                                               # conditions need one real batch
            self.x0_fn = self._x0_builder(np.stack(list(wins) * self.sub)[:self.sub])
        zs = []
        for i in range(0, len(wins), self.sub):
            v = torch.from_numpy(np.stack(wins[i:i + self.sub])).permute(0, 4, 1, 2, 3).to(self.device)
            zs.append(self.pipe.encode(v.to(self.pipe.torch_dtype).div(127.5).sub(1.0)).float())
        return torch.cat(zs, 0).clone()

    @torch.no_grad()
    def decode(self, z):
        out = []
        for i in range(0, z.shape[0], self.sub):
            x = self.pipe.decode(z[i:i + self.sub].to(self.pipe.torch_dtype)).float()
            out += [((u.clamp(-1, 1) + 1) / 2 * 255).round().byte().permute(1, 2, 3, 0).cpu().numpy() for u in x]
        return out

    def _x0_builder(self, wins_uint8):
        """x0(x, sigma) for the empty prompt, no CFG, no conditional frames."""
        pipe = self.pipe; sub = self.sub
        from cosmos_predict2.conditioner import DataType
        conds = {}

        def conditions_for(nb):
            if nb in conds:
                return conds[nb]
            v = torch.from_numpy(wins_uint8[:nb].copy()).permute(0, 4, 1, 2, 3).to(self.device)
            batch = pipe._get_data_batch_input(v, prompt="", negative_prompt="", num_latent_conditional_frames=0)
            batch["t5_text_embeddings"] = batch["t5_text_embeddings"].expand(nb, -1, -1).contiguous()
            batch["fps"] = batch["fps"].expand(nb).contiguous()
            batch["padding_mask"] = batch["padding_mask"].expand(nb, -1, -1, -1).contiguous()
            condition, _ = pipe.conditioner.get_condition_uncondition(batch)
            _, x0, _ = pipe.get_data_and_condition(batch)
            c = condition.edit_data_type(DataType.VIDEO)
            c = c.set_video_condition(gt_frames=x0, random_min_num_conditional_frames=pipe.config.min_num_conditional_frames,
                                      random_max_num_conditional_frames=pipe.config.max_num_conditional_frames, num_conditional_frames=0)
            c = c.edit_for_inference(is_cfg_conditional=True, num_conditional_frames=0)
            _, c, _, _ = pipe.broadcast_split_for_model_parallelsim(x0, c, None, None)
            assert float(c.condition_video_input_mask_B_C_T_H_W.abs().max()) == 0.0
            conds[nb] = c
            return c

        @torch.no_grad()
        def x0_fn(x, sigma):
            outs = []
            for i in range(0, x.shape[0], sub):
                xb = x[i:i + sub]; cond = conditions_for(xb.shape[0])
                s = torch.full((xb.shape[0],), float(sigma), device=self.device, dtype=torch.float32)
                outs.append(pipe.denoise(xb, s, cond).x0)
            return torch.cat(outs, 0)
        return x0_fn

    def levels(self): return self.ts
    def start(self, s):
        s_edm = min(s / (1 - s), self.sg[0]) if s < 1 else self.sg[0]
        return next(i for i, v in enumerate(self.sg[:-1]) if v <= s_edm + 1e-6)
    def noised(self, z, i, eps): return z + self.sg[i] * eps
    def n_ctx(self): return len(self.sg)                                       # context re-noised at sigma_min before finish
    def reset(self, gen): self.x0_prev = None
    def x0(self, x, i): return self.x0_fn(x, self.sg[i])

    def step(self, x, x0, i, noise):
        x, self.x0_prev = self.sch.step(x0_pred=x0, i=i, sample=x, x0_prev=self.x0_prev); return x

    def finish(self, x, guide):                                                # last denoise at sigma_min
        return guide(self.x0_fn(x, self.sg[-1]))


# ---------------------------------------------------------------- one pair
# Method A (responsibility guidance), applied inside the official sampler:
#   frames before the onset come from the --prefix clip (context, RePaint: re-noised to every level); the windows from the
#   onset on are regenerated. Their masked latents (onset..end) start as (z_P + z_I) / 2 noised to sigma' with the model's
#   own forward process (sigma' 1.0 = pure noise; the run starts at the first official level whose noise <= sigma').
#   Each step, the x0 estimate (empty prompt, no CFG) is compared with z_P / z_I inside the mask:
#     d_k = sum_mask (x0 - z_k)^2,  r_P = sigmoid((d_I - d_P) / (tau * dPI)),  dPI = sum_mask (z_P - z_I)^2,
#     x0 <- x0 + lambda (r_P z_P + (1 - r_P) z_I - x0)      (lambda 1.0 = overwrite with the chosen side);
#   after `sigma-hard` of the steps (1 = only the last step) r_P is hardened to argmax (frozen unless --no-freeze) and the official step is rebuilt
#   from the guided x0. The last r_P is the sample's "choice" (P / I).
def generate_pair(M: Model, job: dict, a) -> dict:
    """All (sigma', lambda, seed) outputs of one pair; returns the meta dict (outputs with r_P trace and choice)."""
    cid = job["cid"]; out = Path(job["out"]); t0 = time.time()
    names = [(s, lam, sd, f"resp_l{lam:g}_s{s:.2f}_p{a.prefix}_seed{sd}.mp4") for s in a.sigmas for lam in a.lambdas for sd in a.seeds]
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
                x0 = guide(M.x0(x, i), k >= a.sigma_hard * N if a.sigma_hard < 1 else k == N - 1)
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
            "seeds": a.seeds, "lambdas": a.lambdas, "tau": a.tau, "sigma_hard": a.sigma_hard, "freeze": a.freeze,
            "prefix": a.prefix, "prompt": "", "cfg": None, "outputs": {**prev.get("outputs", {}), **choices},
            "seconds": round(time.time() - t0, 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


# ---------------------------------------------------------------- one pair, overlapping windows (--overlap)
# Windows start every `stride` frames (stride = 93 - overlap; --overlap half = stride 48, overlap 45 frames).
# They are generated ONE AFTER ANOTHER: the first window ending after the
# onset, then the next ones until the clip end is covered. In each window the frames before its frontier F (the onset
# for the first window, the end of the previous generated window afterwards) are final: the --prefix clip before the
# onset, earlier generated output after it. They are the window's context, encoded from the clip assembled so far and
# reset every level as the onset context above (RePaint); the latents from the one holding F on are generated.
# The first window decides r_P exactly as generate_pair (dist rule, --sigma-hard, freeze); every later window is guided
# toward that choice (r_P one-hot at every step). Only frames >= F of a window are pasted, so every overlapped frame
# keeps the EARLIER window's output. All seeds of one (sigma', lambda) run as one batch, each with its own noise
# generator (noise seed = 42 + 1000 seed + 100 sigma', as generate_pair).
HALF_STRIDE = 48             # --overlap half: windows every 48 of the 93 frames, overlap 45 (stride a multiple of 4: the 4x temporal VAE latents line up)


def overlap_plan(n: int, chunk: int, stride: int, onset: int) -> list[tuple[int, int]]:
    """The generated windows [(a, b)], in order, on the grid a = 0, stride, 2 stride, ..."""
    starts = list(range(0, n, stride)); w0 = next(i for i, lo in enumerate(starts) if lo + chunk > onset); plan = []
    for lo in starts[w0:]:
        plan.append((lo, lo + chunk))
        if lo + chunk >= n:
            break
    return plan


def generate_pair_overlap(M: Model, job: dict, a) -> dict:
    """generate_pair with overlapping windows generated in sequence (see above); same output names and meta keys."""
    cid = job["cid"]; out = Path(job["out"]); t0 = time.time(); out.mkdir(parents=True, exist_ok=True)
    vi, fps = read_video(job["bad"], a.size); vp, _ = read_video(job["good"], a.size)
    n = min(len(vi), len(vp)); vi, vp = vi[:n], vp[:n]; vc = vi if a.prefix == "I" else vp
    end = job["end_frame"] if job["category"] == "SphereFallingDown" else n - 1
    plan = overlap_plan(n, M.chunk, a.stride, job["onset"]); S = len(a.seeds); choices = {}; windows = {}
    print(f"  {cid}: {len(plan)} windows, stride {a.stride} (overlap {M.chunk - a.stride})", flush=True)
    with torch.no_grad():
        zis = [M.encode([take_window(vi, lo, hi)]) for lo, hi in plan]; zps = [M.encode([take_window(vp, lo, hi)]) for lo, hi in plan]
        for s in a.sigmas:
            for lam in a.lambdas:
                names = [f"resp_l{lam:g}_s{s:.2f}_p{a.prefix}_seed{sd}.mp4" for sd in a.seeds]
                if all((out / f).exists() for f in names) and not a.force:
                    continue
                full = [vc.copy() for _ in range(S)]
                gens = [torch.Generator(device=M.device).manual_seed(42 + 1000 * sd + round(s * 100)) for sd in a.seeds]
                choice = None; trace = [[] for _ in range(S)]; F = job["onset"]; wmeta = []
                for (lo, hi), zi1, zp1 in zip(plan, zis, zps):
                    zc = M.encode([take_window(f, lo, hi) for f in full])
                    zi, zp = zi1.expand_as(zc).contiguous(), zp1.expand_as(zc).contiguous()
                    n_lat = zi.shape[M.t_axis]; local = F - lo
                    first = 0 if local <= 0 else next(l for l in range(n_lat) if latent_frames(l, n_lat, M.chunk)[1] >= local)
                    m = torch.zeros(n_lat, dtype=torch.bool); m[first:] = True
                    shape = [1] * zi.dim(); shape[M.t_axis] = n_lat
                    m = m.view(shape).to(zi.device).expand_as(zi).float(); keep = m == 0
                    z_mix = torch.where(keep, zc, (zp + zi) / 2); dPI = float(((zp1 - zi1) ** 2 * m[:1]).sum())
                    red = tuple(range(1, zc.dim())); bshape = [-1] + [1] * (zc.dim() - 1)
                    noise = lambda: torch.cat([torch.randn(zc.shape[1:], generator=g, device=M.device, dtype=zc.dtype)[None] for g in gens])
                    i0 = M.start(s); L = M.levels(); N = len(L) - i0; M.reset(gens if S > 1 else gens[0])

                    def guide(x0, hard):
                        if choice is not None:                                   # later windows: the first window's choice
                            rP = choice
                        elif a.freeze and hard and trace[0]:
                            rP = torch.tensor([tr[-1] for tr in trace], device=M.device)
                        else:
                            dP = ((x0 - zp) ** 2 * m).sum(red); dI = ((x0 - zi) ** 2 * m).sum(red)
                            rP = torch.sigmoid((dI - dP) / (a.tau * dPI))
                        if hard:
                            rP = (rP > 0.5).float()
                        if choice is None:
                            for k in range(S):
                                trace[k].append(round(float(rP[k]), 3))
                        target = rP.view(bshape) * zp + (1 - rP.view(bshape)) * zi
                        return torch.where(keep, x0, x0 + lam * (target - x0))

                    x = M.init(z_mix, s, i0, noise())
                    x = torch.where(keep, M.noised(zc, i0, noise()), x); n_ctx = M.n_ctx()
                    for k, i in enumerate(range(i0, len(L))):
                        x0 = guide(M.x0(x, i), k >= a.sigma_hard * N if a.sigma_hard < 1 else k == N - 1)
                        x = M.step(x, x0, i, noise)
                        if i + 1 < n_ctx:
                            x = torch.where(keep, M.noised(zc, i + 1, noise()), x)
                    x = M.finish(x, lambda x0: guide(x0, True))
                    x = torch.where(keep, zc, x)
                    if choice is None:
                        choice = torch.tensor([float(tr[-1] > 0.5) for tr in trace], device=M.device)
                    dec = M.decode(x); e = min(hi, n)
                    for k in range(S):
                        full[k][F:e] = dec[k][F - lo:e - lo]                     # frames < F keep the earlier output
                    wmeta.append({"window": [lo, hi], "frontier": F, "first_generated_latent": first}); F = e
                for k, (sd, fname) in enumerate(zip(a.seeds, names)):
                    full[k][:job["onset"]] = vc[:job["onset"]]
                    write_video(out / fname, full[k], fps)
                    choices[fname] = {"choice": "P" if trace[k][-1] > 0.5 else "I", "rP_trace": trace[k], "start_level": float(L[i0]), "steps": N}
                windows[f"{s:.2f}"] = wmeta
                print(f"  {cid} s={s:.2f} lam={lam:g} seeds {a.seeds} done at {time.time()-t0:.0f}s choices "
                      f"{''.join(choices[f]['choice'] for f in names)}", flush=True)
    prev = json.load(open(out / "meta.json")) if (out / "meta.json").exists() else {}
    meta = {"model": MODEL_NAME, "method": "resp", "sampler": "official", "schedule": M.desc, "code": Path(__file__).name,
            "cid": cid, "attribute": job["attribute"], "category": job["category"], "good": job["good"], "bad": job["bad"],
            "onset_frame": job["onset"], "end_frame": end, "frames": n, "fps": fps, "size": a.size, "chunk": M.chunk,
            "overlap": M.chunk - a.stride, "stride": a.stride, "windows": [list(p) for p in plan],
            "decision": "first generated window; later windows guided to its choice", "stitch": "overlapped frames keep the earlier window",
            "windows_by_sigma": {**prev.get("windows_by_sigma", {}), **windows}, "sigmas": ",".join(f"{s:g}" for s in a.sigmas),
            "seeds": a.seeds, "lambdas": a.lambdas, "tau": a.tau, "sigma_hard": a.sigma_hard, "freeze": a.freeze,
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
    ap.add_argument("--gt-csv", type=Path, default=Path(GT_CSV_FULL), help="the pairs: videos, attribute, scene, onset (default: %(default)s; the 283-pair subset is intphys2_gt.csv)")
    ap.add_argument("--sigmas", default="1.0", help="sigma' start levels (flow sigma; 1.0 = pure noise)")
    ap.add_argument("--lambdas", default="0.5", help="guidance strengths")
    ap.add_argument("--seeds", default="0,1,3,4,5", help="comma-separated seeds, e.g. 0 or 0,1,3,4,5 (noise seed = 42 + 1000 seed + 100 sigma')")
    ap.add_argument("--prefix", choices=("I", "P"), default="I", help="clip providing the frames before the onset (context)")
    ap.add_argument("--tau", type=float, default=0.1, help="responsibility temperature")
    ap.add_argument("--sigma-hard", type=float, default=1.0, help="fraction of the steps after which r_P is hardened to argmax; 1 = only the last step")
    ap.add_argument("--no-freeze", dest="freeze", action="store_false", help="re-evaluate the hardened choice every step (default: frozen)")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--sub-batch", type=int, default=SUB_BATCH, help="windows per model forward")
    ap.add_argument("--overlap", default="0", help="frames shared by consecutive windows: 0 = non-overlapping windows generated in one batch (default); half = 45 of the 93 frames (windows every 48 frames); or a frame count. With overlap the windows are generated in sequence, each continuing the previous one (see generate_pair_overlap)")
    a = ap.parse_args()
    a.sigmas = sorted({float(v) for v in a.sigmas.split(",")}); a.lambdas = [float(v) for v in a.lambdas.split(",")]
    a.seeds = sorted({int(v) for v in a.seeds.split(",") if v.strip()})
    a.stride = HALF_STRIDE if a.overlap == "half" else CHUNK - int(a.overlap)
    if not 0 < a.stride <= CHUNK:
        raise SystemExit(f"--overlap {a.overlap}: must be half or 0..{CHUNK - 1}")

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
                            for s in a.sigmas for lam in a.lambdas for sd in a.seeds)
    todo = [p for p in pairs if a.force or not is_done(p)]
    outs = [f"resp_l{lam:g}_s{s:.2f}_p{a.prefix}_seed{sd}.mp4" for s in a.sigmas for lam in a.lambdas for sd in a.seeds]
    metadata = src.parent / "metadata.csv"           # IntPhys2 input only: the scoring sets need it
    for p in pairs:
        if p not in todo:
            export_sets(p, out_root, outs, metadata)
    print(f"{MODEL_KEY}: {len(pairs)} pair(s), {len(pairs) - len(todo)} already done -> {out_root}", flush=True)
    if not todo:
        print("ALL JOBS DONE", flush=True); return
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "run_meta.json").write_text(json.dumps({
        "model": MODEL_NAME, "code": Path(__file__).name, "input": str(src), "gt_csv": str(a.gt_csv), "size": a.size, "chunk": CHUNK,
        "sigmas": a.sigmas, "lambdas": a.lambdas, "seeds": a.seeds, "prefix": a.prefix, "tau": a.tau, "sigma_hard": a.sigma_hard,
        "freeze": a.freeze, "overlap": CHUNK - a.stride, "stride": a.stride, "sampler": "official"}, indent=2))

    tl = time.time(); print(f"loading {MODEL_NAME}...", flush=True)
    M = Model(a.sub_batch)
    print(f"loaded in {time.time()-tl:.0f}s, {torch.cuda.memory_allocated(M.device)/2**30:.1f} GiB, chunk {M.chunk}, "
          f"sub-batch {a.sub_batch}; sampler: {M.desc}", flush=True)
    for i, p in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {p['cid']}", flush=True); t0 = time.time()
        (generate_pair_overlap if a.stride < CHUNK else generate_pair)(M, p, a)
        export_sets(p, out_root, outs, metadata)
        print(f"JOB DONE {p['cid']} {time.time()-t0:.0f}s", flush=True)
    print("ALL JOBS DONE", flush=True)


if __name__ == "__main__":
    main()
