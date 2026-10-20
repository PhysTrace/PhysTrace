"""The MLLM that describes one window as seen and as expected from the earlier ones, served by vLLM.

The window goes in as one video, the earlier ones as text; pick the GPU with CUDA_VISIBLE_DEVICES.
"""
from __future__ import annotations

import atexit
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch
from transformers import AutoProcessor
from vllm import LLM
from vllm.config import ReasoningConfig

from PhysTrace.video_models.mllm_prompts import FRAMES_PER_PATCH, check_mllm_frames, retry_request

ATTEMPTS = 3
# Ours, not the model's: neither checkpoint's generation_config.json carries a seed and vLLM
# leaves it unset. The official config samples (do_sample, temperature 0.7), so a run is only
# reproducible once one is pinned. Vary it deliberately to measure decoding spread.
DEFAULT_SEED = 42
# Reasoning tokens a reasoning backend may spend per window before vLLM closes </think> for it.
# Normal windows ended within 2783 tokens in the 09-28/29 runs (median ~300); a window that loops
# ("Wait, ...") would otherwise run to max_new_tokens and come back with no answer at all.
DEFAULT_THINKING_BUDGET = 3000
# Room kept for a call's prompt out of max_model_len: the system prompt, the history, the window's
# video and its request measured 1-4.5k tokens on 09-29. The rest is what a call may generate.
PROMPT_TOKENS_RESERVED = 8192
# Frames per second of window, not a fixed count: the official processors take a target fps
# (Qwen3-VL defaults to 2, Cosmos-Reason2 is documented at 4) and derive the count from the
# clip's duration. 4 matches Cosmos-Reason2's documented rate.
FRAMES_PER_SECOND = 4


def frames_for(window_seconds: float) -> int:
    """How many frames one window of `window_seconds` is sampled down to, rounded to an even count (frame pairs)."""
    count = round(window_seconds * FRAMES_PER_SECOND / FRAMES_PER_PATCH) * FRAMES_PER_PATCH
    if count < FRAMES_PER_PATCH:
        raise ValueError(f'A {window_seconds:g}s window is shorter than two sampled frames; '
                         f'raise --window-seconds or FRAMES_PER_SECOND')
    return count


def even_indices(length: int, count: int) -> list[int]:
    """Evenly spaced window-local indices spanning the window, endpoints included."""
    if length < 1 or count < 1:
        raise ValueError('Frame counts must be positive')
    return np.linspace(0, length - 1, count).round().astype(int).tolist()


@dataclass(frozen=True)
class Backend:
    """One model family: where it lives, how it answers, and the budget it was measured with.

    `max_model_len` covers prompt and output; `vllm_budget_gib` becomes vLLM's share of the whole device.
    """

    name: str
    model_dir: str      # relative to the checkpoint root, or absolute
    reasoning: bool
    max_new_tokens: int
    vllm_budget_gib: float
    max_model_len: int

    def path(self, checkpoint_root) -> str:
        """Where this backend's checkpoint sits under `checkpoint_root`."""
        return str(Path(checkpoint_root) / self.model_dir)


BACKENDS = {backend.name: backend for backend in (
    # 8B: 16.8GiB of weights, so CLIP's 25GiB fits beside it on one 80GB card.
    Backend('cosmos-reason2', 'Cosmos-Reason2-8B', reasoning=True, max_new_tokens=4096, vllm_budget_gib=28.0,
            max_model_len=40960 + PROMPT_TOKENS_RESERVED),
    Backend('qwen3-vl-thinking', 'Qwen3-VL-8B-Thinking', reasoning=True, max_new_tokens=4096, vllm_budget_gib=28.0,
            max_model_len=40960 + PROMPT_TOKENS_RESERVED),
    # 32B: 62.1GiB of weights, so vLLM takes most of a card and CLIP needs another one.
    # The Thinking checkpoint's chat template opens <think> itself, and its reasoning needs Cosmos' token budget.
    Backend('qwen3-vl-32b-thinking', 'Qwen3-VL-32B-Thinking', reasoning=True, max_new_tokens=4096,
            vllm_budget_gib=72.0, max_model_len=16384),
)}
DEFAULT_BACKEND = 'qwen3-vl-thinking'


@dataclass(frozen=True)
class Ask:
    """What one call asks for: captions per field, write and reasoning limits, and frames per window.

    None keeps the predictor's own setting; a `thinking_budget` of 0 leaves the reasoning uncapped.
    """

    num_prompts: int
    max_new_tokens: int | None = None
    thinking_budget: int | None = None
    num_frames: int | None = None


@dataclass(frozen=True)
class Prediction:
    """One answer: captions of the clip as the model sees it and as the earlier clips lead it to
    expect it, and whether it trusts this clip."""

    state: list[str]
    prediction: list[str]
    trusted: bool

    FIELDS = ('state', 'prediction', 'trusted')

    def as_dict(self) -> dict:
        """The three fields as a plain dict."""
        return {field: getattr(self, field) for field in self.FIELDS}


def _unwrap(raw: str) -> str:
    """Strip whatever the model wrapped its JSON in: reasoning tags, answer tags, fences.

    A thinking checkpoint's output carries only the closing </think>; the answer is what follows it.
    """
    text = raw.strip()
    if '</think>' in text:
        text = text.split('</think>', 1)[1].strip()
    elif text.startswith('<think>'):
        raise ValueError('Reasoning was truncated before the answer; raise --max-new-tokens')
    if text.startswith('<answer>'):
        # A missing closing tag is fine; the JSON parser below still checks the object.
        text = re.sub(r'^<answer>\s*|\s*</answer>$', '', text, flags=re.DOTALL).strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text, count=1).strip().removesuffix('```').rstrip()
    return text


def parse_answer(raw: str, num_prompts: int) -> Prediction:
    """Read one answer, or raise so the caller can ask again."""
    answer = json.loads(_unwrap(raw))
    if not isinstance(answer, dict):
        raise ValueError('Expected a JSON object')
    missing = [field for field in Prediction.FIELDS if field not in answer]
    if missing:
        raise ValueError('Missing required field(s): ' + ', '.join(missing))
    if not isinstance(answer['trusted'], bool):
        raise ValueError('trusted must be true or false')
    return Prediction(_captions(answer['state'], 'state', num_prompts),
                      _captions(answer['prediction'], 'prediction', num_prompts), answer['trusted'])


def _captions(value, field: str, count: int) -> list[str]:
    """The first `count` captions of one field; a lone string counts as a list of one."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or len(value) < count:
        raise ValueError(f'{field} must be a list of {count} caption(s)')
    captions = value[:count]
    if not all(isinstance(caption, str) and caption.strip() for caption in captions):
        raise ValueError(f'every {field} caption must be a non-empty string')
    return [caption.strip() for caption in captions]


def drop_flash_attn_stubs() -> None:
    """Remove the fake flash_attn modules the CLIP loader leaves in `sys.modules`.

    vLLM would otherwise take its flash-attn path on them and fail.
    """
    for name in [name for name in sys.modules
                 if name == 'flash_attn' or name.startswith('flash_attn.')]:
        del sys.modules[name]


class VideoPredictor:
    """One vLLM engine of one backend, asked about one window at a time.

    `num_frames`, `max_new_tokens`, `temperature` and `thinking_budget` are defaults that a call's `Ask` may override.
    """

    def __init__(self, backend: Backend, model_path: str, *,  # noqa: PLR0913 - engine knobs
                 window_seconds: float, num_frames: int | None = None,
                 max_new_tokens: int | None = None, seed: int | None = DEFAULT_SEED,
                 temperature: float | None = None, gpu_fraction: float | None = None,
                 max_model_len: int | None = None, thinking_budget: int | None = DEFAULT_THINKING_BUDGET):
        # Before the engine starts, never after: vLLM decides here whether flash-attn exists.
        drop_flash_attn_stubs()
        self.backend, self.model_path = backend, model_path
        # The command line's fixed count, kept apart so a run of another window length can still derive its own.
        self.fixed_frames = check_mllm_frames(num_frames) if num_frames else None
        self.num_frames = self.fixed_frames or frames_for(window_seconds)
        self.max_model_len = max_model_len or backend.max_model_len
        # What a call may generate: the engine's context less the room kept for the prompt.
        self.output_limit = self.max_model_len - PROMPT_TOKENS_RESERVED
        if self.output_limit < 1:
            raise ValueError(f'max_model_len {self.max_model_len} leaves no room past the {PROMPT_TOKENS_RESERVED} '
                             f'tokens kept for the prompt')
        self.max_new_tokens, self.thinking_budget = self.limits(Ask(1, max_new_tokens or backend.max_new_tokens,
                                                                    thinking_budget or 0))
        self.processor = AutoProcessor.from_pretrained(model_path)
        # Thinking checkpoints open <think> in the generation prompt; Cosmos-Reason2 writes both tags itself.
        self.opens_think = self.processor.apply_chat_template(
            [{'role': 'user', 'content': 'x'}], tokenize=False, add_generation_prompt=True).rstrip().endswith('<think>')
        if gpu_fraction is None:
            gpu_fraction = self._fit_to_device(backend.vllm_budget_gib)
        try:
            # vLLM refuses a thinking_token_budget unless the engine knows the <think> delimiters;
            # on for every reasoning backend, so a run can set a budget without reloading.
            reasoning = {'reasoning_config': ReasoningConfig()} if backend.reasoning else {}
            self.llm = LLM(model=model_path, dtype='bfloat16', limit_mm_per_prompt={'video': 1},
                           max_model_len=self.max_model_len, enable_prefix_caching=True,
                           gpu_memory_utilization=gpu_fraction, disable_log_stats=True, **reasoning)
        except Exception as error:
            free, total = torch.cuda.mem_get_info()
            raise RuntimeError(
                f'vLLM could not start with gpu_memory_utilization={gpu_fraction:.2f} '
                f'({gpu_fraction * total / 1024 ** 3:.1f}GiB requested, {free / 1024 ** 3:.1f}GiB '
                f'free). That budget has to hold the weights, the CUDA graphs and a KV cache '
                f'for max_model_len={self.max_model_len}; lower --max-model-len, raise --gpu-fraction '
                f'if the device has room, free more of it (CUDA_VISIBLE_DEVICES), or put CLIP on '
                f'another device with --clip-device.') from error
        # Temperature, top_p, top_k and the penalties come from the checkpoint's own
        # generation_config.json rather than being restated here, so they track the model.
        # The seed is the one thing added on top; see DEFAULT_SEED.
        self.sampling = self.llm.get_default_sampling_params()
        self.sampling.max_tokens = self.max_new_tokens
        self.sampling.seed = seed
        self.sampling.thinking_token_budget = self.thinking_budget
        if temperature is not None:
            # Only when a run asks for it: `--temperature 0` makes decoding greedy, which is
            # what a comparison between two prompts needs -- at the official temperature the
            # same window can come back as a different scene from one run to the next.
            self.sampling.temperature = temperature
        self.temperature = self.sampling.temperature
        # vLLM runs its engine in a child process; without this it outlives a killed run and
        # keeps the GPU memory it reserved.
        atexit.register(self.close)

    @staticmethod
    def _fit_to_device(budget_gib: float) -> float:
        """The backend's vLLM budget as a share of this device, checked against what is free."""
        free, total = torch.cuda.mem_get_info()
        budget = budget_gib * 1024 ** 3
        if free < budget:
            raise RuntimeError(
                f'vLLM needs {budget_gib:.0f}GiB but only {free / 1024 ** 3:.1f}GiB of the '
                f'{total / 1024 ** 3:.0f}GiB device is free. Give it a device of its own '
                f'(CUDA_VISIBLE_DEVICES), put CLIP on another one (--clip-device), or pass a '
                f'smaller --gpu-fraction and accept a shorter context.')
        fraction = min(0.9, budget / total)
        print(f'[vllm] {free / 1024 ** 3:.1f}GiB free of {total / 1024 ** 3:.1f}GiB '
              f'-> gpu_memory_utilization={fraction:.2f} ({budget_gib:.0f}GiB for vLLM)',
              flush=True)
        return fraction

    def close(self) -> None:
        """Shut the engine's child process down, releasing the GPU memory it reserved."""
        engine = getattr(getattr(self, 'llm', None), 'llm_engine', None)
        if engine is not None and hasattr(engine, 'shutdown'):
            engine.shutdown()

    def limits(self, ask: Ask) -> tuple[int, int | None]:
        """The generation limit and thinking budget a call runs with: the ask's, else the defaults."""
        max_new_tokens = ask.max_new_tokens or self.max_new_tokens
        budget = self.thinking_budget if ask.thinking_budget is None else (ask.thinking_budget or None)
        # An Instruct backend never opens <think>, so a budget would change nothing there.
        budget = budget if self.backend.reasoning else None
        if not 1 <= max_new_tokens <= self.output_limit:
            raise ValueError(f'max_new_tokens must be in 1..{self.output_limit} (max_model_len {self.max_model_len} '
                             f'less {PROMPT_TOKENS_RESERVED} for the prompt), got {max_new_tokens}')
        if budget and budget >= max_new_tokens:
            raise ValueError(f'thinking budget {budget} leaves no room for the answer within '
                             f'max_new_tokens={max_new_tokens}; raise the limit or lower the budget')
        return max_new_tokens, budget

    def frames_per_window(self, window_seconds: float, requested: int | None = None) -> int:
        """How many frames a run with windows of `window_seconds` shows the model per window.

        A run's own count wins, then the predictor's fixed count, then the window length at `FRAMES_PER_SECOND`.
        """
        if requested is not None:
            return check_mllm_frames(requested)
        return self.fixed_frames or frames_for(window_seconds)

    def sample_indices(self, length: int, num_frames: int | None = None) -> list[int]:
        """Which frames of a `length`-frame window the model sees (window-local); None keeps `num_frames`."""
        return even_indices(length, num_frames or self.num_frames)

    def describe(self, frames: list, fps: float, request: str,  # noqa: PLR0913
                 system: str, history: str, ask: Ask) -> tuple[Prediction, list[str], list[str]]:
        """Answer for one window: `frames` is that window, `history` the earlier windows as text.

        Returns the parsed answer, every attempt's raw text (where the reasoning survives) and its rendered prompt.
        """
        system_turn = self.render_system(system)
        indices = self.sample_indices(len(frames), ask.num_frames)
        rgb = np.stack([np.ascontiguousarray(frames[index][:, :, ::-1]) for index in indices])
        sampling = self.sampling.clone()
        sampling.max_tokens, sampling.thinking_token_budget = self.limits(ask)
        instruction, error, raws, prompts = request, None, [], []
        for attempt in range(ATTEMPTS):
            if attempt:
                instruction = retry_request(request, error, ask.num_prompts)
            content = [{'type': 'video'}, {'type': 'text', 'text': instruction}]
            if history:
                # The chat template joins the parts with nothing between them.
                content.insert(0, {'type': 'text', 'text': history + '\n'})
            messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': content}]
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if not prompt.startswith(system_turn):
                raise ValueError(f'The chat template of {self.model_path} does not open the prompt with the system '
                                 f'turn alone, so the stored inputs cannot leave it out')
            prompts.append(prompt[len(system_turn):])
            raws.append(self._generate(prompt, rgb, (fps, indices, len(frames)), sampling))
            try:
                if self.opens_think and '</think>' not in raws[-1]:
                    # The opening tag is in the prompt, so a cut-off answer is bare reasoning text.
                    raise ValueError('Reasoning was truncated before the answer; raise the generation limit')
                return parse_answer(raws[-1], ask.num_prompts), raws, prompts
            except (ValueError, TypeError, json.JSONDecodeError) as failure:
                error = str(failure)   # ask again with the same frames and an explicit correction
        raise ValueError(f'MLLM returned invalid output after {ATTEMPTS} attempts: {error}\n'
                         f'Raw output: {raws[-1]}')

    def render_system(self, system: str) -> str:
        """The system turn as the chat template opens every prompt with it."""
        return self.processor.apply_chat_template([{'role': 'system', 'content': system}], tokenize=False,
                                                  add_generation_prompt=False)

    def _generate(self, prompt: str, rgb: np.ndarray, timing: tuple[float, list[int], int], sampling) -> str:
        """One generation of a rendered prompt over this window's frames.

        `timing` is the video's fps, the window-local frame indices and the frame count (vLLM's video metadata).
        """
        fps, indices, total_frames = timing
        metadata = {'total_num_frames': total_frames, 'fps': fps, 'frames_indices': indices,
                    'duration': total_frames / fps, 'do_sample_frames': False}
        output = self.llm.generate(
            # No `size` override: the resolution budget is the checkpoint's own
            # (video_preprocessor_config.json), which leaves a 512x512 IntPhys2 frame at
            # native resolution instead of resampling it down.
            {'prompt': prompt, 'multi_modal_data': {'video': (rgb, metadata)},
             'mm_processor_kwargs': {'do_sample_frames': False}},
            sampling, use_tqdm=False)[0]
        return output.outputs[0].text
