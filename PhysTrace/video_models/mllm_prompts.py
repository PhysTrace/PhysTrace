"""The texts the MLLM is asked with, and the config that carries them.

One system and one segment file per IntPhys2 condition, read from a prompt-set folder by `load_prompt_set`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PhysTrace.video_windows import Window

# Keyed by metadata.csv's `condition` column, spelled as it spells them.
ATTRIBUTES = ('solidity', 'permanence', 'immutability', 'continuity')
PROMPT_SET_DIR = Path(__file__).resolve().parents[2] / 'IntPhys2Timing' / 'prompts'
# An earlier clip once it is history: one line of text in the current request, its video not resent.
# A text line and not an assistant turn: the model copies the shape of its own earlier turns over the
# system prompt's, and a state-only turn made it answer state-only or split the JSON (9 of 13 retries, 09-30).
# No CLIP score here or in the request (09-30): the score picks the candidate windows, not the model.
HISTORY_HEADER = 'Earlier clips:'
HISTORY_LINE = 'Clip {segment_number} ({start_sec:.2f}-{end_sec:.2f} s): {state}'

RETRY_TEMPLATE = (
    '\nYour previous output was invalid: {error}. '
    'Return one JSON object with the keys "state", "prediction" and "trusted", as the system prompt '
    'describes. Keep reasoning brief (under 150 words). Complete all JSON braces and answer tags. '
    '{closing}'
)
# Qwen3-VL's vision encoder patches frames in pairs (temporal_patch_size 2); an odd count gets its last
# frame repeated, which costs the tokens of a pair without showing anything new.
FRAMES_PER_PATCH = 2
# `history_windows` that shows the model no earlier window: the no-history ablation.
NO_HISTORY = -1


def check_mllm_frames(count) -> int:
    """A frame count the MLLM can take per window: a positive multiple of FRAMES_PER_PATCH."""
    if isinstance(count, bool) or not isinstance(count, int) or count < FRAMES_PER_PATCH or count % FRAMES_PER_PATCH:
        raise ValueError(f'MLLM frames per window must be an even integer >= {FRAMES_PER_PATCH}, got {count!r}: '
                         f'the vision encoder patches frames in pairs')
    return count


@dataclass(frozen=True)
class PromptConfig:
    """What one run asks with: the two texts and the counts that travel with them.

    `history_windows` 0 shows all trusted windows, NO_HISTORY (-1) none; `thinking_budget` 0 leaves reasoning uncapped.
    """

    segment_template: str
    system_prompt: str
    num_prompts: int = 1
    history_windows: int = 0
    trust_all: bool = False
    lowest_windows: int = 1
    max_new_tokens: int | None = None
    thinking_budget: int | None = None
    mllm_frames: int | None = None

    def __post_init__(self) -> None:
        """Fill each template with stand-in values, so a broken one fails here and not mid-run."""
        # The upper end of the two limits depends on the engine, so VideoPredictor.limits checks it.
        minimums = (('num_prompts', 1), ('history_windows', NO_HISTORY), ('lowest_windows', 0), ('max_new_tokens', 1),
                    ('thinking_budget', 0))
        for field, minimum in minimums:
            value = getattr(self, field)
            if value is None and field in ('max_new_tokens', 'thinking_budget'):
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f'{field} must be an integer >= {minimum}, got {value!r}')
        if self.max_new_tokens and self.thinking_budget and self.thinking_budget >= self.max_new_tokens:
            raise ValueError(f'thinking_budget {self.thinking_budget} leaves no room for the answer within '
                             f'max_new_tokens {self.max_new_tokens}')
        if not isinstance(self.trust_all, bool):
            raise ValueError(f'trust_all must be true or false, got {self.trust_all!r}')
        if self.mllm_frames is not None:
            check_mllm_frames(self.mllm_frames)
        placeholders = dict(
            segment_template=_WINDOW_PLACEHOLDERS,
            system_prompt=_SYSTEM_PLACEHOLDERS,
        )
        for field, values in placeholders.items():
            text = getattr(self, field)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f'{field} must be a non-empty string')
            try:
                text.format(**values)
            except (KeyError, IndexError, ValueError) as error:
                raise ValueError(f'{field} is not a usable template: {error}') from error


_WINDOW_PLACEHOLDERS = {'segment_number': 1, 'start_sec': 0.0, 'end_sec': 1.0}
_SYSTEM_PLACEHOLDERS = {'num_prompts': 1}


def load_prompt_set(folder: Path, **counts: int) -> dict[str, PromptConfig]:
    """Every condition's prompts from one prompt-set folder, checked before any model loads."""
    prompt_set = {}
    for attribute in ATTRIBUTES:
        texts = {part: (folder / f'{attribute}.{part}.txt').read_text(encoding='utf-8') for part in ('system', 'segment')}
        try:
            prompt_set[attribute] = PromptConfig(system_prompt=texts['system'], segment_template=texts['segment'], **counts)
        except ValueError as error:
            raise ValueError(f'{folder}: {attribute} prompts are invalid: {error}') from error
    return prompt_set


def system_prompt(prompts: PromptConfig) -> str:
    """The system turn: the run's own text, with the caption count filled in."""
    return prompts.system_prompt.format(num_prompts=prompts.num_prompts)


def window_request(window: Window, prompts: PromptConfig) -> str:
    """The user turn for one window: describe it as seen and as expected."""
    return prompts.segment_template.format(segment_number=window.index + 1,
                                           start_sec=window.start_sec, end_sec=window.end_sec)


def history_line(window: Window, state: list[str]) -> str:
    """The line an earlier window is remembered by: which clip it was and its state captions."""
    return HISTORY_LINE.format(segment_number=window.index + 1, start_sec=window.start_sec, end_sec=window.end_sec,
                               state=' | '.join(state))


def history_block(lines: list[str]) -> str:
    """The earlier clips as the text that opens a request; empty when there are none."""
    return '\n'.join([HISTORY_HEADER, *lines]) if lines else ''


def retry_request(request: str, error: str, num_prompts: int) -> str:
    """The same request again, naming what was wrong with the answer to it."""
    captions = f'{num_prompts} non-empty string{"s" if num_prompts > 1 else ""}'
    closing = f'"state" and "prediction" must each hold {captions}.'
    return request + RETRY_TEMPLATE.format(error=error, closing=closing)
