"""The same MLLM step as mllm.VideoPredictor, answered by a hosted model through OpenRouter's chat API.

The window's sampled frames go in as images, in order, where the vLLM path sends them as one video: through
`video_url` every provider resamples the clip itself (Gemini at 1 fps, the Qwen hosts at about 1-2 fps; probed
10-24 on a 60-frame 2 s clip, which came back as 2-3 frames), while images keep exactly the frames vLLM sees.
"""
from __future__ import annotations

import base64
import json
import os
import random
import time

import cv2
import requests

from PhysTrace.video_models.mllm import (
    ATTEMPTS,
    Ask,
    Backend,
    Prediction,
    even_indices,
    frames_for,
    parse_answer,
)
from PhysTrace.video_models.mllm_prompts import check_mllm_frames, retry_request

API_URL = 'https://openrouter.ai/api/v1/chat/completions'

DEFAULT_API_KEY = '<YOUR OPENROUTER-API KEY>'
# HTTP failures worth asking again: rate limits, timeouts and the providers' own outages.
RETRY_STATUS = {408, 429, 500, 502, 503, 504, 520, 522, 524, 529}
HTTP_ATTEMPTS = 8
TIMEOUT_SECONDS = 600
JPEG_QUALITY = 95


class OpenRouterAuthError(RuntimeError):
    """The key is refused or out of credit: every later call would fail the same way."""


class PlainTemplate:
    """Stands in for the HF processor's chat template: the API takes messages, so the stored text is our own."""

    @staticmethod
    def _text(content) -> str:
        if isinstance(content, str):
            return content
        return ''.join(part['text'] if part['type'] == 'text' else f"<{part['frames']} frames>"
                       for part in content)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False) -> str:
        """`<|role|>` and the turn's text, one turn after another."""
        text = ''.join(f"<|{message['role']}|>\n{self._text(message['content'])}\n" for message in messages)
        return text + ('<|assistant|>\n' if add_generation_prompt else '')


class OpenRouterPredictor:
    """One hosted model, asked about one window at a time; calls from several threads run concurrently.

    Takes the arguments VideoPredictor takes; the engine ones (`gpu_fraction`) mean nothing here.
    """

    def __init__(self, backend: Backend, model_path: str, *,  # noqa: PLR0913 - VideoPredictor's knobs
                 window_seconds: float, num_frames: int | None = None,
                 max_new_tokens: int | None = None, seed: int | None = None,
                 temperature: float | None = None, gpu_fraction: float | None = None,
                 max_model_len: int | None = None, thinking_budget: int | None = None):
        self.backend, self.model_path = backend, model_path
        self.fixed_frames = check_mllm_frames(num_frames) if num_frames else None
        self.num_frames = self.fixed_frames or frames_for(window_seconds)
        self.max_model_len = max_model_len or backend.max_model_len
        self.output_limit = backend.max_new_tokens
        self.seed, self.temperature = seed, temperature   # None leaves the provider's default
        self.max_new_tokens, self.thinking_budget = self.limits(Ask(1, max_new_tokens or backend.max_new_tokens,
                                                                    thinking_budget or 0))
        self.processor = PlainTemplate()
        self.api_key = os.environ.get('OPENROUTER_API_KEY', DEFAULT_API_KEY)

    def close(self) -> None:
        """Nothing to release: no engine runs here."""

    def limits(self, ask: Ask) -> tuple[int, int | None]:
        """The generation limit and thinking budget a call runs with: the ask's, else the defaults.

        The budget goes out as `reasoning.max_tokens`, which the providers take as a hint, not a cap
        (Qwen3.5-9B reasoned 3870 tokens under a budget of 200 on 10-24); only `max_new_tokens` is hard.
        """
        max_new_tokens = ask.max_new_tokens or self.max_new_tokens
        budget = self.thinking_budget if ask.thinking_budget is None else (ask.thinking_budget or None)
        budget = budget if self.backend.reasoning else None
        if not 1 <= max_new_tokens <= self.output_limit:
            raise ValueError(f'max_new_tokens must be in 1..{self.output_limit} for {self.model_path}, '
                             f'got {max_new_tokens}')
        if budget and budget >= max_new_tokens:
            raise ValueError(f'thinking budget {budget} leaves no room for the answer within '
                             f'max_new_tokens={max_new_tokens}; raise the limit or lower the budget')
        return max_new_tokens, budget

    def frames_per_window(self, window_seconds: float, requested: int | None = None) -> int:
        """How many frames a run with windows of `window_seconds` shows the model per window (as VideoPredictor)."""
        if requested is not None:
            return check_mllm_frames(requested)
        return self.fixed_frames or frames_for(window_seconds)

    def sample_indices(self, length: int, num_frames: int | None = None) -> list[int]:
        """Which frames of a `length`-frame window the model sees (window-local); None keeps `num_frames`."""
        return even_indices(length, num_frames or self.num_frames)

    def render_system_turn(self, system: str) -> str:
        """The system turn as every stored prompt opens with it."""
        return self.processor.apply_chat_template([{'role': 'system', 'content': system}])

    def describe(self, frames: list, fps: float, request: str,  # noqa: PLR0913
                 system: str, history: str, ask: Ask) -> tuple[Prediction, list[str], list[str]]:
        """Answer for one window, as VideoPredictor.describe: the parsed answer, every attempt's raw text, its prompt.

        A raw text is the provider's reasoning in <think> tags followed by the answer; only the answer is parsed.
        """
        system_turn = self.render_system_turn(system)
        indices = self.sample_indices(len(frames), ask.num_frames)
        images = [self._image_part(frames[index]) for index in indices]
        max_tokens, budget = self.limits(ask)
        instruction, error, raws, prompts = request, None, [], []
        for attempt in range(ATTEMPTS):
            if attempt:
                instruction = retry_request(request, error, ask.num_prompts)
            # The vLLM path's order: the history, the window's frames, then the request.
            shown = [{'type': 'frames', 'frames': len(images)}, {'type': 'text', 'text': instruction}]
            if history:
                shown.insert(0, {'type': 'text', 'text': history + '\n'})
            prompt = self.processor.apply_chat_template(
                [{'role': 'system', 'content': system}, {'role': 'user', 'content': shown}], add_generation_prompt=True)
            prompts.append(prompt[len(system_turn):])
            content = [part for item in shown for part in (images if item['type'] == 'frames' else [item])]
            reasoning, answer, finish = self._generate(system, content, max_tokens, budget)
            raws.append(f'<think>\n{reasoning}\n</think>\n\n{answer}' if reasoning else answer)
            try:
                if finish == 'length' and not answer.strip():
                    raise ValueError('Reasoning used the whole generation limit before the answer; '
                                     'raise the generation limit')
                return parse_answer(answer, ask.num_prompts), raws, prompts
            except (ValueError, TypeError, json.JSONDecodeError) as failure:
                error = str(failure)
        raise ValueError(f'MLLM returned invalid output after {ATTEMPTS} attempts: {error}\n'
                         f'Raw output: {raws[-1]}')

    @staticmethod
    def _image_part(frame) -> dict:
        """One BGR frame as a JPEG data URL."""
        ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            raise ValueError('Could not encode a frame as JPEG')
        return {'type': 'image_url',
                'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(jpeg.tobytes()).decode()}}

    def _body(self, system: str, content: list, max_tokens: int, budget: int | None) -> dict:
        """The request body of one call."""
        body = {'model': self.model_path,
                'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': content}],
                'max_tokens': max_tokens}
        if self.backend.reasoning:
            body['reasoning'] = {'enabled': True, **({'max_tokens': budget} if budget else {})}
        if self.seed is not None:
            body['seed'] = self.seed
        if self.temperature is not None:
            body['temperature'] = self.temperature
        if self.backend.media_resolution:
            body['media_resolution'] = self.backend.media_resolution
        provider = {'quantizations': list(self.backend.quantizations), 'ignore': list(self.backend.ignore)}
        if provider := {key: value for key, value in provider.items() if value}:
            body['provider'] = provider
        return body

    def _generate(self, system: str, content: list, max_tokens: int, budget: int | None) -> tuple[str, str, str]:
        """One call: the reasoning text, the answer text and the finish reason; network failures are retried."""
        body = self._body(system, content, max_tokens, budget)
        headers = {'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'}
        failure = None
        for attempt in range(HTTP_ATTEMPTS):
            if attempt:
                time.sleep(min(60, 2 ** attempt) * random.uniform(0.5, 1.0))
            try:
                response = requests.post(API_URL, headers=headers, json=body, timeout=TIMEOUT_SECONDS)
            except requests.RequestException as error:
                failure = f'{type(error).__name__}: {error}'
                continue
            if response.status_code in (401, 402, 403):
                raise OpenRouterAuthError(f'OpenRouter refused the call ({response.status_code}): {response.text[:500]}')
            if response.status_code in RETRY_STATUS:
                failure = f'HTTP {response.status_code}: {response.text[:300]}'
                continue
            if not response.ok:   # a request the API will never take as sent
                raise RuntimeError(f'OpenRouter rejected the call ({response.status_code}): {response.text[:1000]}')
            reply = response.json()
            if 'choices' not in reply:   # a provider error relayed with status 200
                failure = f"no choices: {json.dumps(reply.get('error', reply))[:300]}"
                continue
            choice = reply['choices'][0]
            message = choice['message']
            return message.get('reasoning') or '', message.get('content') or '', choice.get('finish_reason') or ''
        raise RuntimeError(f'OpenRouter call to {self.model_path} failed {HTTP_ATTEMPTS} times; last: {failure}')
