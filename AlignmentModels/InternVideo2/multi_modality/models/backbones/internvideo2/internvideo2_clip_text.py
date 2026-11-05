import logging
import numpy as np
import torch
import torch.nn.functional as F
from peft import get_peft_model, LoraConfig, TaskType
from torch import nn
from transformers import LlamaForCausalLM, LlamaConfig

from transformers import LlamaTokenizer

logger = logging.getLogger(__name__)


class LLaMA(nn.Module):
    def __init__(
            self,
            use_flash_attn: bool = True,
            transformer_width: int = 4096,
            llama_path: str = None,
            use_lora: bool = True,
            clip_embed_dim: int = 768,
        ):
        super().__init__()
        
        self.use_flash_attn = use_flash_attn
        self.transformer_width = transformer_width
        
        """ text encoder of InternVL """
        llama_config = LlamaConfig.from_pretrained(llama_path, local_files_only=True)
        llama_config.causal = True
        llama_config.use_flash_attention = use_flash_attn
        # model = LlamaForCausalLM.from_pretrained(  # LLAMA model
        #     llama_path, torch_dtype=torch.float16, config=llama_config, local_files_only=True)
        model = LlamaForCausalLM(config=llama_config)
        if not use_lora:
            self.transformer = model.model
        else:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.1)
            model = get_peft_model(model, peft_config)
            self.transformer = model.base_model.model.model
        
        self.transformer.gradient_checkpointing = True
        self.text_projection = nn.Parameter(torch.empty(transformer_width, clip_embed_dim))
            
    def forward(self, text):
        text_key_padding_mask = text > 0
        
        x = self.transformer(input_ids=text, attention_mask=text_key_padding_mask).last_hidden_state
        x = x[torch.arange(x.shape[0]), text_key_padding_mask.sum(1) - 1]
        x = x @ self.text_projection

        return x


class Tokenizer(nn.Module):
    """Chinese-Alpaca LLaMA tokenizer wrapper, reproducing what the upstream
    code produced under transformers 4.x:

        [BOS=1] summarize:<text> [EOS=2] [PAD=0 ...]   (right-padded to 80)

    LLaMA.forward() builds the attention mask as `text > 0` and reads the
    hidden state at index `(text > 0).sum(1) - 1`, i.e. the EOS position, so
    pad id must be 0 (= <unk>) and padding must be on the right.

    transformers 5.x changed LlamaTokenizer defaults (tokenizers backend,
    `add_bos_token=False`, `padding_side="left"`), which silently drops the
    BOS token and breaks the EOS lookup. Everything below is therefore set
    explicitly, and the result is asserted once at construction time.
    """

    def __init__(self, tokenizer_path="your_model_path/chinese_alpaca_lora_7b"):
        super(Tokenizer, self).__init__()
        self.tokenizer = LlamaTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True,
            legacy=False,
            add_bos_token=True,
            add_eos_token=True,
        )
        self.tokenizer.add_bos_token = True
        self.tokenizer.add_eos_token = True
        self.tokenizer.pad_token = self.tokenizer.unk_token  # upstream `" "` -> resolves to <unk> (id 0)
        self.tokenizer.padding_side = "right"

        ids = self.forward(["sanity check"])[0].tolist()
        assert ids[0] == self.tokenizer.bos_token_id == 1, f"BOS missing: {ids[:8]}"
        assert self.tokenizer.pad_token_id == 0, f"pad_token_id must be 0, got {self.tokenizer.pad_token_id}"
        n_real = sum(1 for i in ids if i > 0)
        assert ids[n_real - 1] == self.tokenizer.eos_token_id == 2, f"EOS not at last real position: {ids[:n_real + 2]}"
        assert all(i == 0 for i in ids[n_real:]), "padding must be on the right"

    def forward(self, text):
        text = ["summarize:" + item for item in text]
        text = self.tokenizer(text, return_tensors="pt", max_length=80, truncation=True, padding="max_length").input_ids
        return text
