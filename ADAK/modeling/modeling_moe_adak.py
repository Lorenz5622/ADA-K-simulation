# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
import math
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import h5py

from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers.utils import logging, add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
import torch.nn.functional as F
from .configuration_moe import MoEConfig

logger = logging.get_logger(__name__)
ACTOR_LR = 5e-4
CRITIC_LR = 5e-4
KL_PENALTY_RATIO = 0.02
ENTROPY_RATIO = 0.1
_CONFIG_FOR_DOC = "LlamaConfig"
MAX_K = 6
test_flag = False # 为true给训练使用

# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
                
class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: MoEConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = EnhancedSwitchMLP(
            config=config,
            layer_idx=layer_idx,
        )
        self.input_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        chosen_k: int  = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        residual = hidden_states

        # hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.input_norm(hidden_states)


        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        # hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.post_attention_norm(hidden_states)
        if output_router_logits:
            out = self.mlp(hidden_states, output_router_logits)
            # print(f"LlamaDecoderLayer hidden_states: {out['hidden_states']}")
            # print(f"LlamaDecoderLayer router_logits: {out['router_logits']}")
            return out
        hidden_states = self.mlp(hidden_states, output_router_logits)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class MoEPreTrainedModel(PreTrainedModel):
    config_class = MoEConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LlamaModel):
            module.gradient_checkpointing = value

LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""
@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class MoEModel(MoEPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: MoEConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value
    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_router_logits=False,
        # dynamic_k: List[int] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        collected_hidden_states = []

        
        if output_router_logits:
            all_router_logits_for_warm = []
            all_hidden_states_for_warm = []
            for idx, decoder_layer in enumerate(self.layers):
                out = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    output_router_logits=output_router_logits,
                )
                all_router_logits_for_warm.append(out['router_logits'])
                all_hidden_states_for_warm.append(out['hidden_states'])
                # print(f"MoEModel out.router_logits: {out['router_logits']}")
                # print(f"MoEModel out.hidden_states: {out['hidden_states']}")
            return {
                "all_router_logits_for_warm": all_router_logits_for_warm,
                "all_hidden_states_for_warm": all_hidden_states_for_warm,
            }
        for idx, decoder_layer in enumerate(self.layers):
            # print(idx)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                    output_router_logits=output_router_logits,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]
            # print(f"hidden_states{idx}.shape:{hidden_states.shape}")

            collected_hidden_states.append(hidden_states)

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)


        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        # print(f"return dict: {return_dict}")
        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        if test_flag:
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
                # hidden_states_all=collected_hidden_states,
                )
        else: 
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
                # hidden_states_all=collected_hidden_states,
            )

class MoEForCausalLM(MoEPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = MoEModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.saved_logits = []
        self.collected_hidden_states = []
        self.ppo_buffer = []
        self.num_layers = config.num_hidden_layers
        # self.alloc_optimizer = torch.optim.Adam(self.get_allocator_params(), lr=1e-4)

        
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
    def get_allocator_params(self):
        params = []
        for layer in self.model.layers:
            if hasattr(layer.mlp, "allocator"):
                params += list(layer.mlp.allocator.parameters())
        return params
    
    def get_actor_critic_params(self):
        params = []
        for layer in self.model.layers:
            if hasattr(layer.mlp, "actor_critic"):
                params += list(layer.mlp.actor_critic.parameters())
        return params
    
    def clear_ppo_buffer(self):
        self.ppo_buffer = []
    
    def ppo_update(self, ppo_batches):
        """
        PPO 更新 allocator 和 actor-critic 网络
        """
        clip_range = 0.2
        total_loss = 0.0
        actor_losses = []
        critic_losses = []
        
        # 1) 找到所有 allocator 参数
        alloc_params = []
        for layer in self.model.layers:
            if hasattr(layer.mlp, "allocator"):
                alloc_params += list(layer.mlp.allocator.parameters())

        if len(alloc_params) == 0:
            print("⚠ Warning: No allocator found.")
            return

        # 2) 逐 batch 计算 PPO 损失
        for batch in ppo_batches:
            layer_id = batch["layer_id"]
            state = batch["state"]             # [B,S,H]
            action = batch["action"]           # [B,S]
            old_log_prob = batch["old_log_prob"]  # [B,S]
            advantage = batch["advantage"]     # [B,S] - 修正维度
            old_alloc_logits = batch["old_alloc_logits"]
            critic_values = batch["critic_values"]  # [B,S,1]
            # 对应层的 allocator 和 actor-critic
            layer = self.model.layers[layer_id]
            allocator = layer.mlp.allocator
            actor_critic = layer.mlp.actor_critic

            # 3) 通过 allocator 重新计算 log_prob_new
            logits, probs = allocator(state)   # [B,S,K]
            log_probs = torch.log(probs + 1e-8)
            
            new_log_prob = torch.gather(
                log_probs,
                dim=-1,
                index=action.unsqueeze(-1)
            ).squeeze(-1)  # [B,S]

            # 4) 计算 PPO ratio
            ratio = torch.exp(new_log_prob - old_log_prob)

            # 5) PPO clipped 对象函数
            # 确保 advantage 和 ratio 维度一致
            if advantage.dim() == 2 and advantage.size(1) == 1:
                # 如果 advantage 是 [B,1]，扩展为 [B,S]
                advantage = advantage.expand_as(ratio)
            
            unclipped = ratio * advantage      # [B,S]
            clipped_ratio = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
            clipped = clipped_ratio * advantage

            ppo_loss = -torch.mean(torch.min(unclipped, clipped))
            
            # =====================================================
            # ★ 插入位置：在这里加入 Entropy Bonus + KL Penalty
            # =====================================================

            # --- Entropy bonus -----------------------------------
            # probs:  [B,S,K]
            # log_probs: [B,S,K]
            old_log_probs = torch.log_softmax(old_alloc_logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1).mean()    # scalar
            
            # --- KL penalty (new vs old) -------------------------
            # 需要 old_log_probs 展开成 shape [B,S,K]
            old_log_probs_full = torch.log_softmax(old_alloc_logits, dim=-1)      # [B,S,K]
            log_probs_new_full = torch.log(probs + 1e-8)                      # [B,S,K]

            kl = (probs * (log_probs_new_full - old_log_probs_full)).sum(dim=-1).mean()

            # 组合 loss（entropy 惩罚为负号，因为要最大化 entropy）
            # loss = ppo_loss - ENTROPY_RATIO * entropy + KL_PENALTY_RATIO * kl
            allocator_loss = ppo_loss + KL_PENALTY_RATIO * kl
            if layer_id == 23:
                print(f"[PPO] layer={layer_id} | "
                f"ppo_loss={ppo_loss.item():.5f} | "
                f"kl={kl.item():.5f} | "
                f"loss={allocator_loss.item():.5f}")
            total_loss += allocator_loss

            # 6) 计算 Actor-Critic 损失
            # 通过 actor-critic 网络重新计算动作概率和状态值
            _, actor_probs, new_critic_values = actor_critic(state)  # [B,S,K], [B,S,K], [B,S,1]
            
            # 计算 Actor Loss (PPO)
            log_probs = torch.log(actor_probs + 1e-8)
            new_log_prob = torch.gather(
                log_probs,
                dim=-1,
                index=action.unsqueeze(-1)
            ).squeeze(-1)  # [B,S]
            
            ratio = torch.exp(new_log_prob - old_log_prob)
            
            # 确保 advantage 和 ratio 维度一致
            if advantage.dim() == 2 and advantage.size(1) == 1:
                # 如果 advantage 是 [B,1]，扩展为 [B,S]
                advantage = advantage.expand_as(ratio)
            
            unclipped = ratio * advantage      # [B,S]
            clipped_ratio = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
            clipped = clipped_ratio * advantage
            
            actor_loss = -torch.mean(torch.min(unclipped, clipped))
            actor_losses.append((layer_id, actor_loss))
            
            # 计算 Critic Loss (MSE)
            # 确保 critic_values 和 new_critic_values 维度一致
            if critic_values.dim() == 3 and critic_values.size(2) == 1:
                critic_values = critic_values.squeeze(-1)  # [B,S]
            if new_critic_values.dim() == 3 and new_critic_values.size(2) == 1:
                new_critic_values = new_critic_values.squeeze(-1)  # [B,S]
                
            critic_loss = F.mse_loss(new_critic_values, critic_values)
            critic_losses.append((layer_id, critic_loss))

        # 7) 反向传播和参数更新
        # 更新 allocator 现在不需要了
        # self.alloc_optimizer.zero_grad()
        # total_loss.backward()
        # torch.nn.utils.clip_grad_norm_(alloc_params, 1.0)
        # self.alloc_optimizer.step()
        
        # 更新每个层的 actor 和 critic 网络
        for (layer_id, actor_loss), (_, critic_loss) in zip(actor_losses, critic_losses):
            layer = self.model.layers[layer_id]
            
            # 更新 actor 网络
            layer.mlp.actor_optimizer.zero_grad()
            actor_loss.backward()
            layer.mlp.actor_optimizer.step()

            # 更新 critic 网络
            layer.mlp.critic_optimizer.zero_grad()
            critic_loss.backward()
            layer.mlp.critic_optimizer.step()
        
        avg_actor_loss = sum(loss.item() for _, loss in actor_losses) / len(actor_losses) if actor_losses else 0.0
        avg_critic_loss = sum(loss.item() for _, loss in critic_losses) / len(critic_losses) if critic_losses else 0.0
        print(f"[PPO Allocator] Update complete. Allocator Loss = {total_loss.item():.4f}, Actor Loss = {avg_actor_loss:.4f}, Critic Loss = {avg_critic_loss:.4f}")

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_router_logits=False,
        # dynamic_k: List[int] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""
        # print("------------"+dynamic_k+"----------------")
        # print(f"dynamic_k: {dynamic_k}")
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            output_router_logits=output_router_logits,
            # dynamic_k=dynamic_k,
        )
        if output_router_logits:
            return outputs
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        ppo_all_layers = []
        for layer in self.model.layers:
            if hasattr(layer.mlp, "ppo_buffer"):
                ppo_all_layers.extend(layer.mlp.ppo_buffer)
                layer.mlp.ppo_buffer = []  # 清空，避免累积
        self.ppo_buffer.extend(ppo_all_layers)
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]
        # print(f"input_ids: {input_ids}")
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                # "dynamic_k": kwargs.get("dynamic_k"),
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past

@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LLAMA_START_DOCSTRING,
)
def top_p_sampling_batched_all_sequence(logits, top_p=0.9, temperature=1.0):
    """
    Apply Top-p sampling to every element in the sequence for each item in the batch.
    Returns the selected token indices and the corresponding threshold indices.
    
    :param logits: Logits from a language model with shape (sequence length, batch size, L)
    :param top_p: Cumulative probability threshold (float)
    :param temperature: Sampling temperature (float)
    :return: Tuple of tensors (selected token indices, threshold indices) for each position in each sequence in the batch
    """
    # Apply temperature
    logits = logits / temperature
    
    # Convert logits to probabilities
    # probabilities = torch.softmax(logits, dim=-1)
    # Sort probabilities and their indices in descending order
    sorted_probs, sorted_indices = torch.sort(logits, descending=True)

    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative_probs > top_p

    # Find the threshold indices
    threshold_indices = mask.long().argmax(dim=-1)
    threshold_mask = torch.nn.functional.one_hot(threshold_indices, num_classes=sorted_indices.size(-1)).bool()
    
    mask = mask & ~threshold_mask
    sorted_indices = torch.where(mask, -1, sorted_indices)
    sorted_probs = torch.where(mask, 0.0, sorted_probs)   
    return sorted_probs, sorted_indices

class LlamaForSequenceClassification(MoEPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


# class AdaKAllocator(nn.Module):
#     """
#     Per-layer allocator W_alloc in the paper.
#     Computes P(k | h) for dynamic top-k expert routing.
#     """
#     def __init__(self, hidden_size, max_k, hidden_dim = None):
#         super().__init__()
#         self.max_k = max_k
#         if hidden_dim is None:
#             hidden_dim = hidden_size  # 默认隐藏层=H
#         self.alloc = nn.Sequential(
#             nn.Linear(hidden_size, hidden_dim),
#             nn.GELU(),                    # 激活函数
#             nn.Linear(hidden_dim, max_k)  # 输出 logits
#         )
        

#     def forward(self, hidden_states):
#         """
#         hidden_states: [B,S,H]
#         return:
#             alloc_logits: [B,S,K]
#             alloc_probs:  [B,S,K]
#         """
#         logits = self.alloc(hidden_states)      # [B,S,K]
#         logits = torch.clamp(logits, -30, 30)   # avoid softmax overflow
#         probs = F.softmax(logits, dim=-1)
#         probs = torch.clamp(probs, 1e-8, 1.0)

#         # logits = logits.to(hidden_states.dtype)
#         # probs = probs.to(hidden_states.dtype)
#         return logits, probs

def dynamic_topk_indices_and_scores(router_logits, sampled_k):
    """
    router_logits: [B, S, E]
    sampled_k: [B, S]

    返回：
        topk_indices_full: [B, S, E]
            若 expert e 被选中 → e
            未选中 → -1
        selected_scores: [B, S, E]
            选中专家保留 logits 分数
            未选中 → 0
    """

    B, S, E = router_logits.shape
    logits = router_logits.clone()
    max_k = min(sampled_k.max().item(), MAX_K)

    # 保存 top-k 轮次选中 expert index，长度 max_k，每个是 [B,S]
    all_indices = []
    active_mask = sampled_k > 0

    for _ in range(max_k):
        # 当前最大 expert id
        idx = logits.argmax(dim=-1)  # [B, S]

        idx = torch.where(active_mask, idx, torch.full_like(idx, -1))
        all_indices.append(idx)

        # 屏蔽被选中的 expert
        safe = idx.clone()
        safe[safe == -1] = 0
        logits.scatter_(dim=-1, index=safe.unsqueeze(-1), value=float('-inf'))

        sampled_k -= 1
        active_mask = sampled_k > 0

    # all_indices → [B,S,max_k]
    topk_indices = torch.stack(all_indices, dim=-1)

    # ---- 你要求的对位格式 ----
    # 初始化为 -1
    topk_indices_full = torch.full(
        (B, S, E),
        fill_value=-1,
        device=router_logits.device,
        dtype=torch.long
    )

    # 对于每一轮，将 selected expert index 写到对应位置
    for i in range(max_k):
        idx_i = topk_indices[..., i]  # [B,S]
        valid = idx_i != -1

        if valid.any():
            # scatter 的 index 必须转成 [B,S,1]
            safe_idx = idx_i.clone()
            safe_idx = torch.clamp(safe_idx, 0, E-1)

            topk_indices_full.scatter_(
                dim=-1,
                index=safe_idx.unsqueeze(-1),
                src=safe_idx.unsqueeze(-1)   # expert_id 对位写入
            )

    # ---- selected_scores: 对位填概率 ----
    selected_scores = torch.where(
        topk_indices_full != -1,
        router_logits,
        torch.zeros_like(router_logits)
    )

    return topk_indices_full, selected_scores

class EnhancedSwitchMLP(nn.Module):
    """
    Ada-K Routing MLP (MoE FFN)
    - frozen router W_r
    - trainable allocator W_alloc
    - dynamic top-k routing controlled by allocator
    - PPO trajectory storage for allocator
    """
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_num = layer_idx
        self.hidden_size  = config.hidden_size
        self.num_experts  = config.num_experts
        self.max_k        = MAX_K     # 论文中的 K_max
        self.k_counter = []
        self.actor_critic = ActorCritic(config.hidden_size, max_k=config.num_experts)
        self.actor_optimizer = torch.optim.Adam(self.actor_critic.actor.parameters(), lr=ACTOR_LR)
        self.critic_optimizer = torch.optim.Adam(self.actor_critic.critic.parameters(), lr=CRITIC_LR)
        # Optimizers for actor and critic networks
        # =============================
        # Experts (全部冻结)
        # =============================
        self.experts = torch.nn.ModuleList()
        for i in range(config.num_experts):
            self.experts.append(LlamaMLP(config.hidden_size, config.intermediate_size, config.hidden_act))
        for expert in self.experts:
            for p in expert.parameters():
                p.requires_grad = False

        # =============================
        # Router (W_r) —— MoE routing
        # 冻结，用于给专家排序
        # =============================
        self.router = torch.nn.Linear(config.hidden_size, config.num_experts, bias=False)
        for p in self.router.parameters():
            p.requires_grad = False

        # =============================
        # Allocator (W_alloc) —— 论文新增
        # 可训练，通过 PPO 更新
        # =============================
        # self.allocator = AdaKAllocator(
        #     hidden_size=self.hidden_size,
        #     max_k=self.max_k,
        #     hidden_dim=50,
        # )
        
        # =============================
        # Actor-Critic Network
        # =============================
        self.actor_critic = ActorCritic(
            hidden_size=self.hidden_size,
            max_k=self.max_k,
            hidden_dim=50,
        )

        # =============================
        # PPO Buffer（每层独有）
        # =============================
        self.ppo_buffer = []

    def compute_pseudo_k_top_p(self, router_logits, top_p=0.9, temperature=1.0):
        """
        router_logits: [B,S,E]   每层 Router 输出
        输出 pseudo_k: [B,S]     每个 token 的 expert count（k 值）
        """

        # 你的函数要求: logits shape = (seq, batch, E)
        # 但 router_logits 是 (B,S,E)，需要转换
        route = router_logits.permute(1, 0, 2).contiguous()  # -> [S,B,E]

        sorted_probs, sorted_indices = top_p_sampling_batched_all_sequence(
            route, top_p=top_p, temperature=temperature
        )   # sorted_probs: [S,B,E], sorted_indices: [S,B,E]

        # cumulative probabilities (你函数内部 mask 了 top-p外的值)
        cum = torch.cumsum(sorted_probs, dim=-1)             # [S,B,E]

        # 找到 cum >= top_p 的最小 index
        mask = cum >= top_p                                 # [S,B,E]
        pseudo_k = mask.float().argmax(dim=-1) + 1          # [S,B]

        # 转回 [B,S]
        pseudo_k = pseudo_k.permute(1,0).contiguous()
        return pseudo_k
    # ==========================================================
    # Forward：论文 Ada-K routing 的完整执行流程
    # ==========================================================
    def forward(self, hidden_states, output_router_logits=False):
        """
        hidden_states: [B,S,H]
        returns:
            routed_output: [B,S,H]
            （PPO 数据已写入 self.ppo_buffer）
        """
        
        B, S, H = hidden_states.shape

        # ------------------------------------------------------
        # 1. Router logits —— 冻结，只作为专家排序依据（论文 §3.3）
        # ------------------------------------------------------
        router_logits = self.router(hidden_states).float()  # [B,S,E]
        router_logits = torch.nn.functional.softmax(router_logits, dim=2)
        if output_router_logits:
            return {
                "hidden_states": hidden_states.detach(),
                "router_logits": router_logits.detach(),
            }
        # ------------------------------------------------------
        # 2. 使用Actor-Critic获取动作概率和状态值
        # ------------------------------------------------------
        actor_logits, actor_probs, critic_values = self.actor_critic(hidden_states)   # [B,S,K], [B,S,K], [B,S,1]
        sampled_k = actor_probs.argmax(dim=-1)


        # 处理sampled_k
        sampled_k_action = sampled_k.clone()
        true_k = torch.clamp(sampled_k_action + 1, max=self.max_k)
        topk_indices, topk_scores = dynamic_topk_indices_and_scores(router_logits=router_logits, sampled_k=true_k,)
        
        if self.training:
            # 在线统计直方图，不保存所有 token 的 k
            if not hasattr(self, "k_hist"):
                self.k_hist = torch.zeros(self.max_k, device=hidden_states.device)

            # sampled_k ∈ [0..K-1]
            bcount = torch.bincount(sampled_k_action.reshape(-1), minlength=self.max_k)
            self.k_hist += bcount

        # ------------------------------------------------------
        # 4 new. 模仿原模型的output方式处理
        # ------------------------------------------------------
        hidden_flat = hidden_states.view(-1, hidden_states.size(2)) 
        topk_weights = topk_scores.view(-1, topk_scores.size(2)) 
        topk_ind = topk_indices.view(-1, topk_indices.size(2))

        output_total = torch.zeros_like(hidden_flat).to(hidden_flat)
        for expert_num, expert in enumerate(self.experts):
            sample_ind, expert_ind = torch.where(topk_ind == expert_num) 
            hidden = hidden_flat[sample_ind.unsqueeze(1), :] 
            expert_output = expert(hidden)
            output_total[sample_ind] += torch.mul(expert_output.squeeze(1), topk_weights[sample_ind,expert_ind].unsqueeze(1))


        output_total = output_total.view(B, S, H)

        # ------------------------------------------------------
        # 6. 保存 PPO 信息 —— 分层存储
        # ------------------------------------------------------
        logits = actor_logits            # [B,S,K]
        probs  = actor_probs             # [B,S,K]
        log_probs = torch.log(probs + 1e-8)

        old_log_prob = torch.gather(
            log_probs,
            dim=-1,
            index=sampled_k.unsqueeze(-1)
        ).squeeze(-1)  # [B,S]

        # 保存完整 PPO 轨迹
        self.ppo_buffer.append({
            "layer_id": self.layer_num,
            "state": hidden_states.detach().float(),   # [B,S,H]
            "action": sampled_k.detach(),      # [B,S]
            "old_log_prob": old_log_prob.detach().float(),  # [B,S]
            "old_alloc_logits": actor_logits.detach(),
            "critic_values": critic_values.detach().float(),  # [B,S,1]
        })
        
        return output_total

class ActorCritic(nn.Module):
    """
    Actor-Critic module for PPO training of the allocator.
    - Actor: allocator network that outputs action probabilities
    - Critic: value network that estimates state values
    """
    def __init__(self, hidden_size, max_k, hidden_dim=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_k = max_k
        
        if hidden_dim is None:
            hidden_dim = hidden_size
            
        # Actor network (policy network)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, max_k)
        )
        
        # Critic network (value network)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, hidden_states):
        """
        hidden_states: [B,S,H]
        returns:
            actor_logits: [B,S,K] - logits for action probabilities
            actor_probs: [B,S,K] - action probabilities
            critic_values: [B,S,1] - estimated state values
        """
        # Actor forward pass
        actor_logits = self.actor(hidden_states)  # [B,S,K]
        actor_logits = torch.clamp(actor_logits, -30, 30)  # avoid softmax overflow
        actor_probs = F.softmax(actor_logits, dim=-1)
        actor_probs = torch.clamp(actor_probs, 1e-8, 1.0)
        
        # Critic forward pass
        critic_values = self.critic(hidden_states)  # [B,S,1]
        
        return actor_logits, actor_probs, critic_values
        
    def get_action_probabilities(self, hidden_states):
        """
        Get action probabilities for given states
        hidden_states: [B,S,H]
        returns:
            probs: [B,S,K]
        """
        _, probs, _ = self.forward(hidden_states)
        return probs
        
    def get_state_values(self, hidden_states):
        """
        Get state values for given states
        hidden_states: [B,S,H]
        returns:
            values: [B,S,1]
        """
        _, _, values = self.forward(hidden_states)
        return values
        
    def evaluate_actions(self, hidden_states, actions):
        """
        Evaluate actions for given states
        hidden_states: [B,S,H]
        actions: [B,S]
        returns:
            log_probs: [B,S]
            entropy: scalar
            values: [B,S,1]
        """
        actor_logits, actor_probs, values = self.forward(hidden_states)
        
        # Calculate log probabilities
        log_probs = torch.log(actor_probs + 1e-8)
        action_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=actions.unsqueeze(-1)
        ).squeeze(-1)  # [B,S]
        
        # Calculate entropy
        entropy = -(actor_probs * log_probs).sum(dim=-1).mean()  # scalar
        
        return action_log_probs, entropy, values