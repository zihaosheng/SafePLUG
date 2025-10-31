#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoConfig, AutoModelForCausalLM, LlamaConfig,
                          LlamaForCausalLM, LlamaModel)
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..safeplug_arch import LlavaMetaForCausalLM, LlavaMetaModel


class LlavaConfig(LlamaConfig):
    model_type = "safeplug"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):  # 暂时的理解是LlavaMetaModel提供了vision_tower和mm_projector，LlamaModel提供了LLM的模型结构
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)  # TODO 这不是常规的 super() 用法，而是跳过了 LlamaForCausalLM，直接从 LlavaMetaForCausalLM 开始找。

        self.model = LlavaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        region_masks: Optional[List[torch.Tensor]] = None,
        valid_region_masks_bool: Optional[List[torch.BoolTensor]] = [],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if inputs_embeds is None:  # TODO: 这里很重要，因为generate的时候会重复调用forward，而forward的时候inputs_embeds不是None，但input_ids是None，所以会导致inputs_embeds也会变成None
            (
                input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids, attention_mask, past_key_values, labels, images, region_masks, valid_region_masks_bool
            )
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        # print('==========>attention_mask.shape',attention_mask.shape)
        # if inputs_embeds is not None:
        #     print('==========>inputs_embeds.shape',inputs_embeds.shape)
        # else:
        #     print('==========>inputs_embeds is None')
        #     print('==========>input_ids.shape',input_ids.shape)
        # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]  # 这里outputs[0]是last_hidden_states是[batch_size, seq_len, hidden_size]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            # --------------to ignore invalid region prompt sample
            mask = (shift_labels != -100).any(dim=1)  
            # print('mask11',mask)
            shift_labels = shift_labels[mask]
            shift_logits = shift_logits[mask]
            # --------------

            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        if self.training:
            output_hidden_states = outputs.hidden_states
        else:
            output_hidden_states = hidden_states

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            # past_key_values=None,
            # hidden_states=output_hidden_states,  
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # @torch.no_grad()
    # def generate(
    #     self, 
    #     input_ids: Optional[torch.Tensor] = None,
    #     images: Optional[torch.Tensor] = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     region_masks: Optional[List[torch.Tensor]] = None,
    #     valid_region_masks_bool: Optional[List[torch.BoolTensor]] = [],
    #     past_key_values: Optional[List[torch.FloatTensor]] = None,
    #     labels: Optional[torch.LongTensor] = None,
    #     **kwargs
    # ):
    #     print('******************************************************')
    #     print('==========>attention_mask.shape',attention_mask.shape)
    #     print('==========>input_ids.shape',input_ids.shape)
    #     if images is not None:
    #         (
    #             input_ids,
    #             attention_mask,
    #             past_key_values,
    #             inputs_embeds,
    #             labels,
    #         ) = self.prepare_inputs_labels_for_multimodal(
    #             input_ids, attention_mask, past_key_values, labels, images, region_masks, valid_region_masks_bool
    #         )
    #         print('==========>attention_mask.shape',attention_mask.shape)
    #         print('==========>inputs_embeds.shape',inputs_embeds.shape)
    #     else:
    #         inputs_embeds = self.get_model().embed_tokens(input_ids)
    #     print('******************************************************')
        
    #     return super().generate(  # 这里应该会调用prepare_inputs_for_generation，然后会调用prepare_inputs_labels_for_multimodal
    #         attention_mask=attention_mask,
    #         inputs_embeds=inputs_embeds,
    #         **kwargs
    #     )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        images=None,
        region_masks=[],
        valid_region_masks_bool=[],
        **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": images,
                "region_masks": region_masks,
                "valid_region_masks_bool": valid_region_masks_bool,
            }
        )
        return model_inputs


AutoConfig.register("safeplug", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
