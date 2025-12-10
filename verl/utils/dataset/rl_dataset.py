# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import copy
import logging
import os
import re
from collections import defaultdict
from typing import List, Optional, Union

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)

PROMPT2 = """Read and analyze the following text, then you need to provide your reasoning within <think></think> tags. Finally, generate a comprehensive understanding of this text."""

PROMPT3 = """Analyze the following text. Begin your response with "THINKING PROCESS:" and show all your reasoning steps, then conclude with "FINAL UNDERSTANDING:" and your comprehensive analysis."""


def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, *dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


RETRIEVAL_INSTRUCTIONS = {
    "general": "Given a query, retrieve relevant documents that answer the question:",
    "nli": "Given a premise, retrieve a hypothesis that is entailed by the premise:",
    "similarity": "Retrieve semantically similar text:",
    "question_answer": "Given a question, retrieve passages that answer the question:",
    "fact_verification": "Given a claim, retrieve documents that support or refute the claim:",
    "duplicate_detection": "Given a question, retrieve questions that are semantically equivalent to the given question:",
    "msmarco": "Given a web search query, retrieve relevant passages that answer the query:",
    "hotpot_qa": "Given a multi-hop question, retrieve documents that can help answer the question:",
    "fever": "Given a claim, retrieve documents that support or refute the claim:",
    "scifact": "Given a scientific claim, retrieve documents that support or refute the claim:",
    "arguana": "Given a claim, find documents that refute the claim:",
    "trec_covid": "Given a query on COVID-19, retrieve documents that answer the query:",
    "fiqa": "Given a financial question, retrieve user replies that best answer the question:",
    "dbpedia": "Given a query, retrieve relevant entity descriptions from DBPedia:",
    "touche2020": "Given a question, retrieve detailed and persuasive arguments that answer the question:",
    "climate_fever": "Given a claim about climate change, retrieve documents that support or refute the claim:",
    "scidocs": "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper:",
}


class RLHFDataset(Dataset):
    """
    Load and preprocess RLHF data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, (List, ListConfig)):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(
            config.get("cache_dir", "~/.cache/verl/rlhf")
        )
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)

        self.num_workers = config.get(
            "filter_overlong_prompts_workers", max(1, os.cpu_count() // 4)
        )
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        # training mode: 'supervised' or 'unsupervised'
        self.train_mode = config.get("train_mode", "supervised")
        self.serialize_dataset = False
        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = (
            self.data_files if not use_origin_parquet else self.original_data_files
        )
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(
                src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm
            )

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)[
                "train"
            ]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            processor = self.processor
            prompt_key = self.prompt_key
            image_key = self.image_key
            video_key = self.video_key

            if processor is not None:
                from verl.utils.dataset.vision_utils import process_image, process_video

                def doc2len(doc) -> int:
                    messages = self._build_messages(doc)
                    raw_prompt = self.processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False
                    )
                    images = (
                        [process_image(image) for image in messages.pop(image_key)]
                        if image_key in messages
                        else None
                    )
                    videos = (
                        [process_video(video) for video in messages.pop(video_key)]
                        if video_key in messages
                        else None
                    )

                    return len(
                        processor(text=[raw_prompt], images=images, videos=videos)[
                            "input_ids"
                        ][0]
                    )

            else:

                def doc2len(doc) -> int:
                    return len(
                        tokenizer.apply_chat_template(
                            doc[prompt_key], add_generation_prompt=True
                        )
                    )

            self.dataframe = self.dataframe.filter(
                lambda doc: doc2len(doc) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(self.dataframe)}")

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(
                use_origin_parquet=True
            )  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(
                r"old dataloader ckpt file is used, please train from scratch for better ckpt performance"
            )

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                segments = re.split("(<image>|<video>)", content)
                segments = [item for item in segments if item != ""]
                for segment in segments:
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages

    def convert_to_messages(self, text):
        return [
            {
                "role": "system",
                "content": PROMPT2,
            },
            {"role": "user", "content": f"{text}"},
        ]
    
    def build_next_token_prediction_prompt(self, text):
        """
        Build next token prediction format prompt.
        Format: PROMPT2 + "\n" + text + "\n" + "<|endoftext|>"
        Note: In qwen3, <|endoftext|> is pad_token, not eos_token.
        """
        prompt_template = f"{PROMPT2}\n{text}\n<|endoftext|>"
        return prompt_template
    
    def get_eot_token_id(self, tokenizer):
        """
        Get the token ID for <|endoftext|>.
        In qwen3, <|endoftext|> is pad_token, not eos_token.
        """
        # First try pad_token_id (for qwen3)
        if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
            return tokenizer.pad_token_id
        # Fallback: try to convert <|endoftext|> string
        elif hasattr(tokenizer, "convert_tokens_to_ids"):
            try:
                return tokenizer.convert_tokens_to_ids("<|endoftext|>")
            except:
                pass
        return None
    
    def extract_text_from_messages(self, messages):
        """
        Extract text content from messages for next token prediction format.
        Assumes messages is a list of dicts with 'role' and 'content' keys.
        For multimodal content, extracts text parts only.
        """
        text_parts = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                # Multimodal content: extract text parts
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
            elif isinstance(content, str):
                text_parts.append(content)
        # Join all text parts, typically user content is what we want
        # For simplicity, we'll use the last non-system message's content
        user_content = ""
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    user_content = " ".join([
                        item.get("text", "") for item in content 
                        if isinstance(item, dict) and item.get("type") == "text"
                    ])
                elif isinstance(content, str):
                    user_content = content
                break
        return user_content
    
    def truncate_with_eot_preserved(self, token_ids, max_length, tokenizer, truncation="right"):
        """
        Truncate token_ids while preserving <|endoftext|> token at the end.
        Only truncates the content part (query/positive/negative text), keeping prefix and <|endoftext|>.
        Note: In qwen3, <|endoftext|> is pad_token, not eos_token.
        """
        eot_token_id = self.get_eot_token_id(tokenizer)
        
        if eot_token_id is None:
            # No EOT token found, use standard truncation
            if truncation == "left":
                return token_ids[-max_length:]
            elif truncation == "right":
                return token_ids[:max_length]
            elif truncation == "middle":
                left_half = max_length // 2
                right_half = max_length - left_half
                return token_ids[:left_half] + token_ids[-right_half:]
            else:
                return token_ids
        
        if len(token_ids) <= max_length:
            # No truncation needed, but ensure EOT is at the end
            if len(token_ids) == 0 or token_ids[-1] != eot_token_id:
                # Add EOT if not present, but check if we have space
                if len(token_ids) < max_length:
                    return token_ids + [eot_token_id]
                else:
                    # Replace last token with EOT
                    return token_ids[:-1] + [eot_token_id]
            return token_ids
        
        # Check if EOT token exists at the end
        has_eot_at_end = len(token_ids) > 0 and token_ids[-1] == eot_token_id
        
        if has_eot_at_end:
            # Preserve EOT token, truncate content from the right
            # Structure: [prefix_tokens (PROMPT2 + \n)] + [content_tokens (text)] + [\n] + [eot_token]
            # We truncate only the content part (text), keeping prefix and EOT
            
            # Reserve space for EOT token
            available_length = max_length - 1
            
            if truncation == "right":
                # Keep prefix and EOT, truncate content from right
                # Keep first (available_length) tokens + EOT
                truncated = token_ids[:available_length] + [eot_token_id]
                return truncated
            elif truncation == "left":
                # Keep EOT, truncate from left (but this might remove prefix)
                # Keep last (available_length) tokens + EOT
                truncated = token_ids[-(available_length):-1] + [eot_token_id]
                return truncated
            elif truncation == "middle":
                # Keep prefix start, EOT, and truncate middle content
                # Estimate prefix length (PROMPT2 + \n, roughly 30-50 tokens)
                prefix_estimate = min(50, len(token_ids) // 4)  # Rough estimate
                left_half = (available_length - prefix_estimate) // 2 + prefix_estimate
                right_half = available_length - left_half
                if right_half > 0:
                    truncated = token_ids[:left_half] + token_ids[-right_half-1:-1] + [eot_token_id]
                else:
                    truncated = token_ids[:available_length] + [eot_token_id]
                return truncated
            else:
                return token_ids
        else:
            # No EOT at end, add it and truncate if needed
            if truncation == "right":
                truncated = token_ids[:max_length-1] + [eot_token_id]
            elif truncation == "left":
                truncated = token_ids[-(max_length-1):] + [eot_token_id]
            elif truncation == "middle":
                left_half = (max_length - 1) // 2
                right_half = max_length - 1 - left_half
                truncated = token_ids[:left_half] + token_ids[-right_half:] + [eot_token_id]
            else:
                truncated = token_ids + [eot_token_id] if len(token_ids) < max_length else token_ids
            return truncated

    def convert_to_messages_with_instruction(self, text, instruction_type):
        DEFAULT_RETRIEVAL_INSTRUCTION = RETRIEVAL_INSTRUCTIONS["general"]
        if instruction_type in RETRIEVAL_INSTRUCTIONS:
            instruction = RETRIEVAL_INSTRUCTIONS[instruction_type]
        else:
            instruction = DEFAULT_RETRIEVAL_INSTRUCTION
        return [
            {
                "role": "system",
                "content": PROMPT2,
            },
            {"role": "user", "content": f"{instruction} {text}"},
        ]

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        messages = self._build_messages(row_dict)
        model_inputs = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            raw_prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            multi_modal_data = {}

            images = None
            if (
                self.image_key in row_dict
                and row_dict.get(self.image_key, None) is not None
            ):
                images = [
                    process_image(image) for image in row_dict.pop(self.image_key)
                ]
                multi_modal_data["image"] = images

            videos = None
            if (
                self.video_key in row_dict
                and row_dict.get(self.video_key, None) is not None
            ):
                videos = [
                    process_video(video) for video in row_dict.pop(self.video_key)
                ]
                multi_modal_data["video"] = [video.numpy() for video in videos]

            model_inputs = self.processor(
                text=[raw_prompt], images=images, videos=videos, return_tensors="pt"
            )

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data
            row_dict["multi_modal_inputs"] = dict(model_inputs)

            # second_per_grid_ts isn't used for training, just for mrope
            row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            # Extract text from messages for next token prediction format
            main_text = self.extract_text_from_messages(messages)
            raw_prompt = self.build_next_token_prediction_prompt(main_text)
            model_inputs = self.tokenizer(
                raw_prompt, return_tensors="pt", add_special_tokens=False
            )
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")
            
            # Truncate while preserving <|endoftext|>
            if input_ids.shape[1] > self.max_prompt_length:
                truncated_ids = self.truncate_with_eot_preserved(
                    input_ids[0].tolist(), 
                    self.max_prompt_length, 
                    self.tokenizer, 
                    self.truncation
                )
                input_ids = torch.tensor([truncated_ids], dtype=input_ids.dtype)
                # Generate attention_mask: all tokens are valid (1), including <|endoftext|> at the end
                # Note: In qwen3, <|endoftext|> is pad_token, but we want to keep it unmasked
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            
            if self.train_mode == "supervised":

                query = row_dict["multi_input"]["query"]
                negative_doc = row_dict["multi_input"]["negative_document"]

                # Build next token prediction format prompts
                query_raw_prompt = self.build_next_token_prediction_prompt(query)
                negative_doc_raw_prompt = self.build_next_token_prediction_prompt(negative_doc)

                query_inputs = self.tokenizer(
                    query_raw_prompt, return_tensors="pt", add_special_tokens=False
                )
                negative_doc_inputs = self.tokenizer(
                    negative_doc_raw_prompt,
                    return_tensors="pt",
                    add_special_tokens=False,
                )

                query_input_ids = query_inputs.pop("input_ids")
                query_attention_mask = query_inputs.pop("attention_mask")
                negative_doc_input_ids = negative_doc_inputs.pop("input_ids")
                negative_doc_attention_mask = negative_doc_inputs.pop("attention_mask")
                
                # Truncate query and negative_doc while preserving <|endoftext|>
                if query_input_ids.shape[1] > self.max_prompt_length:
                    truncated_query_ids = self.truncate_with_eot_preserved(
                        query_input_ids[0].tolist(),
                        self.max_prompt_length,
                        self.tokenizer,
                        self.truncation
                    )
                    query_input_ids = torch.tensor([truncated_query_ids], dtype=query_input_ids.dtype)
                    # Generate attention_mask: all tokens are valid (1), including <|endoftext|> at the end
                    query_attention_mask = torch.ones_like(query_input_ids, dtype=torch.long)
                
                if negative_doc_input_ids.shape[1] > self.max_prompt_length:
                    truncated_neg_ids = self.truncate_with_eot_preserved(
                        negative_doc_input_ids[0].tolist(),
                        self.max_prompt_length,
                        self.tokenizer,
                        self.truncation
                    )
                    negative_doc_input_ids = torch.tensor([truncated_neg_ids], dtype=negative_doc_input_ids.dtype)
                    # Generate attention_mask: all tokens are valid (1), including <|endoftext|> at the end
                    negative_doc_attention_mask = torch.ones_like(negative_doc_input_ids, dtype=torch.long)
            else:
                query_input_ids = None
                query_attention_mask = None
                negative_doc_input_ids = None
                negative_doc_attention_mask = None

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        # Also process query and negative_doc inputs to ensure consistent lengths
        if self.train_mode == "supervised":
            query_input_ids, query_attention_mask = verl_F.postprocess_data(
                input_ids=query_input_ids,
                attention_mask=query_attention_mask,
                max_length=self.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.truncation,
            )

            negative_doc_input_ids, negative_doc_attention_mask = (
                verl_F.postprocess_data(
                    input_ids=negative_doc_input_ids,
                    attention_mask=negative_doc_attention_mask,
                    max_length=self.max_prompt_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    left_pad=True,
                    truncation=self.truncation,
                )
            )

        if (
            self.processor is not None
            and "Qwen2VLImageProcessor"
            in self.processor.image_processor.__class__.__name__
        ):
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)

        else:
            position_ids = compute_position_id_with_mask(attention_mask)
            if self.train_mode == "supervised":
                query_position_ids = compute_position_id_with_mask(query_attention_mask)
                negative_doc_position_ids = compute_position_id_with_mask(
                    negative_doc_attention_mask
                )

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        if self.train_mode == "supervised":
            row_dict["query_input_ids"] = query_input_ids[0]
            row_dict["query_attention_mask"] = query_attention_mask[0]
            row_dict["query_position_ids"] = query_position_ids[0]
            row_dict["negative_doc_input_ids"] = negative_doc_input_ids[0]
            row_dict["negative_doc_attention_mask"] = negative_doc_attention_mask[0]
            row_dict["negative_doc_position_ids"] = negative_doc_position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if self.train_mode == "supervised":
            query_raw_prompt_ids = self.tokenizer.encode(
                query_raw_prompt, add_special_tokens=False
            )
            negative_doc_raw_prompt_ids = self.tokenizer.encode(
                negative_doc_raw_prompt, add_special_tokens=False
            )

        # Truncate while preserving <|endoftext|>
        if len(raw_prompt_ids) > self.max_prompt_length:
            raw_prompt_ids = self.truncate_with_eot_preserved(
                raw_prompt_ids, self.max_prompt_length, self.tokenizer, self.truncation
            )

        # Also process query and negative_doc raw_prompt_ids
        if self.train_mode == "supervised":
            if len(query_raw_prompt_ids) > self.max_prompt_length:
                query_raw_prompt_ids = self.truncate_with_eot_preserved(
                    query_raw_prompt_ids, self.max_prompt_length, self.tokenizer, self.truncation
                )

            if len(negative_doc_raw_prompt_ids) > self.max_prompt_length:
                negative_doc_raw_prompt_ids = self.truncate_with_eot_preserved(
                    negative_doc_raw_prompt_ids, self.max_prompt_length, self.tokenizer, self.truncation
                )

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        if self.train_mode == "supervised":
            row_dict["query_raw_prompt_ids"] = query_raw_prompt_ids
            row_dict["negative_doc_raw_prompt_ids"] = negative_doc_raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get(
            "interaction_kwargs", {}
        )
        need_tools_kwargs = row_dict.get("extra_info", {}).get(
            "need_tools_kwargs", self.need_tools_kwargs
        )
        if need_tools_kwargs and not tools_kwargs:
            logger.warning(
                "tools_kwargs is empty for index {}, data source: {}",
                index,
                row_dict["data_source"],
            )
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()
