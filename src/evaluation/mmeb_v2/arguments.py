from dataclasses import dataclass, field
from transformers import TrainingArguments
from typing import List


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="Qwen/Qwen3-VL-Embedding-2B",
        metadata={"help": "huggingface model name or path"}
    )
    normalize: bool = field(
        default=True, 
        metadata={"help": "normalize query and passage representations"}
    )
    instruction: str = field(
        default="Represent the user's input.", 
        metadata={"help": "default instruction for the model"}
    )
    # vLLM arguments
    use_vllm: bool = field(
        default=False,
        metadata={"help": "Use vLLM API service for inference instead of transformers"}
    )
    vllm_api_url: str = field(
        default="http://192.168.9.146:9099/v1",
        metadata={"help": "vLLM API service endpoint URL"}
    )

    api_timeout: int = field(
        default=60,
        metadata={"help": "API request timeout in seconds"}
    )

@dataclass
class DataArguments:
    dataset_config: str = field(
        default=None, 
        metadata={"help": "yaml file with dataset configuration"}
    )
    data_basedir: str = field(
        default=None, 
        metadata={"help": "Expect an absolute path to the base directory of all datasets. If set, it will be prepended to each dataset path"}
    )
    encode_output_path: str = field(
        default=None, 
        metadata={"help": "encode output path"}
    )
    # (optional)
    rerank_output_path: str = field(
        default=None,
        metadata={"help": "Where to save rerank results. Default: `data_args.encode_output_path`/rerank_output"}
    )

@dataclass
class EvalArguments(TrainingArguments):
    pass

@dataclass
class RerankArguments:
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path or HF repo id for Qwen3-VL reranker."}
    )
    instruction: str = field(
        default="Given a search query, retrieve relevant candidates that answer the query.",
        metadata={"help": "Instruction passed to reranker."}
    )

    # topk setting
    topk: int = field(default=100, metadata={"help": "TopK from embedding retrieval to rerank."})

    vllm_api_url: str = field(
        default="http://192.168.9.146:9099/v1",
        metadata={"help": "vLLM API service endpoint URL"}
    )