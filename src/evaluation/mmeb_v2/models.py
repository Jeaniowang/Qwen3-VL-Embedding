from typing import Dict, Optional, Any, List, Union, Literal, cast
import os

import torch
import torch.distributed as dist
from torch import nn, Tensor



try:
    from openai import OpenAI
    from openai._types import NOT_GIVEN, NotGiven
    from openai.types.chat import ChatCompletionMessageParam
    from openai.types.create_embedding_response import CreateEmbeddingResponse
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class Qwen3VLEmbedderVLLM:
    """Qwen3VL Embedder using vLLM Embedding API service for inference.
    
    Refer to d.py for the reference implementation using OpenAI client.
    """

    def __init__(
        self,
        model_name_or_path: str,
        default_instruction: str = "Represent the user's input.",
        vllm_api_url: Optional[str] = None,
        **kwargs
    ):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI Python client is not installed. Please install it to use this feature.")

        self.model_name_or_path = model_name_or_path
        self.default_instruction = default_instruction
        
        # Default to vLLM API server base URL (without /v1 suffix, OpenAI client adds it)
        base_url = vllm_api_url or "http://192.168.9.146:9099/v1"
        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key="EMPTY",  # vLLM doesn't require API key
            base_url=base_url,
        )

    def _create_chat_embeddings(
        self,
        messages: list[ChatCompletionMessageParam],
        model: str,
        encoding_format: Literal["base64", "float"] | NotGiven = NOT_GIVEN,
        continue_final_message: bool = False,
        add_special_tokens: bool = False,
    ) -> CreateEmbeddingResponse:
        """
        Convenience function for accessing vLLM's Chat Embeddings API,
        which is an extension of OpenAI's existing Embeddings API.
        Reference: d.py create_chat_embeddings function.
        """
        # return cast(
        #     CreateEmbeddingResponse,
        #     self.client.post(
        #         "/embeddings",
        #         cast_to=CreateEmbeddingResponse,
        #         body={
        #             "messages": messages,
        #             "model": model,
        #             "encoding_format": encoding_format,
        #             "continue_final_message": continue_final_message,
        #             "add_special_tokens": add_special_tokens,
        #         },
        #     )
        # )

        return self.client.post(
            "/embeddings",
            cast_to=CreateEmbeddingResponse,
            body={
                "messages": messages,
                "model": model,
                "encoding_format": encoding_format,
                "continue_final_message": continue_final_message,
                "add_special_tokens": add_special_tokens,
            },
        )


    def _build_messages(
        self,
        inputs: List[Dict[str, Any]]
    ) -> List[List[ChatCompletionMessageParam]]:
        """Build messages in OpenAI chat format for vLLM embedding API."""
        all_messages = []

        for inp in inputs:
            instruction = inp.get('instruction', self.default_instruction)
            text = inp.get('text')
            image = inp.get('image')
            video = inp.get('video')

            # Build system message
            system_message: ChatCompletionMessageParam = {
                "role": "system",
                "content": [
                    {"type": "text", "text": instruction},
                ],
            }

            # Build user message content
            user_content: List[Dict[str, Any]] = []

            # Add image content
            if image is not None:
                if isinstance(image, list):
                    for img in image:
                        if isinstance(img, str):
                            # Support both URL and local file path
                            if img.startswith(('http://', 'https://')):
                                user_content.append({
                                    "type": "image_url",
                                    "image_url": {"url": img}
                                })
                            else:
                                # Local file path - convert to file:// URL
                                if not img.startswith('file://'):
                                    img = 'file://' + os.path.abspath(img)
                                user_content.append({
                                    "type": "image_url",
                                    "image_url": {"url": img}
                                })
                        else:
                            raise TypeError(f"Unsupported image type: {type(img)}")
                else:
                    if isinstance(image, str):
                        if image.startswith(('http://', 'https://')):
                            user_content.append({
                                "type": "image_url",
                                "image_url": {"url": image}
                            })
                        else:
                            if not image.startswith('file://'):
                                image = 'file://' + os.path.abspath(image)
                            user_content.append({
                                "type": "image_url",
                                "image_url": {"url": image}
                            })
                    else:
                        raise TypeError(f"Unsupported image type: {type(image)}")

            # Add video content (vLLM may support this)
            if video is not None:
                if isinstance(video, str):
                    if not video.startswith('file://'):
                        video = 'file://' + os.path.abspath(video)
                    user_content.append({
                        "type": "text",
                        "text": f"[VIDEO: {video}]",
                    })
                else:
                    user_content.append({
                        "type": "text",
                        "text": "[VIDEO]"
                    })

            # Add text content (must have text in user message for vLLM API)
            if text is not None:
                user_content.append({
                    "type": "text",
                    "text": text,
                })

            # Handle empty input
            if not user_content:
                user_content.append({"type": "text", "text": "NULL"})

            # Build user message
            user_message: ChatCompletionMessageParam = {
                "role": "user",
                "content": user_content,
            }

            # Build assistant message (required by vLLM embedding API)
            assistant_message: ChatCompletionMessageParam = {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": ""},
                ],
            }

            # Combine messages
            messages: List[ChatCompletionMessageParam] = [
                system_message,
                user_message,
                assistant_message,
            ]

            all_messages.append(messages)

        return all_messages

    @torch.no_grad()
    def process(
        self,
        inputs: List[Dict[str, Any]],
        normalize: bool = True
    ) -> torch.Tensor:
        """Process inputs and generate embeddings using vLLM Embedding API.
        
        Args:
            inputs: List of input dicts with 'text', 'image', 'video', 'instruction' keys.
            normalize: Whether to normalize embeddings.
        
        Returns:
            Tensor of embeddings with shape [batch_size, embedding_dim].
        """
        # Build messages for each input
        all_messages = self._build_messages(inputs)

        # Call API for each input (vLLM API processes one at a time in this format)
        all_embeddings = []
        for messages in all_messages:
            response = self._create_chat_embeddings(
                messages=messages,
                model=self.model_name_or_path,
                encoding_format="float",
                continue_final_message=True,
                add_special_tokens=True,
            )
            embedding = response.data[0].embedding
            all_embeddings.append(embedding)

        # Convert to tensor
        embeddings = torch.tensor(all_embeddings, dtype=torch.float32)

        # Normalize if requested
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

        return embeddings


class MMEBEmbeddingModel(nn.Module):
    """Simplified MMEBModel for Qwen3VL embeddings."""

    def __init__(self,
                 encoder: Union[Qwen3VLEmbedderVLLM],
                 normalize: bool = True,
                 temperature: float = 0.02,
                 use_vllm: bool = False):
        super().__init__()
        self.encoder = encoder
        self.normalize = normalize
        self.temperature = temperature
        self.use_vllm = use_vllm
        

        self.cross_entropy = None
        self.is_ddp = False
        self.process_rank = 0
        self.world_size = 1

    @property
    def device(self):
        if self.use_vllm:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.encoder.model.device

    @property
    def config(self):
        if self.use_vllm:
            return None
        return self.encoder.model.config

    @classmethod
    def load(cls,
             model_name_or_path: str,
             normalize: bool = True,
             temperature: float = 0.02,
             instruction: Optional[str] = None,
             use_vllm: bool = False,
             vllm_kwargs: Optional[Dict] = None,
             **kwargs) -> "MMEBEmbeddingModel":
        """Load model from pretrained checkpoint or vLLM service.
        
        Args:
            model_name_or_path: huggingface model name or path, or vLLM API URL
            normalize: whether to normalize embeddings
            temperature: temperature for similarity computation
            instruction: default instruction for the model
            use_vllm: whether to use vLLM service for inference
            vllm_kwargs: additional kwargs for vLLM initialization
            **kwargs: additional kwargs for model loading
        """
        default_instruction = kwargs.pop('default_instruction', instruction)

        vllm_kwargs = vllm_kwargs or {}
        encoder = Qwen3VLEmbedderVLLM(
            model_name_or_path=model_name_or_path,
            default_instruction=default_instruction or "Represent the user's input.",
            **vllm_kwargs
        )

        return cls(
            encoder=encoder,
            normalize=normalize,
            temperature=temperature,
            use_vllm=use_vllm
        )

    def save(self, output_dir: str):
        self.encoder.model.save_pretrained(output_dir)
        self.encoder.processor.save_pretrained(output_dir)

    def encode_input(self, inputs: Union[Dict, List[Dict]]) -> Tensor:
        """Encode inputs using the Qwen3VL embedder.
        
        Args:
            inputs: Dict containing 'text', 'image', 'video', 'instruction' etc.
                    Can be a single dict, a list of dicts, or dict with 'inputs' key.
        """
        # Handle vLLM case specially
        if self.use_vllm:
            # Support dict with 'inputs' key
            if isinstance(inputs, dict) and 'inputs' in inputs:
                inputs = inputs['inputs']
            # Ensure inputs is a list
            if isinstance(inputs, dict):
                inputs = [inputs]
            return self.encoder.process(inputs, normalize=self.normalize)

        # Original transformers-based logic
        # 如果是预处理过的 tensor 输入，直接 forward
        if isinstance(inputs, dict) and 'input_ids' in inputs:
            outputs = self.encoder.forward(inputs)
            hidden_state = outputs['last_hidden_state']
            attention_mask = outputs['attention_mask']
            pooled = self._pooling_last(hidden_state, attention_mask)
            if self.normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
            return pooled
        
        # Handle dict with 'inputs' key
        if isinstance(inputs, dict) and 'inputs' in inputs:
            inputs = inputs['inputs']
        
        # 否则使用 embedder 的 process 方法
        if isinstance(inputs, dict):
            inputs = [inputs]
        return self.encoder.process(inputs, normalize=self.normalize)

    def _pooling_last(self, hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
        """Pool the last non-padded token."""
        last_pos = attention_mask.flip(dims=[1]).argmax(dim=1)
        col = attention_mask.shape[1] - last_pos - 1
        row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[row, col]

    def forward(self,
                qry: Dict[str, Tensor] = None,
                tgt: Dict[str, Tensor] = None) -> Dict:
        """Forward pass for contrastive learning / evaluation."""
        qry_reps = self.encode_input(qry) if qry else None
        tgt_reps = self.encode_input(tgt) if tgt else None

        if qry_reps is None or tgt_reps is None:
            return {"qry_reps": qry_reps, "tgt_reps": tgt_reps}

        if self.is_ddp:
            all_qry = self._dist_gather(qry_reps)
            all_tgt = self._dist_gather(tgt_reps)
        else:
            all_qry, all_tgt = qry_reps, tgt_reps

        scores = torch.matmul(all_qry, all_tgt.T) / self.temperature
        target = torch.arange(scores.size(0), device=scores.device)
        target = target * (all_qry.size(0) // all_tgt.size(0))
        loss = self.cross_entropy(scores, target)

        if self.is_ddp:
            loss = loss * self.world_size
        return {"loss": loss, "qry_reps": qry_reps, "tgt_reps": tgt_reps}

    def _dist_gather(self, t: Tensor) -> Tensor:
        """Gather tensors across distributed processes."""
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        return torch.cat(all_tensors, dim=0)

    def compute_similarity(self, q_reps: Tensor, p_reps: Tensor) -> Tensor:
        """Compute similarity matrix between query and passage representations."""
        return torch.matmul(q_reps, p_reps.T)


if __name__ == '__main__':
    model = MMEBEmbeddingModel.load(
        model_name_or_path=r'Qwen/Qwen3-VL-Embedding-2B',
        attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16, device_map='cuda'
    )

    inputs = {
        'inputs': [
            {
                'text': "a woman breaks an egg",
                'instruction': 'Find images that corresponds to the given summary.',
            },
            {
                'text': "a woman breaks two eggs in a bowl",
                'instruction': 'Find images that corresponds to the given summary.',
            },
            {
                'image': r'https://ofasys-multimodal-wlcb-5-toshanghai.oss-cn-shanghai.aliyuncs.com/embedding_proj/linqi.lmx/codes/Qwen3-VL-Embedding/data/examples/0.jpeg?OSSAccessKeyId=LTAI5tSh9E5b7zDCb5uC8EsS&Expires=1925459675&Signature=fkEZFNVHMP3QaF49Qp2I%2Fz1GG6E%3D',
            },
            {
                'image': r'https://ofasys-multimodal-wlcb-5-toshanghai.oss-cn-shanghai.aliyuncs.com/embedding_proj/linqi.lmx/codes/Qwen3-VL-Embedding/data/examples/1.jpg?OSSAccessKeyId=LTAI5tSh9E5b7zDCb5uC8EsS&Expires=1925459699&Signature=PDrsu6gx5ivcskrxuISu2JUlRDc%3D',
            },
        ],
    }

    embeddings = model.encode_input(**inputs)

    print(
        f'Embeddings:\n{embeddings[:, :10].tolist()}\n{embeddings[:, -10:].tolist()}\n'
        f'Score:\n{model.compute_similarity(embeddings, embeddings).tolist()}\n'
    )