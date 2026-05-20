import torch
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

from src.utils.device import get_device


class FrozenVLMBackbone:
    """
    Frozen PaliGemma-3B backbone for visual + language feature extraction.
    Runs in 4-bit quantization on MPS (M3 MacBook Pro).
    """
    def __init__(self, model_id: str = "google/paligemma-3b-pt-224"):
        self.device = get_device()
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        # 4-bit quantization — critical for M3 memory budget
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=str(self.device),
        )

        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad_(False)
    
    @torch.no_grad()
    def encode(
        self,
        images: list[Image.Image],
        instructions: list[str]
    ) -> torch.Tensor:
        """
        Return conditioning embeddings for the flow head.

        Shape:
            (B, 2D), where B is the batch size and D is the hidden dimension.

        The returned embedding concatenates:
            1. the first token embedding
            2. the mean-pooled remaining token embeddings
        """
        inputs = self.processor(
            images=images,
            text=instructions,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        
        outputs = self.model(
            **inputs,
            output_hidden_states=True,
        )
        # Last hidden state: (B, seq_len, D)
        hidden = outputs.hidden_states[-1]
        
        # CLS token
        cls_emb = hidden[:, 0, :]                  # (B, D)
        # Mean pool instruction tokens
        lang_emb = hidden[:, 1:, :].mean(dim=1)    # (B, D)
        
        return torch.cat([cls_emb, lang_emb], dim=-1)  # (B, 2D)