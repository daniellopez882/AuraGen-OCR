from typing import Optional

import torch
from transformers.utils import is_flash_attn_2_available

from auragen.common.load import ModelLoader
from auragen.common.auragen.config import auragenModelConfig
from auragen.common.auragen import auragenModel, auragenXLAModel
from auragen.common.auragen.processor import auragenOCRProcessor
from auragen.common.auragen.processor.tokenizer import auragenOCRTokenizer
from auragen.common.util import is_flash_attn_2_supported
from auragen.common.xla import get_compile_args
from auragen.logging import get_logger
from auragen.settings import settings

logger = get_logger()


class FoundationModelLoader(ModelLoader):
    def __init__(self, checkpoint: Optional[str] = None):
        super().__init__(checkpoint)

        if self.checkpoint is None:
            self.checkpoint = settings.FOUNDATION_MODEL_CHECKPOINT

    def model(
        self,
        device=settings.TORCH_DEVICE_MODEL,
        dtype=None,
        attention_implementation: Optional[str] = None,
    ) -> auragenModel:
        if device is None:
            device = settings.TORCH_DEVICE_MODEL
        if dtype is None:
            # See https://github.com/pytorch/pytorch/issues/118122 - T4 (device version 7.5) will return true since it supports
            # emulated bf16, but falls back to very slow kernels, especially for SDPA
            dtype = settings.MODEL_DTYPE_BFLOAT
            if device == "cuda" and not torch.cuda.is_bf16_supported(
                including_emulation=False
            ):
                # If the device is cuda, we check if bf16 is supported, and if not, we use float16
                dtype = settings.MODEL_DTYPE
        elif dtype == torch.float16:
            dtype = torch.bfloat16  # Model weights in bfloat16

        config = auragenModelConfig.from_pretrained(self.checkpoint)

        if attention_implementation is not None:
            config.decoder._attn_implementation = attention_implementation
            config.vision_encoder._attn_implementation = attention_implementation
        elif is_flash_attn_2_available() and is_flash_attn_2_supported(device):
            config.decoder._attn_implementation = "flash_attention_2"
            config.vision_encoder._attn_implementation = "flash_attention_2"
        elif device == "xla":
            config.decoder._attn_implementation = "sdpa"
            config.vision_encoder._attn_implementation = "sdpa"
        else:
            config.decoder._attn_implementation = "sdpa"
            config.vision_encoder._attn_implementation = "sdpa"

        model_cls = auragenModel
        if device == "xla":
            model_cls = auragenXLAModel

        config._attn_implementation_autoset = True
        config.vision_encoder._attn_implementation_autoset = True
        config.decoder._attn_implementation_autoset = True

        model = model_cls.from_pretrained(
            self.checkpoint, dtype=dtype, config=config, ignore_mismatched_sizes=True
        ).to(device)
        model = model.eval()

        if settings.COMPILE_ALL or settings.COMPILE_FOUNDATION:
            torch._dynamo.config.cache_size_limit = 1000
            torch._dynamo.config.suppress_errors = True
            torch._dynamo.config.specialize_int = False
            torch._dynamo.config.allow_unspec_int_on_nn_module = True
            torch._dynamo.config.capture_scalar_outputs = True
            torch._dynamo.config.recompile_limit = 32

            logger.info(
                f"Compiling foundation model {self.checkpoint} on device {device} with dtype {dtype}"
            )
            compile_args = get_compile_args(device)
            model.vision_encoder = torch.compile(model.vision_encoder, **compile_args)
            model.decoder = torch.compile(model.decoder, **compile_args)

        logger.debug(
            f"Loaded recognition model {self.checkpoint} from {auragenModel.get_local_path(self.checkpoint)} onto device {model.device} with dtype {dtype}, using decoder attention mechanism {model.config.decoder._attn_implementation}, encoder attention mechanism {model.config.vision_encoder._attn_implementation}."
        )
        return model

    def processor(
        self, device=settings.TORCH_DEVICE_MODEL, dtype=settings.MODEL_DTYPE_BFLOAT
    ) -> auragenOCRProcessor:
        config: auragenModelConfig = auragenModelConfig.from_pretrained(self.checkpoint)

        ocr_tokenizer = auragenOCRTokenizer(
            special_tokens=config.special_ocr_tokens, model_checkpoint=self.checkpoint
        )

        processor = auragenOCRProcessor(
            ocr_tokenizer=ocr_tokenizer,
            blank_bbox_token_id=config.blank_bbox_token_id,
            num_register_tokens=config.num_register_tokens,
            sequence_length=None,
            patch_size=config.vision_encoder.patch_size,
            merge_size=config.vision_encoder.spatial_merge_size,
            model_device=device,
            num_beacon_tokens=config.num_beacon_tokens,
            beacon_token_interval=config.beacon_token_interval,
        )

        return processor

