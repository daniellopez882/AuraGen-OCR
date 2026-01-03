from typing import Optional

import torch

from auragen.common.load import ModelLoader
from auragen.logging import get_logger
from auragen.settings import settings
from auragen.table_rec.model.config import (
    auragenTableRecConfig,
    auragenTableRecDecoderConfig,
    DonutSwinTableRecConfig,
)
from auragen.table_rec.model.encoderdecoder import TableRecEncoderDecoderModel
from auragen.table_rec.processor import auragenTableRecProcessor

logger = get_logger()


class TableRecModelLoader(ModelLoader):
    def __init__(self, checkpoint: Optional[str] = None):
        super().__init__(checkpoint)

        if self.checkpoint is None:
            self.checkpoint = settings.TABLE_REC_MODEL_CHECKPOINT

    def model(
        self,
        device=settings.TORCH_DEVICE_MODEL,
        dtype=settings.MODEL_DTYPE,
        attention_implementation: Optional[str] = None,
    ) -> TableRecEncoderDecoderModel:
        if device is None:
            device = settings.TORCH_DEVICE_MODEL
        if dtype is None:
            dtype = settings.MODEL_DTYPE

        if device == "mps":
            logger.warning(
                "`TableRecEncoderDecoderModel` is not compatible with mps backend. Defaulting to cpu instead"
            )
            device = "cpu"
            dtype = "float32"

        config = auragenTableRecConfig.from_pretrained(self.checkpoint)
        decoder_config = config.decoder
        decoder = auragenTableRecDecoderConfig(**decoder_config)
        config.decoder = decoder

        encoder_config = config.encoder
        encoder = DonutSwinTableRecConfig(**encoder_config)
        config.encoder = encoder

        model = TableRecEncoderDecoderModel.from_pretrained(
            self.checkpoint, config=config, dtype=dtype
        )

        model = model.to(device)
        model = model.eval()

        if settings.COMPILE_ALL or settings.COMPILE_TABLE_REC:
            torch.set_float32_matmul_precision("high")
            torch._dynamo.config.cache_size_limit = 16
            torch._dynamo.config.suppress_errors = False

            logger.info(
                f"Compiling table recognition model {self.checkpoint} on device {device} with dtype {dtype}"
            )
            compile_args = {"backend": "openxla"} if device == "xla" else {}
            model.encoder = torch.compile(model.encoder, **compile_args)
            model.decoder = torch.compile(model.decoder, **compile_args)

        logger.debug(
            f"Loaded table recognition model {self.checkpoint} from {TableRecEncoderDecoderModel.get_local_path(self.checkpoint)} onto device {device} with dtype {dtype}"
        )
        return model

    def processor(
        self, device=settings.TORCH_DEVICE_MODEL, dtype=settings.MODEL_DTYPE
    ) -> auragenTableRecProcessor:
        processor = auragenTableRecProcessor(self.checkpoint)

        processor.token_pad_id = 0
        processor.token_eos_id = 1
        processor.token_bos_id = 1
        processor.token_query_end_id = 4
        return processor

