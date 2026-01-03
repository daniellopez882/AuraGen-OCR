from typing import Dict

import torch

from auragen.common.predictor import BasePredictor
from auragen.detection import DetectionPredictor
from auragen.layout import LayoutPredictor
from auragen.logging import configure_logging
from auragen.ocr_error import OCRErrorPredictor
from auragen.foundation import FoundationPredictor
from auragen.recognition import RecognitionPredictor
from auragen.table_rec import TableRecPredictor
from auragen.settings import settings

configure_logging()


def load_predictors(
    device: str | torch.device | None = None, dtype: torch.dtype | str | None = None
) -> Dict[str, BasePredictor]:
    return {
        "layout": LayoutPredictor(FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)),
        "ocr_error": OCRErrorPredictor(device=device, dtype=dtype),
        "recognition": RecognitionPredictor(FoundationPredictor(checkpoint=settings.RECOGNITION_MODEL_CHECKPOINT)),
        "detection": DetectionPredictor(device=device, dtype=dtype),
        "table_rec": TableRecPredictor(device=device, dtype=dtype),
    }

