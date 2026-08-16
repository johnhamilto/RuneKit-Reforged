"""Text recognition via the macOS Vision framework."""
import io
from typing import List, Tuple, TypedDict

from PIL import Image

try:
    import Quartz
    import Vision

    HAS_VISION = True
except ImportError:
    HAS_VISION = False


class OcrLine(TypedDict):
    text: str
    confidence: float
    # normalized (x, y, w, h), origin bottom-left
    box: Tuple[float, float, float, float]


def ocr_lines(image: Image.Image) -> List[OcrLine]:
    if not HAS_VISION:
        raise RuntimeError(
            "Text recognition requires the macOS Vision framework (pyobjc-framework-Vision)"
        )

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    data = Quartz.CFDataCreate(None, buf.getvalue(), len(buf.getvalue()))
    src = Quartz.CGImageSourceCreateWithData(data, None)
    cgimg = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)

    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cgimg, None)
    ok, error = handler.performRequests_error_([request], None)
    if not ok:
        raise RuntimeError(f"Vision text recognition failed: {error}")

    lines: List[OcrLine] = []
    for obs in request.results() or []:
        candidate = obs.topCandidates_(1)[0]
        box = obs.boundingBox()
        lines.append(
            {
                "text": str(candidate.string()),
                "confidence": float(candidate.confidence()),
                "box": (
                    float(box.origin.x),
                    float(box.origin.y),
                    float(box.size.width),
                    float(box.size.height),
                ),
            }
        )
    return lines
