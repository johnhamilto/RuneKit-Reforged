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


def ocr_lines(image: Image.Image, fast: bool = False) -> List[OcrLine]:
    """fast trades accuracy for a several-fold speedup; good enough to spot
    interface titles when screening frames."""
    if not HAS_VISION:
        raise RuntimeError(
            "Text recognition requires the macOS Vision framework (pyobjc-framework-Vision)"
        )

    buf = io.BytesIO()
    if fast:  # screening path; JPEG encodes several times faster than PNG
        image.convert("RGB").save(buf, format="JPEG", quality=85)
    else:
        image.convert("RGB").save(buf, format="PNG")
    data = Quartz.CFDataCreate(None, buf.getvalue(), len(buf.getvalue()))
    src = Quartz.CGImageSourceCreateWithData(data, None)
    cgimg = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)

    request = Vision.VNRecognizeTextRequest.alloc().init()
    if fast:
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelFast)
        request.setUsesLanguageCorrection_(False)
    else:
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
