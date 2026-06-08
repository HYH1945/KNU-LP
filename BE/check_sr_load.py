import sys
from pathlib import Path

BE_ROOT = Path(__file__).resolve().parent
VENDORS_ROOT = BE_ROOT / "vendors"

sys.path.insert(0, str(BE_ROOT))
sys.path.insert(1, str(VENDORS_ROOT))

print("=" * 50)
print("Testing SR model load")
print("=" * 50)

try:
    import taming

    print(f"taming package path: {taming.__file__}")

    from pipeline.superresolution.sr_model import _get_model

    print("Instantiating RGDiffSR model. This can take a while...")
    model = _get_model()
    device = next(model.parameters()).device
    print(f"SUCCESS: SR model loaded on device: {device}")
except Exception:
    print("\nFAILED: SR model load failed.")
    print("-" * 50)
    import traceback

    traceback.print_exc()
    print("-" * 50)
