import sys
import os

# BE 디렉토리를 path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("=" * 50)
print("Testing SR (Super-Resolution) Model Load...")
print("=" * 50)

try:
    # LDM instantiate_from_config가 가동될 수 있는지 taming 임포트 우선 확인
    import taming
    print(f"taming-transformers package path: {taming.__file__}")
    
    # 실제 모델 인스턴스화 함수 로드
    from pipeline.superresolution.sr_model import _get_model
    
    print("Instantiating RGDiffSR model (this may take 10-30 seconds)...")
    model = _get_model()
    device = next(model.parameters()).device
    print(f"SUCCESS: SR model loaded perfectly on device: {device}!")
except Exception as e:
    print("\nFAILED: SR model load failed.")
    print("-" * 50)
    import traceback
    traceback.print_exc()
    print("-" * 50)
