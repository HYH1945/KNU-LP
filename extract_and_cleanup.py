import os
import zipfile
import glob

def process_data():
    data_dir = r"c:\Users\jech0\Desktop\projects\KNU-LP\AI\Task_1\denoiser\185.CCTV_기반_차량정보_및_교통정보_계측_데이터\01-1.정식개방데이터\Training\01.원천데이터"
    zip_path = os.path.join(data_dir, "TS_차량번호판인식_교차로_[cr01]비산사거리_04번.zip")
    
    # 50장만 저장할 테스트 폴더 생성
    target_dir = r"c:\Users\jech0\Desktop\projects\KNU-LP\AI\Task_1\denoiser\test_images"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    print("1. 대용량 압축 파일에서 테스트용 이미지 50장만 추출합니다...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            # 압축 파일 내의 이미지 파일 목록 가져오기
            image_files = [f for f in z.namelist() if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            extracted_count = 0
            for f in image_files[:50]:
                # 경로가 복잡하게 얽혀있을 수 있으므로 파일 이름만 따서 저장
                filename = os.path.basename(f)
                if not filename:
                    continue
                
                source = z.open(f)
                target_path = os.path.join(target_dir, filename)
                
                with open(target_path, "wb") as target:
                    target.write(source.read())
                extracted_count += 1
                
        print(f"   -> {extracted_count}장 추출 성공! 저장 위치: {target_dir}")
    except Exception as e:
        print(f"압축 해제 중 에러 발생: {e}")

    print("\n2. 용량 확보를 위해 원본 대용량 파일(.zip 및 .part)을 삭제합니다...")
    # .part 파일들 삭제
    part_files = glob.glob(os.path.join(data_dir, "*.zip.part*"))
    for f in part_files:
        try:
            os.remove(f)
            print(f"   -> 삭제 완료: {os.path.basename(f)}")
        except Exception as e:
            print(f"   -> 삭제 실패 ({os.path.basename(f)}): {e}")

    # 병합된 .zip 파일 삭제
    if os.path.exists(zip_path):
        try:
            os.remove(zip_path)
            print(f"   -> 삭제 완료: {os.path.basename(zip_path)}")
        except Exception as e:
            print(f"   -> 삭제 실패 ({os.path.basename(zip_path)}): {e}")
            
    print("\n모든 처리가 완료되었습니다!")

if __name__ == '__main__':
    process_data()
