import os
import glob

def merge_parts():
    # 데이터 폴더 경로
    data_dir = r"c:\Users\jech0\Desktop\projects\KNU-LP\AI\Task_1\denoiser\185.CCTV_기반_차량정보_및_교통정보_계측_데이터\01-1.정식개방데이터\Training\01.원천데이터"
    
    # 합칠 대상 패턴 찾기
    part_files = glob.glob(os.path.join(data_dir, "*.zip.part*"))
    
    if not part_files:
        print("분할 압축 파일을 찾을 수 없습니다.")
        return

    # 확장자의 part 숫자 기준으로 정렬 (part0, part1073741824, ...)
    def extract_number(filepath):
        ext = filepath.split('.zip.part')[-1]
        return int(ext) if ext.isdigit() else -1

    part_files.sort(key=extract_number)
    
    # 결과 파일명 생성
    base_name = part_files[0].split('.zip.part')[0] + '.zip'
    output_file = os.path.join(data_dir, base_name)
    
    print(f"다음 {len(part_files)}개의 파일을 병합합니다:")
    for f in part_files:
        print(f" - {os.path.basename(f)}")
        
    print(f"\n출력 파일: {output_file}")
    print("병합 중입니다. 잠시만 기다려주세요...")
    
    # 청크 단위로 읽어서 쓰기 (메모리 부족 방지)
    chunk_size = 1024 * 1024 * 64  # 64MB
    
    with open(output_file, 'wb') as outfile:
        for part_file in part_files:
            with open(part_file, 'rb') as infile:
                while True:
                    data = infile.read(chunk_size)
                    if not data:
                        break
                    outfile.write(data)
                    
    print("\n✅ 병합이 성공적으로 완료되었습니다!")

if __name__ == '__main__':
    merge_parts()
