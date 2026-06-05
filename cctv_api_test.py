import requests
import cv2

def test_cctv_api():
    API_KEY = "4e21bfcebe5945b7af9e7af960169c04"
    
    # 서울/경기 부근 좌표 (대략적)
    minX = 126.800000
    maxX = 127.200000
    minY = 37.400000
    maxY = 37.700000
    
    # ITS CCTV API URL (JSON 응답 요청)
    url = f"https://openapi.its.go.kr:9443/cctvInfo?apiKey={API_KEY}&type=all&cctvType=1&minX={minX}&maxX={maxX}&minY={minY}&maxY={maxY}&getType=json"
    
    print(f"API 요청 전송 중... URL: {url}")
    try:
        response = requests.get(url, verify=False) # SSL 에러 방지
        print(f"응답 코드: {response.status_code}")
        
        if response.status_code != 200:
            print("API 요청 실패!")
            print(response.text)
            return

        data = response.json()
        
        # 데이터 파싱
        cctv_list = data.get('response', {}).get('data', [])
        
        if not cctv_list:
            print("해당 영역에 CCTV 정보가 없습니다.")
            return
            
        print(f"총 {len(cctv_list)}개의 CCTV 정보를 찾았습니다.")
        
        # 첫 번째 CCTV의 스트리밍 주소 가져오기
        first_cctv = cctv_list[0]
        cctv_name = first_cctv.get('cctvname', 'Unknown')
        stream_url = first_cctv.get('cctvurl', '')
        
        print(f"\n[{cctv_name}] CCTV 스트리밍 주소:")
        print(stream_url)
        
        if not stream_url:
            print("스트리밍 URL이 없습니다.")
            return
            
        # OpenCV로 프레임 추출 테스트
        print("\nOpenCV로 영상 스트림 연결 시도 중...")
        cap = cv2.VideoCapture(stream_url)
        
        if not cap.isOpened():
            print("스트림 연결에 실패했습니다.")
            return
            
        ret, frame = cap.read()
        if ret:
            print(f"프레임 추출 성공! 이미지 크기: {frame.shape}")
            cv2.imwrite("cctv_sample.jpg", frame)
            print("결과물 저장 완료: cctv_sample.jpg")
        else:
            print("스트림에는 연결되었으나 프레임을 읽어오지 못했습니다.")
            
        cap.release()
        
    except Exception as e:
        print(f"에러 발생: {e}")

if __name__ == "__main__":
    import urllib3
    urllib3.disable_warnings() # SSL 경고 무시
    test_cctv_api()
