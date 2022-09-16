# Backend

## 목적
Frontend 에서 전달받은 이미지 파일에서 components detection 과 detecting 된 components를 이용하여 전류와 전압의 수치, 회로도를 반환하는 API server. 

## 시작하기
backend 폴더에서 다음의 명령어를 통해 시작. 

```
python server.py --port {available_port} --debug {True or False}
```

## 파일 리스트

### server.py
Backend 구동을 위한 main 코드. 
### findComponents.py
컴포넌트 detection 을 위한 코드. 

## component_predict
### get_component_predict.py
전기소자 예측 모델 결과를 확인하기 위한 코드.


## data_processing
### diagram.py
Detection 한 회로에 대해 이미지 파일로 변환. 
### calcVoltageAndCurrent.py
Detection 된 components 에서 전류와 전압 계산. 


## model
모델 파일 폴더.

## model_pipeline
### model_pipeline.py
N개의 모델을 동작시키기 위한 코드.


## pinmap_data
브레드보드 핀들에 대한 이미지 기준 픽셀 좌표값.


## resistance_value
### get_resistance_value.py
저항값 검출을 위한 코드.