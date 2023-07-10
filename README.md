# AWS_Sagemaker
AWS를 공부합니다.

## 과제1.경력에 따른 직원 월급 예측
1. Sagemaker Studio 사용 방법
2. S3 기초와 데이터를 S3로 변환하는 방법
3. Sagemaker Studio에서 다양한 모델 훈련 방법
4. 훈련된 모델을 엔드포인트에서 효율적으로 사용하고 이를 기반으로 추론하는 방법
5. sklearn을 활용한 머신 러닝 학습(Devide dataset into training and testing)
6. 데이터 프레임 조작을 위한 판다스 라이브러리 학습
7. histograms, Seaborn pairplot, matplotlib and correlation matrices(상관 행렬)을 활용한 탐구 데이터 분석
8. Boto3 AWS SDK를 활용한 데이터 업로드 방법
9. 컨테이너, 계정 접근 관리(IAM), Elastic Inference(딥러닝 추론 가속화)에 대한 개념
10. AWS SageMaker를 활용해 회귀 작업을 수행하기 위한 선형 학습 모델 훈련
11. 훈련 모델 배포, 테스트 및 추론 
---


## 과제2. 연령, BMI(Body mass index), 흡연 습관, 사는 지역 기반 개인의 의료 보험료 예측




1. 람다 함수(파이썬)을 적용하고 범주형(카테고리) 변수를 더미 변수로 변환시키는 방법
2. Seaborn 라이브러리를 통한 직선을 근사하는 방법
3. sklearn을 활용한 데이터 스케일링 수행
4. 다중 선형 회귀에 대한 이론과 직관 이해
5. RMSE(평균 제곱근 오차), MSE(평균 제곱 오차 손실), MAE(평균 절댓값 오차), R2(결정계수)와 같은 회귀 핵심 성과 지표(KPI) 학습

6. SageMaker 내장 선형 학습자 알고리즘을 통한 예측 가능 모델 구축, 훈련 배포하는 방법 학습

7. Sigmoid, ReLu, tanh와 같은 활성화 함수 사용

8. 역전파 알고리즘, 경사 하강법 이론
9. Tensorflow 및 Keras API를 활용한 인공 신경망 구축 및 훈련 방법
---
## 과거 데이터 기반으로 주간 소매업 판매 예측  

1. 예측 모델을 구축하기 위한 XGBoost 알고리즘 
2. Pandas를 사용한 다수의 데이터 프레임 병함
3. 편향과 분산의 차이점 및 트레이드오프 수행
4. L1와 L2 정규화에 대한 개념
5. 데이터 프레임 조작 수행 
6. 누락된 데이터 및 null 다루는 방법
7. Gradient boosted trees and Extreme Gradient boosting 알고리즘 이론 (어떤 회귀나 분류에 적용 가능)
8. 결정 트리와 결정 트리 앙상블
9. XGBoost 알고리즘의 장단점 리스트 제작
10. 회귀 작업을 수행하기 위한 XGBoost 기반 알고리즘 구축, 훈련, 배포
11. 세이지 메이커 하이퍼 파라미터 최적화 툴 
12. 튜닝 후 적합한 모델 배포 방법

## 심혈관 질환 예측 (회귀x 비지도 학습)

1. SageMaker를 통한 비지도 학습 수행
2. 차원 축소를 수행하기 위한 주성분 분석(PDA) 알고리즘
3. XGBoost 알고리즘을 활용한 출력
4. 콜레스테롤 수치, 혈압, 신체 활동과 같은 심혈관 질환의 존재를 감지할 수 있는 특징 > 주성분 분석을 위함 과정
5. 차원 축소를 수행하기 위해 PCA 적용 
6. AWS를 활용하여 훈련된 PCA 알고리즘을 통한 추론 수행
7. 분류 작업을 수행하기 위한 XGBoost 알고리즘 적용 방법 진행 

- 이전 사례에서 회귀를 위한 XGBoost를 적용하는 방법을 배웠다면, 여기서는 분류를 위함
8. 오차 행렬, 정밀도, 재현율, F1 스코어와 같은 분류 모델을 평가하는 방법 

9. ROC(Receiver operation characteristics) 과 AUC(Area Under Curve) 차이 학습
10. sklearn을 통해 grid search와 하이퍼 파라미터 최적화 작업 수행

---
## 딥러닝을 통한 교통 신호 분류 
1. 이미지 분류를 위한 딥러닝 활용
2. CNN
3. 현실의 데이터 세트를 이용한 이미지 분류를 수행하기 위해 LeNet학습
4. LeNet 네트워크를 테스트 데이터로 평가
5. keras API 를 활용한 모델 구축 방법 
6. Tensorflow를 통한 세이지 메이커 평가

7. SageMaker SDK의 Tensorflow estimators를 사용해서 모델을 빌드, 훈련, 추론하는 방법
8. dropout regularization 기술을 활용한 딥러닝 모델의 성능을 향상

---
## 세이지 메이커 스튜디오 사용 방법 
1. AutoML tool
2. SageMaker에서 실험들을 관리하는 방법
3. debug 툴 및 XGBoost models debug 방법
4. 후보 생성 및 데이터 분석 노트와 같은 보고서 생성 방법