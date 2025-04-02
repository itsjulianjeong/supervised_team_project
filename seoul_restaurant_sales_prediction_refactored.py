#!/usr/bin/env python
# coding: utf-8
# %%

# %% [markdown]
# # 서울시 요식업 평균 매출 예측 모델
# 서울시 요식업 매출 데이터 분석 및 예측 모델 구축

# %% [markdown]
# ## 1. 데이터 로딩과 전처리

# %% [markdown]
# ### 1.1 라이브러리 임포트

# %%
# 기본 라이브러리
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, gc, joblib
from pathlib import Path

# 한글 폰트 설정
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# 머신러닝 라이브러리
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# 결과 저장 디렉토리 생성
Path('plots').mkdir(exist_ok=True)
Path('models').mkdir(exist_ok=True)

# %% [markdown]
# ### 1.2 데이터 로드

# %%
# 원본 데이터 파일 로드
sales_df = pd.read_csv("data/서울시 상권분석서비스(추정매출-상권).csv", encoding="cp949")
work_df = pd.read_csv("data/서울시 상권분석서비스(직장인구-상권).csv", encoding="cp949")
street_df = pd.read_csv("data/서울시 상권분석서비스(길단위인구-상권).csv", encoding="cp949")

# %% [markdown]
# ### 1.3 데이터 기본 정보 확인

# %%
# 데이터셋 크기 출력
print("\n=== 데이터셋 기본 정보 ===")
print(f"매출 데이터: {sales_df.shape[0]}행, {sales_df.shape[1]}열")
print(f"직장인구 데이터: {work_df.shape[0]}행, {work_df.shape[1]}열")
print(f"유동인구 데이터: {street_df.shape[0]}행, {street_df.shape[1]}열")

# %% [markdown]
# ### 1.4 매출 데이터 확인

# %%
# 데이터 샘플 및 컬럼 확인
print("\n=== 매출 데이터 샘플 ===")
print(sales_df.head(3))
print("\n=== 매출 데이터 컬럼 목록 ===")
print(sales_df.columns.tolist())

# %% [markdown]
# ### 1.5 결측치 및 요식업 코드 확인

# %%
# 결측치 확인
print("\n=== 결측치 확인 ===")
missing_sales = sales_df.isnull().sum()
missing_sales = missing_sales[missing_sales > 0]
if len(missing_sales) > 0:
    print("매출 데이터 결측치:")
    print(missing_sales)
else:
    print("매출 데이터에 결측치가 없습니다.")

# 요식업 코드 분포 확인
print("\n=== 요식업 서비스 코드 분포 ===")
cs_codes = sales_df["서비스_업종_코드"].str.startswith("CS").value_counts()
print(f"CS로 시작하는 코드 수: {cs_codes.get(True, 0)}/{len(sales_df)}")

# %% [markdown]
# ### 1.6 요식업 데이터 필터링

# %%
# 요식업(CS1) 데이터만 추출
restaurant_sales = sales_df[sales_df["서비스_업종_코드"].str.startswith("CS1")].copy()
print(f"\n요식업(CS1) 데이터 추출: {len(restaurant_sales)}행")

# 업종 분포 확인
if '서비스_업종_코드_명' in restaurant_sales.columns:
    print("\n=== 요식업 유형 분포 (상위 10개) ===")
    print(restaurant_sales['서비스_업종_코드_명'].value_counts().head(10))

# %% [markdown]
# ### 1.7 데이터 전처리 함수 정의

# %%
# 문자열을 숫자로 변환하는 함수 제거 (불필요)

# %% [markdown]
# ### 1.8 데이터 전처리 적용

# %%
# 데이터셋 이름 지정 코드 제거 (불필요)

# %% [markdown]
# ### 1.9 상권별 평균 매출 계산

# %%
# 상권별, 분기별 평균매출 계산
grouped = restaurant_sales.groupby(["상권_코드_명", "기준_년분기_코드"])[["당월_매출_금액"]].mean().reset_index()
grouped.rename(columns={"당월_매출_금액": "평균매출"}, inplace=True)

# 상권 구분 코드 추출 (상권별로 고유한 값)
commercial_type = sales_df[["상권_코드_명", "상권_구분_코드_명"]].drop_duplicates()

# 평균매출 데이터에 상권 구분 코드 병합
grouped = pd.merge(grouped, commercial_type, on="상권_코드_명", how="left")

# 평균매출 통계 확인
print("\n=== 평균매출 기본 통계량 ===")
print(grouped["평균매출"].describe())

# %% [markdown]
# ### 1.10 매출 분포 시각화

# %%
# 평균매출 분포 시각화
plt.figure(figsize=(10, 5))

# 원본 매출 분포
plt.subplot(1, 2, 1)
sns.histplot(grouped["평균매출"], kde=True)
plt.title("평균매출 분포")
plt.xlabel("평균매출")

# 로그 변환 매출 분포
plt.subplot(1, 2, 2)
sns.histplot(np.log1p(grouped["평균매출"]), kde=True)
plt.title("로그 변환 평균매출 분포")
plt.xlabel("log(평균매출+1)")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 1.11 데이터 병합

# %%
# 매출, 유동인구, 직장인구 데이터 병합
# 먼저 매출데이터(grouped)와 유동인구 데이터(street_df) 병합
restaurant_data = pd.merge(grouped, street_df, on=["상권_코드_명", "기준_년분기_코드"], how="left")

# 그 다음 직장인구 데이터(work_df) 병합
restaurant_data = pd.merge(restaurant_data, work_df, on=["상권_코드_명", "기준_년분기_코드"], how="left")

# 병합 결과 확인
print("\n=== 병합 결과 확인 ===")
print(f"병합 후 데이터: {restaurant_data.shape[0]}행, {restaurant_data.shape[1]}열")
print(f"원본 평균매출 데이터: {grouped.shape[0]}행, {grouped.shape[1]}열")

# 결측치 확인
missing_after_merge = restaurant_data.isnull().sum().sum()
missing_percent = (missing_after_merge / (restaurant_data.shape[0] * restaurant_data.shape[1])) * 100
print(f"병합 후 총 결측치: {missing_after_merge}개 ({missing_percent:.2f}%)")

# %% [markdown]
# ### 1.12 병합 결과 확인

# %%
# 병합 결과 샘플 출력
print("\n=== 병합 결과 샘플 ===")
sample_cols = ['상권_코드_명', '기준_년분기_코드', '상권_구분_코드_명', '평균매출']
if '총_유동인구_수' in restaurant_data.columns:
    sample_cols.append('총_유동인구_수')
if '총_직장_인구_수' in restaurant_data.columns:
    sample_cols.append('총_직장_인구_수')
print(restaurant_data[sample_cols].head(3))

# %% [markdown]
# ### 1.13 결측치 처리

# %%
# 결측치 처리
# 수치형 컬럼 - 중앙값으로 대체
numeric_cols = restaurant_data.select_dtypes(include=['number']).columns
for col in numeric_cols:
    if restaurant_data[col].isnull().any():
        restaurant_data[col] = restaurant_data[col].fillna(restaurant_data[col].median())

# 문자열 컬럼 - 최빈값으로 대체
object_cols = restaurant_data.select_dtypes(include=['object']).columns
for col in object_cols:
    if restaurant_data[col].isnull().any():
        restaurant_data[col] = restaurant_data[col].fillna(restaurant_data[col].mode().iloc[0] if not restaurant_data[col].mode().empty else "알수없음")

# %% [markdown]
# ## 2. 특성 엔지니어링

# %% [markdown]
# ### 2.0 특성 정의

# %%
# 모델링에 사용할 특성 정의
numeric_features = [
    "총_유동인구_수", "남성_유동인구_수", "여성_유동인구_수", 
    "총_직장_인구_수", "남성_직장_인구_수", "여성_직장_인구_수", 
    "초년_유동인구_수", "중년_유동인구_수", "노년_유동인구_수", 
    "초년_직장_인구_수", "중년_직장_인구_수", "노년_직장_인구_수"
]

categorical_features = ["상권_구분_코드_명", "기준_년분기_코드"]

# %% [markdown]
# ### 2.1 연령대별 인구 통합 특성

# %%
# 연령대 그룹핑 (초년/중년/노년)
restaurant_data["초년_유동인구_수"] = restaurant_data["연령대_10_유동인구_수"] + restaurant_data["연령대_20_유동인구_수"]
restaurant_data["중년_유동인구_수"] = restaurant_data["연령대_30_유동인구_수"] + restaurant_data["연령대_40_유동인구_수"]
restaurant_data["노년_유동인구_수"] = restaurant_data["연령대_50_유동인구_수"] + restaurant_data["연령대_60_이상_유동인구_수"]
restaurant_data["초년_직장_인구_수"] = restaurant_data["연령대_10_직장_인구_수"] + restaurant_data["연령대_20_직장_인구_수"]
restaurant_data["중년_직장_인구_수"] = restaurant_data["연령대_30_직장_인구_수"] + restaurant_data["연령대_40_직장_인구_수"]
restaurant_data["노년_직장_인구_수"] = restaurant_data["연령대_50_직장_인구_수"] + restaurant_data["연령대_60_이상_직장_인구_수"]

# %% [markdown]
# ### 2.2 성별 인구 비율 계산

# %%
# 성별 인구 비율 계산
restaurant_data['남성_유동인구_비율'] = restaurant_data['남성_유동인구_수'] / restaurant_data['총_유동인구_수']
restaurant_data['여성_유동인구_비율'] = restaurant_data['여성_유동인구_수'] / restaurant_data['총_유동인구_수']
restaurant_data['남성_직장인구_비율'] = restaurant_data['남성_직장_인구_수'] / restaurant_data['총_직장_인구_수']
restaurant_data['여성_직장인구_비율'] = restaurant_data['여성_직장_인구_수'] / restaurant_data['총_직장_인구_수']

# 0으로 나누는 경우 처리
for col in ['남성_유동인구_비율', '여성_유동인구_비율', '남성_직장인구_비율', '여성_직장인구_비율']:
    restaurant_data[col] = restaurant_data[col].fillna(0)

# %% [markdown]
# ### 2.3 성별 분포 시각화

# %%
# 성별 인구 분포 시각화
plt.figure(figsize=(16, 10))

# 1. 성별 인구 분포 (절대값)
plt.subplot(2, 2, 1)
gender_data = restaurant_data[['남성_유동인구_수', '여성_유동인구_수', '남성_직장_인구_수', '여성_직장_인구_수']].mean()
sns.barplot(x=gender_data.index, y=gender_data.values)
plt.title('평균 성별 인구 분포')
plt.ylabel('평균 인구 수')
plt.xticks(rotation=45)

# 2. 성별 비율 분포
plt.subplot(2, 2, 2)
gender_ratio = restaurant_data[['남성_유동인구_비율', '여성_유동인구_비율', '남성_직장인구_비율', '여성_직장인구_비율']]
sns.boxplot(data=gender_ratio)
plt.title('성별 인구 비율 분포')
plt.ylabel('비율')
plt.xticks(rotation=45)

# 3. 성별 유동인구 비율과 매출 관계
plt.subplot(2, 2, 3)
sns.scatterplot(x='남성_유동인구_비율', y='평균매출', data=restaurant_data, alpha=0.5, label='남성')
sns.scatterplot(x='여성_유동인구_비율', y='평균매출', data=restaurant_data, alpha=0.5, label='여성')
plt.title('성별 유동인구 비율과 평균매출')
plt.xlabel('성별 유동인구 비율')
plt.ylabel('평균매출')
plt.legend()

# 4. 성별 직장인구 비율과 매출 관계
plt.subplot(2, 2, 4)
sns.scatterplot(x='남성_직장인구_비율', y='평균매출', data=restaurant_data, alpha=0.5, label='남성')
sns.scatterplot(x='여성_직장인구_비율', y='평균매출', data=restaurant_data, alpha=0.5, label='여성')
plt.title('성별 직장인구 비율과 평균매출')
plt.xlabel('성별 직장인구 비율')
plt.ylabel('평균매출')
plt.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2.4 상권 유형 및 인구 분석

# %%
# 상권 유형별 인구 및 매출 분석과 인구 비율 분석 통합
plt.figure(figsize=(16, 15))

# 1. 상권 구분별 평균 매출
plt.subplot(3, 2, 1)
commercial_sales = restaurant_data.groupby('상권_구분_코드_명')['평균매출'].mean().sort_values(ascending=False).reset_index()
sns.barplot(x='상권_구분_코드_명', y='평균매출', data=commercial_sales)
plt.title('상권 유형별 평균 매출')
plt.xlabel('상권 구분')
plt.ylabel('평균 매출')
plt.xticks(rotation=45)

# 2. 상권 구분별 평균 유동인구
plt.subplot(3, 2, 2)
commercial_floating = restaurant_data.groupby('상권_구분_코드_명')['총_유동인구_수'].mean().sort_values(ascending=False).reset_index()
sns.barplot(x='상권_구분_코드_명', y='총_유동인구_수', data=commercial_floating)
plt.title('상권 유형별 평균 유동인구')
plt.xlabel('상권 구분')
plt.ylabel('평균 유동인구 수')
plt.xticks(rotation=45)

# 3. 상권 구분별 평균 직장인구
plt.subplot(3, 2, 3)
commercial_working = restaurant_data.groupby('상권_구분_코드_명')['총_직장_인구_수'].mean().sort_values(ascending=False).reset_index()
sns.barplot(x='상권_구분_코드_명', y='총_직장_인구_수', data=commercial_working)
plt.title('상권 유형별 평균 직장인구')
plt.xlabel('상권 구분')
plt.ylabel('평균 직장인구 수')
plt.xticks(rotation=45)

# 4. 유동인구/직장인구 비율과 매출 관계
plt.subplot(3, 2, 4)
# 유동인구/직장인구 비율 계산
restaurant_data['유동인구_직장인구_비율'] = restaurant_data['총_유동인구_수'] / restaurant_data['총_직장_인구_수']
restaurant_data['유동인구_직장인구_비율'] = restaurant_data['유동인구_직장인구_비율'].replace([np.inf, -np.inf], np.nan).fillna(0)
sns.scatterplot(x='유동인구_직장인구_비율', y='평균매출', 
                data=restaurant_data[restaurant_data['유동인구_직장인구_비율'] < 10])
plt.title('유동인구/직장인구 비율과 평균매출')
plt.xlabel('유동인구/직장인구 비율')
plt.ylabel('평균매출')

# 5. 유동인구와 매출 (로그 스케일)
plt.subplot(3, 2, 5)
non_zero_data = restaurant_data[(restaurant_data['총_유동인구_수'] > 0) & (restaurant_data['평균매출'] > 0)]
sns.scatterplot(x=np.log1p(non_zero_data['총_유동인구_수']), y=np.log1p(non_zero_data['평균매출']))
plt.title('유동인구와 매출 (로그-로그 스케일)')
plt.xlabel('log(총 유동인구 수 + 1)')
plt.ylabel('log(평균매출 + 1)')

# 6. 직장인구와 매출 (로그 스케일)
plt.subplot(3, 2, 6)
non_zero_data = restaurant_data[(restaurant_data['총_직장_인구_수'] > 0) & (restaurant_data['평균매출'] > 0)]
sns.scatterplot(x=np.log1p(non_zero_data['총_직장_인구_수']), y=np.log1p(non_zero_data['평균매출']))
plt.title('직장인구와 매출 (로그-로그 스케일)')
plt.xlabel('log(총 직장인구 수 + 1)')
plt.ylabel('log(평균매출 + 1)')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2.5 복합 특성 생성

# %%
# 복합 특성 생성
# 1. 총 인구 수
restaurant_data['총_인구_수'] = restaurant_data['총_유동인구_수'] + restaurant_data['총_직장_인구_수']

# 2. 젊은층 비율
restaurant_data['초년_인구_비율'] = (restaurant_data['초년_유동인구_수'] + restaurant_data['초년_직장_인구_수']) / restaurant_data['총_인구_수']
restaurant_data['초년_인구_비율'] = restaurant_data['초년_인구_비율'].replace([np.inf, -np.inf], np.nan).fillna(0)

# 3. 여성 비율
restaurant_data['여성_인구_비율'] = (restaurant_data['여성_유동인구_수'] + restaurant_data['여성_직장_인구_수']) / restaurant_data['총_인구_수']
restaurant_data['여성_인구_비율'] = restaurant_data['여성_인구_비율'].replace([np.inf, -np.inf], np.nan).fillna(0)

# 복합 특성 시각화
plt.figure(figsize=(16, 10))

# 1. 총 인구수와 매출 관계
plt.subplot(2, 2, 1)
sns.scatterplot(x='총_인구_수', y='평균매출', data=restaurant_data, alpha=0.5)
plt.title('총 인구수와 평균매출')
plt.xlabel('총 인구수 (유동+직장)')
plt.ylabel('평균매출')

# 2. 초년 인구 비율과 매출
plt.subplot(2, 2, 2)
sns.scatterplot(x='초년_인구_비율', y='평균매출', data=restaurant_data, alpha=0.5)
plt.title('초년 인구 비율과 평균매출')
plt.xlabel('초년 인구 비율')
plt.ylabel('평균매출')

# 3. 여성 인구 비율과 매출
plt.subplot(2, 2, 3)
sns.scatterplot(x='여성_인구_비율', y='평균매출', data=restaurant_data, alpha=0.5)
plt.title('여성 인구 비율과 평균매출')
plt.xlabel('여성 인구 비율')
plt.ylabel('평균매출')

# 4. 초년 인구 비율과 여성 인구 비율 관계
plt.subplot(2, 2, 4)
h = plt.scatter(restaurant_data['초년_인구_비율'], restaurant_data['여성_인구_비율'], 
                c=restaurant_data['평균매출'], cmap='viridis', alpha=0.6, s=50)
plt.colorbar(h, label='평균매출')
plt.title('초년 인구 비율과 여성 인구 비율')
plt.xlabel('초년 인구 비율')
plt.ylabel('여성 인구 비율')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2.6 수치형 특성 분포 확인

# %%
# 주요 수치형 특성 분포
plt.figure(figsize=(20, 15))
for i, feature in enumerate(numeric_features[:9]):  # 상위 9개 특성
    plt.subplot(3, 3, i+1)
    # 원본 분포
    sns.histplot(restaurant_data[feature], kde=True, color='blue', alpha=0.4, label='원본')
    # 로그 변환 분포
    if (restaurant_data[feature] > 0).any():
        log_data = np.log1p(restaurant_data[feature].replace(0, np.nan).dropna())
        sns.histplot(log_data, kde=True, color='red', alpha=0.4, label='로그변환')
    plt.title(f'{feature} 분포')
    plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2.7 타겟-특성 관계 시각화

# %%
# 특성-타겟 관계 시각화
plt.figure(figsize=(20, 15))
for i, feature in enumerate(numeric_features[:9]):  # 상위 9개 특성
    plt.subplot(3, 3, i+1)
    sns.scatterplot(x=restaurant_data[feature], y=restaurant_data['평균매출'], alpha=0.5)
    # 회귀선 추가
    sns.regplot(x=restaurant_data[feature], y=restaurant_data['평균매출'], 
                scatter=False, line_kws={"color":"red"})
    plt.title(f'{feature} vs 평균매출')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2.8 범주형 특성 분석

# %%
# 상권 구분별 매출 분포
plt.figure(figsize=(12, 8))
sns.boxplot(x='상권_구분_코드_명', y='평균매출', data=restaurant_data)
plt.title('상권 구분별 평균매출 분포')
plt.xlabel('상권 구분')
plt.ylabel('평균매출')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2.9 특성 간 관계 (페어플롯)

# %%
# 주요 특성 간 페어플롯
top_features = ['평균매출', '총_유동인구_수', '총_직장_인구_수', 
               '초년_유동인구_수', '중년_직장_인구_수']
sns.pairplot(restaurant_data[top_features], height=2.5)
plt.suptitle('주요 특성 간 페어플롯', y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2.10 특성 간 상호작용

# %%
# 유동인구와 직장인구의 상호작용 시각화
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='총_유동인구_수',
    y='총_직장_인구_수',
    size='평균매출',
    sizes=(20, 500),
    hue='평균매출',
    palette='viridis',
    data=restaurant_data
)
plt.title('유동인구와 직장인구의 상호작용이 매출에 미치는 영향')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2.11 특성 엔지니어링 검증

# %%
# 생성된 특성 검증
print("\n=== 특성 엔지니어링 검증 ===")
# 무작위 샘플로 검증
sample_idx = np.random.randint(0, len(restaurant_data), size=3)
sample_data = restaurant_data.iloc[sample_idx]

print("초년 유동인구 계산 검증:")
for idx in sample_idx:
    row = restaurant_data.iloc[idx]
    original_sum = row["연령대_10_유동인구_수"] + row["연령대_20_유동인구_수"]
    calculated = row["초년_유동인구_수"]
    match = np.isclose(original_sum, calculated)
    print(f"인덱스 {idx}: 원본합({original_sum}) vs 계산값({calculated}) - 일치: {match}")

# %% [markdown]
# ### 2.12 연령대별 인구 분포

# %%
# 연령대별 인구 분포 
age_cols = ["초년_유동인구_수", "중년_유동인구_수", "노년_유동인구_수"]
plt.figure(figsize=(12, 6))

# 연령대별 유동인구 분포
plt.subplot(1, 2, 1)
sns.boxplot(data=restaurant_data[age_cols])
plt.title("연령대별 유동인구 분포")
plt.ylabel("인구 수")

# 연령대별 직장인구 분포
age_work_cols = ["초년_직장_인구_수", "중년_직장_인구_수", "노년_직장_인구_수"]
plt.subplot(1, 2, 2)
sns.boxplot(data=restaurant_data[age_work_cols])
plt.title("연령대별 직장인구 분포")
plt.ylabel("인구 수")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2.13 최종 특성 선택

# %%
# 모델링을 위해 필요한 기본 특성(numeric_features + categorical_features + 타겟변수) 유지
# 추가로 생성된 특성(engineered_features)은 사용하지 않음

# 타겟 변수 추가
target_variable = ["평균매출"]

# 필요한 열만 선택 (추가 생성 특성 제외)
required_cols = numeric_features + categorical_features + target_variable
restaurant_data = restaurant_data[required_cols]

# %% [markdown]
# ## 3. 전처리 파이프라인 구성

# %% [markdown]
# ### 3.1 전처리 파이프라인 정의

# %%
# 수치형/범주형 특성 전처리 파이프라인
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
        ]), categorical_features)
    ])

# %% [markdown]
# ### 3.2 데이터 준비 및 변환

# %%
# 특성과 타겟 분리
X = restaurant_data[numeric_features + categorical_features]
y = restaurant_data["평균매출"]

# 파이프라인으로 데이터 변환
preprocessor.fit(X)
X_transformed = preprocessor.transform(X)

# %% [markdown]
# ### 3.3 변환된 특성 이름 추출

# %%
# 변환된 특성 이름 추출
transformed_feature_names = numeric_features.copy()

# 범주형 특성의 원핫인코딩 이름 추가
cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
categorical_feature_names = cat_encoder.get_feature_names_out(categorical_features)
transformed_feature_names.extend(categorical_feature_names)

# 변환된 데이터프레임 생성
transformed_df = pd.DataFrame(X_transformed, columns=transformed_feature_names)
transformed_df["평균매출"] = y  # 타겟 변수 추가

# %% [markdown]
# ## 4. 특성 간 상관관계 분석

# %% [markdown]
# ### 4.1 상관계수 계산 및 분석

# %%
# 상관계수 계산
correlation_matrix = transformed_df.corr()
correlation_with_target = correlation_matrix['평균매출'].abs().sort_values(ascending=False)

# %% [markdown]
# ### 4.2 매출과의 상관관계 출력

# %%
# 상관계수 출력
print("\n평균 매출과 상관관계가 강한 특성 (상위 15개, 절대값 기준):")
print(correlation_with_target.head(15))

# %% [markdown]
# ### 4.3 상관계수 히트맵

# %%
# 상관계수 히트맵 (상위 15개 특성)
top_features = correlation_with_target.index[:15]
plt.figure(figsize=(15, 12))
sns.heatmap(correlation_matrix.loc[top_features, top_features], 
            annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('특성 간 상관계수 히트맵 (상위 15개)')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. 모델 훈련 및 평가

# %% [markdown]
# ### 5.1 데이터 분할 및 모델 정의

# %%
# 훈련/테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 정의
models = {
    'Linear Regression': Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ]),
    'Polynomial Regression': Pipeline([
        ('preprocessor', preprocessor),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('regressor', LinearRegression())
    ]),
    'Ridge': Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', Ridge(alpha=1.0))
    ]),
    'Lasso': Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', Lasso(alpha=0.1))
    ]),
    'Random Forest': Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42))
    ]),
    'Gradient Boosting': Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
    ])
}

# 결과 저장용 리스트
results = []

# %% [markdown]
# ### 5.2 모델 훈련 및 평가

# %%
# 각 모델 학습 및 평가
for name, model in models.items():
    print(f"\n{name} 모델 학습 중...")
    # 모델 학습
    model.fit(X_train, y_train)
    
    # 교차 검증
    print("교차 검증 중...")
    cv_rmse_scores = -cross_val_score(model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
    cv_r2_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    cv_mae_scores = -cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    
    # 교차 검증 결과 출력
    print(f"  CV RMSE: {cv_rmse_scores.mean():.2f} ± {cv_rmse_scores.std():.2f}")
    print(f"  CV R²: {cv_r2_scores.mean():.4f} ± {cv_r2_scores.std():.4f}")
    print(f"  CV MAE: {cv_mae_scores.mean():.2f} ± {cv_mae_scores.std():.2f}")
    
    # 훈련 세트 성능
    y_train_pred = model.predict(X_train)
    train_rmse = root_mean_squared_error(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    # 테스트 세트 성능
    y_pred = model.predict(X_test)
    test_rmse = root_mean_squared_error(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    
    # 결과 저장
    results.append({
        'Model': name,
        'Train RMSE': train_rmse,
        'CV RMSE': cv_rmse_scores.mean(),
        'CV RMSE STD': cv_rmse_scores.std(),
        'Test RMSE': test_rmse,
        'Train MAE': train_mae,
        'CV MAE': cv_mae_scores.mean(),
        'Test MAE': test_mae,
        'Train R2': train_r2,
        'CV R2': cv_r2_scores.mean(),
        'Test R2': test_r2
    })
    
    print(f"- 훈련 세트: RMSE={train_rmse:.2f}, MAE={train_mae:.2f}, R2={train_r2:.4f}")
    print(f"- 테스트 세트: RMSE={test_rmse:.2f}, MAE={test_mae:.2f}, R2={test_r2:.4f}")

# %% [markdown]
# ### 5.3 모델 성능 비교

# %%
# 모델 성능 비교 (테스트 RMSE 기준 정렬)
results_df = pd.DataFrame(results).sort_values('Test RMSE')
print("\n모델 평가 결과 (테스트 RMSE 기준 정렬):")
print(results_df[['Model', 'Train RMSE', 'CV RMSE', 'Test RMSE', 'Train R2', 'CV R2', 'Test R2']])

# %% [markdown]
# ### 5.4 모델 성능 시각화

# %%
# 모델별 성능 시각화
plt.figure(figsize=(15, 10))

# RMSE 비교
plt.subplot(2, 1, 1)
models_list = results_df['Model'].tolist()
train_rmse_list = results_df['Train RMSE'].tolist()
cv_rmse_list = results_df['CV RMSE'].tolist()
test_rmse_list = results_df['Test RMSE'].tolist()

x = np.arange(len(models_list))
width = 0.25

plt.bar(x - width, train_rmse_list, width, label='Train RMSE')
plt.bar(x, cv_rmse_list, width, label='CV RMSE')
plt.bar(x + width, test_rmse_list, width, label='Test RMSE')

plt.xlabel('모델')
plt.ylabel('RMSE')
plt.title('모델별 RMSE 비교')
plt.xticks(x, models_list, rotation=45)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# R2 비교
plt.subplot(2, 1, 2)
train_r2_list = results_df['Train R2'].tolist()
cv_r2_list = results_df['CV R2'].tolist()
test_r2_list = results_df['Test R2'].tolist()

plt.bar(x - width, train_r2_list, width, label='Train R²')
plt.bar(x, cv_r2_list, width, label='CV R²')
plt.bar(x + width, test_r2_list, width, label='Test R²')

plt.xlabel('모델')
plt.ylabel('R²')
plt.title('모델별 R² 비교')
plt.xticks(x, models_list, rotation=45)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 5.5 최고 성능 모델 확인

# %%
# 최고 성능 모델 확인 (테스트 RMSE 기준)
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]
print(f"\n테스트 RMSE 기준 최고 성능 모델: {best_model_name}")
print(f"테스트 성능: RMSE={results_df.iloc[0]['Test RMSE']:.2f}, R2={results_df.iloc[0]['Test R2']:.4f}")

# %% [markdown]
# RandomForest 모델이 가장 우수한 성능을 보였습니다. 다음 단계에서는 이 모델의 하이퍼파라미터 튜닝을 진행하겠습니다.

# %% [markdown]
# ## 6. 최적 모델 튜닝

# %% [markdown]
# ### 6.1 RandomForest 모델 하이퍼파라미터 탐색

# %%
# RandomForest 모델 최적화 파이프라인 구성
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# 하이퍼파라미터 탐색 범위
rf_params = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [None, 10, 20],
    'regressor__min_samples_split': [2, 5, 10]
}

# %% [markdown]
# ### 6.2 그리드 서치 수행

# %%
# 그리드 서치 수행
rf_gs = GridSearchCV(rf_pipeline, rf_params, cv=5, n_jobs=-1, 
                    scoring={'rmse': 'neg_root_mean_squared_error', 
                             'mae': 'neg_mean_absolute_error',
                             'r2': 'r2'},
                    refit='rmse')
rf_gs.fit(X_train, y_train)
print("최적 RandomForest 파라미터:", rf_gs.best_params_)
best_rf = rf_gs.best_estimator_

# %% [markdown]
# ### 6.3 하이퍼파라미터 조합별 성능 확인

# %%
# GridSearchCV 결과를 데이터프레임으로 변환하여 출력
cv_results_df = pd.DataFrame(rf_gs.cv_results_)

# 각 하이퍼파라미터 조합과 성능 지표 출력
print("\n=== 하이퍼파라미터 조합별 성능 ===")
# RMSE 값은 음수로 저장되어 있으므로 양수로 변환
cv_results_df['mean_test_rmse'] = -cv_results_df['mean_test_rmse']
cv_results_df['mean_test_mae'] = -cv_results_df['mean_test_mae']

# 필요한 열만 선택하여 RMSE 기준으로 정렬
params_and_scores = cv_results_df[['param_regressor__n_estimators', 
                                   'param_regressor__max_depth', 
                                   'param_regressor__min_samples_split',
                                   'mean_test_rmse', 'mean_test_mae', 
                                   'mean_test_r2', 'rank_test_rmse']]
print(params_and_scores.sort_values('rank_test_rmse'))

# %% [markdown]
# ### 6.4 최적화된 모델 성능 평가

# %%
# 최적화된 모델의 성능 평가
# 교차 검증 성능
cv_rmse_best = -rf_gs.cv_results_['mean_test_rmse'][rf_gs.best_index_]
cv_mae_best = -rf_gs.cv_results_['mean_test_mae'][rf_gs.best_index_]
cv_r2_best = rf_gs.cv_results_['mean_test_r2'][rf_gs.best_index_]

# 학습 데이터 성능
y_train_pred_best = best_rf.predict(X_train)
train_rmse_best = root_mean_squared_error(y_train, y_train_pred_best)
train_mae_best = mean_absolute_error(y_train, y_train_pred_best)
train_r2_best = r2_score(y_train, y_train_pred_best)

# 테스트 데이터 성능
y_test_pred_best = best_rf.predict(X_test)
test_rmse_best = root_mean_squared_error(y_test, y_test_pred_best)
test_mae_best = mean_absolute_error(y_test, y_test_pred_best)
test_r2_best = r2_score(y_test, y_test_pred_best)

# %% [markdown]
# ### 6.5 최적화 결과 분석

# %%
# 결과 출력
print("\n최적화된 RandomForest 모델 평가:")
best_model_results = pd.DataFrame({
    'Metric': ['RMSE', 'MAE', 'R²'],
    'Train': [train_rmse_best, train_mae_best, train_r2_best],
    'CV': [cv_rmse_best, cv_mae_best, cv_r2_best],
    'Test': [test_rmse_best, test_mae_best, test_r2_best]
})
print(best_model_results)

# %% [markdown]
# 하이퍼파라미터 튜닝을 통해 RandomForest 모델의 성능을 최적화했습니다. 다음 단계에서는 최적화된 모델과 기본 모델의 성능을 비교하고, 특성 중요도 분석 및 예측 결과를 확인하겠습니다.

# %% [markdown]
# ## 7. 최적 모델 분석 및 특성 중요도

# %% [markdown]
# ### 7.1 최적화된 RandomForest 모델의 성능 확인

# %%
# 최적화된 RandomForest 모델과 기본 RandomForest 모델 비교
print("\n최적화된 RandomForest 모델과 기본 RandomForest 모델 성능 비교:")

# 기본 RandomForest 모델 성능
base_rf_perf = results_df[results_df['Model']=='Random Forest']
base_rf_rmse = base_rf_perf['Test RMSE'].values[0]
base_rf_r2 = base_rf_perf['Test R2'].values[0]

# 최적화된 RandomForest 모델 성능
optimized_rf_rmse = test_rmse_best
optimized_rf_r2 = test_r2_best

print(f"기본 RandomForest 모델: RMSE={base_rf_rmse:.2f}, R2={base_rf_r2:.4f}")
print(f"최적화된 RandomForest 모델: RMSE={optimized_rf_rmse:.2f}, R2={optimized_rf_r2:.4f}")

# 성능 향상 계산
rmse_improvement = ((base_rf_rmse - optimized_rf_rmse) / base_rf_rmse) * 100
r2_improvement = ((optimized_rf_r2 - base_rf_r2) / base_rf_r2) * 100 if base_rf_r2 > 0 else 0
print(f"\n성능 개선 수치:")
print(f"- RMSE: {rmse_improvement:.2f}% 감소")
print(f"- R2: {r2_improvement:.2f}% 증가")

# 최종 모델 정의
final_model = best_rf  # 최적화된 RandomForest 모델 선택
final_model_name = "최적화된 RandomForest"
final_rmse = optimized_rf_rmse
final_r2 = optimized_rf_r2
final_mae = test_mae_best  # 최종 MAE 값 정의

print(f"최종 모델 성능: RMSE={final_rmse:.2f}, MAE={final_mae:.2f}, R2={final_r2:.4f}")

# %% [markdown]
# ### 7.2 특성 중요도 추출 및 시각화

# %%
# RandomForest 모델의 특성 중요도 분석
# 모델에서 특성 중요도 추출
regressor = final_model.named_steps['regressor']
feature_importances = regressor.feature_importances_

# 전처리기에서 변환된 특성 이름 가져오기
preprocessor = final_model.named_steps['preprocessor']
num_transformer = preprocessor.named_transformers_['num']
cat_transformer = preprocessor.named_transformers_['cat']

# 수치형 특성 이름
feature_names = numeric_features.copy()

# 범주형 특성 변환 이름 추출
cat_feature_names = cat_transformer.named_steps['onehot'].get_feature_names_out(categorical_features)
feature_names.extend(cat_feature_names)

# 특성 중요도 데이터프레임 생성
importance_df = pd.DataFrame({
    'feature': feature_names[:len(feature_importances)],
    'importance': feature_importances
}).sort_values('importance', ascending=False)

# 상위 10개 특성 시각화
plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=importance_df.head(10))
plt.title(f'{final_model_name} 모델의 상위 10개 중요 특성')
plt.xlabel('중요도')
plt.ylabel('특성')
plt.tight_layout()
plt.show()

print("\n상위 10개 중요 특성:")
for idx, row in importance_df.head(10).iterrows():
    print(f"- {row['feature']}: {row['importance']:.4f}")

# %% [markdown]
# ### 7.3 예측 결과 분석

# %%
# 예측 결과 분석
print("\n예측 결과 분석 중...")
y_pred_final = final_model.predict(X_test)

# 예측 결과 데이터프레임
prediction_df = pd.DataFrame({
    '실제값': y_test,
    '예측값': y_pred_final,
    '오차': y_test - y_pred_final,
    '절대오차': np.abs(y_test - y_pred_final),
    '상대오차(%)': np.abs((y_test - y_pred_final) / y_test) * 100
})

# 오차 통계 출력
print(f"평균 절대 오차: {prediction_df['절대오차'].mean():.2f}")
print(f"중간값 절대 오차: {prediction_df['절대오차'].median():.2f}")
print(f"평균 상대 오차: {prediction_df['상대오차(%)'].mean():.2f}%")

# %% [markdown]
# ### 7.4 예측 결과 시각화

# %%
# 예측 결과 시각화
plt.figure(figsize=(15, 10))

# 예측값 vs 실제값 산점도
plt.subplot(2, 2, 1)
plt.scatter(y_test, y_pred_final, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.title('실제값 vs 예측값')
plt.xlabel('실제 평균매출')
plt.ylabel('예측 평균매출')
plt.grid(True, linestyle='--', alpha=0.5)

# 잔차 히스토그램
plt.subplot(2, 2, 2)
sns.histplot(prediction_df['오차'], kde=True)
plt.axvline(x=0, color='r', linestyle='--')
plt.title('예측 오차 분포')
plt.xlabel('오차')
plt.ylabel('빈도')

# 예측값 대비 잔차 산점도
plt.subplot(2, 2, 3)
plt.scatter(y_pred_final, prediction_df['오차'], alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('예측값 대비 잔차')
plt.xlabel('예측값')
plt.ylabel('잔차')
plt.grid(True, linestyle='--', alpha=0.5)

# 실제값 대비 상대오차 산점도
plt.subplot(2, 2, 4)
plt.scatter(y_test, prediction_df['상대오차(%)'], alpha=0.5)
plt.title('실제값 대비 상대오차(%)')
plt.xlabel('실제값')
plt.ylabel('상대오차(%)')
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. 예측 결과 및 결론

# %% [markdown]
# ### 8.1 요식업 매출 예측 모델 결론

# %% [markdown]
# 분석 결과, 최적화된 RandomForest 모델이 가장 우수한 성능을 보여 최종 모델로 선정되었습니다. 이 모델은 하이퍼파라미터 튜닝을 통해 기본 RandomForest 모델보다 향상된 성능을 보여주었으며, 특성 중요도 분석을 통해 매출 예측에 중요한 요인들을 파악할 수 있었습니다. 예측 결과 분석에서는 평균 절대 오차와 상대 오차가 합리적인 수준으로 나타났습니다. 이 모델은 새로운 상권의 예상 매출을 예측하는 데 활용할 수 있을 것입니다.

# %%
# 모델 결론 출력
print("\n최종 모델 성능 요약:")

# 상위 5개 중요 특성 출력
top_features_str = ", ".join([f"{row['feature']}" for _, row in importance_df.head(5).iterrows()])
print(f"모델: {final_model_name} (R²={final_r2:.4f})")
print(f"상위 5개 중요 특성: {top_features_str}")
print(f"평균 절대 오차: {prediction_df['절대오차'].mean():.2f}원")
print(f"평균 상대 오차: {prediction_df['상대오차(%)'].mean():.2f}%")

# %% [markdown]
# ### 8.2 새로운 상권 예측 예시

# %%
# 새로운 상권 예측 예시
print("\n새로운 상권 매출 예측 예시:")
# 테스트 데이터의 첫 번째 샘플을 예시로 사용
sample_data = X_test.iloc[0:1].copy()
sample_prediction = final_model.predict(sample_data)[0]
actual_value = y_test.iloc[0]

print(f"샘플 상권 정보:")
for col in categorical_features:
    if col in sample_data.columns:
        print(f"- {col}: {sample_data[col].values[0]}")
        
for col in numeric_features[:3]:  # 주요 특성 3개만 표시
    if col in sample_data.columns:
        print(f"- {col}: {sample_data[col].values[0]:,.2f}")

print(f"\n예측 평균 매출: {sample_prediction:,.2f}원")
print(f"실제 평균 매출: {actual_value:,.2f}원")
print(f"예측 오차: {abs(actual_value - sample_prediction):,.2f}원 ({abs(actual_value - sample_prediction) / actual_value * 100:.2f}%)")

# %% [markdown]
# ### 8.3 모델 및 데이터 저장

# %%
# 모델 저장
model_filename = f'models/{final_model_name.replace(" ", "_").lower()}_model.pkl'
joblib.dump(final_model, model_filename)
print(f"\n최종 모델 저장 완료: {model_filename}")

# 최종 정제된 데이터 저장
restaurant_data.to_csv("data/서울시_요식업_정제데이터.csv", index=False, encoding="cp949")
print("정제된 데이터 저장 완료: data/서울시_요식업_정제데이터.csv")

