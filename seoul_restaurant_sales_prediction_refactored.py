#!/usr/bin/env python
# coding: utf-8
# %%

# %% [markdown]
# # 서울시 요식업 평균 매출 예측 모델
#
# 서울시 요식업 매출 데이터를 분석하고 예측 모델을 구축하여 창업 의사결정에 도움이 될 수 있는 인사이트를 제공합니다.

# %% [markdown]
# ## 1. 데이터 로딩과 전처리

# %% [markdown]
# ### 1.1 필요한 라이브러리 임포트 및 환경 설정

# %%
# 필요한 라이브러리 임포트
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, gc, joblib
from pathlib import Path

# 한글 폰트 설정
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# 머신러닝 관련 라이브러리 임포트
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# 결과 저장용 디렉토리 생성
Path('plots').mkdir(exist_ok=True)
Path('models').mkdir(exist_ok=True)

# %% [markdown]
# ### 1.2 데이터 로드

# %%
# 원본 데이터 로드
sales_df = pd.read_csv("data/서울시 상권분석서비스(추정매출-상권).csv", encoding="cp949")
work_df = pd.read_csv("data/서울시 상권분석서비스(직장인구-상권).csv", encoding="cp949")
street_df = pd.read_csv("data/서울시 상권분석서비스(길단위인구-상권).csv", encoding="cp949")

# %% [markdown]
# ### 1.3 데이터 기본 정보 확인

# %%
# 데이터셋 기본 정보 출력
print("\n=== 데이터셋 기본 정보 ===")
print(f"매출 데이터: {sales_df.shape[0]}행, {sales_df.shape[1]}열")
print(f"직장인구 데이터: {work_df.shape[0]}행, {work_df.shape[1]}열")
print(f"유동인구 데이터: {street_df.shape[0]}행, {street_df.shape[1]}열")

# %% [markdown]
# ### 1.4 매출 데이터 샘플 및 컬럼 확인

# %%
# 데이터 샘플 확인
print("\n=== 매출 데이터 샘플 ===")
print(sales_df.head(3))

# 데이터 컬럼 확인
print("\n=== 매출 데이터 컬럼 목록 ===")
print(sales_df.columns.tolist())

# %% [markdown]
# ### 1.5 결측치 및 요식업 코드 분포 확인

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

# 요식업 코드 및 유형 확인
print("\n=== 요식업 서비스 코드 분포 ===")
cs_codes = sales_df["서비스_업종_코드"].str.startswith("CS").value_counts()
print(f"CS로 시작하는 코드 수: {cs_codes.get(True, 0)}/{len(sales_df)}")

# %% [markdown]
# ### 1.6 요식업 데이터 필터링

# %%
# 요식업 데이터만 필터링 (CS1 코드)
restaurant_sales = sales_df[sales_df["서비스_업종_코드"].str.startswith("CS1")].copy()
print(f"\n요식업(CS1) 데이터 추출: {len(restaurant_sales)}행")

# 요식업 코드 분포 확인
if '서비스_업종_코드_명' in restaurant_sales.columns:
    print("\n=== 요식업 유형 분포 (상위 10개) ===")
    print(restaurant_sales['서비스_업종_코드_명'].value_counts().head(10))

# %% [markdown]
# ### 1.7 데이터 전처리 함수 정의

# %%
# 문자열을 숫자로 변환하는 전처리 함수
def convert_to_numeric(df):
    # 변환 전 데이터 타입 정보(출력용)
    print(f"\n=== {df.name if hasattr(df, 'name') else '데이터'} 변환 전 타입 ===")
    print(df.dtypes.value_counts())
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    print(f"숫자형 컬럼 수: {len(numeric_cols)}")
    
    # 변환 전 샘플 데이터 확인 (첫 3개 컬럼만)
    if len(numeric_cols) > 0:
        sample_cols = numeric_cols[:min(3, len(numeric_cols))]
        print(f"\n변환 전 숫자 데이터 샘플 (첫 {len(sample_cols)}개 컬럼):")
        print(df[sample_cols].head(3))
    
    # 숫자 변환 수행
    for col in numeric_cols:
        # 실제 변환
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 변환 후 체크 (결측치가 생겼는지)
        missing_after = df[col].isnull().sum()
        if missing_after > 0:
            print(f"컬럼 '{col}' 변환 후 결측치: {missing_after}개")
            
    # 변환 후 데이터 타입 정보(출력용)
    print(f"\n=== {df.name if hasattr(df, 'name') else '데이터'} 변환 후 타입 ===")
    print(df.dtypes.value_counts())
    
    return df

# %% [markdown]
# ### 1.8 데이터 전처리 적용

# %%
# 데이터셋 이름 지정 (출력용)
restaurant_sales.name = '요식업 매출'
work_df.name = '직장인구'
street_df.name = '유동인구'

# 데이터프레임 전처리 적용
restaurant_sales = convert_to_numeric(restaurant_sales)
work_df = convert_to_numeric(work_df)
street_df = convert_to_numeric(street_df)

# %% [markdown]
# ### 1.9 평균 매출 계산

# %%
# 평균매출 계산
grouped = restaurant_sales.groupby(["상권_코드_명", "기준_년분기_코드"])[["당월_매출_금액"]].mean().reset_index()
grouped.rename(columns={"당월_매출_금액": "평균매출"}, inplace=True)

# 평균매출 분포 확인
print("\n=== 평균매출 기본 통계량 ===")
print(grouped["평균매출"].describe())

# %% [markdown]
# ### 1.10 매출 분포 시각화

# %%
# 매출 분포 히스토그램
plt.figure(figsize=(10, 5))

# 원본 매출 분포
plt.subplot(1, 2, 1)
sns.histplot(grouped["평균매출"], kde=True)
plt.title("평균매출 분포")
plt.xlabel("평균매출")

# 로그 변환 매출 분포 (왜도 확인)
plt.subplot(1, 2, 2)
sns.histplot(np.log1p(grouped["평균매출"]), kde=True)
plt.title("로그 변환 평균매출 분포")
plt.xlabel("log(평균매출+1)")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 1.11 데이터 병합

# %%
# 데이터 병합
merged = pd.merge(grouped, street_df, on=["상권_코드_명", "기준_년분기_코드"], how="left")
merged = pd.merge(merged, work_df, on=["상권_코드_명", "기준_년분기_코드"], how="left")

# 병합 결과 확인
print("\n=== 병합 결과 확인 ===")
print(f"병합 후 데이터: {merged.shape[0]}행, {merged.shape[1]}열")
print(f"원본 평균매출 데이터: {grouped.shape[0]}행, {grouped.shape[1]}열")

# 병합 후 결측치 확인
missing_after_merge = merged.isnull().sum().sum()
missing_percent = (missing_after_merge / (merged.shape[0] * merged.shape[1])) * 100
print(f"병합 후 총 결측치: {missing_after_merge}개 ({missing_percent:.2f}%)")

# %% [markdown]
# ### 1.12 병합 결과 샘플 확인

# %%
# 병합 결과 샘플 확인
print("\n=== 병합 결과 샘플 ===")
# 병합된 데이터의 주요 컬럼 샘플 출력
sample_cols = ['상권_코드_명', '기준_년분기_코드', '평균매출']
if '총_유동인구_수' in merged.columns:
    sample_cols.append('총_유동인구_수')
if '총_직장_인구_수' in merged.columns:
    sample_cols.append('총_직장_인구_수')
print(merged[sample_cols].head(3))

# %% [markdown]
# ### 1.13 상권 구분 코드 병합 및 결측치 처리

# %%
# 상권 구분 코드 병합
duplicate_drop = sales_df[["상권_코드_명", "상권_구분_코드_명"]].drop_duplicates()
restaurant_data = pd.merge(merged, duplicate_drop, on="상권_코드_명", how="left")

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
# 수치형 변수와 범주형 변수 정의 - 시각화와 모델링에 일관되게 사용
numeric_features = [
    "총_유동인구_수", "남성_유동인구_수", "여성_유동인구_수", 
    "총_직장_인구_수", "남성_직장_인구_수", "여성_직장_인구_수", 
    "초년_유동인구_수", "중년_유동인구_수", "노년_유동인구_수", 
    "초년_직장_인구_수", "중년_직장_인구_수", "노년_직장_인구_수"
]

categorical_features = ["상권_구분_코드_명", "기준_년분기_코드"]

# %% [markdown]
# ### 2.1 연령대별 인구 통합 특성 생성

# %%
# 연령대별 인구 통합 
restaurant_data["초년_유동인구_수"] = restaurant_data["연령대_10_유동인구_수"] + restaurant_data["연령대_20_유동인구_수"]
restaurant_data["중년_유동인구_수"] = restaurant_data["연령대_30_유동인구_수"] + restaurant_data["연령대_40_유동인구_수"]
restaurant_data["노년_유동인구_수"] = restaurant_data["연령대_50_유동인구_수"] + restaurant_data["연령대_60_이상_유동인구_수"]
restaurant_data["초년_직장_인구_수"] = restaurant_data["연령대_10_직장_인구_수"] + restaurant_data["연령대_20_직장_인구_수"]
restaurant_data["중년_직장_인구_수"] = restaurant_data["연령대_30_직장_인구_수"] + restaurant_data["연령대_40_직장_인구_수"]
restaurant_data["노년_직장_인구_수"] = restaurant_data["연령대_50_직장_인구_수"] + restaurant_data["연령대_60_이상_직장_인구_수"]

# %% [markdown]
# ### 2.2 성별 인구 분포 및 영향 분석

# %%
# 성별 유동인구 비율 계산
restaurant_data['남성_유동인구_비율'] = restaurant_data['남성_유동인구_수'] / restaurant_data['총_유동인구_수']
restaurant_data['여성_유동인구_비율'] = restaurant_data['여성_유동인구_수'] / restaurant_data['총_유동인구_수']
restaurant_data['남성_직장인구_비율'] = restaurant_data['남성_직장_인구_수'] / restaurant_data['총_직장_인구_수']
restaurant_data['여성_직장인구_비율'] = restaurant_data['여성_직장_인구_수'] / restaurant_data['총_직장_인구_수']

# 결측치 처리 (0으로 나누는 경우)
for col in ['남성_유동인구_비율', '여성_유동인구_비율', '남성_직장인구_비율', '여성_직장인구_비율']:
    restaurant_data[col] = restaurant_data[col].fillna(0)

# %% [markdown]
# ### 2.3 성별 분포 시각화

# %%
# 성별 인구 분포 시각화
plt.figure(figsize=(16, 10))

# 1. 유동인구와 직장인구의 성별 분포 (절대값)
plt.subplot(2, 2, 1)
gender_data = restaurant_data[['남성_유동인구_수', '여성_유동인구_수', '남성_직장_인구_수', '여성_직장_인구_수']].mean()
sns.barplot(x=gender_data.index, y=gender_data.values)
plt.title('평균 성별 인구 분포')
plt.ylabel('평균 인구 수')
plt.xticks(rotation=45)

# 2. 유동인구와 직장인구의 성별 비율 분포 (상자 그림)
plt.subplot(2, 2, 2)
gender_ratio = restaurant_data[['남성_유동인구_비율', '여성_유동인구_비율', '남성_직장인구_비율', '여성_직장인구_비율']]
sns.boxplot(data=gender_ratio)
plt.title('성별 인구 비율 분포')
plt.ylabel('비율')
plt.xticks(rotation=45)

# 3. 성별 인구 비율과 평균매출 관계 (산점도)
plt.subplot(2, 2, 3)
sns.scatterplot(x='남성_유동인구_비율', y='평균매출', data=restaurant_data, alpha=0.5, label='남성 유동인구')
sns.scatterplot(x='여성_유동인구_비율', y='평균매출', data=restaurant_data, alpha=0.5, label='여성 유동인구')
plt.title('성별 유동인구 비율과 평균매출 관계')
plt.xlabel('성별 유동인구 비율')
plt.ylabel('평균매출')
plt.legend()

# 4. 성별 직장인구 비율과 평균매출 관계 (산점도)
plt.subplot(2, 2, 4)
sns.scatterplot(x='남성_직장인구_비율', y='평균매출', data=restaurant_data, alpha=0.5, label='남성 직장인구')
sns.scatterplot(x='여성_직장인구_비율', y='평균매출', data=restaurant_data, alpha=0.5, label='여성 직장인구')
plt.title('성별 직장인구 비율과 평균매출 관계')
plt.xlabel('성별 직장인구 비율')
plt.ylabel('평균매출')
plt.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2.4 상권 및 시간 카테고리 분석

# %%
# 범주형 변수 분석 (상권_구분_코드_명, 기준_년분기_코드)
plt.figure(figsize=(10, 6))

# 상권 구분별 평균 매출
sns.barplot(x='상권_구분_코드_명', y='평균매출', data=restaurant_data.groupby('상권_구분_코드_명')['평균매출'].mean().sort_values(ascending=False).reset_index())
plt.title('상권 구분별 평균 매출')
plt.xlabel('상권 구분')
plt.ylabel('평균 매출')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 참고: 기준_년분기_코드는 단순히 원핫인코딩 용도로만 사용

# %% [markdown]
# ### 2.5 상권 유형별 분석

# %%
# 상권 구분 코드별 분석
plt.figure(figsize=(16, 12))

# 1. 상권 구분별 평균 매출
plt.subplot(2, 2, 1)
commercial_sales = restaurant_data.groupby('상권_구분_코드_명')['평균매출'].mean().sort_values(ascending=False).reset_index()
sns.barplot(x='상권_구분_코드_명', y='평균매출', data=commercial_sales)
plt.title('상권 유형별 평균 매출')
plt.xlabel('상권 구분')
plt.ylabel('평균 매출')
plt.xticks(rotation=45)

# 2. 상권 구분별 평균 유동인구
plt.subplot(2, 2, 2)
commercial_floating = restaurant_data.groupby('상권_구분_코드_명')['총_유동인구_수'].mean().sort_values(ascending=False).reset_index()
sns.barplot(x='상권_구분_코드_명', y='총_유동인구_수', data=commercial_floating)
plt.title('상권 유형별 평균 유동인구')
plt.xlabel('상권 구분')
plt.ylabel('평균 유동인구 수')
plt.xticks(rotation=45)

# 3. 상권 구분별 평균 직장인구
plt.subplot(2, 2, 3)
commercial_working = restaurant_data.groupby('상권_구분_코드_명')['총_직장_인구_수'].mean().sort_values(ascending=False).reset_index()
sns.barplot(x='상권_구분_코드_명', y='총_직장_인구_수', data=commercial_working)
plt.title('상권 유형별 평균 직장인구')
plt.xlabel('상권 구분')
plt.ylabel('평균 직장인구 수')
plt.xticks(rotation=45)

# 4. 상권 구분별 연령대 분포 (스택 바 차트)
plt.subplot(2, 2, 4)
age_commercial = restaurant_data.groupby('상권_구분_코드_명')[['초년_유동인구_수', '중년_유동인구_수', '노년_유동인구_수']].mean().reset_index()
age_commercial_melted = pd.melt(age_commercial, id_vars='상권_구분_코드_명', var_name='연령대', value_name='평균인구수')
sns.barplot(x='상권_구분_코드_명', y='평균인구수', hue='연령대', data=age_commercial_melted)
plt.title('상권 유형별 연령대 분포')
plt.xlabel('상권 구분')
plt.ylabel('평균 인구 수')
plt.xticks(rotation=45)
plt.legend(title='연령대')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2.6 인구 수 비율과 밀도 분석

# %%
# 유동인구와 직장인구의 비율 및 밀도 계산
restaurant_data['유동인구_직장인구_비율'] = restaurant_data['총_유동인구_수'] / restaurant_data['총_직장_인구_수']
restaurant_data['유동인구_직장인구_비율'] = restaurant_data['유동인구_직장인구_비율'].replace([np.inf, -np.inf], np.nan).fillna(0)

# 시각화
plt.figure(figsize=(16, 10))

# 1. 유동인구 대 직장인구 비율 분포
plt.subplot(2, 2, 1)
sns.histplot(restaurant_data['유동인구_직장인구_비율'].clip(0, 10), bins=30, kde=True)
plt.title('유동인구/직장인구 비율 분포 (0-10 범위)')
plt.xlabel('유동인구/직장인구 비율')
plt.ylabel('빈도')

# 2. 유동인구 대 직장인구 비율과 매출 관계
plt.subplot(2, 2, 2)
sns.scatterplot(x='유동인구_직장인구_비율', y='평균매출', data=restaurant_data[restaurant_data['유동인구_직장인구_비율'] < 10])
plt.title('유동인구/직장인구 비율과 평균매출 관계')
plt.xlabel('유동인구/직장인구 비율')
plt.ylabel('평균매출')

# 3. 유동인구와 매출의 로그-로그 관계
plt.subplot(2, 2, 3)
non_zero_data = restaurant_data[(restaurant_data['총_유동인구_수'] > 0) & (restaurant_data['평균매출'] > 0)]
sns.scatterplot(x=np.log1p(non_zero_data['총_유동인구_수']), y=np.log1p(non_zero_data['평균매출']))
plt.title('유동인구와 매출의 로그-로그 관계')
plt.xlabel('log(총 유동인구 수 + 1)')
plt.ylabel('log(평균매출 + 1)')

# 4. 직장인구와 매출의 로그-로그 관계
plt.subplot(2, 2, 4)
non_zero_data = restaurant_data[(restaurant_data['총_직장_인구_수'] > 0) & (restaurant_data['평균매출'] > 0)]
sns.scatterplot(x=np.log1p(non_zero_data['총_직장_인구_수']), y=np.log1p(non_zero_data['평균매출']))
plt.title('직장인구와 매출의 로그-로그 관계')
plt.xlabel('log(총 직장인구 수 + 1)')
plt.ylabel('log(평균매출 + 1)')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2.7 복합 특성 생성 및 분석

# %%
# 복합 특성 생성
# 1. 인구 밀도 지표: 유동인구와 직장인구의 합
restaurant_data['총_인구_수'] = restaurant_data['총_유동인구_수'] + restaurant_data['총_직장_인구_수']

# 2. 젊은층 비율: 초년 인구 / 총 인구
restaurant_data['초년_인구_비율'] = (restaurant_data['초년_유동인구_수'] + restaurant_data['초년_직장_인구_수']) / restaurant_data['총_인구_수']
restaurant_data['초년_인구_비율'] = restaurant_data['초년_인구_비율'].replace([np.inf, -np.inf], np.nan).fillna(0)

# 3. 여성 비율: 여성 인구 / 총 인구
restaurant_data['여성_인구_비율'] = (restaurant_data['여성_유동인구_수'] + restaurant_data['여성_직장_인구_수']) / restaurant_data['총_인구_수']
restaurant_data['여성_인구_비율'] = restaurant_data['여성_인구_비율'].replace([np.inf, -np.inf], np.nan).fillna(0)

# 복합 특성 시각화
plt.figure(figsize=(16, 10))

# 1. 총 인구수와 평균매출 관계
plt.subplot(2, 2, 1)
sns.scatterplot(x='총_인구_수', y='평균매출', data=restaurant_data, alpha=0.5)
plt.title('총 인구수와 평균매출 관계')
plt.xlabel('총 인구수 (유동+직장)')
plt.ylabel('평균매출')

# 2. 초년 인구 비율과 평균매출 관계
plt.subplot(2, 2, 2)
sns.scatterplot(x='초년_인구_비율', y='평균매출', data=restaurant_data, alpha=0.5)
plt.title('초년 인구 비율과 평균매출 관계')
plt.xlabel('초년 인구 비율')
plt.ylabel('평균매출')

# 3. 여성 인구 비율과 평균매출 관계
plt.subplot(2, 2, 3)
sns.scatterplot(x='여성_인구_비율', y='평균매출', data=restaurant_data, alpha=0.5)
plt.title('여성 인구 비율과 평균매출 관계')
plt.xlabel('여성 인구 비율')
plt.ylabel('평균매출')

# 4. 초년 인구 비율과 여성 인구 비율의 상호작용
plt.subplot(2, 2, 4)
h = plt.scatter(restaurant_data['초년_인구_비율'], restaurant_data['여성_인구_비율'], 
                c=restaurant_data['평균매출'], cmap='viridis', alpha=0.6, s=50)
plt.colorbar(h, label='평균매출')
plt.title('초년 인구 비율과 여성 인구 비율의 상호작용')
plt.xlabel('초년 인구 비율')
plt.ylabel('여성 인구 비율')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2.8 특성별 분포 시각화

# %%
# 주요 수치형 특성의 분포 시각화
plt.figure(figsize=(20, 15))
for i, feature in enumerate(numeric_features[:9]):  # 상위 9개 특성만 시각화
    plt.subplot(3, 3, i+1)
    # 원본 분포
    sns.histplot(restaurant_data[feature], kde=True, color='blue', alpha=0.4, label='원본')
    # 로그 변환 분포 (0보다 큰 값만)
    if (restaurant_data[feature] > 0).any():
        log_data = np.log1p(restaurant_data[feature].replace(0, np.nan).dropna())
        sns.histplot(log_data, kde=True, color='red', alpha=0.4, label='로그변환')
    plt.title(f'{feature} 분포')
    plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2.9 특성과 타겟 변수 간의 관계 시각화

# %%
# 주요 특성과 타겟 변수(평균매출) 간의 관계 시각화
plt.figure(figsize=(20, 15))
for i, feature in enumerate(numeric_features[:9]):  # 상위 9개 특성만 시각화
    plt.subplot(3, 3, i+1)
    sns.scatterplot(x=restaurant_data[feature], y=restaurant_data['평균매출'], alpha=0.5)
    # 회귀선 추가
    sns.regplot(x=restaurant_data[feature], y=restaurant_data['평균매출'], 
                scatter=False, line_kws={"color":"red"})
    plt.title(f'{feature} vs 평균매출')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2.10 범주형 특성과 타겟 변수 관계

# %%
# 범주형 특성과 타겟 변수 관계
plt.figure(figsize=(12, 8))
sns.boxplot(x='상권_구분_코드_명', y='평균매출', data=restaurant_data)
plt.title('상권 구분별 평균매출 분포')
plt.xlabel('상권 구분')
plt.ylabel('평균매출')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 참고: 기준_년분기_코드는 수가 많아 시각화에 적합하지 않으므로 원핫인코딩만 적용

# %% [markdown]
# ### 2.11 다변량 분석 - 페어플롯

# %%
# 중요 특성들 간의 관계 페어플롯
# 모든 특성을 사용하면 너무 복잡하므로 주요 특성 5개만 선택
top_features = ['평균매출', '총_유동인구_수', '총_직장_인구_수', 
                '초년_유동인구_수', '중년_직장_인구_수']
sns.pairplot(restaurant_data[top_features], height=2.5)
plt.suptitle('주요 특성 간 페어플롯', y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2.12 특성 간 상호작용 분석

# %%
# 유동인구와 직장인구의 상호작용 효과 시각화
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
plt.title('유동인구와 직장인구의 상호작용이 평균매출에 미치는 영향')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2.13 특성 엔지니어링 검증

# %%
# 특성 엔지니어링 검증 출력
print("\n=== 특성 엔지니어링 검증 ===")
# 샘플 데이터로 원본 값과 계산된 값 비교
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
# ### 2.14 연령대별 인구 분포 시각화

# %%
# 연령대별 인구 분포 시각화
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
# ### 2.15 필요한 컬럼만 선택

# %%
# 필요한 컬럼만 선택
keep_cols = [
    "기준_년분기_코드", "상권_코드_명", "상권_구분_코드_명", "평균매출",
    "총_유동인구_수", "남성_유동인구_수", "여성_유동인구_수", 
    "총_직장_인구_수", "남성_직장_인구_수", "여성_직장_인구_수",
    "초년_유동인구_수", "중년_유동인구_수", "노년_유동인구_수", 
    "초년_직장_인구_수", "중년_직장_인구_수", "노년_직장_인구_수",
    # 추가 생성 특성
    "남성_유동인구_비율", "여성_유동인구_비율", "남성_직장인구_비율", "여성_직장인구_비율",
    "유동인구_직장인구_비율", "총_인구_수", "초년_인구_비율", "여성_인구_비율"
]

# 컬럼 존재 여부 확인 후 필터링
existing_cols = [col for col in keep_cols if col in restaurant_data.columns]
restaurant_data = restaurant_data[existing_cols]

# 불필요한 컬럼 제거 (연도, 분기 등 시각화용 임시 컬럼이 있다면 삭제)
drop_cols = ['연도', '분기']
for col in drop_cols:
    if col in restaurant_data.columns:
        restaurant_data = restaurant_data.drop(columns=[col])

# %% [markdown]
# ## 3. 전처리 파이프라인 구성

# %% [markdown]
# ### 3.1 특성 정의 및 파이프라인 구성

# %%
# 전처리 파이프라인 구성 (특성은 2.0에서 정의된 것을 사용)
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
# 데이터 준비 - 특성과 타겟 분리
X = restaurant_data[numeric_features + categorical_features]
y = restaurant_data["평균매출"]

# 파이프라인으로 데이터 변환 (전체 데이터)
preprocessor.fit(X)
X_transformed = preprocessor.transform(X)

# %% [markdown]
# ### 3.3 변환된 특성 이름 추출

# %%
# 변환된 특성 이름 추출
transformed_feature_names = numeric_features.copy()

# 범주형 특성의 원핫인코딩된 이름 추출
cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
categorical_feature_names = cat_encoder.get_feature_names_out(categorical_features)
transformed_feature_names.extend(categorical_feature_names)

# 변환된 데이터를 DataFrame으로 변환
transformed_df = pd.DataFrame(X_transformed, columns=transformed_feature_names)
transformed_df["평균매출"] = y  # 타겟 변수 추가

# %% [markdown]
# ## 4. 특성 간 상관관계 분석

# %% [markdown]
# ### 4.1 상관계수 계산 및 분석

# %%
# 변환된 데이터로 상관계수 계산
correlation_matrix = transformed_df.corr()

# 평균매출과의 상관계수 내림차순 정렬
correlation_with_target = correlation_matrix['평균매출'].sort_values(ascending=False)

# %% [markdown]
# ### 4.2 상관관계 결과 출력

# %%
# 상관계수 출력
print("\n평균 매출과 상관관계가 높은 특성 (상위 15개):")
print(correlation_with_target.head(15))

# %% [markdown]
# ### 4.3 상관계수 히트맵 시각화

# %%
# 상관계수 히트맵 시각화 - 상위 15개 특성만 시각화
top_features = correlation_with_target.index[:15]
plt.figure(figsize=(15, 12))
sns.heatmap(correlation_matrix.loc[top_features, top_features], annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('특성 간 상관계수 히트맵 (상위 15개)')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. 모델 훈련 및 평가

# %% [markdown]
# ### 5.1 데이터 분할 및 모델 정의

# %%
# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 정의
models = {
    'Linear Regression': Pipeline([
        ('preprocessor', preprocessor),
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

# 모델 평가 결과 저장용 리스트
results = []

# %% [markdown]
# ### 5.2 모델 훈련 및 평가 루프

# %%
# 각 모델별 학습 및 평가
for name, model in models.items():
    print(f"\n{name} 모델 학습 중...")
    # 모델 학습
    model.fit(X_train, y_train)
    
    # 교차 검증 - 추가적인 성능 지표 사용
    print("교차 검증 중...")
    cv_rmse_scores = -cross_val_score(model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
    cv_r2_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    cv_mae_scores = -cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    
    # 교차 검증 결과 출력
    print(f"  CV RMSE: {cv_rmse_scores.mean():.2f} ± {cv_rmse_scores.std():.2f}")
    print(f"  CV R²: {cv_r2_scores.mean():.4f} ± {cv_r2_scores.std():.4f}")
    print(f"  CV MAE: {cv_mae_scores.mean():.2f} ± {cv_mae_scores.std():.2f}")
    
    # 훈련 세트 예측
    y_train_pred = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    # 테스트 세트 예측
    y_pred = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    
    # 과적합 계산
    overfitting_rmse = ((train_rmse - test_rmse) / train_rmse) * 100 if train_rmse > 0 else 0
    overfitting_r2 = ((train_r2 - test_r2) / train_r2) * 100 if train_r2 > 0 else 0
    
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
        'Test R2': test_r2,
        'RMSE Overfitting (%)': overfitting_rmse,
        'R2 Overfitting (%)': overfitting_r2
    })
    
    print(f"- 훈련 세트: RMSE={train_rmse:.2f}, MAE={train_mae:.2f}, R2={train_r2:.4f}")
    print(f"- 테스트 세트: RMSE={test_rmse:.2f}, MAE={test_mae:.2f}, R2={test_r2:.4f}")
    
    if overfitting_rmse > 10 or overfitting_r2 > 10:
        print(f"  ! 경고: 과적합 가능성 (RMSE 차이: {overfitting_rmse:.1f}%, R² 차이: {overfitting_r2:.1f}%)")

# %% [markdown]
# ### 5.3 평가 결과 분석

# %%
# 결과 정렬 및 출력
results_df = pd.DataFrame(results).sort_values('Test RMSE')
print("\n모델 평가 결과 (테스트 RMSE 기준 정렬):")
print(results_df[['Model', 'Train RMSE', 'CV RMSE', 'Test RMSE', 'Train R2', 'CV R2', 'Test R2']])

# %% [markdown]
# ### 5.4 모델 성능 시각화

# %%
# 학습-테스트 성능 격차 시각화
plt.figure(figsize=(15, 10))

# RMSE 비교 시각화
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
plt.title('모델별 학습, 교차검증, 테스트 RMSE 비교')
plt.xticks(x, models_list, rotation=45)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# R2 비교 시각화
plt.subplot(2, 1, 2)
train_r2_list = results_df['Train R2'].tolist()
cv_r2_list = results_df['CV R2'].tolist()
test_r2_list = results_df['Test R2'].tolist()

plt.bar(x - width, train_r2_list, width, label='Train R²')
plt.bar(x, cv_r2_list, width, label='CV R²')
plt.bar(x + width, test_r2_list, width, label='Test R²')

plt.xlabel('모델')
plt.ylabel('R²')
plt.title('모델별 학습, 교차검증, 테스트 R² 비교')
plt.xticks(x, models_list, rotation=45)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 5.5 과적합 분석

# %%
# 과적합 분석 시각화
plt.figure(figsize=(12, 6))

# RMSE 기준 과적합 시각화
plt.subplot(1, 2, 1)
plt.bar(models_list, results_df['RMSE Overfitting (%)'])
plt.title('모델별 RMSE 과적합 정도 (%)')
plt.xlabel('모델')
plt.ylabel('RMSE 차이 (%)')
plt.xticks(rotation=45)

# R2 기준 과적합 시각화
plt.subplot(1, 2, 2)
plt.bar(models_list, results_df['R2 Overfitting (%)'])
plt.title('모델별 R² 과적합 정도 (%)')
plt.xlabel('모델')
plt.ylabel('R² 차이 (%)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

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
# ### 6.3 최적화된 모델 성능 평가

# %%
# 최적화된 모델의 성능 평가
# 교차 검증 성능
cv_rmse_best = -rf_gs.cv_results_['mean_test_rmse'][rf_gs.best_index_]
cv_mae_best = -rf_gs.cv_results_['mean_test_mae'][rf_gs.best_index_]
cv_r2_best = rf_gs.cv_results_['mean_test_r2'][rf_gs.best_index_]

# 학습 데이터 성능
y_train_pred_best = best_rf.predict(X_train)
train_rmse_best = np.sqrt(mean_squared_error(y_train, y_train_pred_best))
train_mae_best = mean_absolute_error(y_train, y_train_pred_best)
train_r2_best = r2_score(y_train, y_train_pred_best)

# 테스트 데이터 성능
y_test_pred_best = best_rf.predict(X_test)
test_rmse_best = np.sqrt(mean_squared_error(y_test, y_test_pred_best))
test_mae_best = mean_absolute_error(y_test, y_test_pred_best)
test_r2_best = r2_score(y_test, y_test_pred_best)

# 과적합 계산
overfitting_rmse_best = (train_rmse_best - test_rmse_best) / train_rmse_best * 100 if train_rmse_best > 0 else 0
overfitting_r2_best = (train_r2_best - test_r2_best) / train_r2_best * 100 if train_r2_best > 0 else 0

# %% [markdown]
# ### 6.4 최적화 결과 분석

# %%
# 결과 출력
print("\n최적화된 RandomForest 모델 평가:")
best_model_results = pd.DataFrame({
    'Metric': ['RMSE', 'MAE', 'R²', 'Overfitting (%)'],
    'Train': [train_rmse_best, train_mae_best, train_r2_best, '-'],
    'CV': [cv_rmse_best, cv_mae_best, cv_r2_best, '-'],
    'Test': [test_rmse_best, test_mae_best, test_r2_best, '-'],
    'Train-Test': [f"{train_rmse_best-test_rmse_best:.2f}", 
                  f"{train_mae_best-test_mae_best:.2f}", 
                  f"{train_r2_best-test_r2_best:.4f}",
                  f"RMSE: {overfitting_rmse_best:.1f}%, R²: {overfitting_r2_best:.1f}%"]
})
print(best_model_results)

# 과적합 경고
if overfitting_rmse_best > 10:
    print(f"\n주의: 최적화된 모델에서 과적합이 발생했을 수 있습니다. (RMSE 차이: {overfitting_rmse_best:.1f}%)")
    print("정규화 매개변수를 조정하거나 특성 선택을 다시 검토해 보세요.")

# %% [markdown]
# ## 7. 최적 모델 분석 및 특성 중요도

# %% [markdown]
# ### 7.1 최고 성능 모델 선택

# %%
# 최고 성능 모델 선택
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]
print(f"선택된 최적 모델: {best_model_name}")
print(f"테스트 성능: RMSE={results_df.iloc[0]['Test RMSE']:.2f}, R2={results_df.iloc[0]['Test R2']:.4f}")

# %% [markdown]
# ### 7.2 특성 중요도 추출 및 시각화

# %%
# 특성 중요도 분석 (트리 기반 모델인 경우)
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    # 모델에서 특성 중요도 추출
    regressor = best_model.named_steps['regressor']
    feature_importances = regressor.feature_importances_
    
    # 전처리기에서 변환된 특성 이름 가져오기
    preprocessor = best_model.named_steps['preprocessor']
    num_transformer = preprocessor.named_transformers_['num']
    cat_transformer = preprocessor.named_transformers_['cat']
    
    # 수치형 특성 이름
    feature_names = numeric_features.copy()
    
    # 범주형 특성 변환 이름 추가
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
    plt.title(f'{best_model_name} 모델의 상위 10개 중요 특성')
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
y_pred_best = best_model.predict(X_test)

# 예측 성능 평가
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_best))
test_mae = mean_absolute_error(y_test, y_pred_best)
test_r2 = r2_score(y_test, y_pred_best)

# 예측 결과 데이터프레임
prediction_df = pd.DataFrame({
    '실제값': y_test,
    '예측값': y_pred_best,
    '오차': y_test - y_pred_best,
    '절대오차': np.abs(y_test - y_pred_best),
    '상대오차(%)': np.abs((y_test - y_pred_best) / y_test) * 100
})

# 예측 성능 지표 출력
print(f"최종 모델 성능: RMSE={test_rmse:.2f}, MAE={test_mae:.2f}, R²={test_r2:.4f}")
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
plt.scatter(y_test, y_pred_best, alpha=0.5)
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
plt.scatter(y_pred_best, prediction_df['오차'], alpha=0.5)
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

# %%
# 모델 결론 출력
print("\n요식업 매출 예측 모델 결론:")
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    top_features_str = ", ".join([f"{row['feature']}" for _, row in importance_df.head(5).iterrows()])
    print(f"1. {best_model_name} 모델이 가장 우수한 성능을 보여 최종 모델로 선정되었습니다 (R²={test_r2:.4f})")
    print(f"2. 가장 중요한 상위 5개 특성은 {top_features_str}입니다")
    print(f"3. 평균 예측 오차는 {prediction_df['절대오차'].mean():.2f}원이며, 상대 오차는 {prediction_df['상대오차(%)'].mean():.2f}%입니다")
else:
    print(f"1. {best_model_name} 모델이 가장 우수한 성능을 보여 최종 모델로 선정되었습니다 (R²={test_r2:.4f})")
    print(f"2. 평균 예측 오차는 {prediction_df['절대오차'].mean():.2f}원이며, 상대 오차는 {prediction_df['상대오차(%)'].mean():.2f}%입니다")
print(f"3. 이 모델은 새로운 상권의 예상 매출을 예측하는 데 활용할 수 있습니다")

# %% [markdown]
# ### 8.2 새로운 상권 예측 예시

# %%
# 새로운 상권 예측 예시
print("\n새로운 상권 매출 예측 예시:")
# 테스트 데이터의 첫 번째 샘플을 예시로 사용
sample_data = X_test.iloc[0:1].copy()
sample_prediction = best_model.predict(sample_data)[0]
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
joblib.dump(best_model, f'models/{best_model_name.replace(" ", "_").lower()}_model.pkl')
print(f"\n최종 모델 저장 완료: models/{best_model_name.replace(' ', '_').lower()}_model.pkl")

# 최종 정제된 데이터 저장
restaurant_data.to_csv("data/서울시_요식업_정제데이터.csv", index=False, encoding="cp949")
print("정제된 데이터 저장 완료: data/서울시_요식업_정제데이터.csv")

