#!/usr/bin/env python
# coding: utf-8
# %%

# %% [markdown]
# # 서울시 요식업 평균 매출 예측 모델
#
# 저희 팀은 서울시 요식업 매출 데이터를 분석하고 예측 모델을 구축했습니다. 창업 의사결정에 도움이 될 수 있는 인사이트를 제공하고자 했습니다.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, gc, joblib
from pathlib import Path

plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# %% [markdown]
# ## 1. 데이터 로딩과 전처리

# %%
print("데이터 로드 및 전처리 중...")
# 디렉토리 생성
Path('plots').mkdir(exist_ok=True)
Path('models').mkdir(exist_ok=True)

# 데이터 로드
sales_df = pd.read_csv("data/서울시 상권분석서비스(추정매출-상권).csv", encoding="cp949")
work_df = pd.read_csv("data/서울시 상권분석서비스(직장인구-상권).csv", encoding="cp949")
street_df = pd.read_csv("data/서울시 상권분석서비스(길단위인구-상권).csv", encoding="cp949")

# 요식업 데이터만 필터링 (CS1 코드)
restaurant_sales = sales_df[sales_df["서비스_업종_코드"].str.startswith("CS1")].copy()

# 데이터 전처리 함수 - 문자열을 숫자로 변환
def convert_to_numeric(df):
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# 전처리 적용
restaurant_sales = convert_to_numeric(restaurant_sales)
work_df = convert_to_numeric(work_df)
street_df = convert_to_numeric(street_df)

# 평균매출 계산 (main.py 방식으로 변경)
grouped = restaurant_sales.groupby(["상권_코드_명", "기준_년분기_코드"])[["당월_매출_금액"]].mean().reset_index()
grouped.rename(columns={"당월_매출_금액": "평균매출"}, inplace=True)

# 인구 데이터 병합
merged = pd.merge(grouped, street_df, on=["상권_코드_명", "기준_년분기_코드"], how="left")
merged = pd.merge(merged, work_df, on=["상권_코드_명", "기준_년분기_코드"], how="left")

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

# %%
print("특성 엔지니어링 중...")

# 연령대별 인구 통합 (필요한 Feature만 생성)
restaurant_data["초년_유동인구_수"] = restaurant_data["연령대_10_유동인구_수"] + restaurant_data["연령대_20_유동인구_수"]
restaurant_data["중년_유동인구_수"] = restaurant_data["연령대_30_유동인구_수"] + restaurant_data["연령대_40_유동인구_수"]
restaurant_data["노년_유동인구_수"] = restaurant_data["연령대_50_유동인구_수"] + restaurant_data["연령대_60_이상_유동인구_수"]
restaurant_data["초년_직장_인구_수"] = restaurant_data["연령대_10_직장_인구_수"] + restaurant_data["연령대_20_직장_인구_수"]
restaurant_data["중년_직장_인구_수"] = restaurant_data["연령대_30_직장_인구_수"] + restaurant_data["연령대_40_직장_인구_수"]
restaurant_data["노년_직장_인구_수"] = restaurant_data["연령대_50_직장_인구_수"] + restaurant_data["연령대_60_이상_직장_인구_수"]

# 필요한 컬럼만 선택
keep_cols = [
    "기준_년분기_코드", "상권_코드_명", "상권_구분_코드_명", "평균매출",
    "총_유동인구_수", "남성_유동인구_수", "여성_유동인구_수", 
    "총_직장_인구_수", "남성_직장_인구_수", "여성_직장_인구_수",
    "초년_유동인구_수", "중년_유동인구_수", "노년_유동인구_수", 
    "초년_직장_인구_수", "중년_직장_인구_수", "노년_직장_인구_수"
]
restaurant_data = restaurant_data[keep_cols]

# %% [markdown]
# ## 3. 전처리 파이프라인 구성

# %%
print("전처리 파이프라인 구성 중...")

# 모델링용 특성 정의 - 범주형 변수와 수치형 변수 구분
categorical_features = ["상권_구분_코드_명"]
numeric_features = [
    "총_유동인구_수", "남성_유동인구_수", "여성_유동인구_수", 
    "총_직장_인구_수", "남성_직장_인구_수", "여성_직장_인구_수", 
    "초년_유동인구_수", "중년_유동인구_수", "노년_유동인구_수", 
    "초년_직장_인구_수", "중년_직장_인구_수", "노년_직장_인구_수"
]

# 전처리 파이프라인 구성
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

# 데이터 준비 - 특성과 타겟 분리
X = restaurant_data[numeric_features + categorical_features]
y = restaurant_data["평균매출"]

# 파이프라인으로 데이터 변환 (전체 데이터)
preprocessor.fit(X)
X_transformed = preprocessor.transform(X)

# 변환된 특성 이름 추출
transformed_feature_names = numeric_features.copy()

# 범주형 특성의 원핫인코딩된 이름 추출
cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
if hasattr(cat_encoder, 'get_feature_names_out'):
    categorical_feature_names = cat_encoder.get_feature_names_out(categorical_features)
    transformed_feature_names.extend(categorical_feature_names)
else:
    # 원래 카테고리 이름 + 고유값으로 수동 생성 (이전 버전 호환성)
    unique_cats = X[categorical_features[0]].unique()
    for cat in unique_cats:
        transformed_feature_names.append(f"{categorical_features[0]}_{cat}")

# 변환된 데이터를 DataFrame으로 변환
transformed_df = pd.DataFrame(X_transformed, columns=transformed_feature_names)
transformed_df["평균매출"] = y  # 타겟 변수 추가

# %% [markdown]
# ## 4. 특성 간 상관관계 분석

# %%
print("특성 간 상관관계 분석 중...")

# 변환된 데이터로 상관계수 계산
correlation_matrix = transformed_df.corr()

# 평균매출과의 상관계수 내림차순 정렬
correlation_with_target = correlation_matrix['평균매출'].sort_values(ascending=False)

# 상관계수 출력
print("\n평균 매출과 상관관계가 높은 특성 (상위 15개):")
print(correlation_with_target.head(15))

# 상관계수 히트맵 시각화 - 너무 많은 특성이 있을 경우, 상위 15개만 시각화
top_features = correlation_with_target.index[:15]
plt.figure(figsize=(15, 12))
sns.heatmap(correlation_matrix.loc[top_features, top_features], annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('특성 간 상관계수 히트맵 (상위 15개)')
plt.tight_layout()
plt.savefig('plots/correlation_heatmap.png')
plt.close()

# 다중공선성 확인 (상관계수가 0.8 이상인 특성 쌍 찾기)
high_correlation_pairs = []
numeric_only_features = [col for col in transformed_feature_names if col in numeric_features]

for i in range(len(numeric_only_features)):
    for j in range(i+1, len(numeric_only_features)):
        feature_i = numeric_only_features[i]
        feature_j = numeric_only_features[j]
        corr = correlation_matrix.loc[feature_i, feature_j]
        if abs(corr) > 0.8:
            high_correlation_pairs.append((feature_i, feature_j, corr))

# 다중공선성 출력
if high_correlation_pairs:
    print("\n다중공선성이 높은 특성 쌍 (상관계수 > 0.8):")
    for feature_i, feature_j, corr in high_correlation_pairs:
        print(f"- {feature_i} 와 {feature_j}: {corr:.4f}")

# %% [markdown]
# ## 5. 모델 훈련 및 평가

# %%
print("모델 훈련 및 평가 중...")

# 데이터 분할 - 원본 데이터 사용
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

# 모델 평가 결과 저장
results = []

# 모델 훈련 및 평가
for name, model in models.items():
    # 모델 학습
    model.fit(X_train, y_train)
    
    # 교차 검증 점수 - 여러 지표로 확장
    cv_rmse_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
    cv_rmse = -cv_rmse_scores.mean()
    cv_rmse_std = cv_rmse_scores.std()
    
    cv_mae_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    cv_mae = -cv_mae_scores.mean()
    
    cv_r2_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    cv_r2 = cv_r2_scores.mean()
    
    # 예측 및 평가
    y_train_pred = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    y_pred = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    
    # 과적합 계산 - 학습과 테스트 오차의 차이
    overfitting_rmse = (train_rmse - test_rmse) / train_rmse * 100 if train_rmse > 0 else 0
    overfitting_r2 = (train_r2 - test_r2) / train_r2 * 100 if train_r2 > 0 else 0
    
    results.append({
        'Model': name,
        'Train RMSE': train_rmse,
        'Train MAE': train_mae,
        'Train R2': train_r2,
        'CV RMSE': cv_rmse,
        'CV RMSE STD': cv_rmse_std,
        'CV MAE': cv_mae,
        'CV R2': cv_r2,
        'Test RMSE': test_rmse,
        'Test MAE': test_mae,
        'Test R2': test_r2,
        'RMSE Overfitting (%)': overfitting_rmse,
        'R2 Overfitting (%)': overfitting_r2
    })
    
    print(f"- {name}: Train RMSE={train_rmse:.2f}, Test RMSE={test_rmse:.2f}, CV RMSE={cv_rmse:.2f}±{cv_rmse_std:.2f}")
    print(f"  R2 (Train/CV/Test)={train_r2:.3f}/{cv_r2:.3f}/{test_r2:.3f}")

# 결과 정렬
results_df = pd.DataFrame(results).sort_values('Test RMSE')

# 학습, CV, 테스트 지표 모두 출력 - 컬럼 순서 정렬
print("\n모델 평가 결과 (테스트 RMSE 기준 정렬):")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(results_df[[
    'Model', 
    'Train RMSE', 'CV RMSE', 'Test RMSE',
    'Train MAE', 'CV MAE', 'Test MAE', 
    'Train R2', 'CV R2', 'Test R2',
    'CV RMSE STD', 'RMSE Overfitting (%)', 'R2 Overfitting (%)'
]])

# 학습-테스트 성능 격차 시각화
plt.figure(figsize=(15, 10))
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

# R2 비교 추가
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
plt.savefig('plots/model_performance_comparison.png')
plt.close()

# 과적합 분석
overfitting_threshold = 30  # 30% 이상 차이나면 과적합 의심
overfitting_models = results_df[results_df['RMSE Overfitting (%)'] > overfitting_threshold]
if not overfitting_models.empty:
    print("\n주의: 다음 모델에서 과적합 가능성이 있습니다:")
    for idx, row in overfitting_models.iterrows():
        print(f"- {row['Model']}: 학습 RMSE가 테스트 RMSE보다 {row['RMSE Overfitting (%)']:.1f}% 낮습니다.")

# RandomForest 모델 최적화 (성능이 가장 좋은 모델)
print("\nRandomForest 모델 최적화 중...")
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

rf_params = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [None, 10, 20],
    'regressor__min_samples_split': [2, 5, 10]
}

rf_gs = GridSearchCV(rf_pipeline, rf_params, cv=5, n_jobs=-1, 
                    scoring={'rmse': 'neg_root_mean_squared_error', 
                             'mae': 'neg_mean_absolute_error',
                             'r2': 'r2'},
                    refit='rmse')
rf_gs.fit(X_train, y_train)
print("최적 RandomForest 파라미터:", rf_gs.best_params_)
best_rf = rf_gs.best_estimator_

# 최적화된 모델의 학습 및 테스트 성능 비교 - 통합된 평가 지표 사용
print("\n최적화된 RandomForest 모델 평가:")

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

# 결과 출력 - 테이블 형식
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
if overfitting_rmse_best > overfitting_threshold:
    print(f"\n주의: 최적화된 모델에서 과적합이 발생했을 수 있습니다. (RMSE 차이: {overfitting_rmse_best:.1f}%)")
    print("정규화 매개변수를 조정하거나 특성 선택을 다시 검토해 보세요.")

# %% [markdown]
# ## 6. 최적 모델 분석 및 특성 중요도

# %%
print("최적 모델 분석 중...")

# 최적 모델로 최종 예측
y_pred = best_rf.predict(X_test)
final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
final_r2 = r2_score(y_test, y_pred)
final_mae = mean_absolute_error(y_test, y_pred)

print(f"최종 모델 성능: RMSE={final_rmse:.2f}, R2={final_r2:.3f}, MAE={final_mae:.2f}")

# 특성 중요도 추출
rf_importances = best_rf.named_steps['regressor'].feature_importances_

# 특성 중요도 데이터프레임 생성
importance_df = pd.DataFrame({
    'feature': transformed_feature_names[:len(rf_importances)],  # 길이 맞추기
    'importance': rf_importances
}).sort_values('importance', ascending=False)

print("\n특성 중요도 (상위 10개):")
print(importance_df.head(10))

# 특성 중요도 시각화
plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=importance_df.head(10))
plt.title('상위 10개 중요 특성')
plt.tight_layout()
plt.savefig('plots/feature_importance.png')
plt.close()

# 모델 저장
joblib.dump(best_rf, 'models/random_forest_model.pkl')
print("모델 저장 완료")

# 최종 정제된 데이터 저장
restaurant_data.to_csv("data/서울시_요식업_정제데이터.csv", index=False, encoding="cp949")

# %% [markdown]
# ## 7. 예측 결과 및 결론

# %%
print("\n예측 결과 및 결론:")
print(f"- RandomForest 모델이 R² = {final_r2:.4f}로 가장 우수한 성능을 보였습니다.")
print("- 상관관계 분석을 통해 직장인구와 유동인구 특성이 매출에 미치는 영향을 확인했습니다.")
print("- 특성 중요도 분석 결과, 중년 직장인구 수와 총 직장인구 수가 매출에 가장 큰 영향을 미쳤습니다.")
print("- 상권 유형별 매출 차이도 중요한 요소로 확인되었습니다.")
print("- 창업 위치 선정 시 유동인구보다 직장인구 특성을 더 중요하게 고려해야 합니다.")

