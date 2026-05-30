# 🏥 ER Forecast (응급실 수요 예측 시스템)

**ER Forecast**는 한국 응급실 방문 패턴을 학습하여 미래의 환자 수요를 예측하는 고성능 CLI 도구입니다. 외부 머신러닝 라이브러리(Python, Scikit-learn 등)에 의존하지 않고, **순수 Rust(Pure Rust)**로 구현된 Gradient Boosted Trees(GBM) 엔진을 탑재하고 있습니다.

[![Rust Edition 2024](https://img.shields.io/badge/Rust-Edition%202024-blue?logo=rust)](https://doc.rust-lang.org/edition-guide/rust-2024/index.html)
[![Rust 1.85+](https://img.shields.io/badge/rustc-1.85%2B-orange?logo=rust)](https://blog.rust-lang.org/2025/02/20/Rust-1.85.0/)
[![Version](https://img.shields.io/badge/version-2.0.0-green)](Cargo.toml)

---

## ✨ 핵심 기능

- **한국형 패턴 시뮬레이터**: 요일 효과(주말 급증), 공휴일 가중치, 계절성, 기온 및 미세먼지 영향을 반영한 가상 데이터 생성. `--seed` 옵션으로 완전한 재현성 보장.
- **순수 Rust GBM 엔진**: 밑바닥부터 구현된 Gradient Boosting 알고리즘. Pre-sorted 분할로 트리 빌드 속도를 최적화.
- **Early Stopping**: 검증 MAE가 일정 라운드 동안 개선되지 않으면 자동 조기 종료. 과적합 방지 및 최적 트리 수 자동 결정.
- **신뢰 구간 예측**: 학습 잔차 표준편차 기반 95% 신뢰 구간(`lower_bound` / `upper_bound`)을 예측값과 함께 출력.
- **피처 중요도**: Split-count 기반 피처 중요도를 막대 그래프로 시각화.
- **고급 피처 엔지니어링**: Lag, Rolling Stats(평균·표준편차), Cyclical Encoding 자동 생성. 16개 피처.
- **유연한 예측 인터페이스**: JSON 및 CSV 형식 지원. 기온·미세먼지 실측값 주입(`--temps`, `--dusts`) 가능.

---

## 🚀 시작하기

### 요구 사항

```
Rust 1.85.0 이상 (Edition 2024)
```

```bash
rustup update stable
```

### 빌드

```bash
git clone <repository-url>
cd patient-demand-forecast
cargo build --release
```

### 기본 워크플로우

#### 1. 시뮬레이션 데이터 생성

한국 응급실 특성을 반영한 2년치 학습 데이터를 생성합니다. `--seed`로 재현성을 보장합니다.

```bash
cargo run --release -- simulate \
  --days 730 \
  --start 2023-01-01 \
  --output data/simulated.csv \
  --seed 42
```

#### 2. 모델 학습

Early Stopping과 하이퍼파라미터를 직접 지정할 수 있습니다. 학습 완료 후 Train/Val 지표와 피처 중요도가 출력됩니다.

```bash
cargo run --release -- train \
  --input data/simulated.csv \
  --model-out models/er_model.json \
  --val-ratio 0.2 \
  --n-estimators 200 \
  --max-depth 5 \
  --learning-rate 0.05 \
  --early-stopping-rounds 15
```

출력 예시:
```
  [Round  70] val_MAE = 8.03 | best = 8.02 | no_improve = 1/15
  ✓ Early stopping at round 84 (best val_MAE=8.02)

── 학습 결과 ─────────────────────────────────────
  [Train] MAE=4.48  MAPE=3.51%  RMSE=5.68  R²=0.9320
  [Val  ] MAE=8.02  MAPE=6.32%  RMSE=9.61  R²=0.7820

── 피처 중요도 (Split Count 기준) ────────────────
  fine_dust         11.6%  █████
  dow_sin           11.2%  ████
  temp_avg          10.8%  ████
  ...
```

#### 3. 수요 예측 (Horizon: 14일)

예측값과 함께 95% 신뢰 구간이 출력됩니다. 기온·미세먼지 실측값을 주입하면 정확도가 향상됩니다.

```bash
# 기본 (계절 평균 기온/미세먼지 사용)
cargo run --release -- predict \
  --model models/er_model.json \
  --start 2025-06-01 \
  --horizon 14 \
  --history data/simulated.csv \
  --format json
```

```bash
# 실측 기상 데이터 주입
cargo run --release -- predict \
  --model models/er_model.json \
  --start 2025-06-01 \
  --horizon 3 \
  --history data/simulated.csv \
  --format csv \
  --temps "22.1,23.5,21.0" \
  --dusts "35.0,42.0,28.0"
```

출력 예시 (JSON):
```json
[
  {
    "date": "2025-06-01",
    "predicted_visits": 128.4,
    "lower_bound": 117.2,
    "upper_bound": 139.6
  }
]
```

#### 4. 모델 성능 평가

MAE, MAPE, RMSE, R² 지표와 피처 중요도, 예측 오차 Top 5를 출력합니다. `--output`으로 예측 결과를 CSV로 저장할 수 있습니다.

```bash
cargo run --release -- evaluate \
  --model models/er_model.json \
  --input data/simulated.csv \
  --output data/eval_predictions.csv
```

---

## 📊 피처 구성 (16개)

| 카테고리 | 피처명 | 설명 |
| :--- | :--- | :--- |
| **과거 지표** | `lag_7 / lag_14 / lag_28` | 7·14·28일 전의 실제 환자 수 |
| **통계 지표** | `roll_mean_7 / roll_mean_14` | 최근 7·14일간 이동 평균 |
| **통계 지표** | `roll_std_7 / roll_std_14` | 최근 7·14일간 변동성 (표준편차) |
| **시간 주기** | `dow_sin / dow_cos` | 요일(0-6) 순환성 인코딩 |
| **시간 주기** | `month_sin / month_cos` | 월(1-12) 계절적 순환성 인코딩 |
| **특수 날짜** | `is_holiday` | 한국 법정 공휴일 여부 |
| **특수 날짜** | `is_pre_holiday` | 연휴 전날 여부 |
| **환경 변수** | `temp_avg` | 일평균 기온 (극단 기후 영향 반영) |
| **환경 변수** | `fine_dust` | 미세먼지 농도 (PM2.5) |

---

## ⚙️ CLI 옵션 전체 정리

### `simulate`

| 옵션 | 기본값 | 설명 |
| :--- | :--- | :--- |
| `--days` | `730` | 생성할 기간 (일수) |
| `--start` | `2023-01-01` | 시작 날짜 |
| `--output` | `data/simulated.csv` | 출력 CSV 경로 |
| `--seed` | `42` | 랜덤 시드 (재현성 보장) |

### `train`

| 옵션 | 기본값 | 설명 |
| :--- | :--- | :--- |
| `--input` | `data/simulated.csv` | 학습 데이터 경로 |
| `--model-out` | `models/er_model.json` | 모델 저장 경로 |
| `--val-ratio` | `0.2` | 검증 데이터 비율 |
| `--n-estimators` | `100` | 최대 트리 수 |
| `--max-depth` | `5` | 트리 최대 깊이 |
| `--learning-rate` | `0.1` | 학습률 |
| `--min-samples-leaf` | `5` | 리프 최소 샘플 수 |
| `--early-stopping-rounds` | `10` | 조기 종료 기준 라운드 |

### `predict`

| 옵션 | 기본값 | 설명 |
| :--- | :--- | :--- |
| `--model` | `models/er_model.json` | 모델 경로 |
| `--start` | *(필수)* | 예측 시작 날짜 |
| `--horizon` | `14` | 예측 기간 (일수) |
| `--history` | `data/simulated.csv` | Lag 계산용 과거 데이터 |
| `--format` | `json` | 출력 형식 (`json` \| `csv`) |
| `--temps` | *(계절 평균)* | 기온 오버라이드 (쉼표 구분) |
| `--dusts` | *(계절 평균)* | 미세먼지 오버라이드 (쉼표 구분) |

### `evaluate`

| 옵션 | 기본값 | 설명 |
| :--- | :--- | :--- |
| `--model` | `models/er_model.json` | 모델 경로 |
| `--input` | `data/simulated.csv` | 평가 데이터 경로 |
| `--output` | *(없음)* | 예측 결과 CSV 저장 경로 (선택) |

---

## 🛠 프로젝트 구조

```text
src/
├── main.rs               # CLI 진입점 및 인자 파싱 (clap)
├── commands/
│   ├── simulate.rs       # 시뮬레이션 데이터 생성
│   ├── train.rs          # 모델 학습 + Early Stopping + 피처 중요도 출력
│   ├── predict.rs        # 자기회귀 예측 + 신뢰 구간
│   └── evaluate.rs       # 성능 평가 + 오차 Top 5
├── data/
│   ├── loader.rs         # CSV I/O
│   └── features.rs       # 피처 엔지니어링 (Lag, Rolling, Cyclical)
├── models/
│   └── gbm.rs            # GBM 구현 (Pre-sorted 분할, 피처 중요도)
└── utils/
    └── metrics.rs        # MAE / MAPE / RMSE / R²
```

---

## 🔬 v2.0 주요 변경 사항

| 항목 | v1.0 | v2.0 |
| :--- | :--- | :--- |
| 재현성 | ❌ 전역 RNG 사용 | ✅ `--seed`로 완전 재현 |
| Early Stopping | ❌ 없음 | ✅ `--early-stopping-rounds` |
| 신뢰 구간 | ❌ 없음 | ✅ 95% 구간 자동 출력 |
| 피처 중요도 | ❌ 미구현 | ✅ Split-count 막대 시각화 |
| 검증 지표 | ❌ 학습 후 미사용 | ✅ Train/Val MAE·MAPE·RMSE·R² |
| 환경 피처 (예측 시) | ❌ 상수(15.0 / 30.0) 고정 | ✅ 계절 평균 자동 계산 + 실측값 주입 |
| 트리 빌드 성능 | ❌ 매 노드 O(n log n) 정렬 | ✅ Pre-sorted로 O(n) 분할 |
| 피처 수 | 14개 | 16개 (`roll_std_14` 추가) |
| `--history` 인자 | ❌ 하드코딩 | ✅ 명시적 경로 지정 |

---

## ⚖️ 라이선스

이 프로젝트는 학습 및 연구 목적으로 제작되었습니다. 자유롭게 활용하시되, 의료 현장 적용 시 반드시 전문가의 검증을 거치시기 바랍니다.
