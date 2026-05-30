# 🏥 ER Forecast (응급실 수요 예측 시스템)

**ER Forecast**는 한국 응급실 방문 패턴을 학습하여 미래의 환자 수요를 예측하는 고성능 CLI 도구입니다. 외부 머신러닝 라이브러리(Python, Scikit-learn 등)에 의존하지 않고, **순수 Rust(Pure Rust)**로 구현된 Gradient Boosted Trees(GBM) 엔진을 탑재하고 있습니다.

[![Rust Edition 2024](https://img.shields.io/badge/Rust-Edition%202024-blue?logo=rust)](https://doc.rust-lang.org/edition-guide/rust-2024/index.html)
[![Version](https://img.shields.io/badge/version-1.0.0-green)](Cargo.toml)

---

## ✨ 핵심 기능

-   **한국형 패턴 시뮬레이터**: 요일 효과(주말 급증), 공휴일 가중치, 계절성, 기온 및 미세먼지 영향을 반영한 가상 데이터 생성.
-   **순수 Rust GBM 엔진**: 밑바닥부터 구현된 Gradient Boosting 알고리즘을 통해 외부 의존성 없는 빠른 학습 및 추론.
-   **고급 피처 엔지니어링**: 과거 방문량(Lag), 이동 평균(Rolling Stats), 주기적 인코딩(Cyclical Encoding) 자동 생성.
-   **유연한 예측 인터페이스**: JSON 및 CSV 형식을 지원하여 타 시스템(대시보드 등)과의 연동 용이.

---

## 🚀 시작하기

### 설치 및 빌드

최신 안정판 Rust(1.85.0+ 권장)가 설치되어 있어야 합니다.

```bash
git clone <repository-url>
cd patient-demand-forecast
cargo build --release
```

### 기본 워크플로우

#### 1. 시뮬레이션 데이터 생성 (730일치)
한국 응급실 특성을 반영한 2년치 학습 데이터를 생성합니다.
```bash
cargo run -- simulate --days 730 --start 2023-01-01 --output data/simulated.csv
```

#### 2. 모델 학습
생성된 데이터를 바탕으로 GBM 모델을 학습시킵니다.
```bash
cargo run -- train --input data/simulated.csv --model-out models/lgbm.json --val-ratio 0.2
```

#### 3. 수요 예측 (Horizon: 14일)
특정 날짜로부터 2주간의 수요를 예측합니다. (JSON 출력)
```bash
cargo run -- predict --model models/lgbm.json --start 2025-06-01 --horizon 14 --format json
```

#### 4. 모델 성능 평가
학습된 모델의 오차율(MAE, MAPE, RMSE)을 확인합니다.
```bash
cargo run -- evaluate --model models/lgbm.json --input data/simulated.csv
```

---

## 📊 피처 구성 (Features)

| 카테고리 | 피처명 | 설명 |
| :--- | :--- | :--- |
| **과거 지표** | `lag_7/14/28` | 7·14·28일 전의 실제 환자 수 |
| **통계 지표** | `roll_mean_7/14` | 최근 7·14일간의 환자 수 이동 평균 |
| **통계 지표** | `roll_std_7` | 최근 7일간의 변동성 (표준편차) |
| **시간 주기** | `dow_sin/cos` | 요일(0-6)의 순환성 인코딩 |
| **시간 주기** | `month_sin/cos` | 월(1-12)의 계절적 순환성 인코딩 |
| **특수 날짜** | `is_holiday` | 한국 공휴일 여부 (가중치 적용) |
| **특수 날짜** | `is_pre_holiday` | 연휴 전날 여부 |
| **환경 변수** | `temp_avg` | 일평균 기온 (극단적 기후 영향 반영) |
| **환경 변수** | `fine_dust` | 미세먼지 농도 (PM2.5) |

---

## 🛠 프로젝트 구조

```text
src/
├── main.rs           # CLI 진입점 및 인자 파싱 (clap)
├── commands/         # 각 실행 명령 (simulate, train, predict, evaluate)
├── data/             # 데이터 로더 및 피처 엔지니어링 로직
├── models/           # GBM(Gradient Boosted Trees) 알고리즘 구현
└── utils/            # 성능 지표(Metrics) 및 공통 유틸리티
```

---

## ⚖️ 라이선스

이 프로젝트는 학습 및 연구 목적으로 제작되었습니다. 자유롭게 활용하시되, 의료 현장 적용 시 반드시 전문가의 검증을 거치시기 바랍니다.
