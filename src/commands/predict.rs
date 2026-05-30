use anyhow::Result;
use chrono::{NaiveDate, Datelike};
use std::f64::consts::PI;

use crate::data::{loader, features, DailyRecord};
use crate::models::{gbm::GBMModel, ForecastResult};

pub fn run(
    model_path: &str,
    start_str: &str,
    horizon: u32,
    history_path: &str,
    format: &str,
    temp_override: Option<Vec<f64>>,
    dust_override: Option<Vec<f64>>,
) -> Result<()> {
    let model = GBMModel::load(model_path)?;
    let start_date = NaiveDate::parse_from_str(start_str, "%Y-%m-%d")?;

    // 과거 데이터 로드 (Lag 피처 계산에 필요)
    let mut records = loader::load_csv(history_path)?;
    records.sort_by_key(|r| r.date);

    anyhow::ensure!(
        records.len() >= 28,
        "과거 데이터가 28일 미만입니다 ({} 행). Lag 피처를 계산할 수 없습니다.",
        records.len()
    );

    let mut results = Vec::with_capacity(horizon as usize);

    for i in 0..horizon {
        let current_date = start_date + chrono::Duration::days(i as i64);

        // 기온/미세먼지: 오버라이드 → 계절 평균 fallback
        let temp = match &temp_override {
            Some(v) => v[i as usize],
            None    => seasonal_temp(current_date),
        };
        let dust = match &dust_override {
            Some(v) => v[i as usize],
            None    => seasonal_dust(current_date),
        };

        // 더미 레코드 추가 (visits=0, 예측 후 업데이트)
        let dummy = DailyRecord {
            date:           current_date,
            visits:         0,
            day_of_week:    current_date.weekday().num_days_from_monday(),
            month:          current_date.month(),
            is_holiday:     is_korean_holiday(current_date),
            is_pre_holiday: is_pre_holiday(current_date),
            temp_avg:       (temp * 10.0).round() / 10.0,
            fine_dust:      (dust * 10.0).round() / 10.0,
        };
        records.push(dummy);

        // 피처 생성 → 마지막 행 추론
        let all_features = features::generate_features(&records);
        let feat_row = all_features
            .last()
            .ok_or_else(|| anyhow::anyhow!("피처 생성 실패 (인덱스 {})", i))?;

        let x = crate::models::gbm::row_to_features(feat_row);
        let (pred, lower, upper) = model.predict_with_interval(&x);

        // 예측값을 다음 루프의 Lag로 활용 (자기회귀)
        if let Some(r) = records.last_mut() {
            r.visits = pred.round() as u32;
        }

        results.push(ForecastResult {
            date: current_date,
            predicted_visits: pred,
            lower_bound: lower,
            upper_bound: upper,
        });
    }

    // ── 출력 ────────────────────────────────────────────────────
    match format {
        "json" => println!("{}", serde_json::to_string_pretty(&results)?),
        "csv"  => {
            println!("date,predicted_visits,lower_bound,upper_bound");
            for r in &results {
                println!("{},{:.1},{:.1},{:.1}",
                    r.date, r.predicted_visits, r.lower_bound, r.upper_bound);
            }
        }
        _ => anyhow::bail!("지원하지 않는 출력 형식: '{}' (json | csv)", format),
    }

    Ok(())
}

// ── 환경 변수 계절 평균 (오버라이드 없을 때 fallback) ────────────────────────

fn seasonal_temp(date: NaiveDate) -> f64 {
    let day_of_year = date.ordinal() as f64;
    12.5 + 17.5 * ((2.0 * PI * (day_of_year - 30.0) / 365.0).sin())
}

fn seasonal_dust(date: NaiveDate) -> f64 {
    match date.month() {
        3..=5  => 45.0,
        6..=8  => 20.0,
        9..=11 => 30.0,
        _      => 40.0,
    }
}

fn is_korean_holiday(date: NaiveDate) -> bool {
    matches!(
        (date.month(), date.day()),
        (1,1)|(3,1)|(5,5)|(6,6)|(8,15)|(10,3)|(10,9)|(12,25)
    )
}

fn is_pre_holiday(date: NaiveDate) -> bool {
    is_korean_holiday(date + chrono::Duration::days(1))
}
