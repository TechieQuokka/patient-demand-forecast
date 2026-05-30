use std::f64::consts::PI;
use chrono::Datelike;
use crate::data::{DailyRecord, FeatureRow};

/// DailyRecord 슬라이스 → FeatureRow 목록
/// 최소 28개 레코드가 선행되어야 피처를 생성할 수 있음.
pub fn generate_features(records: &[DailyRecord]) -> Vec<FeatureRow> {
    records
        .iter()
        .enumerate()
        .filter(|(i, _)| *i >= 28)
        .map(|(i, current)| {
            // ── Lag ──────────────────────────────────────────────
            let lag_7  = records[i - 7].visits as f64;
            let lag_14 = records[i - 14].visits as f64;
            let lag_28 = records[i - 28].visits as f64;

            // ── Rolling stats ────────────────────────────────────
            let w7:  Vec<f64> = records[i - 7..i].iter().map(|r| r.visits as f64).collect();
            let w14: Vec<f64> = records[i - 14..i].iter().map(|r| r.visits as f64).collect();

            let roll_mean_7  = mean(&w7);
            let roll_std_7   = std_dev(&w7);
            let roll_mean_14 = mean(&w14);
            let roll_std_14  = std_dev(&w14);

            // ── Cyclical encoding ────────────────────────────────
            let dow   = current.date.weekday().num_days_from_monday() as f64;
            let dow_sin   = (2.0 * PI * dow / 7.0).sin();
            let dow_cos   = (2.0 * PI * dow / 7.0).cos();

            let month = current.date.month() as f64 - 1.0;
            let month_sin = (2.0 * PI * month / 12.0).sin();
            let month_cos = (2.0 * PI * month / 12.0).cos();

            FeatureRow {
                date: current.date,
                visits: current.visits as f64,
                lag_7, lag_14, lag_28,
                roll_mean_7, roll_std_7,
                roll_mean_14, roll_std_14,
                dow_sin, dow_cos,
                month_sin, month_cos,
                is_holiday:     if current.is_holiday     { 1.0 } else { 0.0 },
                is_pre_holiday: if current.is_pre_holiday { 1.0 } else { 0.0 },
                temp_avg:   current.temp_avg,
                fine_dust:  current.fine_dust,
            }
        })
        .collect()
}

// ── 내부 통계 헬퍼 ─────────────────────────────────────────────

pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    data.iter().sum::<f64>() / data.len() as f64
}

pub fn std_dev(data: &[f64]) -> f64 {
    if data.len() < 2 { return 0.0; }
    let m = mean(data);
    let var = data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / data.len() as f64;
    var.sqrt()
}
