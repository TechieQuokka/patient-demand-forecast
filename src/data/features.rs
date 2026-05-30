use std::f64::consts::PI;
use chrono::Datelike;
use crate::data::{DailyRecord, FeatureRow};

pub fn generate_features(records: &[DailyRecord]) -> Vec<FeatureRow> {
    let mut features = Vec::new();

    for i in 28..records.len() {
        let current = &records[i];
        
        // Lags
        let lag_7 = records[i - 7].visits as f64;
        let lag_14 = records[i - 14].visits as f64;
        let lag_28 = records[i - 28].visits as f64;

        // Rolling mean/std
        let window_7: Vec<f64> = records[i-7..i].iter().map(|r| r.visits as f64).collect();
        let roll_mean_7 = mean(&window_7);
        let roll_std_7 = std(&window_7);

        let window_14: Vec<f64> = records[i-14..i].iter().map(|r| r.visits as f64).collect();
        let roll_mean_14 = mean(&window_14);

        // Cyclical encoding
        let dow = current.date.weekday().num_days_from_monday() as f64;
        let dow_sin = (2.0 * PI * dow / 7.0).sin();
        let dow_cos = (2.0 * PI * dow / 7.0).cos();

        let month = current.date.month() as f64 - 1.0;
        let month_sin = (2.0 * PI * month / 12.0).sin();
        let month_cos = (2.0 * PI * month / 12.0).cos();

        features.push(FeatureRow {
            date: current.date,
            visits: current.visits as f64,
            lag_7,
            lag_14,
            lag_28,
            roll_mean_7,
            roll_std_7,
            roll_mean_14,
            dow_sin,
            dow_cos,
            month_sin,
            month_cos,
            is_holiday: if current.is_holiday { 1.0 } else { 0.0 },
            is_pre_holiday: if current.is_pre_holiday { 1.0 } else { 0.0 },
            temp_avg: current.temp_avg,
            fine_dust: current.fine_dust,
        });
    }

    features
}

fn mean(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    data.iter().sum::<f64>() / data.len() as f64
}

fn std(data: &[f64]) -> f64 {
    if data.len() < 2 { return 0.0; }
    let m = mean(data);
    let var = data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / data.len() as f64;
    var.sqrt()
}
