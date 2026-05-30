use serde::{Deserialize, Serialize};
use chrono::NaiveDate;

pub mod loader;
pub mod features;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyRecord {
    pub date: NaiveDate,
    pub visits: u32,
    pub day_of_week: u32,
    pub month: u32,
    pub is_holiday: bool,
    pub is_pre_holiday: bool,
    pub temp_avg: f64,
    pub fine_dust: f64,
}

#[derive(Debug, Clone)]
pub struct FeatureRow {
    pub date: NaiveDate,
    pub visits: f64,
    pub lag_7: f64,
    pub lag_14: f64,
    pub lag_28: f64,
    pub roll_mean_7: f64,
    pub roll_std_7: f64,
    pub roll_mean_14: f64,
    pub roll_std_14: f64,
    pub dow_sin: f64,
    pub dow_cos: f64,
    pub month_sin: f64,
    pub month_cos: f64,
    pub is_holiday: f64,
    pub is_pre_holiday: f64,
    pub temp_avg: f64,
    pub fine_dust: f64,
}
