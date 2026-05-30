use serde::{Deserialize, Serialize};
use chrono::NaiveDate;

pub mod gbm;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelParams {
    pub n_estimators: usize,
    pub max_depth: usize,
    pub learning_rate: f64,
    pub min_samples_leaf: usize,
}

impl Default for ModelParams {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            max_depth: 5,
            learning_rate: 0.1,
            min_samples_leaf: 5,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ForecastResult {
    pub date: NaiveDate,
    pub predicted_visits: f64,
}
