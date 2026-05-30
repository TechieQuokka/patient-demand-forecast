/// Mean Absolute Error
pub fn mae(actual: &[f64], predicted: &[f64]) -> f64 {
    if actual.is_empty() { return 0.0; }
    actual.iter().zip(predicted)
        .map(|(a, p)| (a - p).abs())
        .sum::<f64>() / actual.len() as f64
}

/// Mean Absolute Percentage Error (actual=0인 샘플 제외)
pub fn mape(actual: &[f64], predicted: &[f64]) -> f64 {
    let valid: Vec<(f64, f64)> = actual.iter().zip(predicted)
        .filter(|(a, _)| **a > 1e-6)
        .map(|(a, p)| (*a, *p))
        .collect();
    if valid.is_empty() { return 0.0; }
    valid.iter()
        .map(|(a, p)| ((a - p) / a).abs())
        .sum::<f64>() / valid.len() as f64 * 100.0
}

/// Root Mean Squared Error
pub fn rmse(actual: &[f64], predicted: &[f64]) -> f64 {
    if actual.is_empty() { return 0.0; }
    let mse = actual.iter().zip(predicted)
        .map(|(a, p)| (a - p).powi(2))
        .sum::<f64>() / actual.len() as f64;
    mse.sqrt()
}

/// R² (결정계수)
pub fn r2(actual: &[f64], predicted: &[f64]) -> f64 {
    if actual.is_empty() { return 0.0; }
    let mean = actual.iter().sum::<f64>() / actual.len() as f64;
    let ss_tot: f64 = actual.iter().map(|a| (a - mean).powi(2)).sum();
    let ss_res: f64 = actual.iter().zip(predicted).map(|(a, p)| (a - p).powi(2)).sum();
    if ss_tot < 1e-10 { return 1.0; }
    1.0 - ss_res / ss_tot
}
