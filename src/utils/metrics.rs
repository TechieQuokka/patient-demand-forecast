pub fn mae(actual: &[f64], predicted: &[f64]) -> f64 {
    if actual.is_empty() { return 0.0; }
    actual.iter().zip(predicted.iter())
        .map(|(a, p)| (a - p).abs())
        .sum::<f64>() / actual.len() as f64
}

pub fn mape(actual: &[f64], predicted: &[f64]) -> f64 {
    if actual.is_empty() { return 0.0; }
    let sum: f64 = actual.iter().zip(predicted.iter())
        .filter(|&(a, _)| *a != 0.0)
        .map(|(a, p)| (a - p).abs() / a)
        .sum();
    sum / actual.len() as f64 * 100.0
}

pub fn rmse(actual: &[f64], predicted: &[f64]) -> f64 {
    if actual.is_empty() { return 0.0; }
    let mse = actual.iter().zip(predicted.iter())
        .map(|(a, p)| (a - p).powi(2))
        .sum::<f64>() / actual.len() as f64;
    mse.sqrt()
}
