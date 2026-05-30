use anyhow::Result;
use crate::data::{loader, features};
use crate::models::gbm::{GBMModel, row_to_features};
use crate::utils::metrics;

pub fn run(model_path: &str, input_path: &str, output: Option<&str>) -> Result<()> {
    let model   = GBMModel::load(model_path)?;
    let records = loader::load_csv(input_path)?;
    let rows    = features::generate_features(&records);

    anyhow::ensure!(!rows.is_empty(), "피처 행이 0개입니다. 데이터를 확인하세요.");

    let actual: Vec<f64>    = rows.iter().map(|r| r.visits).collect();
    let predicted: Vec<f64> = rows.iter()
        .map(|r| model.predict_one(&row_to_features(r)))
        .collect();

    // ── 지표 출력 ─────────────────────────────────────────────
    let mae  = metrics::mae(&actual, &predicted);
    let mape = metrics::mape(&actual, &predicted);
    let rmse = metrics::rmse(&actual, &predicted);
    let r2   = metrics::r2(&actual, &predicted);

    println!("── 모델 평가: {} ────────────────────────────────", input_path);
    println!("  샘플 수: {}", rows.len());
    println!("  MAE:  {:.2}  (평균 절대 오차 - 환자 수 기준)", mae);
    println!("  MAPE: {:.2}%  (평균 절대 비율 오차)", mape);
    println!("  RMSE: {:.2}  (제곱근 평균 제곱 오차)", rmse);
    println!("  R²:   {:.4}  (설명력, 1.0이 완벽)", r2);
    println!("  사용 트리 수: {} | 잔차 σ: {:.2}", model.trees.len(), model.residual_std);

    // ── 피처 중요도 ──────────────────────────────────────────
    println!("\n── 피처 중요도 (Split Count) ──────────────────");
    for (name, score) in model.feature_importance() {
        let bar = (score * 40.0).round() as usize;
        println!("  {:16} {:5.1}%  {}", name, score * 100.0, "█".repeat(bar));
    }

    // ── 최악 예측 Top 5 ─────────────────────────────────────
    let mut errors: Vec<(usize, f64)> = actual.iter().zip(predicted.iter())
        .enumerate()
        .map(|(i, (a, p))| (i, (a - p).abs()))
        .collect();
    errors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\n── 예측 오차 Top 5 ────────────────────────────");
    println!("  {:12}  {:>10}  {:>10}  {:>10}", "날짜", "실제", "예측", "오차");
    for (idx, err) in errors.iter().take(5) {
        println!("  {}  {:>10.0}  {:>10.1}  {:>10.1}",
            rows[*idx].date, actual[*idx], predicted[*idx], err);
    }

    // ── CSV 저장 (옵션) ──────────────────────────────────────
    if let Some(path) = output {
        let results: Vec<crate::models::ForecastResult> = rows.iter()
            .zip(predicted.iter())
            .map(|(r, &p)| {
                let margin = 1.96 * model.residual_std;
                crate::models::ForecastResult {
                    date: r.date,
                    predicted_visits: p,
                    lower_bound: (p - margin).max(0.0),
                    upper_bound: p + margin,
                }
            })
            .collect();
        loader::save_predictions_csv(&results, path)?;
        println!("\n✓ 예측 결과 저장: {}", path);
    }

    Ok(())
}
