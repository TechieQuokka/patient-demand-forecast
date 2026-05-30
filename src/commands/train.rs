use anyhow::Result;
use crate::data::{loader, features};
use crate::models::{ModelParams, gbm::GBMModel};
use crate::utils::metrics;

pub fn run(input: &str, model_out: &str, val_ratio: f64, params: ModelParams) -> Result<()> {
    anyhow::ensure!(
        (0.05..0.5).contains(&val_ratio),
        "val_ratio는 0.05~0.5 사이여야 합니다 (현재: {})",
        val_ratio
    );

    println!("▶ 데이터 로딩: {}", input);
    let records = loader::load_csv(input)?;

    println!("▶ 피처 생성 중...");
    let rows = features::generate_features(&records);
    anyhow::ensure!(rows.len() >= 60, "피처 행이 너무 적습니다 ({}개). 최소 88일치 데이터가 필요합니다.", rows.len());

    // 시계열 분할: 앞부분 학습, 뒷부분 검증 (셔플 금지)
    let n_val   = ((rows.len() as f64) * val_ratio).round() as usize;
    let n_train = rows.len() - n_val;
    let train_rows = &rows[..n_train];
    let val_rows   = &rows[n_train..];

    println!("  학습: {} 행 | 검증: {} 행", train_rows.len(), val_rows.len());
    println!("  하이퍼파라미터: n_estimators={}, max_depth={}, lr={}, min_leaf={}, early_stopping={}",
        params.n_estimators, params.max_depth, params.learning_rate,
        params.min_samples_leaf, params.early_stopping_rounds);

    let mut model = GBMModel::new(params);

    println!("\n▶ 모델 학습 시작...");
    model.fit(train_rows, val_rows)?;

    // ── 최종 검증 지표 ──────────────────────────────────────────
    let val_actual: Vec<f64> = val_rows.iter().map(|r| r.visits).collect();
    let val_preds:  Vec<f64> = val_rows.iter()
        .map(|r| model.predict_one(&crate::models::gbm::row_to_features(r)))
        .collect();
    let train_actual: Vec<f64> = train_rows.iter().map(|r| r.visits).collect();
    let train_preds:  Vec<f64> = train_rows.iter()
        .map(|r| model.predict_one(&crate::models::gbm::row_to_features(r)))
        .collect();

    println!("\n── 학습 결과 ─────────────────────────────────────");
    println!("  [Train] MAE={:.2}  MAPE={:.2}%  RMSE={:.2}  R²={:.4}",
        metrics::mae(&train_actual, &train_preds),
        metrics::mape(&train_actual, &train_preds),
        metrics::rmse(&train_actual, &train_preds),
        metrics::r2(&train_actual, &train_preds));
    println!("  [Val  ] MAE={:.2}  MAPE={:.2}%  RMSE={:.2}  R²={:.4}",
        metrics::mae(&val_actual, &val_preds),
        metrics::mape(&val_actual, &val_preds),
        metrics::rmse(&val_actual, &val_preds),
        metrics::r2(&val_actual, &val_preds));
    println!("  사용된 트리 수: {} | 잔차 σ: {:.2}", model.trees.len(), model.residual_std);

    // ── 피처 중요도 ─────────────────────────────────────────────
    println!("\n── 피처 중요도 (Split Count 기준) ────────────────");
    for (name, score) in model.feature_importance() {
        let bar_len = (score * 40.0).round() as usize;
        println!("  {:16} {:5.1}%  {}", name, score * 100.0, "█".repeat(bar_len));
    }

    // ── 저장 ────────────────────────────────────────────────────
    println!("\n▶ 모델 저장: {}", model_out);
    model.save(model_out)?;
    println!("✓ 학습 완료");

    Ok(())
}
