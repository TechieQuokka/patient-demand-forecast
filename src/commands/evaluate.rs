use anyhow::Result;
use crate::data::{loader, features};
use crate::models::gbm::GBMModel;
use crate::utils::metrics;

pub fn run(model_path: &str, input_path: &str) -> Result<()> {
    let model = GBMModel::load(model_path)?;
    let records = loader::load_csv(input_path)?;
    let rows = features::generate_features(&records);

    let actual: Vec<f64> = rows.iter().map(|r| r.visits).collect();
    let predicted: Vec<f64> = rows.iter()
        .map(|r| {
            let x = crate::models::gbm::row_to_features(r);
            model.predict_one(&x)
        })
        .collect();

    let mae = metrics::mae(&actual, &predicted);
    let mape = metrics::mape(&actual, &predicted);
    let rmse = metrics::rmse(&actual, &predicted);

    println!("모델 평가 결과 ({})", input_path);
    println!("----------------------------------");
    println!("MAE:  {:.2}", mae);
    println!("MAPE: {:.2}%", mape);
    println!("RMSE: {:.2}", rmse);
    println!("----------------------------------");
    
    // 간단한 피처 중요도 출력 (Split count 기준)
    // GBMModel에 feature importance를 계산하는 기능이 필요함.
    // 여기서는 생략하거나 간단히 이름만 출력
    println!("피처 목록: {:?}", model.feature_names);

    Ok(())
}
