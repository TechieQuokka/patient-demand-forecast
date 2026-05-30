use anyhow::Result;
use crate::data::{loader, features};
use crate::models::{ModelParams, gbm::GBMModel};

pub fn run(input: &str, model_out: &str, val_ratio: f64) -> Result<()> {
    println!("데이터 로딩 중: {}...", input);
    let records = loader::load_csv(input)?;
    
    println!("피처 생성 중...");
    let rows = features::generate_features(&records);
    
    let n_val = (rows.len() as f64 * val_ratio) as usize;
    let n_train = rows.len() - n_val;
    
    let train_rows = &rows[..n_train];
    let val_rows = &rows[n_train..];
    
    println!("모델 학습 시작 (학습: {}, 검증: {})...", train_rows.len(), val_rows.len());
    let params = ModelParams::default();
    let mut model = GBMModel::new(params);
    
    model.fit(train_rows)?;
    
    println!("모델 저장 중: {}...", model_out);
    model.save(model_out)?;
    
    println!("학습 완료!");
    Ok(())
}
