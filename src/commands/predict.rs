use anyhow::{Result, anyhow};
use chrono::NaiveDate;
use crate::data::{loader, features};
use crate::models::{gbm::GBMModel, ForecastResult};

pub fn run(model_path: &str, start_str: &str, horizon: u32, format: &str) -> Result<()> {
    let model = GBMModel::load(model_path)?;
    let start_date = NaiveDate::parse_from_str(start_str, "%Y-%m-%d")?;
    
    // 예측을 위해 과거 데이터가 필요함 (최소 28일)
    // 여기서는 편의상 data/simulated.csv를 로드한다고 가정
    let history_path = "data/simulated.csv";
    if !std::path::Path::new(history_path).exists() {
        return Err(anyhow!("예측을 위한 과거 데이터({})가 존재하지 않습니다. 먼저 simulate를 실행하세요.", history_path));
    }
    
    let mut records = loader::load_csv(history_path)?;
    records.sort_by_key(|r| r.date);

    let mut results = Vec::new();

    for i in 0..horizon {
        let current_date = start_date + chrono::Duration::days(i as i64);
        
        // 현재 날짜에 대한 피처 생성
        // DailyRecord를 먼저 생성 (방문수는 모르므로 0 또는 더미)
        // 사실 feature generation은 과거 records가 필요함.
        
        // 간단하게 하기 위해:
        // 1. records에 '예측할 날짜'를 추가 (방문수는 dummy)
        // 2. feature_generate 수행
        // 3. 마지막 행의 피처로 예측
        // 4. 예측된 값을 records에 업데이트 (다음 루프의 lag로 사용)
        
        use crate::data::DailyRecord;
        use chrono::Datelike;
        
        let dummy_record = DailyRecord {
            date: current_date,
            visits: 0, // 예측할 대상
            day_of_week: current_date.weekday().num_days_from_monday(),
            month: current_date.month(),
            is_holiday: false, // 단순화
            is_pre_holiday: false,
            temp_avg: 15.0, // 더미
            fine_dust: 30.0, // 더미
        };
        
        records.push(dummy_record);
        let all_features = features::generate_features(&records);
        let current_features = all_features.last().ok_or_else(|| anyhow!("피처 생성 실패"))?;
        
        let x = crate::models::gbm::row_to_features(current_features);
        let pred = model.predict_one(&x);
        
        // 예측값으로 record 업데이트
        if let Some(r) = records.last_mut() {
            r.visits = pred.round() as u32;
        }
        
        results.push(ForecastResult {
            date: current_date,
            predicted_visits: pred,
        });
    }

    if format == "json" {
        println!("{}", serde_json::to_string_pretty(&results)?);
    } else {
        println!("date,predicted_visits");
        for r in results {
            println!("{},{:.1}", r.date, r.predicted_visits);
        }
    }

    Ok(())
}
