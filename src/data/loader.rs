use std::path::Path;
use anyhow::{Context, Result};
use chrono::{NaiveDate, Duration};
use crate::data::DailyRecord;

pub fn date_range(start: NaiveDate, days: u32) -> Vec<NaiveDate> {
    (0..days)
        .map(|i| start + Duration::days(i as i64))
        .collect()
}

pub fn save_csv(records: &[DailyRecord], path: &str) -> Result<()> {
    if let Some(parent) = Path::new(path).parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("디렉토리 생성 실패: {:?}", parent))?;
    }
    let mut wtr = csv::Writer::from_path(path)
        .with_context(|| format!("CSV 파일 열기 실패: {}", path))?;
    for record in records {
        wtr.serialize(record)?;
    }
    wtr.flush()?;
    Ok(())
}

pub fn load_csv(path: &str) -> Result<Vec<DailyRecord>> {
    anyhow::ensure!(
        Path::new(path).exists(),
        "파일을 찾을 수 없습니다: {}\n힌트: 먼저 `simulate` 명령으로 데이터를 생성하세요.",
        path
    );
    let mut rdr = csv::Reader::from_path(path)
        .with_context(|| format!("CSV 읽기 실패: {}", path))?;
    let mut records = Vec::new();
    for (i, result) in rdr.deserialize().enumerate() {
        let record: DailyRecord = result
            .with_context(|| format!("{}번째 행 파싱 실패", i + 1))?;
        records.push(record);
    }
    anyhow::ensure!(!records.is_empty(), "CSV 파일이 비어 있습니다: {}", path);
    Ok(records)
}

/// 예측 결과를 CSV로 저장
pub fn save_predictions_csv(results: &[crate::models::ForecastResult], path: &str) -> Result<()> {
    if let Some(parent) = Path::new(path).parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut wtr = csv::Writer::from_path(path)?;
    wtr.write_record(["date", "predicted_visits", "lower_bound", "upper_bound"])?;
    for r in results {
        wtr.write_record(&[
            r.date.to_string(),
            format!("{:.1}", r.predicted_visits),
            format!("{:.1}", r.lower_bound),
            format!("{:.1}", r.upper_bound),
        ])?;
    }
    wtr.flush()?;
    Ok(())
}
