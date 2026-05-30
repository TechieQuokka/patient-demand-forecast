use std::path::Path;
use anyhow::Result;
use chrono::{NaiveDate, Duration};
use crate::data::DailyRecord;

pub fn date_range(start: NaiveDate, days: u32) -> Vec<NaiveDate> {
    (0..days)
        .map(|i| start + Duration::days(i as i64))
        .collect()
}

pub fn save_csv(records: &[DailyRecord], path: &str) -> Result<()> {
    if let Some(parent) = Path::new(path).parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut wtr = csv::Writer::from_path(path)?;
    for record in records {
        wtr.serialize(record)?;
    }
    wtr.flush()?;
    Ok(())
}

pub fn load_csv(path: &str) -> Result<Vec<DailyRecord>> {
    let mut rdr = csv::Reader::from_path(path)?;
    let mut records = Vec::new();
    for result in rdr.deserialize() {
        let record: DailyRecord = result?;
        records.push(record);
    }
    Ok(records)
}
