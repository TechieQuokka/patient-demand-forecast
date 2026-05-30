use anyhow::Result;
use chrono::{NaiveDate, Datelike};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

use crate::data::{loader, DailyRecord};

/// 한국 공휴일 (월-일 기준, 연도 무관 단순화)
fn is_korean_holiday(date: NaiveDate) -> bool {
    matches!(
        (date.month(), date.day()),
        (1,1)|(3,1)|(5,5)|(6,6)|(8,15)|(10,3)|(10,9)|(12,25)
    )
}

fn is_pre_holiday(date: NaiveDate) -> bool {
    let next = date + chrono::Duration::days(1);
    is_korean_holiday(next)
}

/// 계절별 기온 시뮬레이션 (서울 기준)
fn simulate_temp(date: NaiveDate) -> f64 {
    use std::f64::consts::PI;
    let day_of_year = date.ordinal() as f64;
    // 연중 최저(1월) -5도, 최고(8월) 30도
    let base = 12.5 + 17.5 * ((2.0 * PI * (day_of_year - 30.0) / 365.0).sin());
    base + rand::random::<f64>() * 4.0 - 2.0
}

/// 계절별 미세먼지 (봄 고농도)
fn simulate_fine_dust(date: NaiveDate) -> f64 {
    let month = date.month();
    let base = match month {
        3..=5 => 45.0,   // 봄: 황사
        6..=8 => 20.0,   // 여름: 낮음
        9..=11 => 30.0,  // 가을
        _ => 40.0,       // 겨울
    };
    (base + rand::random::<f64>() * 20.0 - 10.0).max(5.0)
}

/// 요일별 기저 방문수 (월~일)
fn dow_factor(dow: u32) -> f64 {
    match dow {
        0 => 1.0,    // 월
        1 => 0.95,   // 화
        2 => 0.93,   // 수
        3 => 0.95,   // 목
        4 => 1.05,   // 금
        5 => 1.20,   // 토 (주말 급증)
        6 => 1.35,   // 일 (최고)
        _ => 1.0,
    }
}

/// 월별 계절 팩터
fn month_factor(month: u32) -> f64 {
    match month {
        1 | 2 => 1.15,    // 겨울: 호흡기 증가
        3..=5 => 1.05,    // 봄
        6..=8 => 0.95,    // 여름: 상대적 감소
        9..=11 => 1.0,    // 가을
        12 => 1.10,       // 연말
        _ => 1.0,
    }
}

pub fn run(days: u32, start_str: &str, output: &str) -> Result<()> {
    let start = NaiveDate::parse_from_str(start_str, "%Y-%m-%d")?;
    let dates = loader::date_range(start, days);

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let noise = Normal::new(0.0, 8.0)?;

    let base_visits = 120.0_f64;
    let mut records = Vec::new();

    for date in &dates {
        use chrono::Datelike;
        let dow = date.weekday().num_days_from_monday(); // 0=월
        let month = date.month();
        let holiday = is_korean_holiday(*date);
        let pre_holiday = is_pre_holiday(*date);
        let temp = simulate_temp(*date);
        let dust = simulate_fine_dust(*date);

        let mut visits = base_visits
            * dow_factor(dow)
            * month_factor(month);

        // 공휴일 효과: +25%
        if holiday { visits *= 1.25; }
        // 연휴 전날: +10%
        if pre_holiday { visits *= 1.10; }
        // 기온 영향: 극단적 기온(< -5 or > 35)에서 +10%
        if temp < -5.0 || temp > 35.0 { visits *= 1.10; }
        // 미세먼지 영향: PM2.5 > 75 이면 +8%
        if dust > 75.0 { visits *= 1.08; }

        visits += noise.sample(&mut rng);
        visits = visits.max(50.0).round();

        records.push(DailyRecord {
            date: *date,
            visits: visits as u32,
            day_of_week: dow,
            month,
            is_holiday: holiday,
            is_pre_holiday: pre_holiday,
            temp_avg: (temp * 10.0).round() / 10.0,
            fine_dust: (dust * 10.0).round() / 10.0,
        });
    }

    loader::save_csv(&records, output)?;
    println!("시뮬레이션 완료: {} 일치 데이터 → {}", days, output);
    println!("평균 방문수: {:.1}", records.iter().map(|r| r.visits as f64).sum::<f64>() / records.len() as f64);
    Ok(())
}
