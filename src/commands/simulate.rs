use anyhow::Result;
use chrono::{NaiveDate, Datelike};
use rand::SeedableRng;
use rand::Rng;
use rand_distr::{Distribution, Normal};

use crate::data::{loader, DailyRecord};

// ── 공휴일 판정 ───────────────────────────────────────────────────────────────

/// 한국 법정 공휴일 (음력 공휴일은 단순화하여 양력 고정)
fn is_korean_holiday(date: NaiveDate) -> bool {
    matches!(
        (date.month(), date.day()),
        (1, 1)   // 신정
        | (3, 1) // 삼일절
        | (5, 5) // 어린이날
        | (6, 6) // 현충일
        | (8, 15)// 광복절
        | (10, 3)// 개천절
        | (10, 9)// 한글날
        | (12, 25)// 성탄절
    )
}

fn is_pre_holiday(date: NaiveDate) -> bool {
    is_korean_holiday(date + chrono::Duration::days(1))
}

// ── 환경 시뮬레이션 (rng를 명시적으로 전달) ─────────────────────────────────

/// 서울 기준 일평균 기온 시뮬레이션
fn simulate_temp<R: Rng>(date: NaiveDate, rng: &mut R) -> f64 {
    use std::f64::consts::PI;
    let day_of_year = date.ordinal() as f64;
    // 연중 최저(1월) 약 -5°C, 최고(8월) 약 30°C
    let base = 12.5 + 17.5 * ((2.0 * PI * (day_of_year - 30.0) / 365.0).sin());
    let noise: f64 = rng.random_range(-2.0..2.0);
    base + noise
}

/// 계절별 PM2.5 시뮬레이션
fn simulate_fine_dust<R: Rng>(date: NaiveDate, rng: &mut R) -> f64 {
    let base: f64 = match date.month() {
        3..=5  => 45.0, // 봄: 황사
        6..=8  => 20.0, // 여름: 낮음
        9..=11 => 30.0, // 가을
        _      => 40.0, // 겨울
    };
    let noise: f64 = rng.random_range(-10.0..10.0);
    (base + noise).max(5.0)
}

// ── 수요 가중치 ────────────────────────────────────────────────────────────────

fn dow_factor(dow: u32) -> f64 {
    match dow {
        0 => 1.00, // 월
        1 => 0.95, // 화
        2 => 0.93, // 수
        3 => 0.95, // 목
        4 => 1.05, // 금
        5 => 1.20, // 토
        6 => 1.35, // 일 (최고)
        _ => 1.00,
    }
}

fn month_factor(month: u32) -> f64 {
    match month {
        1 | 2    => 1.15, // 겨울: 호흡기 급증
        3..=5    => 1.05, // 봄
        6..=8    => 0.95, // 여름
        9..=11   => 1.00, // 가을
        12       => 1.10, // 연말
        _        => 1.00,
    }
}

// ── 진입점 ───────────────────────────────────────────────────────────────────

pub fn run(days: u32, start_str: &str, output: &str, seed: u64) -> Result<()> {
    let start = NaiveDate::parse_from_str(start_str, "%Y-%m-%d")?;

    // 단일 rng 인스턴스를 모든 헬퍼에 전달 → 시드 고정 시 완전 재현 가능
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let noise_dist = Normal::new(0.0, 8.0)?;

    let dates = loader::date_range(start, days);
    let base_visits = 120.0_f64;
    let mut records = Vec::with_capacity(days as usize);

    for date in &dates {
        let dow      = date.weekday().num_days_from_monday();
        let month    = date.month();
        let holiday  = is_korean_holiday(*date);
        let pre_hol  = is_pre_holiday(*date);
        let temp     = simulate_temp(*date, &mut rng);
        let dust     = simulate_fine_dust(*date, &mut rng);

        let mut visits = base_visits * dow_factor(dow) * month_factor(month);

        if holiday         { visits *= 1.25; }
        if pre_hol         { visits *= 1.10; }
        if temp < -5.0 || temp > 35.0 { visits *= 1.10; }
        if dust > 75.0     { visits *= 1.08; }

        visits += noise_dist.sample(&mut rng);
        visits = visits.max(50.0).round();

        records.push(DailyRecord {
            date:         *date,
            visits:       visits as u32,
            day_of_week:  dow,
            month,
            is_holiday:   holiday,
            is_pre_holiday: pre_hol,
            temp_avg:     (temp * 10.0).round() / 10.0,
            fine_dust:    (dust * 10.0).round() / 10.0,
        });
    }

    loader::save_csv(&records, output)?;

    let avg = records.iter().map(|r| r.visits as f64).sum::<f64>() / records.len() as f64;
    let max = records.iter().map(|r| r.visits).max().unwrap_or(0);
    let min = records.iter().map(|r| r.visits).min().unwrap_or(0);

    println!("✓ 시뮬레이션 완료 (seed={})", seed);
    println!("  기간: {} ~ {} ({} 일)", records.first().unwrap().date, records.last().unwrap().date, days);
    println!("  출력: {}", output);
    println!("  평균 방문수: {:.1} | 최소: {} | 최대: {}", avg, min, max);

    Ok(())
}
