mod commands;
mod data;
mod models;
mod utils;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "er_forecast")]
#[command(version = "2.0.0")]
#[command(about = "응급실 방문 수 예측 CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// 한국 응급실 패턴 기반 시뮬레이션 데이터 생성
    Simulate {
        #[arg(short, long, default_value = "730", help = "생성할 기간 (일수)")]
        days: u32,

        #[arg(short, long, default_value = "2023-01-01", help = "시작 날짜 (YYYY-MM-DD)")]
        start: String,

        #[arg(short, long, default_value = "data/simulated.csv", help = "출력 CSV 경로")]
        output: String,

        #[arg(long, help = "랜덤 시드 (기본값: 42, 재현성 보장)")]
        seed: Option<u64>,
    },

    /// GBM 모델 학습 및 검증
    Train {
        #[arg(short, long, default_value = "data/simulated.csv", help = "학습 데이터 CSV 경로")]
        input: String,

        #[arg(short, long, default_value = "models/er_model.json", help = "모델 저장 경로")]
        model_out: String,

        #[arg(short, long, default_value = "0.2", help = "검증 데이터 비율 (0.0~1.0)")]
        val_ratio: f64,

        #[arg(long, default_value = "100", help = "트리 개수 (n_estimators)")]
        n_estimators: usize,

        #[arg(long, default_value = "5", help = "트리 최대 깊이")]
        max_depth: usize,

        #[arg(long, default_value = "0.1", help = "학습률 (learning rate)")]
        learning_rate: f64,

        #[arg(long, default_value = "5", help = "리프 최소 샘플 수")]
        min_samples_leaf: usize,

        #[arg(long, default_value = "10", help = "Early stopping: 검증 개선 없으면 중단할 라운드 수")]
        early_stopping_rounds: usize,
    },

    /// 학습된 모델로 미래 수요 예측
    Predict {
        #[arg(short, long, default_value = "models/er_model.json", help = "모델 경로")]
        model: String,

        #[arg(short, long, help = "예측 시작 날짜 (YYYY-MM-DD)")]
        start: String,

        #[arg(short = 'n', long, default_value = "14", help = "예측 기간 (일수)")]
        horizon: u32,

        #[arg(long, default_value = "data/simulated.csv", help = "Lag 피처 계산용 과거 데이터 경로")]
        history: String,

        #[arg(short, long, default_value = "json", help = "출력 형식 (json | csv)")]
        format: String,

        #[arg(long, help = "일평균 기온 오버라이드 (쉼표 구분, 예: 10.5,11.0,...). 없으면 계절 평균 사용")]
        temps: Option<String>,

        #[arg(long, help = "미세먼지(PM2.5) 오버라이드 (쉼표 구분). 없으면 계절 평균 사용")]
        dusts: Option<String>,
    },

    /// 모델 성능 평가 (MAE / MAPE / RMSE + 피처 중요도)
    Evaluate {
        #[arg(short, long, default_value = "models/er_model.json", help = "모델 경로")]
        model: String,

        #[arg(short, long, default_value = "data/simulated.csv", help = "평가 데이터 경로")]
        input: String,

        #[arg(long, help = "예측값을 CSV로 저장할 경로 (선택)")]
        output: Option<String>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Simulate { days, start, output, seed } => {
            commands::simulate::run(days, &start, &output, seed.unwrap_or(42))?;
        }
        Commands::Train {
            input, model_out, val_ratio,
            n_estimators, max_depth, learning_rate,
            min_samples_leaf, early_stopping_rounds,
        } => {
            let params = models::ModelParams {
                n_estimators,
                max_depth,
                learning_rate,
                min_samples_leaf,
                early_stopping_rounds,
            };
            commands::train::run(&input, &model_out, val_ratio, params)?;
        }
        Commands::Predict { model, start, horizon, history, format, temps, dusts } => {
            let temp_override = parse_floats(temps.as_deref(), horizon as usize)?;
            let dust_override = parse_floats(dusts.as_deref(), horizon as usize)?;
            commands::predict::run(&model, &start, horizon, &history, &format, temp_override, dust_override)?;
        }
        Commands::Evaluate { model, input, output } => {
            commands::evaluate::run(&model, &input, output.as_deref())?;
        }
    }

    Ok(())
}

/// 쉼표 구분 문자열 → Vec<f64> 파싱
fn parse_floats(s: Option<&str>, expected_len: usize) -> Result<Option<Vec<f64>>> {
    let Some(s) = s else { return Ok(None) };
    let vals: Result<Vec<f64>, _> = s.split(',').map(|v| v.trim().parse::<f64>()).collect();
    let vals = vals.map_err(|e| anyhow::anyhow!("숫자 파싱 오류: {}", e))?;
    if vals.len() != expected_len {
        anyhow::bail!(
            "오버라이드 값 개수({})가 horizon({})과 불일치합니다.",
            vals.len(), expected_len
        );
    }
    Ok(Some(vals))
}
