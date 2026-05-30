mod commands;
mod data;
mod models;
mod utils;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "er_forecast")]
#[command(about = "응급실 방문 수 예측 CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// 시뮬레이션 데이터 생성
    Simulate {
        /// 생성할 기간 (일수)
        #[arg(short, long, default_value = "730")]
        days: u32,

        /// 시작 날짜 (YYYY-MM-DD)
        #[arg(short, long, default_value = "2023-01-01")]
        start: String,

        /// 출력 파일 경로
        #[arg(short, long, default_value = "data/simulated.csv")]
        output: String,
    },

    /// 모델 학습
    Train {
        /// 입력 데이터 경로
        #[arg(short, long, default_value = "data/simulated.csv")]
        input: String,

        /// 모델 저장 경로
        #[arg(short, long, default_value = "models/lgbm.json")]
        model_out: String,

        /// 검증 비율 (0.0~1.0)
        #[arg(short, long, default_value = "0.2")]
        val_ratio: f64,
    },

    /// 예측 실행
    Predict {
        /// 모델 경로
        #[arg(short, long, default_value = "models/lgbm.json")]
        model: String,

        /// 예측 시작 날짜 (YYYY-MM-DD)
        #[arg(short, long)]
        start: String,

        /// 예측 horizon (일수)
        #[arg(short = 'n', long, default_value = "14")]
        horizon: u32,

        /// 출력 형식 (json | csv)
        #[arg(short, long, default_value = "json")]
        format: String,
    },

    /// 모델 성능 평가
    Evaluate {
        /// 모델 경로
        #[arg(short, long, default_value = "models/lgbm.json")]
        model: String,

        /// 테스트 데이터 경로
        #[arg(short, long, default_value = "data/simulated.csv")]
        input: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Simulate { days, start, output } => {
            commands::simulate::run(days, &start, &output)?;
        }
        Commands::Train { input, model_out, val_ratio } => {
            commands::train::run(&input, &model_out, val_ratio)?;
        }
        Commands::Predict { model, start, horizon, format } => {
            commands::predict::run(&model, &start, horizon, &format)?;
        }
        Commands::Evaluate { model, input } => {
            commands::evaluate::run(&model, &input)?;
        }
    }

    Ok(())
}
