use anyhow::Result;
use serde::{Deserialize, Serialize};
use crate::data::FeatureRow;
use crate::utils::metrics;
use super::ModelParams;

// ── 피처 목록 (순서 고정 — row_to_features와 반드시 동일해야 함) ──────────
pub fn feature_names() -> Vec<String> {
    [
        "lag_7", "lag_14", "lag_28",
        "roll_mean_7", "roll_std_7",
        "roll_mean_14", "roll_std_14",
        "dow_sin", "dow_cos",
        "month_sin", "month_cos",
        "is_holiday", "is_pre_holiday",
        "temp_avg", "fine_dust",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

pub fn row_to_features(r: &FeatureRow) -> Vec<f64> {
    vec![
        r.lag_7, r.lag_14, r.lag_28,
        r.roll_mean_7, r.roll_std_7,
        r.roll_mean_14, r.roll_std_14,
        r.dow_sin, r.dow_cos,
        r.month_sin, r.month_cos,
        r.is_holiday, r.is_pre_holiday,
        r.temp_avg, r.fine_dust,
    ]
}

// ── 트리 노드 ────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
pub enum TreeNode {
    Leaf { value: f64 },
    Split {
        feature_idx: usize,
        threshold: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
}

impl TreeNode {
    pub fn predict(&self, x: &[f64]) -> f64 {
        match self {
            TreeNode::Leaf { value } => *value,
            TreeNode::Split { feature_idx, threshold, left, right } => {
                if x[*feature_idx] <= *threshold {
                    left.predict(x)
                } else {
                    right.predict(x)
                }
            }
        }
    }

    /// Split-count 기반 피처 중요도 누적
    pub fn accumulate_importance(&self, importance: &mut Vec<u64>) {
        match self {
            TreeNode::Leaf { .. } => {}
            TreeNode::Split { feature_idx, left, right, .. } => {
                importance[*feature_idx] += 1;
                left.accumulate_importance(importance);
                right.accumulate_importance(importance);
            }
        }
    }
}

// ── GBM 모델 ─────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
pub struct GBMModel {
    pub params: ModelParams,
    pub trees: Vec<TreeNode>,
    pub base_prediction: f64,
    pub feature_names: Vec<String>,
    /// 학습 시 계산된 잔차 표준편차 (신뢰구간 추정용)
    pub residual_std: f64,
    /// 에포크별 검증 MAE 이력
    pub val_mae_history: Vec<f64>,
}

impl GBMModel {
    pub fn new(params: ModelParams) -> Self {
        Self {
            params,
            trees: Vec::new(),
            base_prediction: 0.0,
            feature_names: feature_names(),
            residual_std: 0.0,
            val_mae_history: Vec::new(),
        }
    }

    /// 학습: train_rows로 GBM을 적합, val_rows로 early stopping
    pub fn fit(&mut self, train_rows: &[FeatureRow], val_rows: &[FeatureRow]) -> Result<()> {
        let (train_xs, train_ys) = extract_xy(train_rows);
        let (val_xs, val_ys) = extract_xy(val_rows);

        // 기저 예측값 = 학습 타겟 평균
        self.base_prediction = mean(&train_ys);

        // 초기 잔차
        let mut residuals: Vec<f64> = train_ys.iter()
            .map(|y| y - self.base_prediction)
            .collect();

        // Pre-sort: 각 피처별로 정렬된 (값, 원본인덱스) 목록을 미리 계산
        // → 트리 빌드 시 O(n log n) 정렬을 O(n)으로 줄임
        let n_feat = train_xs[0].len();
        let sorted_indices: Vec<Vec<usize>> = (0..n_feat)
            .map(|f| {
                let mut idx: Vec<usize> = (0..train_xs.len()).collect();
                idx.sort_by(|&a, &b| train_xs[a][f].partial_cmp(&train_xs[b][f]).unwrap());
                idx
            })
            .collect();

        let mut best_val_mae = f64::MAX;
        let mut rounds_no_improve = 0;

        for round in 0..self.params.n_estimators {
            let tree = build_tree_presorted(
                &train_xs,
                &residuals,
                &sorted_indices,
                self.params.max_depth,
                self.params.min_samples_leaf,
            );

            // 잔차 업데이트
            for (i, r) in residuals.iter_mut().enumerate() {
                *r -= self.params.learning_rate * tree.predict(&train_xs[i]);
            }

            // 검증 MAE 계산
            let val_preds: Vec<f64> = val_xs.iter()
                .map(|x| self.predict_one_with_trees(x, &self.trees) + self.params.learning_rate * tree.predict(x))
                .collect();
            let val_mae = metrics::mae(&val_ys, &val_preds);
            self.val_mae_history.push(val_mae);

            self.trees.push(tree);

            // Early stopping
            if val_mae < best_val_mae - 1e-6 {
                best_val_mae = val_mae;
                rounds_no_improve = 0;
            } else {
                rounds_no_improve += 1;
            }

            if (round + 1) % 10 == 0 {
                eprintln!(
                    "  [Round {:3}] val_MAE = {:.2} | best = {:.2} | no_improve = {}/{}",
                    round + 1, val_mae, best_val_mae, rounds_no_improve,
                    self.params.early_stopping_rounds
                );
            }

            if rounds_no_improve >= self.params.early_stopping_rounds {
                eprintln!(
                    "  ✓ Early stopping at round {} (best val_MAE={:.2})",
                    round + 1, best_val_mae
                );
                // 최적 라운드까지 트리 잘라내기
                let best_round = round + 1 - rounds_no_improve;
                self.trees.truncate(best_round);
                break;
            }
        }

        // 최종 잔차 표준편차 (신뢰구간 추정)
        let final_resids: Vec<f64> = train_ys.iter().enumerate()
            .map(|(i, y)| y - self.predict_one(&train_xs[i]))
            .collect();
        self.residual_std = std_dev(&final_resids);

        Ok(())
    }

    /// 추론: 단일 샘플
    pub fn predict_one(&self, x: &[f64]) -> f64 {
        self.predict_one_with_trees(x, &self.trees)
    }

    fn predict_one_with_trees(&self, x: &[f64], trees: &[TreeNode]) -> f64 {
        let raw = trees.iter().fold(self.base_prediction, |acc, t| {
            acc + self.params.learning_rate * t.predict(x)
        });
        raw.max(0.0) // 방문수는 음수 불가
    }

    /// 95% 신뢰구간 포함 예측
    pub fn predict_with_interval(&self, x: &[f64]) -> (f64, f64, f64) {
        let pred = self.predict_one(x);
        let margin = 1.96 * self.residual_std;
        (pred, (pred - margin).max(0.0), pred + margin)
    }

    /// Split-count 기반 피처 중요도 (정규화된 비율)
    pub fn feature_importance(&self) -> Vec<(String, f64)> {
        let mut counts = vec![0u64; self.feature_names.len()];
        for tree in &self.trees {
            tree.accumulate_importance(&mut counts);
        }
        let total: u64 = counts.iter().sum();
        if total == 0 {
            return self.feature_names.iter().map(|n| (n.clone(), 0.0)).collect();
        }
        let mut importance: Vec<(String, f64)> = self.feature_names.iter()
            .zip(counts.iter())
            .map(|(name, &cnt)| (name.clone(), cnt as f64 / total as f64))
            .collect();
        importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        importance
    }

    pub fn save(&self, path: &str) -> Result<()> {
        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self> {
        anyhow::ensure!(
            std::path::Path::new(path).exists(),
            "모델 파일을 찾을 수 없습니다: {}\n힌트: 먼저 `train` 명령으로 모델을 학습하세요.",
            path
        );
        let json = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&json)?)
    }
}

// ── 데이터 헬퍼 ──────────────────────────────────────────────────────────────

fn extract_xy(rows: &[FeatureRow]) -> (Vec<Vec<f64>>, Vec<f64>) {
    let xs = rows.iter().map(row_to_features).collect();
    let ys = rows.iter().map(|r| r.visits).collect();
    (xs, ys)
}

// ── 트리 빌드 (Pre-sorted) ────────────────────────────────────────────────────

/// Pre-sorted 인덱스를 활용한 고속 트리 빌드.
/// active_mask: 현재 노드에 포함된 샘플 여부
fn build_tree_presorted(
    xs: &[Vec<f64>],
    ys: &[f64],
    sorted_indices: &[Vec<usize>],
    depth: usize,
    min_leaf: usize,
) -> TreeNode {
    let active: Vec<usize> = (0..xs.len()).collect();
    build_node(xs, ys, sorted_indices, &active, depth, min_leaf)
}

fn build_node(
    xs: &[Vec<f64>],
    ys: &[f64],
    sorted_indices: &[Vec<usize>],
    active: &[usize],
    depth: usize,
    min_leaf: usize,
) -> TreeNode {
    // 리프 조건
    if depth == 0 || active.len() < min_leaf * 2 {
        let active_ys: Vec<f64> = active.iter().map(|&i| ys[i]).collect();
        return TreeNode::Leaf { value: mean(&active_ys) };
    }

    let active_ys: Vec<f64> = active.iter().map(|&i| ys[i]).collect();
    let total_var = variance(&active_ys) * active_ys.len() as f64;

    // active 집합을 빠르게 조회하기 위한 마스크
    let mut in_active = vec![false; xs.len()];
    for &i in active { in_active[i] = true; }

    let mut best_gain = 0.0_f64; // 0보다 커야 분할
    let mut best_feat = 0;
    let mut best_thresh = 0.0;

    for (feat, sorted_idx) in sorted_indices.iter().enumerate() {
        // active 샘플만 필터링 (정렬 순서 유지)
        let sorted_active: Vec<usize> = sorted_idx.iter()
            .copied()
            .filter(|&i| in_active[i])
            .collect();
        let n = sorted_active.len();
        if n < min_leaf * 2 { continue; }

        // 누적 합/제곱합으로 분산 이득 계산 (O(n))
        let mut left_sum = 0.0_f64;
        let mut left_sq  = 0.0_f64;

        for split in 0..(n - 1) {
            let idx = sorted_active[split];
            let y = ys[idx];
            left_sum += y;
            left_sq  += y * y;

            if split + 1 < min_leaf { continue; }
            if n - split - 1 < min_leaf { break; }

            // 같은 피처 값이면 분할 불가
            let cur_val  = xs[sorted_active[split]][feat];
            let next_val = xs[sorted_active[split + 1]][feat];
            if (cur_val - next_val).abs() < 1e-10 { continue; }

            let nl = (split + 1) as f64;
            let nr = (n - split - 1) as f64;
            let right_sum = active_ys.iter().sum::<f64>() - left_sum;
            let right_sq  = active_ys.iter().map(|y| y * y).sum::<f64>() - left_sq;

            let left_var  = left_sq  - left_sum  * left_sum  / nl;
            let right_var = right_sq - right_sum * right_sum / nr;
            let gain = total_var - left_var - right_var;

            if gain > best_gain {
                best_gain = gain;
                best_feat = feat;
                best_thresh = (cur_val + next_val) / 2.0;
            }
        }
    }

    if best_gain <= 0.0 {
        return TreeNode::Leaf { value: mean(&active_ys) };
    }

    let (left_active, right_active): (Vec<usize>, Vec<usize>) = active
        .iter()
        .partition(|&&i| xs[i][best_feat] <= best_thresh);

    TreeNode::Split {
        feature_idx: best_feat,
        threshold: best_thresh,
        left:  Box::new(build_node(xs, ys, sorted_indices, &left_active,  depth - 1, min_leaf)),
        right: Box::new(build_node(xs, ys, sorted_indices, &right_active, depth - 1, min_leaf)),
    }
}

// ── 통계 헬퍼 ────────────────────────────────────────────────────────────────

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() { return 0.0; }
    v.iter().sum::<f64>() / v.len() as f64
}

fn variance(v: &[f64]) -> f64 {
    if v.len() < 2 { return 0.0; }
    let m = mean(v);
    v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64
}

fn std_dev(v: &[f64]) -> f64 {
    variance(v).sqrt()
}
