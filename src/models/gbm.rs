use anyhow::Result;
use serde::{Deserialize, Serialize};
use crate::data::FeatureRow;
use super::ModelParams;

/// 단일 결정 트리 노드
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
}

/// Gradient Boosted Trees (MSE loss)
#[derive(Debug, Serialize, Deserialize)]
pub struct GBMModel {
    pub params: ModelParams,
    pub trees: Vec<TreeNode>,
    pub base_prediction: f64,
    pub feature_names: Vec<String>,
}

impl GBMModel {
    pub fn new(params: ModelParams) -> Self {
        Self {
            params,
            trees: Vec::new(),
            base_prediction: 0.0,
            feature_names: Vec::new(),
        }
    }

    pub fn fit(&mut self, rows: &[FeatureRow]) -> Result<()> {
        let (xs, ys) = extract_xy(rows);
        self.feature_names = feature_names();
        self.base_prediction = ys.iter().sum::<f64>() / ys.len() as f64;

        let mut residuals: Vec<f64> = ys.iter()
            .map(|y| y - self.base_prediction)
            .collect();

        for _ in 0..self.params.n_estimators {
            let tree = build_tree(
                &xs,
                &residuals,
                self.params.max_depth,
                self.params.min_samples_leaf,
            );
            let preds: Vec<f64> = xs.iter().map(|x| tree.predict(x)).collect();
            for (r, p) in residuals.iter_mut().zip(preds.iter()) {
                *r -= self.params.learning_rate * p;
            }
            self.trees.push(tree);
        }

        Ok(())
    }

    pub fn predict_one(&self, x: &[f64]) -> f64 {
        let mut pred = self.base_prediction;
        for tree in &self.trees {
            pred += self.params.learning_rate * tree.predict(x);
        }
        pred.max(0.0)
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
        let json = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&json)?)
    }
}

fn extract_xy(rows: &[FeatureRow]) -> (Vec<Vec<f64>>, Vec<f64>) {
    let xs = rows.iter().map(row_to_features).collect();
    let ys = rows.iter().map(|r| r.visits).collect();
    (xs, ys)
}

pub fn row_to_features(r: &FeatureRow) -> Vec<f64> {
    vec![
        r.lag_7, r.lag_14, r.lag_28,
        r.roll_mean_7, r.roll_std_7, r.roll_mean_14,
        r.dow_sin, r.dow_cos,
        r.month_sin, r.month_cos,
        r.is_holiday, r.is_pre_holiday,
        r.temp_avg, r.fine_dust,
    ]
}

pub fn feature_names() -> Vec<String> {
    vec![
        "lag_7","lag_14","lag_28",
        "roll_mean_7","roll_std_7","roll_mean_14",
        "dow_sin","dow_cos",
        "month_sin","month_cos",
        "is_holiday","is_pre_holiday",
        "temp_avg","fine_dust",
    ].into_iter().map(String::from).collect()
}

/// 재귀적 트리 빌드 (분산 기반 최적 분할)
fn build_tree(xs: &[Vec<f64>], ys: &[f64], depth: usize, min_leaf: usize) -> TreeNode {
    if depth == 0 || ys.len() < min_leaf * 2 {
        return TreeNode::Leaf { value: mean(ys) };
    }

    let n_features = xs[0].len();
    let mut best_gain = f64::NEG_INFINITY;
    let mut best_feat = 0;
    let mut best_thresh = 0.0;

    let total_var = variance(ys) * ys.len() as f64;

    for feat in 0..n_features {
        let mut vals: Vec<(f64, f64)> = xs.iter().zip(ys.iter())
            .map(|(x, &y)| (x[feat], y))
            .collect();
        vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        for i in min_leaf..(vals.len() - min_leaf) {
            let thresh = (vals[i].0 + vals[i + 1].0) / 2.0;
            let left_y: Vec<f64>  = vals[..=i].iter().map(|v| v.1).collect();
            let right_y: Vec<f64> = vals[i+1..].iter().map(|v| v.1).collect();

            let gain = total_var
                - variance(&left_y) * left_y.len() as f64
                - variance(&right_y) * right_y.len() as f64;

            if gain > best_gain {
                best_gain = gain;
                best_feat = feat;
                best_thresh = thresh;
            }
        }
    }

    if best_gain <= 0.0 {
        return TreeNode::Leaf { value: mean(ys) };
    }

    let (left_idx, right_idx): (Vec<usize>, Vec<usize>) = (0..xs.len())
        .partition(|&i| xs[i][best_feat] <= best_thresh);

    let left_xs:  Vec<Vec<f64>> = left_idx.iter().map(|&i| xs[i].clone()).collect();
    let left_ys:  Vec<f64>      = left_idx.iter().map(|&i| ys[i]).collect();
    let right_xs: Vec<Vec<f64>> = right_idx.iter().map(|&i| xs[i].clone()).collect();
    let right_ys: Vec<f64>      = right_idx.iter().map(|&i| ys[i]).collect();

    TreeNode::Split {
        feature_idx: best_feat,
        threshold: best_thresh,
        left:  Box::new(build_tree(&left_xs,  &left_ys,  depth - 1, min_leaf)),
        right: Box::new(build_tree(&right_xs, &right_ys, depth - 1, min_leaf)),
    }
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() { return 0.0; }
    v.iter().sum::<f64>() / v.len() as f64
}

fn variance(v: &[f64]) -> f64 {
    if v.len() < 2 { return 0.0; }
    let m = mean(v);
    v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64
}
