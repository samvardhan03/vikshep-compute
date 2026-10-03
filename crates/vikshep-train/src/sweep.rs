//! Lambda sweep, frontier table and benchmark report (`spec/VDS-1.md`
//! sections 16.6, 16.7 and 17.7).

use vikshep_numerics::jcs::{Value, object};
use vikshep_stats::report::{EvalConfig, FrontierRow, Report, build_report, evaluate};

use crate::TrainError;
use crate::data::Dataset;
use crate::train::{TrainConfig, TrainedModel, train};

/// Result of a sweep.
#[derive(Clone, Debug)]
pub struct Sweep {
    /// One row per lambda, in sweep order.
    pub rows: Vec<FrontierRow>,
    /// One model per lambda, in sweep order.
    pub models: Vec<TrainedModel>,
}

/// Train one model per `lambda` (same seed, same schedule) on `train_set`
/// and evaluate each on `eval_set`.
pub fn lambda_sweep(
    train_set: &Dataset,
    eval_set: &Dataset,
    base: TrainConfig,
    lambdas: &[f64],
    eval: EvalConfig,
) -> Result<Sweep, TrainError> {
    if eval_set.d != train_set.d {
        return Err(TrainError::new("train and eval feature counts differ"));
    }
    let mut rows = Vec::with_capacity(lambdas.len());
    let mut models = Vec::with_capacity(lambdas.len());
    for &lambda in lambdas {
        let model = train(train_set, TrainConfig { lambda, ..base })?;
        let scores = model.predict(&eval_set.x);
        rows.push(
            evaluate(lambda, &scores, &eval_set.y, &eval_set.w, &eval_set.m, eval)
                .map_err(|e| TrainError::new(e.0))?,
        );
        models.push(model);
    }
    Ok(Sweep { rows, models })
}

/// Configuration of a benchmark run as canonical JSON.
#[must_use]
pub fn config_value(
    base: &TrainConfig,
    lambdas: &[f64],
    lambda_star: f64,
    eval: &EvalConfig,
) -> Value {
    object([
        ("batch_size", Value::Int(base.batch_size as i64)),
        ("epochs", Value::Int(base.epochs as i64)),
        ("gradient", Value::Str(base.gradient.name().into())),
        ("head", Value::Str(base.head.name())),
        ("lambda_star", Value::Num(lambda_star)),
        (
            "lambdas",
            Value::Array(lambdas.iter().map(|&l| Value::Num(l)).collect()),
        ),
        ("lr", Value::Num(base.lr)),
        ("n_bins", Value::Int(eval.n_bins as i64)),
        ("seed", Value::Str(base.seed.to_string())),
        ("target_sig_eff", Value::Num(eval.target_sig_eff)),
    ])
}

/// Run a sweep and build the benchmark report (`lambda_star` must be one of
/// the lambdas, and the lambdas must contain 0 exactly once).
pub fn benchmark(
    train_set: &Dataset,
    eval_set: &Dataset,
    base: TrainConfig,
    lambdas: &[f64],
    lambda_star: f64,
    eval: EvalConfig,
) -> Result<(Sweep, Report), TrainError> {
    let sweep = lambda_sweep(train_set, eval_set, base, lambdas, eval)?;
    let inputs = object([
        ("eval_oid", Value::Str(eval_set.oid())),
        ("eval_events", Value::Int(eval_set.n as i64)),
        ("features", Value::Int(train_set.d as i64)),
        (
            "model_oids",
            Value::Array(sweep.models.iter().map(|m| Value::Str(m.oid())).collect()),
        ),
        ("train_oid", Value::Str(train_set.oid())),
        ("train_events", Value::Int(train_set.n as i64)),
    ]);
    let report = build_report(
        &sweep.rows,
        lambda_star,
        config_value(&base, lambdas, lambda_star, &eval),
        inputs,
    )
    .map_err(|e| TrainError::new(e.0))?;
    Ok((sweep, report))
}
