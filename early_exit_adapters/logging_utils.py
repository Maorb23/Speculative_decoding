def build_layer_metrics_table_for_wandb(step, split, layer_metrics):
    import wandb

    table = wandb.Table(
        columns=[
            "step",
            "split",
            "layer",
            "kl_loss",
            "metric_kl",
            "ce",
            "top1",
            "topk_overlap",
            "accept_exact",
            "accept_sampled",
        ]
    )

    for row in layer_metrics:
        table.add_data(
            step,
            split,
            row["layer"],
            row.get("kl_loss"),
            row.get("metric_kl"),
            row.get("ce"),
            row.get("top1"),
            row.get("topk_overlap"),
            row.get("accept_exact"),
            row.get("accept_sampled"),
        )

    return {f"{split}/layer_metrics_table": table}


def build_layer_metric_plots_for_wandb(step, split, layer_metrics):
    import wandb

    table = wandb.Table(
        columns=[
            "layer",
            "kl_loss",
            "metric_kl",
            "ce",
            "top1",
            "topk_overlap",
            "accept_exact",
            "accept_sampled",
        ]
    )

    for row in layer_metrics:
        table.add_data(
            row["layer"],
            row.get("kl_loss"),
            row.get("metric_kl"),
            row.get("ce"),
            row.get("top1"),
            row.get("topk_overlap"),
            row.get("accept_exact"),
            row.get("accept_sampled"),
        )

    return {
        f"{split}/kl_by_layer": wandb.plot.line(
            table, "layer", "metric_kl", title=f"{split} KL by layer"
        ),
        f"{split}/ce_by_layer": wandb.plot.line(
            table, "layer", "ce", title=f"{split} CE by layer"
        ),
        f"{split}/top1_by_layer": wandb.plot.line(
            table, "layer", "top1", title=f"{split} top1 by layer"
        ),
        f"{split}/accept_exact_by_layer": wandb.plot.line(
            table, "layer", "accept_exact", title=f"{split} accept exact by layer"
        ),
        f"{split}/accept_sampled_by_layer": wandb.plot.line(
            table, "layer", "accept_sampled", title=f"{split} accept sampled by layer"
        ),
    }


def metric_row(layer_index, metrics, top_k, kl_loss=None):
    return {
        "layer": layer_index,
        "kl_loss": kl_loss,
        "metric_kl": metrics["kl_to_teacher"],
        "ce": metrics["ce"],
        "top1": metrics["top1_teacher_agreement"],
        "topk_overlap": metrics[f"top{top_k}_overlap"],
        "accept_exact": metrics["accept_proxy_exact"],
        "accept_sampled": metrics["accept_proxy_sampled"],
    }


def add_rows_to_wandb_log(wandb_log, step, split, rows):
    wandb_log.update(
        build_layer_metric_plots_for_wandb(
            step=step,
            split=split,
            layer_metrics=rows,
        )
    )


def add_train_metric_layer_scalars_to_wandb(
    wandb_log,
    rows,
    metrics_to_log=None,
):
    if metrics_to_log is None:
        metrics_to_log = [
            "kl_loss",
            "metric_kl",
            "ce",
            "top1",
            "accept_exact",
        ]

    for row in rows:
        layer = row["layer"]

        for metric in metrics_to_log:
            value = row.get(metric)
            if value is None:
                continue
            wandb_log[f"train_by_layer/{metric}/layer_{layer:02d}"] = value


def eval_rows_from_metrics(eval_metrics, candidate_layers, top_k):
    baseline_rows = []
    adapted_rows = []
    improvement_rows = []

    for layer in candidate_layers:
        baseline_prefix = f"layer_{layer}/eval_baseline"
        adapted_prefix = f"layer_{layer}/eval_adapted"

        baseline = metric_row(
            layer,
            {
                "kl_to_teacher": eval_metrics[f"{baseline_prefix}/kl_to_teacher"],
                "ce": eval_metrics[f"{baseline_prefix}/ce"],
                "top1_teacher_agreement": eval_metrics[
                    f"{baseline_prefix}/top1_teacher_agreement"
                ],
                f"top{top_k}_overlap": eval_metrics[
                    f"{baseline_prefix}/top{top_k}_overlap"
                ],
                "accept_proxy_exact": eval_metrics[
                    f"{baseline_prefix}/accept_proxy_exact"
                ],
                "accept_proxy_sampled": eval_metrics[
                    f"{baseline_prefix}/accept_proxy_sampled"
                ],
            },
            top_k,
        )

        adapted = metric_row(
            layer,
            {
                "kl_to_teacher": eval_metrics[f"{adapted_prefix}/kl_to_teacher"],
                "ce": eval_metrics[f"{adapted_prefix}/ce"],
                "top1_teacher_agreement": eval_metrics[
                    f"{adapted_prefix}/top1_teacher_agreement"
                ],
                f"top{top_k}_overlap": eval_metrics[
                    f"{adapted_prefix}/top{top_k}_overlap"
                ],
                "accept_proxy_exact": eval_metrics[
                    f"{adapted_prefix}/accept_proxy_exact"
                ],
                "accept_proxy_sampled": eval_metrics[
                    f"{adapted_prefix}/accept_proxy_sampled"
                ],
            },
            top_k,
        )

        improvement = {
            "layer": layer,
            "kl_loss": None,
            "metric_kl": baseline["metric_kl"] - adapted["metric_kl"],
            "ce": baseline["ce"] - adapted["ce"],
            "top1": adapted["top1"] - baseline["top1"],
            "topk_overlap": adapted["topk_overlap"] - baseline["topk_overlap"],
            "accept_exact": adapted["accept_exact"] - baseline["accept_exact"],
            "accept_sampled": adapted["accept_sampled"] - baseline["accept_sampled"],
        }

        baseline_rows.append(baseline)
        adapted_rows.append(adapted)
        improvement_rows.append(improvement)

    return baseline_rows, adapted_rows, improvement_rows


def build_eval_comparison_plots_for_wandb(step, baseline_rows, adapted_rows):
    import wandb

    layers = [row["layer"] for row in baseline_rows]

    def values(rows, key):
        return [row[key] for row in rows]

    return {
        "eval/kl_baseline_vs_adapted_by_layer": wandb.plot.line_series(
            xs=layers,
            ys=[
                values(baseline_rows, "metric_kl"),
                values(adapted_rows, "metric_kl"),
            ],
            keys=["baseline", "adapted"],
            title="eval KL by layer",
            xname="layer",
        ),
        "eval/ce_baseline_vs_adapted_by_layer": wandb.plot.line_series(
            xs=layers,
            ys=[
                values(baseline_rows, "ce"),
                values(adapted_rows, "ce"),
            ],
            keys=["baseline", "adapted"],
            title="eval CE by layer",
            xname="layer",
        ),
        "eval/top1_baseline_vs_adapted_by_layer": wandb.plot.line_series(
            xs=layers,
            ys=[
                values(baseline_rows, "top1"),
                values(adapted_rows, "top1"),
            ],
            keys=["baseline", "adapted"],
            title="eval top1 by layer",
            xname="layer",
        ),
        "eval/accept_exact_baseline_vs_adapted_by_layer": wandb.plot.line_series(
            xs=layers,
            ys=[
                values(baseline_rows, "accept_exact"),
                values(adapted_rows, "accept_exact"),
            ],
            keys=["baseline", "adapted"],
            title="eval accept exact by layer",
            xname="layer",
        ),
        "eval/accept_sampled_baseline_vs_adapted_by_layer": wandb.plot.line_series(
            xs=layers,
            ys=[
                values(baseline_rows, "accept_sampled"),
                values(adapted_rows, "accept_sampled"),
            ],
            keys=["baseline", "adapted"],
            title="eval accept sampled by layer",
            xname="layer",
        ),
    }
