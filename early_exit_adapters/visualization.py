import matplotlib.pyplot as plt


def plot_baseline_metric_by_layer(
    baseline_summary,
    metric="kl_to_teacher",
    title=None,
):
    df = baseline_summary.sort_values("layer_index")

    plt.figure(figsize=(8, 5))
    plt.plot(df["layer_index"], df[metric], marker="o")

    plt.xlabel("Layer index")
    plt.ylabel(metric)
    plt.title(title or f"Baseline {metric} by layer")
    plt.grid(True)
    plt.show()


def plot_baseline_all_metrics(baseline_summary):
    metrics = [
        "kl_to_teacher",
        "ce",
        "mean_gt_prob",
        "top1_teacher_agreement",
        "topk_overlap",
        "accept_proxy_exact",
        "accept_proxy_sampled",
    ]

    for metric in metrics:
        if metric in baseline_summary.columns:
            plot_baseline_metric_by_layer(baseline_summary, metric=metric)


def plot_baseline_normalized_metrics(
    baseline_summary,
    metrics=None,
):
    if metrics is None:
        metrics = [
            "mean_gt_prob",
            "top1_teacher_agreement",
            "topk_overlap",
            "accept_proxy_exact",
            "accept_proxy_sampled",
        ]

    df = baseline_summary.sort_values("layer_index").copy()

    plt.figure(figsize=(9, 5))

    for metric in metrics:
        if metric not in df.columns:
            continue

        values = df[metric].astype(float)
        denom = values.max() - values.min()

        if denom == 0:
            normalized = values * 0.0
        else:
            normalized = (values - values.min()) / denom

        plt.plot(
            df["layer_index"],
            normalized,
            marker="o",
            label=metric,
        )

    plt.xlabel("Layer index")
    plt.ylabel("Normalized value")
    plt.title("Baseline normalized metrics by layer")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_baseline_vs_adapted(
    compare_summary,
    metric="kl_to_teacher",
):
    pivot = (
        compare_summary.pivot(
            index="layer_index",
            columns="run_name",
            values=metric,
        )
        .sort_index()
    )

    plt.figure(figsize=(8, 5))

    for col in pivot.columns:
        plt.plot(
            pivot.index,
            pivot[col],
            marker="o",
            label=col,
        )

    plt.xlabel("Layer index")
    plt.ylabel(metric)
    plt.title(f"{metric}: baseline vs adapted")
    plt.grid(True)
    plt.legend()
    plt.show()
