# @title
"""
visualize_tree_lowram.py

Standalone visualizer for OPT-Tree speculative decoding runs produced by
opt_tree_speculative_simple.py.

It can be used directly after:

    final_text, all_ids, step_infos, report = speculative_decode_tree(...)

or after the function in opt_tree_speculative_simple.py, currently named
`speculative_decode`, returns:

    final_text, all_ids, step_infos, report = speculative_decode(...)

Main programmatic usage:

    from visualize_tree_lowram import visualize_opt_tree_run

    viz = visualize_opt_tree_run(
        report=report,
        tokenizer=tokenizer,
        out_dir="opt_tree_viz",
        gif_path="opt_tree_viz/opt_tree.gif",
        html_path="opt_tree_viz/index.html",
        fps=1.0,
        max_steps=None,
    )

CLI usage from a saved JSON log:

    python visualize_tree_lowram.py \
        --log opt_tree_logs.json \
        --out-dir opt_tree_viz \
        --gif opt_tree_viz/opt_tree.gif \
        --html opt_tree_viz/index.html \
        --fps 1.0

Optional CLI tokenizer decoding:

    python visualize_tree_lowram.py --log opt_tree_logs.json --tokenizer gpt2-medium

Dependencies:
    pip install matplotlib pillow

Optional for GIF creation:
    pip install imageio

Optional for CLI tokenizer loading:
    pip install transformers
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import shutil
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


# -----------------------------
# Styling
# -----------------------------

STYLE = {
    "root_fill": "#F3F6FA",
    "root_edge": "#2F4858",
    "node_fill": "#FFFFFF",
    "node_edge": "#4A5568",
    "accepted_fill": "#C6F6D5",
    "accepted_edge": "#178A3B",
    "current_fill": "#DCEBFF",
    "current_edge": "#1A5FB4",
    "rejected_fill": "#FED7D7",
    "rejected_edge": "#C53030",
    "bonus_fill": "#FFE8A3",
    "bonus_edge": "#B7791F",
    "dim_fill": "#F7FAFC",
    "dim_edge": "#CBD5E0",
    "text": "#1A202C",
    "muted": "#718096",
    "edge": "#A0AEC0",
    "banner_bg": "#F8FAFC",
    "banner_edge": "#CBD5E0",
    "sentence_base": "#111827",
    "sentence_prefix": "#334155",
    "sentence_accept": "#15803D",
    "sentence_reject": "#B91C1C",
    "sentence_target": "#1D4ED8",
}


@dataclass
class FrameResult:
    path: str
    caption: str
    step_index: int
    frame_index: int


# -----------------------------
# Public API
# -----------------------------


def visualize_opt_tree_run(
    *,
    report: Optional[Dict[str, Any]] = None,
    step_infos: Optional[List[Dict[str, Any]]] = None,
    log_path: Optional[str | os.PathLike[str]] = None,
    tokenizer: Optional[Any] = None,
    out_dir: str | os.PathLike[str] = "opt_tree_viz",
    gif_path: Optional[str | os.PathLike[str]] = None,
    html_path: Optional[str | os.PathLike[str]] = None,
    max_steps: Optional[int] = None,
    fps: float = 1.0,
    clean_out_dir: bool = True,
    show_probs: bool = True,
    dpi: int = 100,
    create_gif: bool = False,
    gif_max_size: Tuple[int, int] = (1400, 900),
) -> Dict[str, Any]:
    """
    Render all OPT-Tree verification steps into PNG frames, then optionally GIF/HTML.

    Args:
        report: The final_report returned by speculative_decode/speculative_decode_tree.
        step_infos: The raw step_infos returned by the decoder. Used only if report is None.
        log_path: Path to a saved JSON report, e.g. opt_tree_logs.json.
        tokenizer: Optional HF tokenizer. If passed, node token ids are decoded into text.
        out_dir: Directory where frame PNGs and optional HTML/GIF are written.
        gif_path: Optional output GIF path. If None, defaults to out_dir/opt_tree.gif.
        html_path: Optional output HTML path. If None, defaults to out_dir/index.html.
        max_steps: Optional cap for visualization.
        fps: GIF playback speed. 1.0 means each frame lasts roughly 1 second.
        clean_out_dir: Delete old visualization frames before rendering.
        show_probs: Whether node labels include q and path scores.
        dpi: PNG render DPI. Lower is much safer for notebook RAM.
        create_gif: Whether to create an animated GIF. False by default because GIF
            encoding can consume a lot of CPU RAM. HTML frames are usually safer.
        gif_max_size: Maximum GIF frame size after downscaling.

    Returns:
        dict with frame paths, gif path, html path, and counts.
    """
    loaded_report = _load_report(report=report, step_infos=step_infos, log_path=log_path)
    steps = loaded_report.get("steps", [])

    if max_steps is not None:
        steps = steps[: max(0, max_steps)]

    out = Path(out_dir)
    frames_dir = out / "frames"

    if clean_out_dir and frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    if gif_path is None:
        gif_path = out / "opt_tree.gif"
    else:
        gif_path = Path(gif_path)

    if html_path is None:
        html_path = out / "index.html"
    else:
        html_path = Path(html_path)

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)

    all_frames: List[FrameResult] = []

    # Use a stable canvas size across the whole run to make GIF creation clean.
    max_leaf_count, global_max_depth = _estimate_global_tree_size(steps)

    # The decoder report stores full_text_after_step but not full_text_before_step.
    # Reconstruct the sentence before each step from the original input and the
    # previous step output. This is what lets every verification frame show:
    # Current sentence = before this verification choice
    # Updated sentence = after applying this accepted/rejected/target choice
    text_before_step = str(loaded_report.get("input_text", ""))

    for step_idx, step in enumerate(steps):
        step_frames = render_step_frames(
            step=step,
            tokenizer=tokenizer,
            frames_dir=frames_dir,
            step_index=step_idx,
            show_probs=show_probs,
            dpi=dpi,
            global_leaf_count=max_leaf_count,
            global_max_depth=global_max_depth,
            text_before_step=text_before_step,
        )
        all_frames.extend(step_frames)

        after = step.get("output", {}).get("full_text_after_step")
        if after is not None:
            text_before_step = str(after)

    made_gif = False
    if all_frames:
        # HTML is memory-safe: it references PNG files on disk.
        make_html_viewer(
            frames=all_frames,
            html_path=html_path,
            title="OPT-Tree Speculative Decoding Visualization",
            report_summary=_make_report_summary(loaded_report),
        )

        # GIF creation can be RAM-heavy, so it is opt-in and streamed frame-by-frame.
        if create_gif:
            make_gif(
                frame_paths=[frame.path for frame in all_frames],
                gif_path=gif_path,
                fps=fps,
                max_size=gif_max_size,
            )
            made_gif = True
    else:
        print("No frames were produced. Check that report['steps'] exists and is non-empty.")

    result = {
        "out_dir": str(out),
        "frames_dir": str(frames_dir),
        "num_frames": len(all_frames),
        "frame_paths": [frame.path for frame in all_frames],
        "gif_path": str(gif_path) if made_gif else None,
        "html_path": str(html_path) if all_frames else None,
    }

    print("\nVisualization complete")
    print(f"frames: {result['num_frames']}")
    if result["gif_path"]:
        print(f"gif:    {result['gif_path']}")
    if result["html_path"]:
        print(f"html:   {result['html_path']}")

    return result


# Backward-friendly alias.
def visualize_tree_run(**kwargs) -> Dict[str, Any]:
    return visualize_opt_tree_run(**kwargs)


# -----------------------------
# Step rendering
# -----------------------------


def render_step_frames(
    *,
    step: Dict[str, Any],
    tokenizer: Optional[Any],
    frames_dir: Path,
    step_index: int,
    show_probs: bool,
    dpi: int,
    global_leaf_count: int,
    global_max_depth: int,
    text_before_step: str = "",
) -> List[FrameResult]:
    """Render one speculative step into sentence-evolution frames.

    Important behavior:
      - Every frame has a sentence banner at the top.
      - No final summary/stat frame is produced.
      - Accepted draft choices are green.
      - Rejected draft children are red.
      - The target/big-model inserted choice is blue.
    """
    tree = step.get("tree", {})
    verification = step.get("verification", {})

    nodes = _normalize_nodes(tree.get("nodes", []))
    children_by_parent = _children_by_parent(nodes)
    target_choices = verification.get("target_choices", []) or []

    if step.get("step_index") is not None:
        shown_step_idx = step.get("step_index")
    else:
        shown_step_idx = step_index

    common = {
        "nodes": nodes,
        "children_by_parent": children_by_parent,
        "tokenizer": tokenizer,
        "show_probs": show_probs,
        "dpi": dpi,
        "global_leaf_count": global_leaf_count,
        "global_max_depth": global_max_depth,
        "step_index": shown_step_idx,
    }

    frames: List[FrameResult] = []

    # Frame 0: draft tree exists, before target verification starts.
    # It still shows the current/updated sentence banner, but without a stats box.
    intro_banner = _make_sentence_banner(
        prefix_text=text_before_step,
        accepted_text="",
        new_text="",
        new_kind="prefix",
        rejected_options=[],
    )
    caption = (
        f"Step {shown_step_idx}: small model proposed a tree. "
        "Sentence has not changed yet; verification starts in the next frame."
    )
    path = frames_dir / f"step_{shown_step_idx:03d}_frame_000_tree_built.png"
    _render_tree_frame(
        **common,
        frame_path=path,
        title=f"Step {shown_step_idx} — draft tree before verification",
        subtitle=_step_subtitle(step),
        accepted_ids=set(),
        current_parent_id=None,
        current_child_id=None,
        rejected_ids=set(),
        dim_unrelated=False,
        bonus_token_text=None,
        caption=caption,
        sentence_banner=intro_banner,
    )
    frames.append(FrameResult(str(path), caption, shown_step_idx, 0))

    accepted_so_far_ids: List[int] = []
    accepted_so_far_text = ""

    for i, choice in enumerate(target_choices, start=1):
        parent_id = int(choice.get("parent_node_id", -1))
        hit = bool(choice.get("hit_draft_child", False))
        child_id = choice.get("child_node_id")
        token_text = choice.get("target_token_text")
        if token_text is None:
            token_text = _decode_token(tokenizer, choice.get("target_token_id"))
        token_text = str(token_text)

        if hit and child_id is not None:
            child_id = int(child_id)
            new_kind = "accepted"
            rejected_ids: set[int] = set()
            rejected_options: List[str] = []
            bonus = None
            current_child = child_id
            title = f"Step {shown_step_idx} — accepted draft token"
            caption = (
                f"Step {shown_step_idx}, choice {i}: target chose {_quote_token(token_text)}. "
                "That token exists in the draft tree, so the sentence advances with a green accepted draft token."
            )
            accepted_so_far_ids.append(child_id)
        else:
            # Rejection: all drafted children under this parent are rejected because
            # the target chose a token outside that frontier.
            frontier = children_by_parent.get(parent_id, [])
            rejected_ids = {int(n["node_id"]) for n in frontier}
            rejected_options = [_decode_token(tokenizer, n.get("token_id")) for n in frontier]
            new_kind = "target"
            bonus = token_text
            current_child = None
            title = f"Step {shown_step_idx} — reject draft frontier, insert target token"
            caption = (
                f"Step {shown_step_idx}, choice {i}: target chose {_quote_token(token_text)}, "
                "which was not among the drafted children. The red options are rejected and the blue target token is appended."
            )

        banner = _make_sentence_banner(
            prefix_text=text_before_step,
            accepted_text=accepted_so_far_text,
            new_text=token_text,
            new_kind=new_kind,
            rejected_options=rejected_options,
        )

        path = frames_dir / f"step_{shown_step_idx:03d}_frame_{i:03d}_choice.png"
        _render_tree_frame(
            **common,
            frame_path=path,
            title=title,
            subtitle=f"Target choice: {_quote_token(token_text)}",
            accepted_ids=set(accepted_so_far_ids),
            current_parent_id=parent_id,
            current_child_id=current_child,
            rejected_ids=rejected_ids,
            dim_unrelated=True,
            bonus_token_text=bonus,
            caption=caption,
            sentence_banner=banner,
        )
        frames.append(FrameResult(str(path), caption, shown_step_idx, i))

        if hit and child_id is not None:
            accepted_so_far_text += token_text
        else:
            # A rejection/target insert ends verification for this step.
            break

    return frames


# -----------------------------
# Matplotlib renderer
# -----------------------------


def _render_tree_frame(
    *,
    nodes: List[Dict[str, Any]],
    children_by_parent: Dict[int, List[Dict[str, Any]]],
    tokenizer: Optional[Any],
    show_probs: bool,
    dpi: int,
    global_leaf_count: int,
    global_max_depth: int,
    step_index: int,
    frame_path: Path,
    title: str,
    subtitle: str,
    accepted_ids: set[int],
    current_parent_id: Optional[int],
    current_child_id: Optional[int],
    rejected_ids: set[int],
    dim_unrelated: bool,
    bonus_token_text: Optional[str],
    caption: str,
    sentence_banner: Optional[Dict[str, Any]] = None,
):
    positions = _tree_positions(children_by_parent)

    leaf_count = max(global_leaf_count, 3)
    depth = max(global_max_depth, 2)

    width = min(28, max(12, 1.45 * leaf_count + 4))
    height = min(18, max(7, 1.45 * depth + 4))

    # Reserve top space for the sentence banner. The visualization should focus
    # on sentence evolution, so this banner is intentionally more prominent
    # than the tree metadata.
    height = min(20, height + 2.5)
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_axis_off()
    fig.subplots_adjust(top=0.68, bottom=0.08, left=0.04, right=0.98)

    fig.text(
        0.5,
        0.965,
        title,
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
        color=STYLE["text"],
    )

    if subtitle:
        fig.text(
            0.5,
            0.935,
            subtitle,
            ha="center",
            va="top",
            fontsize=10,
            color=STYLE["muted"],
        )

    if sentence_banner is not None:
        _draw_sentence_banner(fig, sentence_banner)

    nodes_by_id = {int(n["node_id"]): n for n in nodes}

    # Edges first.
    for node in nodes:
        child_id = int(node["node_id"])
        parent_id = int(node.get("parent_id", -1))
        if parent_id not in positions or child_id not in positions:
            continue

        x1, y1 = positions[parent_id]
        x2, y2 = positions[child_id]

        if child_id in accepted_ids:
            color = STYLE["accepted_edge"]
            lw = 2.8
            z = 2
        elif child_id in rejected_ids:
            color = STYLE["rejected_edge"]
            lw = 2.4
            z = 2
        elif dim_unrelated:
            color = "#E2E8F0"
            lw = 1.0
            z = 1
        else:
            color = STYLE["edge"]
            lw = 1.2
            z = 1

        ax.plot([x1, x2], [y1 - 0.10, y2 + 0.10], color=color, linewidth=lw, zorder=z)

    # Bonus target token node on rejection.
    if bonus_token_text is not None and current_parent_id is not None and current_parent_id in positions:
        parent_x, parent_y = positions[current_parent_id]
        siblings = children_by_parent.get(current_parent_id, [])
        if siblings:
            sibling_x = [positions[int(s["node_id"])][0] for s in siblings if int(s["node_id"]) in positions]
            bonus_x = max(sibling_x + [parent_x]) + 1.15
        else:
            bonus_x = parent_x + 1.15
        bonus_y = parent_y - 1.0
        ax.plot(
            [parent_x, bonus_x],
            [parent_y - 0.10, bonus_y + 0.10],
            color=STYLE["bonus_edge"],
            linewidth=2.4,
            linestyle="--",
            zorder=3,
        )
        ax.text(
            bonus_x,
            bonus_y,
            _wrap_label(f"target\n{bonus_token_text}", 14),
            ha="center",
            va="center",
            fontsize=10,
            color=STYLE["text"],
            bbox=dict(
                boxstyle="round,pad=0.42",
                facecolor=STYLE["bonus_fill"],
                edgecolor=STYLE["bonus_edge"],
                linewidth=2.2,
            ),
            zorder=5,
        )
        positions[10**9 + step_index] = (bonus_x, bonus_y)

    # Nodes.
    for node_id in sorted(positions.keys()):
        if node_id >= 10**9:
            continue
        x, y = positions[node_id]

        if node_id == -1:
            label = "PREFIX"
            fill = STYLE["root_fill"]
            edge = STYLE["root_edge"]
            lw = 2.0
            fontsize = 11
        else:
            node = nodes_by_id[node_id]
            label = _node_label(node, tokenizer=tokenizer, show_probs=show_probs)
            fill, edge, lw = _node_style(
                node_id=node_id,
                accepted_ids=accepted_ids,
                rejected_ids=rejected_ids,
                current_parent_id=current_parent_id,
                current_child_id=current_child_id,
                dim_unrelated=dim_unrelated,
            )
            fontsize = 9

        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=STYLE["text"],
            bbox=dict(
                boxstyle="round,pad=0.42",
                facecolor=fill,
                edgecolor=edge,
                linewidth=lw,
            ),
            zorder=4,
        )

    # Bottom caption.
    ax.text(
        0.5,
        0.015,
        _wrap_label(caption, 120),
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        color=STYLE["muted"],
    )

    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    if xs and ys:
        ax.set_xlim(min(xs) - 1.5, max(xs) + 1.5)
        ax.set_ylim(min(ys) - 1.2, max(ys) + 0.8)

    fig.tight_layout()
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(frame_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# GIF + HTML
# -----------------------------


def make_gif(
    *,
    frame_paths: List[str | os.PathLike[str]],
    gif_path: str | os.PathLike[str],
    fps: float = 1.0,
    max_size: Tuple[int, int] = (1400, 900),
):
    """
    Create a GIF from frame images using low-RAM streaming.

    The old version loaded all PNGs into RAM and then created a second padded list.
    That can crash Colab/T4 sessions. This version opens one frame at a time,
    downscales it, pads it, writes it, and immediately releases it.

    Requires:
        pip install imageio
    """
    try:
        import imageio.v2 as imageio
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "GIF creation now uses low-RAM streaming via imageio. Install it with: pip install imageio"
        ) from exc

    gif_path = Path(gif_path)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    fps = max(float(fps), 0.1)
    duration_s = 1.0 / fps

    paths = [Path(p) for p in frame_paths]
    if not paths:
        return

    max_w_limit, max_h_limit = max_size

    # First pass: find the max dimensions after thumbnailing, without keeping images.
    max_w = 1
    max_h = 1
    for path in paths:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((max_w_limit, max_h_limit), Image.Resampling.LANCZOS)
            max_w = max(max_w, im.width)
            max_h = max(max_h, im.height)

    # Second pass: write one frame at a time.
    with imageio.get_writer(gif_path, mode="I", duration=duration_s, loop=0) as writer:
        for path in paths:
            with Image.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail((max_w_limit, max_h_limit), Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", (max_w, max_h), "white")
                x = (max_w - im.width) // 2
                y = (max_h - im.height) // 2
                canvas.paste(im, (x, y))
                writer.append_data(np.asarray(canvas))

def make_html_viewer(
    *,
    frames: List[FrameResult],
    html_path: str | os.PathLike[str],
    title: str,
    report_summary: str,
):
    """Create a simple interactive viewer with slider and previous/next buttons."""
    html_path = Path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)

    frame_items = []
    for frame in frames:
        rel = os.path.relpath(frame.path, start=html_path.parent)
        frame_items.append(
            {
                "src": rel.replace(os.sep, "/"),
                "caption": frame.caption,
                "step": frame.step_index,
                "frame": frame.frame_index,
            }
        )

    data_json = json.dumps(frame_items, ensure_ascii=False)

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0f172a;
      color: #e2e8f0;
    }}
    header {{
      padding: 24px 32px 12px 32px;
    }}
    h1 {{ margin: 0 0 8px 0; font-size: 28px; }}
    .summary {{ color: #94a3b8; max-width: 980px; line-height: 1.45; }}
    .viewer {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 12px 24px 32px 24px;
    }}
    .panel {{
      background: #111827;
      border: 1px solid #334155;
      border-radius: 18px;
      padding: 16px;
      box-shadow: 0 20px 50px rgba(0,0,0,.35);
    }}
    img {{
      width: 100%;
      max-height: 74vh;
      object-fit: contain;
      background: white;
      border-radius: 12px;
      display: block;
    }}
    .controls {{
      display: flex;
      gap: 12px;
      align-items: center;
      margin-top: 14px;
      flex-wrap: wrap;
    }}
    button {{
      background: #2563eb;
      color: white;
      border: 0;
      border-radius: 10px;
      padding: 10px 14px;
      font-weight: 700;
      cursor: pointer;
    }}
    button:hover {{ background: #1d4ed8; }}
    input[type=range] {{ flex: 1; min-width: 220px; }}
    .caption {{
      margin-top: 12px;
      color: #cbd5e1;
      line-height: 1.45;
      font-size: 15px;
    }}
    .counter {{ color: #94a3b8; font-variant-numeric: tabular-nums; }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <div class="summary">{html.escape(report_summary)}</div>
  </header>
  <main class="viewer">
    <div class="panel">
      <img id="frame" alt="OPT-Tree frame" />
      <div class="controls">
        <button onclick="prevFrame()">← Previous</button>
        <button onclick="nextFrame()">Next →</button>
        <input id="slider" type="range" min="0" max="0" value="0" oninput="setFrame(Number(this.value))" />
        <span id="counter" class="counter"></span>
      </div>
      <div id="caption" class="caption"></div>
    </div>
  </main>
  <script>
    const frames = {data_json};
    let idx = 0;
    const img = document.getElementById('frame');
    const slider = document.getElementById('slider');
    const caption = document.getElementById('caption');
    const counter = document.getElementById('counter');
    slider.max = Math.max(0, frames.length - 1);

    function setFrame(i) {{
      idx = Math.max(0, Math.min(frames.length - 1, i));
      const f = frames[idx];
      img.src = f.src;
      caption.textContent = f.caption;
      slider.value = idx;
      counter.textContent = `${{idx + 1}} / ${{frames.length}}`;
    }}
    function nextFrame() {{ setFrame(idx + 1); }}
    function prevFrame() {{ setFrame(idx - 1); }}
    document.addEventListener('keydown', (e) => {{
      if (e.key === 'ArrowRight') nextFrame();
      if (e.key === 'ArrowLeft') prevFrame();
    }});
    if (frames.length) setFrame(0);
  </script>
</body>
</html>
"""

    html_path.write_text(html_text, encoding="utf-8")



# -----------------------------
# Sentence-banner helpers
# -----------------------------


def _make_sentence_banner(
    *,
    prefix_text: str,
    accepted_text: str,
    new_text: str,
    new_kind: str,
    rejected_options: List[str],
) -> Dict[str, Any]:
    """Build banner data for the top of a frame."""
    prefix_text = str(prefix_text or "")
    accepted_text = str(accepted_text or "")
    new_text = str(new_text or "")

    current_segments = [
        {"text": _tail(prefix_text, 135), "kind": "prefix"},
    ]
    if accepted_text:
        current_segments.append({"text": accepted_text, "kind": "accepted"})

    updated_segments = [
        {"text": _tail(prefix_text, 135), "kind": "prefix"},
    ]
    if accepted_text:
        updated_segments.append({"text": accepted_text, "kind": "accepted"})
    if new_text:
        updated_segments.append({"text": new_text, "kind": new_kind})

    return {
        "current_segments": current_segments,
        "updated_segments": updated_segments,
        "rejected_options": rejected_options[:12],
    }


def _draw_sentence_banner(fig, banner: Dict[str, Any]):
    """Draw a compact, readable, colored sentence state banner."""
    # Background panel.
    fig.text(
        0.5,
        0.850,
        "",
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.85",
            facecolor=STYLE["banner_bg"],
            edgecolor=STYLE["banner_edge"],
            linewidth=1.2,
        ),
    )

    fig.text(0.055, 0.887, "Current sentence:", ha="left", va="center", fontsize=11, fontweight="bold", color=STYLE["text"])
    _draw_segmented_line(fig, 0.205, 0.887, banner.get("current_segments", []), max_chars=165)

    fig.text(0.055, 0.842, "Updated sentence:", ha="left", va="center", fontsize=11, fontweight="bold", color=STYLE["text"])
    _draw_segmented_line(fig, 0.205, 0.842, banner.get("updated_segments", []), max_chars=165)

    rejected_options = banner.get("rejected_options", []) or []
    if rejected_options:
        rejected = " | ".join(_clean_visible_token(x) for x in rejected_options)
        fig.text(0.055, 0.797, "Rejected draft options:", ha="left", va="center", fontsize=11, fontweight="bold", color=STYLE["sentence_reject"])
        fig.text(0.235, 0.797, _shorten(rejected, 150), ha="left", va="center", fontsize=10, color=STYLE["sentence_reject"])
    else:
        fig.text(0.055, 0.797, "Legend:", ha="left", va="center", fontsize=10, fontweight="bold", color=STYLE["muted"])
        fig.text(0.125, 0.797, "green = accepted draft  |  red = rejected draft options  |  blue = target/big-model choice", ha="left", va="center", fontsize=10, color=STYLE["muted"])


def _draw_segmented_line(fig, x: float, y: float, segments: List[Dict[str, str]], max_chars: int):
    """Draw colored text segments using a stable monospace-ish width approximation."""
    visible = _truncate_segments(segments, max_chars=max_chars)
    cursor = x
    char_w = 0.0060
    for seg in visible:
        text = _clean_visible_token(seg.get("text", ""))
        if not text:
            continue
        kind = seg.get("kind", "prefix")
        color = _segment_color(kind)
        weight = "bold" if kind in {"accepted", "rejected", "target"} else "normal"
        fig.text(cursor, y, text, ha="left", va="center", fontsize=10.5, color=color, fontweight=weight)
        cursor += max(0.008, len(text) * char_w)


def _truncate_segments(segments: List[Dict[str, str]], max_chars: int) -> List[Dict[str, str]]:
    """Keep the important tail of long sentences while preserving colored suffix tokens."""
    clean = [{"text": str(s.get("text", "")), "kind": str(s.get("kind", "prefix"))} for s in segments]
    total = sum(len(_clean_visible_token(s["text"])) for s in clean)
    if total <= max_chars:
        return clean

    # Preserve non-prefix generated tokens and truncate the prefix tail.
    suffix_len = sum(len(_clean_visible_token(s["text"])) for s in clean[1:])
    prefix_budget = max(20, max_chars - suffix_len - 3)
    if clean:
        clean[0] = {"text": _tail(clean[0]["text"], prefix_budget), "kind": clean[0]["kind"]}
    return clean


def _segment_color(kind: str) -> str:
    if kind == "accepted":
        return STYLE["sentence_accept"]
    if kind == "rejected":
        return STYLE["sentence_reject"]
    if kind == "target":
        return STYLE["sentence_target"]
    return STYLE["sentence_prefix"]


def _clean_visible_token(text: Any) -> str:
    s = str(text)
    return s.replace("\n", "\\n").replace("\t", "\\t")


def _tail(text: Any, max_chars: int) -> str:
    s = _clean_visible_token(text)
    if len(s) <= max_chars:
        return s
    return "…" + s[-max(0, max_chars - 1):]


# -----------------------------
# Data helpers
# -----------------------------


def _load_report(
    *,
    report: Optional[Dict[str, Any]],
    step_infos: Optional[List[Dict[str, Any]]],
    log_path: Optional[str | os.PathLike[str]],
) -> Dict[str, Any]:
    if report is not None:
        return report

    if log_path is not None:
        path = Path(log_path)
        return json.loads(path.read_text(encoding="utf-8"))

    if step_infos is not None:
        # Convert raw step_infos into the same report-like shape.
        steps = []
        for info in step_infos:
            if "step_log" in info:
                steps.append(info["step_log"])
        return {"decode_method": "opt_tree_speculative_decoding", "steps": steps}

    raise ValueError("Pass either report=..., log_path=..., or step_infos=...")


def _normalize_nodes(nodes: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    clean = []
    for node in nodes:
        clean.append(
            {
                "node_id": int(node.get("node_id")),
                "token_id": int(node.get("token_id")),
                "parent_id": int(node.get("parent_id", -1)),
                "depth": int(node.get("depth", 0)),
                "draft_prob": float(node.get("draft_prob", 0.0)),
                "path_score": float(node.get("path_score", 0.0)),
            }
        )
    clean.sort(key=lambda n: (n["depth"], n["node_id"]))
    return clean


def _children_by_parent(nodes: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    children: Dict[int, List[Dict[str, Any]]] = {}
    for node in nodes:
        children.setdefault(int(node["parent_id"]), []).append(node)
    for parent, child_list in children.items():
        child_list.sort(key=lambda n: (-float(n.get("path_score", 0.0)), int(n["node_id"])))
    return children


def _tree_positions(children_by_parent: Dict[int, List[Dict[str, Any]]]) -> Dict[int, Tuple[float, float]]:
    """Simple tidy-tree layout: leaves get consecutive x positions."""
    positions: Dict[int, Tuple[float, float]] = {}
    next_x = 0

    def dfs(node_id: int, depth: int) -> float:
        nonlocal next_x
        children = children_by_parent.get(node_id, [])
        if not children:
            x = float(next_x)
            next_x += 1
        else:
            child_xs = [dfs(int(child["node_id"]), depth + 1) for child in children]
            x = sum(child_xs) / len(child_xs)
        positions[node_id] = (x, -float(depth))
        return x

    dfs(-1, 0)
    return positions


def _estimate_global_tree_size(steps: List[Dict[str, Any]]) -> Tuple[int, int]:
    max_leaves = 3
    max_depth = 2
    for step in steps:
        nodes = _normalize_nodes(step.get("tree", {}).get("nodes", []))
        children = _children_by_parent(nodes)
        positions = _tree_positions(children)
        leaves = sum(1 for node_id in positions if node_id != -1 and not children.get(node_id))
        max_leaves = max(max_leaves, leaves)
        max_depth = max(max_depth, _max_depth(nodes))
    return max_leaves, max_depth


def _max_depth(nodes: List[Dict[str, Any]]) -> int:
    if not nodes:
        return 0
    return max(int(n.get("depth", 0)) for n in nodes)


def _node_style(
    *,
    node_id: int,
    accepted_ids: set[int],
    rejected_ids: set[int],
    current_parent_id: Optional[int],
    current_child_id: Optional[int],
    dim_unrelated: bool,
) -> Tuple[str, str, float]:
    if node_id in rejected_ids:
        return STYLE["rejected_fill"], STYLE["rejected_edge"], 2.4
    if node_id in accepted_ids:
        return STYLE["accepted_fill"], STYLE["accepted_edge"], 2.6
    if current_child_id is not None and node_id == current_child_id:
        return STYLE["current_fill"], STYLE["current_edge"], 3.0
    if current_parent_id is not None and node_id == current_parent_id:
        return STYLE["current_fill"], STYLE["current_edge"], 2.6
    if dim_unrelated:
        return STYLE["dim_fill"], STYLE["dim_edge"], 1.0
    return STYLE["node_fill"], STYLE["node_edge"], 1.2


def _node_label(node: Dict[str, Any], *, tokenizer: Optional[Any], show_probs: bool) -> str:
    tok = _decode_token(tokenizer, node.get("token_id"))
    tok = tok.replace("\n", "\\n")
    tok = _shorten(tok, 18)

    if show_probs:
        label = f"{tok}\nq={node.get('draft_prob', 0.0):.2f}\npath={node.get('path_score', 0.0):.3f}"
    else:
        label = tok
    return _wrap_label(label, 18)


def _decode_token(tokenizer: Optional[Any], token_id: Optional[int]) -> str:
    if token_id is None:
        return "?"
    try:
        token_id = int(token_id)
    except Exception:
        return str(token_id)

    if tokenizer is None:
        return f"id:{token_id}"

    try:
        return tokenizer.decode([token_id], skip_special_tokens=False)
    except Exception:
        return f"id:{token_id}"


def _quote_token(text: Any) -> str:
    s = str(text)
    s = s.replace("\n", "\\n")
    if s == "":
        return "''"
    return repr(_shorten(s, 60))


def _shorten(text: Any, max_chars: int) -> str:
    s = str(text)
    if len(s) <= max_chars:
        return s
    return s[: max(0, max_chars - 1)] + "…"


def _wrap_label(text: str, width: int) -> str:
    lines = []
    for line in str(text).splitlines() or [""]:
        if len(line) <= width:
            lines.append(line)
        else:
            lines.extend(textwrap.wrap(line, width=width, break_long_words=False) or [line])
    return "\n".join(lines)


def _step_subtitle(step: Dict[str, Any]) -> str:
    tree = step.get("tree", {})
    tf = step.get("target_forward", {})
    verification = step.get("verification", {})

    parts = [
        f"nodes={tree.get('actual_nodes', '?')}/{tree.get('node_budget', '?')}",
        f"depth={tree.get('actual_depth', '?')}/{tree.get('max_depth_setting', '?')}",
        f"target={tf.get('target_forward_method', '?')}",
        f"accepted={verification.get('num_accepted_draft_tokens', '?')}",
    ]
    return "  |  ".join(parts)


def _summary_box_text(step: Dict[str, Any]) -> str:
    tree = step.get("tree", {})
    verification = step.get("verification", {})
    output = step.get("output", {})
    target = step.get("target_forward", {})
    running = step.get("running_stats", {})

    lines = [
        f"nodes: {tree.get('actual_nodes', '?')}/{tree.get('node_budget', '?')}",
        f"depth: {tree.get('actual_depth', '?')}/{tree.get('max_depth_setting', '?')}",
        f"accepted draft tokens: {verification.get('num_accepted_draft_tokens', '?')}",
        f"bonus token: {_quote_token(verification.get('bonus_text', ''))}",
        f"step text: {_quote_token(output.get('step_text', ''))}",
        f"target forward: {target.get('target_forward_method', '?')}",
    ]

    if running:
        mean_acc = running.get("mean_acceptance_length")
        toks = running.get("tokens_per_second_wall")
        if mean_acc is not None:
            lines.append(f"mean step length: {mean_acc:.2f}")
        if toks is not None:
            lines.append(f"wall tokens/s: {toks:.2f}")

    return "\n".join(lines)


def _make_report_summary(report: Dict[str, Any]) -> str:
    method = report.get("decode_method", "unknown")
    node_budget = report.get("node_budget", "?")
    max_depth = report.get("max_depth", "?")
    threshold = report.get("threshold", "?")
    branch_top_k = report.get("branch_top_k", "?")
    steps = len(report.get("steps", []))
    stats = report.get("stats", {})

    mean_acc = stats.get("mean_acceptance_length")
    mean_nodes = stats.get("mean_tree_nodes")

    bits = [
        f"method={method}",
        f"steps={steps}",
        f"node_budget={node_budget}",
        f"max_depth={max_depth}",
        f"threshold={threshold}",
        f"branch_top_k={branch_top_k}",
    ]
    if mean_acc is not None:
        bits.append(f"mean_step_length={mean_acc:.2f}")
    if mean_nodes is not None:
        bits.append(f"mean_nodes={mean_nodes:.2f}")

    return " | ".join(bits)


