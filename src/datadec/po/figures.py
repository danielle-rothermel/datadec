"""Spec-driven grid figures over the tidy results tables (contract po-tidy/1).

A spec (JSON, `configs/po/figures/*.json`) lays out `rows` x `cols` panels; each panel draws one axis per entry
of its `metrics` (stacked: accuracy above likelihood). In `value` mode a panel plots summary rows (one point per
model, 95 % item-bootstrap CIs as error bars); in `delta` mode it plots contrast rows (b - a with their CIs).
Panel data are the tidy table joined with models.csv: in value mode the model's attributes; in delta mode the
b model's attributes unprefixed and the a model's prefixed `a_`. Series filters match on those columns
(a list matches any of its values; null matches missing). Value-mode series use the canonical prompt
(instruction `none`) unless their filter names an `instruction_id`.

Spec fields beyond the contract: `x.order` (category order; makes x categorical, points dodged per series),
`x.labels` (tick labels per value), `x.label` (axis label), `x.connect` (draw lines; default off for tasks),
`series.linestyle`, `panel_size` ([width, height] inches per metric axis) and `legend` (`each` panel or only
the `first`). Series colours are palette keys fixed in `PALETTE`, or `by:<field>` to colour each point by
its value of `stage_index`, `precision`, `recipe`, `variant`, `family` or `checkpoint_kind`.

`band: "seed"` draws the default-seed point with its CI and shades the min-max range over all seeds at each x.
`chance_line` draws dashed grey chance levels (zero for margins and deltas). `arrows` draws, from each plotted
point, the matching contrast rows of that type (GEPA deltas, b_key ending `:both`); nothing when none exist.
Every PNG is written with a CSV twin of the plotted numbers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pydantic import BaseModel, ConfigDict, field_validator, model_validator  # noqa: E402

from datadec.po.tidy import BASELINE_INSTRUCTION, TASKS, ContrastType, Role  # noqa: E402

PALETTE = {
    # recipes
    "dclm": "#1f77b4", "dolma": "#ff7f0e", "c4": "#7f7f7f",
    # variants
    "base": "#1f77b4", "instruct": "#d62728",
    # post-training stages, grey to black
    "stage-0": "#bdbdbd", "stage-1": "#969696", "stage-2": "#525252", "stage-3": "#000000",
    # precision
    "fp32-tf32": "#000000", "bf16": "#1f77b4", "int8": "#2ca02c", "nf4": "#ff7f0e",
    # families (DataDecide shown in its recipe colours)
    "pythia": "#2ca02c", "qwen3": "#9467bd", "olmo2": "#8c564b", "olmoe": "#e377c2", "olmo3": "#bcbd22",
    "olmo3.1": "#17becf",
    # checkpoints
    "final": "#000000", "mid": "#6baed6", "peak": "#fd8d3c",
}
BY_FIELDS = {"stage_index": "stage-{}", "precision": "{}", "recipe": "{}", "variant": "{}", "family": "{}",
             "checkpoint_kind": "{}"}
MARKERS = ("o", "s", "^", "v", "D", "P", "X", "*", "<", ">")
LINESTYLES = ("-", "--", ":", "-.")
VALUE_METRICS = ("accuracy", "likelihood", "margin_accuracy", "margin_likelihood")
DELTA_METRICS = ("accuracy", "likelihood")
METRIC_LABEL = {"accuracy": "Accuracy", "likelihood": "Likelihood", "margin_accuracy": "Accuracy − chance",
                "margin_likelihood": "Likelihood − chance"}
DELTA_LABEL = {"accuracy": "Δ Accuracy", "likelihood": "Δ Likelihood"}
X_LABEL = {"params": "Parameters", "checkpoint_fraction": "Fraction of training", "stage_index": "Post-training stage",
           "precision": "Weight precision", "task": ""}
LINE_GREY = "#9e9e9e"
CHANCE_GREY = "#8c8c8c"
LEGEND_GREY = "#525252"
PANEL_SIZE = (2.9, 2.0)  # inches per metric axis


class SeriesSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str
    filter: dict[str, str | int | float | bool | None | list[str | int | float | bool | None]]
    color: str
    band: Literal["seed"] | None = None
    marker: str = "o"
    linestyle: str = "-"

    @field_validator("color")
    @classmethod
    def _color(cls, v: str) -> str:
        if v.startswith("by:"):
            if v[3:] not in BY_FIELDS:
                raise ValueError(f"colour field must be one of {sorted(BY_FIELDS)}, got {v}")
        elif v not in PALETTE:
            raise ValueError(f"colour {v!r} is not a palette key ({sorted(PALETTE)})")
        return v

    @field_validator("marker")
    @classmethod
    def _marker(cls, v: str) -> str:
        if v not in MARKERS:
            raise ValueError(f"marker must be one of {MARKERS}")
        return v

    @field_validator("linestyle")
    @classmethod
    def _linestyle(cls, v: str) -> str:
        if v not in LINESTYLES:
            raise ValueError(f"linestyle must be one of {LINESTYLES}")
        return v


class XSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: str
    log: bool = False
    order: list[str | int | float] | None = None
    labels: dict[str, str] | None = None
    label: str | None = None
    connect: bool | None = None

    @property
    def categorical(self) -> bool:
        return self.order is not None or self.field == "task"

    @property
    def draws_lines(self) -> bool:
        return self.connect if self.connect is not None else self.field != "task"


class ArrowSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    contrast_type: ContrastType


class PanelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str
    task: str | None = None
    tasks: list[str] | None = None
    role: Role = Role.TEST
    x: XSpec
    metrics: list[str]
    series: list[SeriesSpec]
    chance_line: bool = True
    arrows: ArrowSpec | None = None

    @model_validator(mode="after")
    def _check(self) -> PanelSpec:
        if (self.task is None) == (self.tasks is None):
            raise ValueError(f"panel {self.title!r}: give exactly one of task / tasks")
        unknown = [t for t in self.task_list if t not in TASKS]
        if unknown:
            raise ValueError(f"panel {self.title!r}: unknown tasks {unknown}")
        if self.tasks is not None and self.x.field != "task":
            raise ValueError(f"panel {self.title!r}: several tasks need x.field 'task'")
        if not self.series:
            raise ValueError(f"panel {self.title!r}: no series")
        return self

    @property
    def task_list(self) -> list[str]:
        return [self.task] if self.task is not None else list(self.tasks or [])


class FigureSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str
    rows: int
    cols: int
    metric_mode: Literal["value", "delta"]
    panels: list[PanelSpec]
    panel_size: tuple[float, float] = PANEL_SIZE
    legend: Literal["each", "first"] = "each"

    @model_validator(mode="after")
    def _check(self) -> FigureSpec:
        if len(self.panels) > self.rows * self.cols:
            raise ValueError(f"{len(self.panels)} panels do not fit a {self.rows} x {self.cols} grid")
        allowed = VALUE_METRICS if self.metric_mode == "value" else DELTA_METRICS
        n = {len(p.metrics) for p in self.panels}
        if len(n) != 1:
            raise ValueError("every panel needs the same number of metrics")
        for p in self.panels:
            if bad := [m for m in p.metrics if m not in allowed]:
                raise ValueError(f"panel {p.title!r}: metrics {bad} not allowed in {self.metric_mode} mode")
            if self.metric_mode == "delta":
                if p.arrows is not None:
                    raise ValueError(f"panel {p.title!r}: arrows apply to value mode only")
                if missing := [s.label for s in p.series if "contrast_type" not in s.filter]:
                    raise ValueError(f"panel {p.title!r}: delta series need a contrast_type filter: {missing}")
        return self


def load_spec(path: Path) -> FigureSpec:
    return FigureSpec.model_validate(json.loads(Path(path).read_text()))


# ---------------------------------------------------------------- data


def _value_frame(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return tables["summary"].merge(tables["models"], on="model_key", how="left", validate="many_to_one")


def _delta_frame(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    models = tables["models"]
    a = models.add_prefix("a_").rename(columns={"a_model_key": "a_key"})
    c = tables["contrasts"].merge(models.rename(columns={"model_key": "b_key"}), on="b_key", how="left")
    return c.merge(a, on="a_key", how="left")


def _match(df: pd.DataFrame, flt: dict) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col, want in flt.items():
        if col not in df.columns:
            raise KeyError(f"filter column {col!r} is not in the panel data")
        values = want if isinstance(want, list) else [want]
        hit = df[col].isin([v for v in values if v is not None])
        if any(v is None for v in values):
            hit |= df[col].isna()
        mask &= hit
    return mask


def _x_positions(panel: PanelSpec, values: pd.Series) -> tuple[np.ndarray, list, list[str]]:
    """Numeric x for each value plus (tick positions, tick labels) for categorical axes."""
    if not panel.x.categorical:
        return values.to_numpy(float), [], []
    order = list(panel.x.order) if panel.x.order is not None else (
        [t for t in TASKS if t in panel.task_list] if panel.x.field == "task" else sorted(values.unique()))
    lookup = {str(v): i for i, v in enumerate(order)}
    pos = np.array([lookup.get(str(v), np.nan) for v in values], dtype=float)
    labels = [(panel.x.labels or {}).get(str(v), str(v)) for v in order]
    return pos, list(range(len(order))), labels


def _series_points(spec: FigureSpec, panel: PanelSpec, series: SeriesSpec, data: pd.DataFrame,
                   metric: str) -> pd.DataFrame:
    """One row per x: y, lo, hi (and band_lo, band_hi, n_seeds for seed bands)."""
    flt = dict(series.filter)
    if spec.metric_mode == "value" and "instruction_id" not in flt:
        flt["instruction_id"] = BASELINE_INSTRUCTION
    rows = data[_match(data, flt)].copy()
    if spec.metric_mode == "value":
        if metric.startswith("margin_"):
            base = metric.removeprefix("margin_")
            rows["y"], rows["lo"], rows["hi"] = (rows[metric], rows[f"{base}_lo"] - rows[f"chance_{base}"],
                                                 rows[f"{base}_hi"] - rows[f"chance_{base}"])
            rows["chance"] = 0.0
        else:
            rows["y"], rows["lo"], rows["hi"] = rows[metric], rows[f"{metric}_lo"], rows[f"{metric}_hi"]
            rows["chance"] = rows[f"chance_{metric}"]
    else:
        rows = rows[rows["metric"] == metric]
        rows["y"], rows["chance"] = rows["delta"], 0.0
    rows["xval"] = rows[panel.x.field]
    key = "model_key" if spec.metric_mode == "value" else "b_key"
    if series.band == "seed":
        band = rows.groupby("xval", dropna=False)["y"].agg(band_lo="min", band_hi="max", n_seeds="size")
        main = rows[rows["seed_label"] == "default"].merge(band, left_on="xval", right_index=True, how="left")
        missing = band.index.difference(main["xval"])
        rows = pd.concat([main, band.loc[missing].reset_index()], ignore_index=True)
    elif rows["xval"].duplicated().any():
        dup = rows[rows["xval"].duplicated(keep=False)]
        raise ValueError(f"series {series.label!r} in panel {panel.title!r} ({metric}) has several rows per x: "
                         f"{sorted(dup[key])}")
    keep = [c for c in ("task", "xval", "y", "lo", "hi", "chance", "band_lo", "band_hi", "n_seeds", "model_key",
                        "a_key", "b_key", "stage_index", "precision", "recipe", "variant", "family",
                        "checkpoint_kind", "role") if c in rows.columns]
    return rows[keep].reset_index(drop=True)


def _arrows(tables: dict[str, pd.DataFrame], panel: PanelSpec, points: pd.DataFrame, metric: str) -> pd.DataFrame:
    if panel.arrows is None or points.empty or metric not in DELTA_METRICS:
        return pd.DataFrame()
    c = tables["contrasts"]
    c = c[(c["contrast_type"] == panel.arrows.contrast_type) & (c["metric"] == metric) & (c["role"] == panel.role)
          & c["b_key"].str.endswith(":both")]
    if c.empty:
        return pd.DataFrame()
    return points.merge(c[["a_key", "b_key", "task", "delta", "lo", "hi"]].rename(
        columns={"a_key": "model_key", "delta": "arrow_delta", "lo": "arrow_lo", "hi": "arrow_hi"}),
        on=["model_key", "task"], how="inner")


# ---------------------------------------------------------------- drawing


def _style(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#ebebeb", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)


def _params_ticks(ax: plt.Axes) -> None:
    lo, hi = ax.get_xlim()
    ticks = [t for t in (1e6, 1e7, 1e8, 1e9, 1e10, 1e11) if lo <= t <= hi]
    ax.set_xticks(ticks, [f"{t / 1e9:g}B" if t >= 1e9 else f"{t / 1e6:g}M" for t in ticks])
    ax.set_xticks([], minor=True)


def _point_colors(series: SeriesSpec, pts: pd.DataFrame) -> list[str]:
    if not series.color.startswith("by:"):
        return [PALETTE[series.color]] * len(pts)
    field = series.color[3:]
    return [PALETTE.get(BY_FIELDS[field].format(v), LINE_GREY) for v in pts[field]]


def _draw_series(ax: plt.Axes, panel: PanelSpec, series: SeriesSpec, pts: pd.DataFrame, offset: float) -> None:
    if pts.empty:
        return
    pts = pts.assign(_x=_x_positions(panel, pts["xval"])[0] + offset).dropna(subset=["_x"]).sort_values("_x")
    colors = _point_colors(series, pts)
    line_color = LINE_GREY if series.color.startswith("by:") else PALETTE[series.color]
    has_y = pts["y"].notna()
    if "band_lo" in pts and (pts["n_seeds"] > 1).any():
        b = pts[pts["n_seeds"] > 1]
        ax.fill_between(b["_x"], b["band_lo"], b["band_hi"], color=line_color, alpha=0.15, linewidth=0)
    p = pts[has_y]
    if panel.x.draws_lines and len(p) > 1:
        ax.plot(p["_x"], p["y"], color=line_color, linestyle=series.linestyle, linewidth=1.2, zorder=2)
    p_colors = [c for c, h in zip(colors, has_y) if h]
    if "lo" in p:
        for color in dict.fromkeys(p_colors):
            q = p[[c == color for c in p_colors]]
            ax.errorbar(q["_x"], q["y"], yerr=[q["y"] - q["lo"], q["hi"] - q["y"]], fmt="none", ecolor=color,
                        elinewidth=0.9, capsize=1.8, zorder=3)
    by_point = series.color.startswith("by:")
    ax.scatter(p["_x"], p["y"], c=p_colors, marker=series.marker, s=20, zorder=4, edgecolors="white",
               linewidths=0.4, label=None if by_point else series.label)
    if by_point:  # legend proxy in neutral grey; the point colours carry the by-field
        ax.plot([], [], color=LEGEND_GREY, marker=series.marker, markersize=4, linestyle=series.linestyle,
                linewidth=1.0, label=series.label)


def _draw_arrows(ax: plt.Axes, panel: PanelSpec, arrows: pd.DataFrame, offset: float) -> None:
    for _, r in arrows.iterrows():
        x = _x_positions(panel, pd.Series([r["xval"]]))[0][0] + offset
        ax.annotate("", xy=(x, r["y"] + r["arrow_delta"]), xytext=(x, r["y"]),
                    arrowprops=dict(arrowstyle="->", color="#2b2b2b", linewidth=1.0), zorder=5)


def _draw_chance(ax: plt.Axes, panel: PanelSpec, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    if panel.x.field == "task" and frame["chance"].nunique() > 1:
        levels = frame.groupby("task")["chance"].first()
        pos, _, _ = _x_positions(panel, pd.Series(levels.index))
        for x, level in zip(pos, levels):
            ax.hlines(level, x - 0.4, x + 0.4, colors=CHANCE_GREY, linestyles="--", linewidth=0.9, zorder=1)
    else:
        ax.axhline(float(frame["chance"].iloc[0]), color=CHANCE_GREY, linestyle="--", linewidth=0.9, zorder=1)


def render(spec: FigureSpec, tables: dict[str, pd.DataFrame], out_png: Path) -> tuple[Path, Path]:
    """Draw the figure to out_png and its plotted numbers to the CSV twin beside it."""
    data = _value_frame(tables) if spec.metric_mode == "value" else _delta_frame(tables)
    n_metrics = len(spec.panels[0].metrics)
    w, h = spec.panel_size
    fig, axes = plt.subplots(spec.rows * n_metrics, spec.cols, figsize=(w * spec.cols, h * spec.rows * n_metrics),
                             squeeze=False, layout="constrained")
    fig.patch.set_facecolor("white")
    records = []
    for i, panel in enumerate(spec.panels):
        r0, c = divmod(i, spec.cols)
        pdata = data[(data["role"] == panel.role) & data["task"].isin(panel.task_list)]
        k = len(panel.series)
        width = min(0.12, 0.8 / max(k, 1)) if panel.x.categorical else 0.0
        for j, metric in enumerate(panel.metrics):
            ax = axes[r0 * n_metrics + j, c]
            _style(ax)
            drawn = []
            for s_i, series in enumerate(panel.series):
                pts = _series_points(spec, panel, series, pdata, metric)
                offset = (s_i - (k - 1) / 2) * width
                _draw_series(ax, panel, series, pts, offset)
                arrows = _arrows(tables, panel, pts, metric) if spec.metric_mode == "value" else pd.DataFrame()
                if not arrows.empty:
                    _draw_arrows(ax, panel, arrows, offset)
                    pts = pts.merge(arrows[["model_key", "task", "b_key", "arrow_delta", "arrow_lo", "arrow_hi"]]
                                    .rename(columns={"b_key": "arrow_key"}), on=["model_key", "task"], how="left")
                drawn.append(pts)
                records.append(pts.assign(panel=panel.title, metric=metric, series=series.label,
                                          x_field=panel.x.field))
            if panel.chance_line:
                _draw_chance(ax, panel, pd.concat(drawn, ignore_index=True) if drawn else pd.DataFrame())
            if panel.x.log:
                ax.set_xscale("log")
            if panel.x.categorical:
                _, ticks, labels = _x_positions(panel, pd.Series([], dtype=object))
                ax.set_xticks(ticks, labels, rotation=30 if panel.x.field == "task" else 0,
                              ha="right" if panel.x.field == "task" else "center")
            elif panel.x.field == "params":
                _params_ticks(ax)
            ylabel = (METRIC_LABEL if spec.metric_mode == "value" else DELTA_LABEL)[metric]
            ax.set_ylabel(ylabel, fontsize=9)
            if j == 0:
                ax.set_title(panel.title, fontsize=9.5)
                if any(not p.empty for p in drawn):
                    if spec.legend == "each":
                        ax.legend(frameon=False, fontsize=7, loc="best", handletextpad=0.3, borderaxespad=0.2)
                    elif i == 0:  # one shared legend outside the grid
                        handles, labels = ax.get_legend_handles_labels()
                        fig.legend(handles, labels, loc="outside right upper", frameon=False, fontsize=8)
            if j == n_metrics - 1:
                ax.set_xlabel(panel.x.label if panel.x.label is not None else X_LABEL.get(panel.x.field, panel.x.field),
                              fontsize=9)
            else:
                ax.tick_params(labelbottom=False)
    for i in range(len(spec.panels), spec.rows * spec.cols):
        r0, c = divmod(i, spec.cols)
        for j in range(n_metrics):
            axes[r0 * n_metrics + j, c].set_visible(False)
    roles = {p.role for p in spec.panels}
    note = f"{roles.pop().replace('-', ' ').capitalize()} role" if len(roles) == 1 else "mixed roles"
    what = "means" if spec.metric_mode == "value" else "paired differences b − a"
    fig.suptitle(f"{spec.title}\n{note}; {what} with 95 % item-bootstrap CIs", fontsize=10.5)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, facecolor="white")
    plt.close(fig)
    csv = out_png.with_suffix(".csv")
    front = ["panel", "metric", "series", "x_field", "xval"]
    table = pd.concat(records, ignore_index=True) if records else pd.DataFrame(columns=front)
    table = table[front + [c for c in table.columns if c not in front]].rename(columns={"xval": "x"})
    table.to_csv(csv, index=False)
    return out_png, csv
