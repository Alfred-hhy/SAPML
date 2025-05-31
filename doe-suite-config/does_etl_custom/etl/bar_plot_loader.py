import typing
from math import ceil
from enum import Enum
import warnings
import itertools
import logging

import math
import pandas as pd
import os
from typing import Dict, List, Union, Optional

from doespy.design.etl_design import MyETLBaseModel
from doespy.etl.etl_util import expand_factors, escape_tuple_str
from doespy.etl.steps.transformers import Transformer
from doespy.etl.steps.loaders import Loader, PlotLoader

from does_etl_custom.etl.config import setup_plt

import matplotlib.pyplot as plt
import numpy as np

from typing import Tuple, Literal, Any

logger = logging.getLogger(__name__)

class MetricConfig(MyETLBaseModel):

    bar_part_cols: List[str]
    y_label: str
    y_label_pad: float = None


    log_y: bool = False

    log_x: bool = False


    y_unit_multiplicator: float = 1.0
    y_unit_divider: float = 1.0

    y_max: float = None

    y_ticks: List[float] = None


    x_lim: Tuple[float, float] = None
    y_lim: Tuple[float, float] = None
    y_lim_dict: List[Dict] = None

    y_lim_row: List[Tuple[float, float]] = None
    y_ticks_row: List[List[float]] = None

    n_y_ticks: int = None


    plot_cols_filter: Dict[str, List[str]] = None

    bar_cols_filter: Dict[str, List[str]] = None
    bar_cols_filter_last_row: Dict[str, List[str]] = None
    bar_cols_filter_last_row_num: int = 2

    bar_pos_bias: float = 0.0
    bar_pos_bias_last: float = 0.0



class LegendConfig(MyETLBaseModel):
    format: str
    cols: List[str]

    kwargs: Dict[str, Any] = None


class AxLegendConfig(LegendConfig):

    filter: Dict[str, List[str]] = None


def get_label_cols_and_format(full_id, legend_configs: List[AxLegendConfig]):

    for cfg in legend_configs:
        is_match = True
        for col, values in cfg.filter.items():
            if col not in full_id:
                raise ValueError(f"Error in legend_ax, the filter column: {col} is not avaialable ({full_id.keys()})")
            elif full_id[col] not in values:
                is_match = False
                break

        if is_match:
            return cfg.cols, cfg.format

    raise ValueError(f"Error in legend_ax, no match found for {full_id}")


def style_ax_legend(ax, plot_id, legend_configs: List[AxLegendConfig]):


    for cfg in legend_configs:

        if cfg.kwargs is None:
            continue

        is_match = True
        for col, values in cfg.filter.items():
            if col not in plot_id:
                raise ValueError(f"Error in legend_ax, the filter column: {col} is not avaialable ({full_id.keys()})")
            elif plot_id[col] not in values:
                is_match = False
                break

        if is_match:
            handles, labels = [], []
            for handle, label in zip(*ax.get_legend_handles_labels()):
                handles.append(handle)
                labels.append(label)

            unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]

            ax.legend(*zip(*unique), **cfg.kwargs)

            return

    raise ValueError(f"Error in legend_ax, no match found for {plot_id}")



class FigLegendConfig(LegendConfig):

    subplot_idx: Tuple[int, int] = None


    def style_fig_legend(self, fig, axs):

        handles, labels = [], []
        for ax in axs.flat:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                handles.append(handle)
                labels.append(label)

        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]

        container = fig if self.subplot_idx is None else axs[self.subplot_idx[0], self.subplot_idx[1]]
        container.legend(*zip(*unique), **self.kwargs)



class TitleConfig(MyETLBaseModel):
    format: str
    plot_cols: List[str]

class TickLabelConfig(MyETLBaseModel):
    format: str
    cols: List[str]

class SubplotConfig(MyETLBaseModel):
    rows: List[str]
    cols: List[str]

    share_x: Literal['none', 'all', 'row', 'col'] = 'none'
    share_y: Literal['none', 'all', 'row', 'col'] = 'row'



class BarStyle(MyETLBaseModel):

    style: Dict[str, typing.Any]
    filter: Dict[str, List[str]]


def get_bar_style(bar_styles: List[BarStyle], full_id):

    if bar_styles is None:
        return {}

    config = {}

    for style in bar_styles:

        is_match = True
        for col, values in style.filter.items():
            if col not in full_id:
                raise ValueError(f"Error in BarStyle, the filter column: {col} is not avaialable ({full_id.keys()})")
            elif full_id[col] not in values:
                is_match = False
                break

        if is_match:
            for k, v in style.style.items():
                if k not in config:
                    config[k] = v

    return config


from matplotlib.ticker import FuncFormatter

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42



def format_axis_label(value, _):

    def format(value):

        if abs(value) < 0.001:
            formatted_number = f'{value:.4f}'
        elif abs(value) < 0.01:
            formatted_number = f'{value:.3f}'
        elif abs(value) < 0.1:
            formatted_number = f'{value:.2f}'
        else:
            formatted_number = f'{value:.1f}'

        if "." in formatted_number:
            formatted_number = formatted_number.rstrip('0').rstrip('.')

        return formatted_number

    if abs(value) >= 1e12:
        formatted_number = format(value / 1e12)
        formatted_number += "T"
        val = formatted_number
    elif abs(value) >= 1e9:
        formatted_number = format(value / 1e9)
        formatted_number += "B"
        val = formatted_number
    elif abs(value) >= 1e6:
        formatted_number = format(value / 1e6)
        formatted_number += "M"
        val = formatted_number
    elif abs(value) >= 1e3:
        formatted_number = format(value / 1e3)
        formatted_number += "k"
        val = formatted_number
    else:
        formatted_number = format(value)
        val = formatted_number

    if val == "100M":
        val = "0.1B"
    return val

class BarPlotLoader(PlotLoader):

    metrics: Dict[str, MetricConfig]

    subplots: SubplotConfig = None

    plot_cols: List[str]
    group_cols: List[str]

    bar_cols: List[str]

    cols_values_filter: Dict[str, List[str]]

    labels: Dict[str, str]

    legend_fig: FigLegendConfig = None
    legend_ax: List[AxLegendConfig] = None

    title: TitleConfig = None

    x_axis_label: str = None

    group_labels: TickLabelConfig = None


    bar_styles: List[BarStyle] = None

    figure_size: List[float] = [2.5, 2.5]

    bar_width: float = 1.2 # 0.6

    show_debug_info: bool = False

    verbose: bool = False

    def load(self, df: pd.DataFrame, options: Dict, etl_info: Dict) -> None:
        if df.empty:
            return

        plt.set_loglevel("warning")

        plt.close('all')

        output_dir = self.get_output_dir(etl_info)

        df_filtered = self.filter_df(df)

        if self.subplots is not None:
            subplots = Subplots()
            fig, axs = subplots.init_subplots(df=df_filtered, loader=self)
        else:
            fig, axs = plt.subplots(1, 1, figsize=self.figure_size)

        for metric_name, metric_cfg in self.metrics.items():

            logger.info(f"Creating metric {metric_name} plot...")

            for idx_plot, df_plot in df_filtered.groupby(self.plot_cols):

                idx_plot = (idx_plot, ) if not isinstance(idx_plot, tuple) else idx_plot
                plot_id = {k: v for k, v in zip(self.plot_cols, list(idx_plot))}

                logger.info(f"  Creating {plot_id} plot...")

                if self.subplots is None:
                    setup_plt(width=self.figure_size[0], height=self.figure_size[1])

                    fig, ax = plt.subplots(1, 1)
                    subplot_idx = (0, 0)
                else:
                    subplot_idx = subplots.get_subplot_idx(plot_id, metric_name)

                    if subplot_idx is None:
                        continue

                    if len(axs.shape) == 2:
                        ax = axs[subplot_idx[0], subplot_idx[1]]
                    else:
                        ax = axs[subplot_idx[0]]


                df1 = self.aggregate_data(df_plot, metric_cfg)


                group_ticks, bar_ticks = self.plot_bars(ax, subplot_idx, metric_cfg, plot_id, df1)

                if self.legend_ax is not None:
                    style_ax_legend(ax, plot_id, self.legend_ax)

                self.style_y_axis(ax, metric_cfg, subplot_idx, plot_id)

                self.style_x_axis(ax, metric_cfg, subplot_idx, group_ticks, bar_ticks)

                self.debug_info(ax)

                self.style_title(ax, metric_cfg, plot_id, subplot_idx)

                ax.grid(True, axis="y", linestyle=':', color='0.6', zorder=0, linewidth=1.2)

                filename = f"bar_{metric_name}_{escape_tuple_str(idx_plot)}"

                if output_dir is not None:
                    out = os.path.join(output_dir, metric_name)
                    os.makedirs(out, exist_ok=True)

                    self.save_data(df1, filename=filename, output_dir=out)

                    if self.subplots is None:
                        self.save_plot(fig, filename=filename, output_dir=out, use_tight_layout=True, output_filetypes=["pdf"])


        if self.legend_fig is not None and self.subplots is not None:
            self.legend_fig.style_fig_legend(fig, axs)

        fig.subplots_adjust(wspace=0.28)


        if self.subplots is not None and output_dir is not None:
            filename =  f"{etl_info['pipeline']}_debug" if self.show_debug_info else etl_info['pipeline']
            self.save_plot(fig, filename=filename, output_dir=output_dir, use_tight_layout=True, output_filetypes=["pdf"])



    def debug_info(self, ax):

        if self.show_debug_info:
            for p in ax.patches:
                if hasattr(p, 'get_height') and hasattr(p, 'get_width') and hasattr(p, 'get_x'):
                    ax.annotate(f"{p.get_height():0.2f}", (p.get_x() * 1.005 + (p.get_width() / 2), (p.get_y() + p.get_height()) * 1.005), ha='center', va='bottom')

                elif hasattr(p, 'get_x') and hasattr(p, 'get_y'):

                    ax.annotate(f"{p.get_y():0.2f}", (p.get_x() * 1.005, p.get_y() * 1.005), ha='center', va='bottom')



    def filter_df(self, df: pd.DataFrame) -> pd.DataFrame:

        n_rows_intial = len(df)
        df_filtered = df.copy()

        plot_cols = [(col, self.cols_values_filter[col]) for col in self.plot_cols]
        group_cols = [(col, self.cols_values_filter[col]) for col in self.group_cols]
        bar_cols = [(col, self.cols_values_filter[col]) for col in self.bar_cols]

        for col, allowed in plot_cols + group_cols + bar_cols:

            try:
                df_filtered[col] = df_filtered[col].astype(str)
            except KeyError:
                raise KeyError(f"col={col} not in df.columns={df.columns}")

            logger.info(f"Filtering {col} to {allowed}    all={df[col].unique()}")
            df_filtered = df_filtered[df_filtered[col].isin(allowed)]
            df_filtered[col] = pd.Categorical(df_filtered[col], ordered=True, categories=allowed)
        df_filtered.sort_values(by=self.plot_cols + self.group_cols + self.bar_cols, inplace=True)

        logger.info(f"Filtered out {n_rows_intial - len(df_filtered)} rows (based on plot_cols, row_cols, col_cols)  remaining: {len(df_filtered)}")

        return df_filtered


    def aggregate_data(self, df_plot, metric_cfg: MetricConfig):

        df_plot[metric_cfg.bar_part_cols] = df_plot[metric_cfg.bar_part_cols] * metric_cfg.y_unit_multiplicator / metric_cfg.y_unit_divider

        grouped_over_reps = df_plot.groupby(by = self.group_cols + self.bar_cols)
        for group in grouped_over_reps.groups:
            if len(grouped_over_reps.get_group(group)) > 1:
                logger.info(f"Group rows: {grouped_over_reps.get_group(group)}")
                logger.info(f"Const args: {grouped_over_reps.get_group(group)['consistency_args.type']}")


        combined = grouped_over_reps[metric_cfg.bar_part_cols].agg(['mean', 'std'])

        combined[("$total$", "mean")] = combined.loc[:, pd.IndexSlice[metric_cfg.bar_part_cols, "mean"]].sum(axis=1)
        for col in metric_cfg.bar_part_cols:
            combined[(f"$total_share_{col}$", "mean")] = combined[col]["mean"] / combined["$total$"]["mean"]
            combined[(f"$total_factor_{col}$", "mean")] = combined["$total$"]["mean"] / combined[col]["mean"]


        return combined.reset_index()



    def style_y_axis(self, ax, metric_cfg, subplot_idx: Tuple[int, int], plot_id: Dict):



        if metric_cfg.log_y:
            ax.set_yscale('log')

        if  metric_cfg.y_max is not None:
            ax.set_ylim(0, metric_cfg.y_max)
        else:
            if metric_cfg.log_y:
                pass
            else:
                pass

        if metric_cfg.y_lim is not None:
            ax.set_ylim(*metric_cfg.y_lim)

        if metric_cfg.y_lim_dict is not None:
            assert self.subplots is not None, "y_lim_dict only supported for subplots"

            def predicate(x):
                return x['selector'] == plot_id

            lims = next(filter(predicate, metric_cfg.y_lim_dict))
            ax.set_ylim(*lims['value'])

        elif metric_cfg.y_lim_row is not None:

            assert self.subplots is not None, "y_lim_row only supported for subplots"

            col = subplot_idx[1]
            ax.set_ylim(*metric_cfg.y_lim_row[col])


        if metric_cfg.n_y_ticks is not None:
            ymin, ymax = ax.get_ylim()

            ax.set_yticks(np.linspace(ymin, ymax, metric_cfg.n_y_ticks))


        if metric_cfg.y_ticks_row is not None:

            assert self.subplots is not None, "y_lim_row only supported for subplots"

            col = subplot_idx[1]

            ax.set_yticks(metric_cfg.y_ticks_row[col])

        if metric_cfg.y_ticks is not None:
            ax.set_yticks(metric_cfg.y_ticks)



        if self.subplots is None:
            ax.set_ylabel(metric_cfg.y_label)
        elif subplot_idx[1] == 0:


            if metric_cfg.y_label_pad is not None:
                ax.set_ylabel(metric_cfg.y_label, labelpad=metric_cfg.y_label_pad)
            else:
                ax.set_ylabel(metric_cfg.y_label)

        ax.yaxis.set_major_formatter(FuncFormatter(format_axis_label))




    def style_x_axis(self, ax, metric_cfg, subplot_idx: Tuple[int, int], group_ticks, bar_ticks):
        if metric_cfg.log_x:
            ax.set_xscale('log')

        cols, format = (self.group_cols, " ".join([r"{}"] * len(self.group_cols))) if self.group_labels is None else (self.group_labels.cols, self.group_labels.format)

        positions = []
        labels = []

        for pos, d in group_ticks:

            subs = []
            for col in cols:
                if col not in d:
                    raise ValueError(f"Error: Group Label Config:  col={col} not available: {d}")
                subs.append(self.labels.get(d[col], d[col]))


            tick_label = format.format(*subs)

            positions.append(pos)
            labels.append(tick_label)

            ax.set_xticks(positions, labels=labels)



        ax.set_xlabel(self.x_axis_label)

        if metric_cfg.x_lim is not None:
            ax.set_xlim(*metric_cfg.x_lim)


    def style_title(self, ax, metric_cfg, plot_id, subplot_idx):

        if self.title is not None:

            subs = []
            for col in self.title.plot_cols:
                if col not in plot_id:
                    raise ValueError(f"Title Config Error: col={col} not in self.plot_cols={self.plot_cols}")

                lbl = plot_id[col]
                subs.append(self.labels.get(lbl, lbl))
            title = self.title.format.format(*subs)

            if self.subplots is None:
                ax.set_title(title)
            elif subplot_idx[0] == 0:
                ax.set_title(title)


    def get_zoom_info(self, metric_cfg, df_plot, zoom_threshold):
        key = "$total_bar_height$"

        df_plot[key] = df_plot[metric_cfg.bar_part_cols].sum(axis=1)


        group_cols = self.group_cols[0] if len(self.group_cols) == 1 else self.group_cols

        gb = df_plot.groupby(group_cols)
        df_group_height =  gb[key].max()


        df_plot_height = df_group_height.max()


        df_group_height = df_group_height.to_frame('group_height')


        df_group_height["factor_group_plot"] = df_group_height['group_height'] / df_plot_height


        zoom_infos = []
        connect_adjacent = False
        if connect_adjacent:
            temp = []
            for idx_bar_group, v in df_group_height.iterrows():

                if v["factor_group_plot"]  < zoom_threshold:
                    idx_bar_group = (idx_bar_group, ) if not isinstance(idx_bar_group, tuple) else idx_bar_group
                    bar_group_id = {k: v for k, v in zip(self.group_cols, list(idx_bar_group))}
                    temp.append((bar_group_id, v["group_height"]))
                else:
                    logger.info(f"no zoom: {idx_bar_group}     -> {v}")
                    if temp:
                        zoom_infos.append({"bar_group_ids": [x for x, _ in temp], "bar_group_heights": [x for _, x in temp]})
                        temp = []


            if temp:
                zoom_infos.append({"bar_group_ids": [x for x, _ in temp], "bar_group_heights": [x for _, x in temp]})
        else:
            for idx_bar_group, v in df_group_height.iterrows():
                 if v["factor_group_plot"] < zoom_threshold and v["group_height"] > 0.0:
                    idx_bar_group = (idx_bar_group, ) if not isinstance(idx_bar_group, tuple) else idx_bar_group
                    bar_group_id = {k: v for k, v in zip(self.group_cols, list(idx_bar_group))}

                    zoom_infos.append({"bar_group_ids": [bar_group_id], "bar_group_heights": [v["group_height"]]})


        return zoom_infos


    def plot_bars(self, ax, subplot_idx, metric_cfg, plot_id, df1):
        existing_labels = set()

        zoom_threshold = 0.15

        zoom_infos = self.get_zoom_info(metric_cfg=metric_cfg, df_plot=df1, zoom_threshold=zoom_threshold)

        n_bars_per_group, n_bar_groups = self.get_bars_info(df1)

        w = self.bar_width / n_bar_groups

        group_ticks = []
        bar_ticks = []


        group_cols = self.group_cols[0] if len(self.group_cols) == 1 else self.group_cols
        for i_group, (idx_bar_group, df_bar_group) in enumerate(df1.groupby(group_cols)):
            idx_bar_group = (idx_bar_group, ) if not isinstance(idx_bar_group, tuple) else idx_bar_group


            bar_group_id = {k: v for k, v in zip(self.group_cols, list(idx_bar_group))}


            group_pos_center = np.arange(n_bar_groups)[i_group]

            group_pos_left = group_pos_center - (w * n_bars_per_group / 2.)

            group_ticks.append((group_pos_center, {**plot_id, **bar_group_id}))


            bar_cols = self.bar_cols[0] if len(self.bar_cols) == 1 else self.bar_cols

            use_zoom = not metric_cfg.log_y

            zoom = None
            if not use_zoom:
                zoom = None
            else:
                for zinfo in zoom_infos:
                    if bar_group_id == zinfo["bar_group_ids"][0]:
                        zoom_span_n_groups = len(zinfo["bar_group_ids"])


                        zoom_pos_right = np.arange(n_bar_groups)[i_group+zoom_span_n_groups-1]
                        zoom_pos_right += (w * n_bars_per_group / 2.)

                        x_bias = 0.2 * w
                        xlim = (group_pos_left-x_bias, zoom_pos_right+x_bias)

                        ylim = (0, 1.2 * max(zinfo["bar_group_heights"]))


                        x_bound_per_group = {
                            0: 0.15,
                            1: 0.52,
                            2: 0.77,
                        }

                        y_bound = 0.28
                        height_bound = 0.45
                        width_bound = 0.2

                        bias = 0.05
                        bounds=[x_bound_per_group[i_group], y_bound,  zoom_span_n_groups * width_bound + (zoom_span_n_groups-1)*bias, height_bound]


                        inset_cfg = {}
                        inset_cfg["bounds"] = bounds
                        inset_cfg["xlim"] = xlim
                        inset_cfg["ylim"] = ylim
                        inset_cfg["xticks"] = []



                        zinfo["inset_cfg"] = inset_cfg


                        zinfo["y_ticks"] = set()


                        zoom = zinfo
                        break

                    elif bar_group_id in zinfo["bar_group_ids"]:
                        zoom = zinfo
                        break



            for i_bar, (idx_bar, df_bar) in enumerate(df_bar_group.groupby(bar_cols)):
                idx_bar = (idx_bar, ) if not isinstance(idx_bar, tuple) else idx_bar


                bar_id = {k: v for k, v in zip(self.bar_cols, list(idx_bar))}


                if metric_cfg.bar_cols_filter is not None:
                    skip = False
                    for col, allowed in metric_cfg.bar_cols_filter.items():
                        if bar_id[col] not in allowed:
                            skip = True
                    if skip:
                        if self.verbose:
                            logger.info(f"    Skipping bar because {bar_id} does not match metric filter: {metric_cfg.bar_cols_filter}")
                        continue

                if subplot_idx[1] == metric_cfg.bar_cols_filter_last_row_num and metric_cfg.bar_cols_filter_last_row is not None:
                    skip = False

                    for col, allowed in metric_cfg.bar_cols_filter_last_row.items():
                        if bar_id[col] not in allowed:
                            skip = True
                    if skip:
                        if self.verbose:
                            logger.info(f"    Skipping bar because {bar_id} does not match metric filter: {metric_cfg.bar_cols_filter_last_row}")
                        continue

                bar_pos = group_pos_left + (i_bar*w) + (w/2.)

                if subplot_idx[1] == metric_cfg.bar_cols_filter_last_row_num \
                    and metric_cfg.bar_pos_bias == 0.0:
                    bar_pos += metric_cfg.bar_pos_bias_last * w
                else:
                    bar_pos += metric_cfg.bar_pos_bias * w



                bar_ticks.append((bar_pos, {**plot_id, **bar_group_id, **bar_id}))



                bottom = 0

                for bar_part_col in metric_cfg.bar_part_cols:
                    bar_part_id = {"$bar_part_col$": bar_part_col}

                    yerr = df_bar[bar_part_col]["std"].fillna(0)


                    full_id = {**plot_id, **bar_group_id, **bar_id, **bar_part_id}


                    style_config = get_bar_style(self.bar_styles, full_id)


                    label = style_config.pop("label", self.get_label(full_id))



                    ax.bar(bar_pos, df_bar[bar_part_col]["mean"], width=w, label=label, yerr=yerr,  bottom=bottom, **style_config)


                    if zoom is not None:

                        if "axins"  in zoom:
                            axins = zoom["axins"]
                        else:
                            axins = ax.inset_axes(**zoom["inset_cfg"])
                            axins.tick_params(axis='y', labelsize=7, length=1.5, pad=2)
                            axins.yaxis.set_major_formatter(FuncFormatter(format_axis_label))
                            zoom["axins"] = axins

                        axins.bar(bar_pos, df_bar[bar_part_col]["mean"], width=w, label=label, yerr=yerr,  bottom=bottom, **style_config)

                        v = df_bar["$total_bar_height$"].values[0]

                        v = math.ceil(v) if v > 10  else v

                        zoom["y_ticks"].add(v)

                    bottom += df_bar[bar_part_col]["mean"]

        for zinfo in zoom_infos:
            if "axins" in zinfo:
                zinfo["axins"].set_yticks(list(zinfo["y_ticks"]))
            if "axins" in zinfo:
                ax.indicate_inset_zoom(zinfo["axins"])

                zinfo["axins"].grid(True, axis="y", linestyle=':', color='0.6', zorder=0, linewidth=1.2)


        return group_ticks, bar_ticks


    def get_label(self, full_id):


        if self.legend_fig is None and self.legend_ax is None:
            return None

        assert self.legend_fig is None or self.legend_ax is None, "Cannot have both fig and ax legend"

        cols, format = (self.legend_fig.cols, self.legend_fig.format) if self.legend_ax is None else get_label_cols_and_format(full_id, self.legend_ax)

        subs = []
        for col in cols:

            if col not in full_id:
                raise ValueError(f"Legend Config Error: col={col} not available: {full_id}")

            lbl = full_id[col]
            subs.append(self.labels.get(lbl, lbl))

        return format.format(*subs)


    def get_bars_info(self, combined_new):
        n_bars_info = []

        group_cols = self.group_cols[0] if len(self.group_cols) == 1 else self.group_cols
        for _, df_bar_group in combined_new.groupby(group_cols):
            bar_cols = self.bar_cols[0] if len(self.bar_cols) == 1 else self.bar_cols
            n_bars_per_group = df_bar_group.groupby(bar_cols).ngroups
            n_bars_info.append(n_bars_per_group)

        n_bar_groups = len(n_bars_info)
        assert all(x == n_bars_info[0] for x in n_bars_info), "not all bar groups have the same number of bars"
        n_bars_per_group = n_bars_info[0]

        return n_bars_per_group,n_bar_groups




class Subplots:

    def init_subplots(self, df, loader: BarPlotLoader):

        self.loader = loader

        row_keys = set()
        col_keys = set()

        for metric_name, metric_cfg in loader.metrics.items():
            for idx_plot, _ in df.groupby(loader.plot_cols):
                plot_id = {k: v for k, v in zip(loader.plot_cols, list(idx_plot))}

                if metric_cfg.plot_cols_filter is not None:
                    skip = False
                    for col, allowed in metric_cfg.plot_cols_filter.items():
                        if plot_id[col] not in allowed:
                            skip = True
                    if skip:
                        logger.info(f"    Skipping plot because {idx_plot} does not match metric filter: {metric_cfg.plot_cols_filter}")
                        continue

                row_keys.add(self._lookup_key(plot_id, metric_name, relevant_columns=loader.subplots.rows))
                col_keys.add(self._lookup_key(plot_id, metric_name, relevant_columns=loader.subplots.cols))

        def _build_lookup(allowed_keys, relevant_columns):
            lookup = {}
            i = 0
            cross = [loader.metrics.keys() if x == "$metrics$" else loader.cols_values_filter[x] for x in relevant_columns]
            for k in itertools.product(*cross):
                if k in allowed_keys:
                    lookup[k] = i
                    i += 1
            return lookup

        self._row_lookup = _build_lookup(allowed_keys=row_keys, relevant_columns=loader.subplots.rows)
        self._col_lookup = _build_lookup(allowed_keys=col_keys, relevant_columns=loader.subplots.cols)

        nrows = len(self._row_lookup)
        ncols = len(self._col_lookup)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * loader.figure_size[0], nrows * loader.figure_size[1]), sharex=loader.subplots.share_x, sharey=loader.subplots.share_y)
        return fig, axs


    def get_subplot_idx(self, plot_id, metric_name):
        row_key = self._lookup_key(plot_id, metric_name, relevant_columns=self.loader.subplots.rows)
        col_key = self._lookup_key(plot_id, metric_name, relevant_columns=self.loader.subplots.cols)


        if row_key not in self._row_lookup or col_key not in self._col_lookup:
            logger.info(f"Skipping plot because {row_key} or {col_key} not in {self._row_lookup} or {self._col_lookup}")
            return None
        else:
            return (self._row_lookup[row_key], self._col_lookup[col_key])


    def _lookup_key(self, plot_id, metric_name, relevant_columns):
        key = []
        for x in relevant_columns:
            if x == "$metrics$":
                key.append(metric_name)
            else:
                key.append(plot_id[x])
        return tuple(key)
