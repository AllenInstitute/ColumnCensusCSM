import warnings

warnings.simplefilter("ignore")
from meshparty import meshwork
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
import pickle
from loguru import logger
import click
import statsmodels.api as sm
from statsmodels.stats import contingency_tables as cont
from scipy import stats

from common_setup import project_info, project_paths
from affine_transform import minnie_column_transform, minnie_column_transform_nm

tform = minnie_column_transform()
tform_nm = minnie_column_transform_nm()

from caveclient import CAVEclient
import sys

sys.path.append("../cards/")
from src import synapse_budget
from src import card_plotting_code


import matplotlib as mpl

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["font.sans-serif"] = "Helvetica"
mpl.rcParams.update({"pdf.use14corefonts": True})

axon_color = [0.214, 0.495, 0.721]
dendrite_color = [0.894, 0.103, 0.108]

import dotenv
import card_config

visConfig = card_config.VisualizationConfig()


def baseline_data_filename(process_dir, configuration_name, version):
    return f"{process_dir}/{configuration_name}_v{version}_files.pkl"


def morpho_plot(
    nrn,
    layer_bnds,
    height_bounds,
    width_bounds,
    axon_color=axon_color,
    dendrite_color=dendrite_color,
    linewidth=0.5,
    soma_size=25,
    colorize=True,
    bar_start=None,
    bar_length=100,
    text_voffset=10,
    soma=True,
    ax=None,
):

    if colorize == False:
        axon_color = "k"
        dendrite_color = "k"
        axon_pts = tform_nm.apply(nrn.skeleton.vertices)
        dend_pts = np.atleast_2d([])
    else:
        axon_pts = tform_nm.apply(
            nrn.skeleton.vertices[nrn.anno.is_axon.mesh_index.to_skel_mask]
        )
        dend_pts = tform_nm.apply(
            nrn.skeleton.vertices[~nrn.anno.is_axon.mesh_index.to_skel_mask]
        )

    if ax is None:
        ax = plt.gca()
    ax.scatter(
        axon_pts[:, 0],
        axon_pts[:, 1],
        marker=".",
        s=linewidth,
        color=axon_color,
    )
    ax.scatter(
        dend_pts[:, 0],
        dend_pts[:, 1],
        marker=".",
        s=linewidth,
        color=dendrite_color,
    )
    if soma:
        soma_pt = tform_nm.apply(nrn.skeleton.vertices[nrn.skeleton.root]).squeeze()
        ax.scatter(
            soma_pt[0],
            soma_pt[1],
            s=soma_size,
            color="w",
            edgecolor="k",
            linewidth=0.5,
            alpha=1,
        )

    card_plotting_code.plot_layers(
        layer_bnds,
        height_bounds,
        width_bounds,
        color=(0.5, 0.5, 0.5),
        linewidth=0.5,
        linestyle=":",
        ax=ax,
    )

    if bar_start is None:
        bar_start = (layer_bnds[-1] + 50, width_bounds[0])
    if bar_length > 0:
        ax.hlines(
            bar_start[0],
            bar_start[1],
            bar_start[1] + bar_length,
            linewidth=5,
            color="k",
        )
        ax.text(
            bar_start[1] + bar_length / 2,
            bar_start[0] + text_voffset,
            f"{bar_length} $\mu$m",
            color="k",
            ha="center",
            va="top",
            fontsize=8,
        )

    ax.set_aspect("equal")
    sns.despine(ax=ax, offset=5, bottom=True)
    ax.set_xticks([])
    return ax


def extract_baseline_data(bd_filename):
    with open(bd_filename, "rb") as f:
        baseline_data = pickle.load(f)

    cell_type_df = baseline_data.get("cell_type_df")
    layer_bins = baseline_data.get("layer_bins")
    syn_profile_bin_comp_df = baseline_data.get("syn_profile_bin_comp_df")
    syn_profile_count_comp_long = baseline_data.get("syn_profile_count_comp_long")
    syn_profile_comp_count = baseline_data.get("syn_profile_comp_count")
    layerConfig = baseline_data.get("layerConfig")
    catConfig = baseline_data.get("catConfig")
    return (
        cell_type_df,
        layer_bins,
        syn_profile_bin_comp_df,
        syn_profile_count_comp_long,
        syn_profile_comp_count,
        layerConfig,
        catConfig,
    )


def process_presyn_df(nrn, syn_profile_bin_comp_df, catConfig):

    pre_syn_df = nrn.anno["pre_syn"].df

    keep_cols = [
        "id",
        "dist_to_root",
        "syn_depth_um",
        "soma_depth_um",
        "is_apical",
        "is_soma",
        "is_dendrite",
        "comp",
        "cell_type",
        "cell_type_comp",
        "layer",
        "layer_bin",
        "valence",
    ]
    # Add cell type and component information
    pre_syn_df = pre_syn_df.merge(
        syn_profile_bin_comp_df[keep_cols], how="inner", on="id"
    )
    pre_syn_df["cell_type"] = pd.Categorical(
        pre_syn_df["cell_type"], categories=catConfig.order, ordered=True
    )

    # Add target soma depth
    pre_syn_df["comp"] = pd.Categorical(
        pre_syn_df["comp"],
        categories=catConfig.compartments[::-1],
        ordered=True,
    )

    return pre_syn_df


def get_soma_id(oid, soma_df):
    try:
        return soma_df.query("pt_root_id == @oid")["id"].values[0]
    except:
        return None


def get_celltype(oid, cell_type_df):
    nrn_row = cell_type_df.query("pt_root_id == @oid")
    if len(nrn_row) == 1:
        return nrn_row["cell_type"].iloc[0]
    else:
        return None


def _transform_baseline_syn_count(syn_profile_count, bin_column):
    syn_profile_count_long = syn_profile_count.melt(
        value_name="num_syn", ignore_index=False
    ).reset_index()
    syn_profile_count_long = syn_profile_count_long.merge(
        syn_profile_count_long.groupby(bin_column).sum()["num_syn"].rename("bin_total"),
        left_on=bin_column,
        right_index=True,
    )

    syn_profile_count_long["bin_other"] = (
        syn_profile_count_long["bin_total"] - syn_profile_count_long["num_syn"]
    )
    syn_profile_count_long["p"] = (
        syn_profile_count_long["num_syn"] / syn_profile_count_long["bin_total"]
    )
    return syn_profile_count_long


def baseline_counts(syn_df, bin_column, cell_type_column, order):
    syn_profile_count = get_syn_profile_count(
        syn_df,
        bin_column,
        cell_type_column,
        order=order,
    )
    syn_profile_count_long = _transform_baseline_syn_count(
        syn_profile_count, bin_column
    )
    return syn_profile_count, syn_profile_count_long


def get_syn_profile_count(syn_profile_df, bin_column, cell_type_column, order=None):
    syn_profile_count = (
        syn_profile_df.groupby([bin_column, cell_type_column])
        .count()[["id"]]
        .rename(columns={"id": "syn_count"})
        .reset_index()
    )

    syn_profile_count = syn_profile_count.pivot_table(
        index=bin_column, columns=cell_type_column, values="syn_count", fill_value=0
    )

    if order is not None:
        drop_cols = syn_profile_count.columns.values
        drop_cols = drop_cols[~np.isin(drop_cols, order)]
        if len(drop_cols) > 0:
            syn_profile_count = syn_profile_count.drop(columns=drop_cols)

    return syn_profile_count


def full_synapse_data(syn_df_ct, cell_type_column, profile_df_long, order=None):
    syn_df_bin_count, syn_df_bin_count_long = baseline_counts(
        syn_df_ct, "layer_bin", cell_type_column, order
    )

    if len(syn_df_bin_count_long) == 0:
        df = profile_df_long.copy()
        df = df.rename(
            columns={
                "num_syn": "num_syn_overall",
                "bin_total": "bin_total_overall",
                "bin_overall": "bin_other_overall",
                "p": "p_overall",
            }
        )
        df["p"] = pd.NA
        df["num_syn"] = 0
        df["bin_total"] = 0
        df["bin_overall"] = 0
        df["sig"] = False
        df["category"] = "base"
        return df

    syn_df_bin_count_long = baseline_context(
        syn_df_bin_count_long, profile_df_long, cell_type_column
    )
    syn_df_bin_count_long = add_stats(syn_df_bin_count_long)
    return syn_df_bin_count_long


def baseline_context(syn_df_long, baseline_long, cell_type_column):
    bin_lookup = {
        row["layer_bin"]: row["bin_total"]
        for _, row in syn_df_long.drop_duplicates("layer_bin").iterrows()
    }
    syn_df_long = syn_df_long.merge(
        baseline_long,
        on=["layer_bin", cell_type_column],
        how="right",
        suffixes=(None, "_overall"),
    )
    syn_df_long["bin_total"] = syn_df_long["layer_bin"].apply(
        lambda x: bin_lookup.get(x, 0)
    )
    syn_df_long["num_syn"] = syn_df_long["num_syn"].fillna(0)
    syn_df_long["bin_other"] = syn_df_long["bin_total"] - syn_df_long["num_syn"]
    syn_df_long["p"] = syn_df_long["num_syn"] / syn_df_long["bin_total"]
    syn_df_long["num_syn"] = syn_df_long["num_syn"].astype(int)
    syn_df_long["bin_other"] = syn_df_long["bin_other"].astype(int)
    return syn_df_long


def fisher_exact_rows(row):
    if row["bin_total"] == 0:
        return np.nan
    oddsratio, p_value = stats.fisher_exact(
        [
            [row["num_syn"], row["bin_other"]],
            [row["num_syn_overall"], row["bin_other_overall"]],
        ]
    )
    return p_value


def binom_test_ci(row):
    if row["bin_total"] == 0:
        return np.nan, np.nan
    lb, ub = sm.stats.proportion_confint(
        row["num_syn"], row["bin_total"], method="agresti_coull"
    )
    return lb, ub


def _categorize_sig(row):
    if not row["sig"]:
        return "base"
    elif row["p"] < row["p_overall"]:
        return "low"
    else:
        return "high"


def add_stats(syn_bin_long, alpha=0.01):
    lbub = np.vstack(syn_bin_long.apply(binom_test_ci, axis=1))
    pval = syn_bin_long.apply(fisher_exact_rows, axis=1)
    syn_bin_long["lci"] = lbub[:, 0].squeeze()
    syn_bin_long["uci"] = lbub[:, 1].squeeze()
    sig, pval_adj, _, _ = sm.stats.multipletests(pval, method="holm-sidak")
    syn_bin_long["p_value"] = pval_adj
    syn_bin_long["sig"] = sig
    syn_bin_long["category"] = syn_bin_long.apply(_categorize_sig, axis=1)
    return syn_bin_long


def create_binning(df, min_max, bin_spacing, column="dist_to_root"):
    bins = np.arange(-1, min_max + 2 * bin_spacing + 1, bin_spacing)
    return bins
    # actual_max = df[column].max()
    # bins = np.arange(-1, np.max((actual_max, min_max)), bin_spacing)
    # return np.append(bins, np.inf)


def target_dist_from_soma(
    df, color, bins=np.arange(-1, 750, 20), max_dist=750, offset=5, ax=None
):

    df = df.copy()
    df["dist_to_root"] = np.where(
        df["dist_to_root"] < max_dist, df["dist_to_root"], max_dist
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 2), facecolor="w")
    ax2 = ax.twinx()

    ax2.set_yticks([], minor=False)
    ax2.set_yticks(np.arange(0, 1.01, 0.25), minor=True)
    ax2.grid(True, axis="y", linestyle=":", which="both")

    sns.histplot(
        x="dist_to_root",
        cumulative=True,
        stat="density",
        element="step",
        fill=True,
        alpha=0.1,
        data=df,
        color=(0.4, 0.4, 0.4),
        ax=ax2,
        bins=np.arange(-1, 750, 1),
    )

    sns.histplot(
        x="dist_to_root",
        stat="count",
        element="step",
        fill=False,
        data=df,
        color=color,
        line_kws={"linewidth": 4},
        ax=ax,
        alpha=1,
        bins=bins,
    )

    ax2.plot(
        np.median(df["dist_to_root"]) * np.ones(2),
        [0, 1],
        color="k",
        linestyle=":",
        linewidth=1,
    )
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("")
    ax2.xaxis.set_visible(False)

    ax.set_ylabel("# Syn", color=color)
    ax.tick_params(axis="y", labelcolor=color)
    sns.despine(ax=ax2, left=True, right=True, top=True, offset=offset)
    sns.despine(ax=ax, right=True, bottom=False, offset=offset)
    ax.set_xticks(np.arange(0, max(bins[~np.isinf(bins)]), 50), minor=True)
    _ = ax.set_xlabel("Dist. to soma ($\mu m$)")
    ax.set_xlim(-5, max_dist + visConfig.dist_bin_spacing)
    return ax, ax2


def compartment_target_distance_plot(
    df,
    palettes,
    layer_bounds,
    height_bounds,
    size=10,
    alpha=0.5,
    markers=("o", "d", "X"),
    ax=None,
    max_dist=None,
    soma_depth=None,
):
    if ax is None:
        ax = plt.gca()
    if max_dist is None:
        max_dist = np.max(df["dist_to_root"])

    palette, palette_apical, palette_soma = palettes
    marker, marker_apical, marker_soma = markers

    df = df.copy()
    df["dist_to_root"] = np.where(
        df["dist_to_root"] < max_dist, df["dist_to_root"], max_dist
    )

    ax = sns.scatterplot(
        y="soma_depth_um",
        x="dist_to_root",
        hue="valence",
        edgecolor=None,
        palette=palette,
        data=df.query("not is_apical and not is_soma"),
        marker=marker,
        s=size,
        alpha=alpha,
        ax=ax,
        rasterized=True,
    )

    ax = sns.scatterplot(
        x="dist_to_root",
        y="soma_depth_um",
        hue="valence",
        edgecolor=None,
        palette=palette_apical,
        data=df.query("is_apical"),
        marker=marker_apical,
        s=size,
        alpha=alpha,
        ax=ax,
        rasterized=True,
    )

    ax = sns.scatterplot(
        x="dist_to_root",
        y="soma_depth_um",
        hue="valence",
        edgecolor=None,
        palette=palette_soma,
        data=df.query("is_soma"),
        marker=marker_soma,
        s=size,
        alpha=alpha,
        ax=ax,
        rasterized=True,
    )
    if soma_depth is not None:
        ax.plot(
            [0, max_dist],
            [soma_depth, soma_depth],
            linestyle="--",
            marker=None,
            color="k",
            linewidth=2,
            alpha=0.5,
        )

    layer_x, layer_y = card_plotting_code.generate_layer_data(
        layer_bounds, (-5, max_dist + 5), axis="x"
    )
    layer_ticks = card_plotting_code.generate_layer_ticks(layer_bounds, height_bounds)
    layer_tick_labels = ["L1", "L2/3", "L4", "L5", "L6", "WM"]

    ax.plot(
        layer_x,
        layer_y,
        color=(0.6, 0.6, 0.6),
        linewidth=1,
        linestyle="-",
        alpha=0.3,
        zorder=-10,
    )

    ax.set_xlabel("Target distance to root")

    ax.set_yticks(layer_ticks)
    ax.set_yticklabels(layer_tick_labels)
    ax.set_ylabel("Target Soma Depth")

    ax.set_xlim(-5, max_dist + visConfig.dist_bin_spacing)
    ax.set_ylim(*height_bounds)

    ax.invert_yaxis()
    ax.legend().remove()
    #     ax.set_aspect('equal')
    ax.xaxis.set_label_position("top")
    sns.despine(ax=ax, offset=5, top=False, bottom=True)
    return ax


def synapse_location_plot(
    df,
    palette,
    layer_bounds,
    height_bounds,
    size=10,
    alpha=0.5,
    marker="o",
    ax=None,
    soma_depth=None,
):
    if ax is None:
        ax = plt.gca()
    ax = sns.scatterplot(
        y="syn_depth_um",
        x="soma_depth_um",
        hue="valence",
        edgecolor="k",
        palette=palette,
        data=df,
        marker=marker,
        s=size,
        alpha=alpha,
        ax=ax,
    )

    ax.plot(height_bounds, height_bounds, color="k", linestyle=":", linewidth=0.5)

    if soma_depth is not None:
        ax.plot(
            [soma_depth],
            [soma_depth],
            "wo",
            markersize=5,
            markeredgecolor="k",
            linewidth=0.5,
            alpha=1,
        )

    layer_x, layer_y = card_plotting_code.generate_layer_data(
        layer_bounds, height_bounds, axis="x"
    )
    layer_ticks = card_plotting_code.generate_layer_ticks(layer_bounds, height_bounds)
    layer_tick_labels = ["L1", "L2/3", "L4", "L5", "L6", "WM"]

    ax.plot(
        layer_x, layer_y, color="k", linewidth=0.5, linestyle="-", alpha=0.5, zorder=0
    )
    ax.plot(
        layer_y, layer_x, color="k", linewidth=0.5, linestyle="-", alpha=0.5, zorder=0
    )

    ax.set_xticks(layer_ticks)
    ax.set_xticklabels(layer_tick_labels)
    ax.set_xlabel("Target Soma Depth")

    ax.set_yticks(layer_ticks)
    ax.set_yticklabels(layer_tick_labels)
    ax.set_ylabel("Synapse Depth")

    ax.set_xlim(*height_bounds)
    ax.set_ylim(*height_bounds)

    ax.invert_yaxis()
    ax.legend().remove()
    ax.set_aspect("equal")
    ax.xaxis.set_label_position("top")
    sns.despine(ax=ax, offset=5, top=False, bottom=True)
    return ax


def output_count_bar(
    syn_df,
    num_rows=None,
    labels=None,
    cell_type="cell_type",
    hue="comp",
    hue_order=None,
    min_xmax=10,
    palette=None,
    first_category=None,
    grid=True,
    ax=None,
):
    if ax is None:
        ax = plt.gca()

    sns.histplot(
        y=cell_type,
        data=syn_df,
        hue=hue,
        multiple="stack",
        hue_order=hue_order,
        palette=palette,
        ax=ax,
        linewidth=0,
        shrink=0.5,
        legend=False,
        zorder=10,
        alpha=1,
    )

    if grid:
        ax.grid(axis="x", zorder=-10)

    if num_rows is not None:
        if first_category is not None:
            offset = (
                np.flatnonzero(syn_df[cell_type].cat.categories == first_category) - 1
            )
            ax.set_ylim(offset + num_rows + 0.5, offset + 0.5)
        else:
            ax.set_ylim(num_rows - 0.5, -0.5)

    sns.despine(ax=ax, top=False, bottom=True)

    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Synapse Count")
    if labels is not None:
        ax.set_yticklabels(labels)

    if ax.get_xlim()[1] < min_xmax:
        ax.set_xlim(0, min_xmax)

    ax.set_ylabel(None)
    return ax


def target_summary_plot(
    df,
    res_df,
    fig,
    gridspec,
    groups,
    num_rows,
    palette,
    cell_type="cell_type",
    component="comp",
    cell_type_component="cell_type_comp",
    first_category=None,
    width_ratio=4,
    preference_width=1.0,
    wspace=0.3,
    min_xmax=10,
):
    g = gridspec.subgridspec(
        nrows=num_rows,
        ncols=3,
        wspace=wspace,
        width_ratios=(1, width_ratio, width_ratio * preference_width),
    )
    ax_bar = fig.add_subplot(g[:, 1])

    if len(df) == 0:
        return

    output_count_bar(
        df.dropna(axis=0),
        num_rows=num_rows,
        labels=None,
        cell_type=cell_type,
        hue=component,
        min_xmax=min_xmax,
        palette=palette,
        first_category=first_category,
        ax=ax_bar,
    )

    axes_spec = []
    for col in range(num_rows):
        axes_spec.append(fig.add_subplot(g[col, 2]))
    card_plotting_code.output_preference_index(
        res_df,
        groups,
        axes=axes_spec,
        extra_mult=2,
        cell_type_column=cell_type_component,
        palette=visConfig.spec_palette,
    )


def card_figure_v2(
    nrn,
    pre_syn_df,
    soma_depth,
    syn_df_bin_count_long,
    res_df,
    visConfig,
    layerConfig,
    catConfig,
    soma=True,
    title=None,
    oid=None,
):
    if oid is None:
        oid = nrn.seg_id

    fig = plt.figure(figsize=(33, 5), dpi=72, facecolor="w")

    gmain = fig.add_gridspec(1, 6, wspace=0.05, width_ratios=(6, 1, 5, 4, 4, 4))

    ax_morpho = fig.add_subplot(gmain[0, 0])
    morpho_plot(
        nrn,
        layerConfig.layer_bounds,
        layerConfig.height_bounds,
        layerConfig.width_bounds,
        linewidth=1,
        bar_length=200,
        axon_color=visConfig.axon_color,
        dendrite_color=visConfig.dendrite_color,
        ax=ax_morpho,
        soma=soma,
    )
    ax_morpho.set_title(title)

    ax_syn_dens = fig.add_subplot(gmain[0, 1])

    syn_df = pd.concat([nrn.anno.pre_syn.df, nrn.anno.post_syn.df], ignore_index=True)
    syn_df["syn_depth"] = tform.apply_project("y", syn_df["ctr_pt_position"])
    card_plotting_code.synapse_depth_distribution_plot(
        oid,
        syn_df,
        layerConfig.height_bounds,
        layerConfig.layer_bounds,
        soma_depth=soma_depth,
        bw=visConfig.depth_bandwidth,
        post_color=visConfig.postsyn_color,
        pre_color=visConfig.presyn_color,
        ax=ax_syn_dens,
    )
    ax_syn_dens.set_ylabel(None)

    ax_syns = fig.add_subplot(gmain[0, 2])
    synapse_location_plot(
        pre_syn_df,
        visConfig.valence_palette,
        layerConfig.layer_bounds,
        layerConfig.height_bounds,
        size=10,
        alpha=0.6,
        marker=visConfig.markers[0],
        soma_depth=soma_depth,
        ax=ax_syns,
    )

    g1 = gmain[0, 3]
    g = g1.subgridspec(2, 2, wspace=0.4, width_ratios=(1, 1), height_ratios=(4, 1))
    ax_e_tall = fig.add_subplot(g[0, 0])

    if len(pre_syn_df) > 0:
        bins = create_binning(
            pre_syn_df, visConfig.min_dist_bin_max, visConfig.dist_bin_spacing
        )
    else:
        bins = np.array([visConfig.min_dist_bin_max])

    compartment_target_distance_plot(
        pre_syn_df.query('valence == "Exc"'),
        (visConfig.valence_palette, visConfig.apical_palette, visConfig.soma_palette),
        layerConfig.layer_bounds,
        layerConfig.height_bounds,
        size=15,
        alpha=0.5,
        markers=visConfig.markers,
        ax=ax_e_tall,
        max_dist=visConfig.min_dist_bin_max,
        soma_depth=soma_depth,
    )

    ax_e_hist = fig.add_subplot(g[1, 0], sharex=ax_e_tall)
    target_dist_from_soma(
        pre_syn_df.query('valence =="Exc"'),
        color=visConfig.e_color,
        max_dist=visConfig.min_dist_bin_max,
        ax=ax_e_hist,
        bins=bins,
    )

    ax_i_tall = fig.add_subplot(g[0, 1], sharey=ax_e_tall)
    compartment_target_distance_plot(
        pre_syn_df.query('valence == "Inh"'),
        (visConfig.valence_palette, visConfig.apical_palette, visConfig.soma_palette),
        layerConfig.layer_bounds,
        layerConfig.height_bounds,
        size=15,
        alpha=0.5,
        markers=visConfig.markers,
        ax=ax_i_tall,
        max_dist=np.max(bins[~np.isinf(bins)]),
        soma_depth=soma_depth,
    )

    ax_i_hist = fig.add_subplot(g[1, 1], sharex=ax_i_tall)
    target_dist_from_soma(
        pre_syn_df.query('valence=="Inh"'),
        color=visConfig.i_color,
        max_dist=visConfig.min_dist_bin_max,
        ax=ax_i_hist,
        bins=bins,
    )

    g_excbar = gmain[0, 4]
    num_rows = max([len(catConfig.e_groups), len(catConfig.i_groups)])
    target_summary_plot(
        pre_syn_df.query('valence == "Exc"'),
        res_df,
        fig,
        g_excbar,
        catConfig.e_groups,
        num_rows,
        visConfig.e_component_palette,
        width_ratio=3,
        preference_width=0.75,
    )

    g_inhbar = gmain[0, 5]
    target_summary_plot(
        pre_syn_df.query('valence == "Inh"'),
        res_df,
        fig,
        g_inhbar,
        catConfig.i_groups,
        num_rows,
        visConfig.i_component_palette,
        first_category=catConfig.i_groups[0],
        width_ratio=3,
        preference_width=0.75,
    )
    return fig


def preprocess_neuron(
    oid,
    cell_type_df,
    soma_df,
    layer_bins,
    target_df,
    syn_profile_count_comp_long,
    catConfig,
    skeleton_folder=None,
    nrn=None,
    soma=True,
    nrn_soma_id=None,
    nrn_subtype=None,
):

    if nrn is None:
        nrn = meshwork.load_meshwork(f"{skeleton_folder}/{oid}.h5")

    pre_syn_df = process_presyn_df(nrn, target_df, catConfig)

    if soma:
        soma_position = nrn.skeleton.vertices[nrn.skeleton.root]
        soma_depth = soma_position[1] / 1000
    else:
        soma_position = None
        soma_depth = None

    if nrn_soma_id == None:
        nrn_soma_id = get_soma_id(oid, soma_df)

    if nrn_subtype == None:
        nrn_subtype = get_celltype(oid, cell_type_df)

    syn_df_bin_count_long = full_synapse_data(
        pre_syn_df,
        "cell_type_comp",
        syn_profile_count_comp_long,
        order=catConfig.order_comp,
    )

    res_df = synapse_budget.stratified_tables(
        syn_df_bin_count_long, "cell_type_comp", order=catConfig.order_comp
    )

    return (
        nrn,
        pre_syn_df,
        soma_depth,
        nrn_soma_id,
        nrn_subtype,
        syn_df_bin_count_long,
        res_df,
    )


def get_card_oids(params, client, timestamp=None):
    column_table = params.get("COLUMN_TABLE", project_info.column_table)
    column_df = client.materialize.query_table(
        column_table,
        filter_in_dict={
            "classification_system": [
                "aibs_coarse_inhibitory",
                "aibs_coarse_excitatory",
            ]
        },
        timestamp=timestamp,
    )
    oids = column_df.query('classification_system == "aibs_coarse_inhibitory"')[
        "pt_root_id"
    ].values
    return oids


@click.command()
@click.option("--environment", "-e")
def generate_cards(environment):
    params = dotenv.dotenv_values(f"{environment}.env")
    version = params.get("MAT_VERSION")

    bd_filename = baseline_data_filename(
        f"{project_paths.data}/temp", environment, version
    )
    card_dir = f"{project_paths.plots}/cards_v2/{environment}/v{version}"
    try:
        os.makedirs(card_dir)
    except:
        pass

    specificity_data_dir = (
        f"{project_paths.data}/specificity_data/{environment}/v{version}"
    )
    try:
        os.makedirs(specificity_data_dir)
    except:
        pass

    client = CAVEclient(project_info.datastack)
    # client.materialize.version = version
    timestamp = client.materialize.get_timestamp(version=version)
    print(f"Making cards for v{client.materialize.version}")
    logger.add("card_log.log")

    (
        cell_type_df,
        layer_bins,
        target_df,
        syn_profile_count_comp_long,
        syn_profile_comp_count,
        layerConfig,
        catConfig,
    ) = extract_baseline_data(bd_filename)

    soma_df = client.materialize.query_table(
        project_info.soma_table, filter_equal_dict={"cell_type": "neuron"}, timestamp=timestamp
    )

    inhib_oids = get_card_oids(params, client, timestamp=timestamp)
    for oid in tqdm.tqdm(inhib_oids):
        logger.info(f"Working on {oid}")
        try:
            (
                nrn,
                pre_syn_df,
                soma_depth,
                nrn_soma_id,
                nrn_subtype,
                syn_df_bin_count_long,
                res_df,
            ) = preprocess_neuron(
                oid,
                cell_type_df,
                soma_df,
                layerConfig.layer_bins,
                target_df,
                syn_profile_count_comp_long,
                catConfig,
                f"{project_paths.skeletons}/skeleton_files",
            )

            title = f"{nrn_subtype}$\sim${nrn_soma_id}$\sim${oid}"

            fig = card_figure_v2(
                nrn,
                pre_syn_df,
                soma_depth,
                syn_df_bin_count_long,
                res_df,
                visConfig,
                layerConfig,
                catConfig,
                title=title,
            )
            logger.info(f"Saved preview file for {oid}")
            fig.savefig(
                f"{card_dir}/{nrn_subtype}_id{nrn_soma_id}_{oid}.pdf",
                bbox_inches="tight",
            )
            res_df.reset_index(drop=True).to_feather(
                f"{specificity_data_dir}/res_{oid}.feather"
            )
            syn_df_bin_count_long.reset_index(drop=True).to_feather(
                f"{specificity_data_dir}/bin_count_{oid}.feather"
            )
            plt.close(fig)
        except Exception as e:
            logger.debug(f"{oid} | {e}")
    pass


if __name__ == "__main__":
    generate_cards()
