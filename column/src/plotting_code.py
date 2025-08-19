import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import cloudvolume
import matplotlib.pyplot as plt
from neuron_info import syn_depth
from synapse_budget import *

voxel_resolution = [4, 4, 40]


def target_category_location_plot(
    oid,
    syn_df,
    target_df,
    cell_type_column,
    direction="post",
    hue_column=None,
    palette=None,
    order=None,
    size=2,
    voxel_resolution=[4, 4, 40],
    jitter=1,
    alpha=0.25,
    ax=None,
):
    if ax is None:
        ax = plt.gca()

    if direction == "post":
        df = syn_df.query("pre_pt_root_id == @oid").reset_index(drop=True)
        df["syn_depth"] = (
            df["ctr_pt_position"].apply(lambda x: x[1] * voxel_resolution[1]) / 1000
        )
        df = df.merge(target_df, left_on="post_pt_root_id", right_on="pt_root_id")
    elif direction == "pre":
        df = syn_df.query("post_pt_root_id == @oid").reset_index(drop=True)
        df["syn_depth"] = (
            df["ctr_pt_position"].apply(lambda x: x[1] * voxel_resolution[1]) / 1000
        )
        df = df.merge(target_df, left_on="pre_pt_root_id", right_on="pt_root_id")
    else:
        raise ValueError('Direction must be either "pre" or "post')

    df["syn_depth_delta"] = df["soma_depth_um"] - df["syn_depth"]

    sns.stripplot(
        x=cell_type_column,
        y="syn_depth_delta",
        hue=hue_column,
        palette=palette,
        size=size,
        order=order,
        data=df,
        ax=ax,
        jitter=jitter,
        alpha=alpha,
    )
    ax.grid(b=True, axis="y", which="major")
    ax.legend().remove()
    ax.set_ylabel("Cell type")
    return ax


def synapse_depth_distribution_plot(
    oid,
    syn_df,
    depth_bounds=None,
    layers=None,
    ax=None,
    soma_depth=None,
    bw=0.05,
    title=None,
    voxel_resolution=[4, 4, 40],
):
    if ax is None:
        ax = plt.gca()

    if depth_bounds is None:
        syn_dep = syn_df["ctr_pt_position"].apply(lambda x: x[1])
        depth_bounds = [np.min(syn_dep), np.max(syn_dep)]

    syn_nrn_pre_df = syn_df.query("pre_pt_root_id == @oid").reset_index(drop=True)
    syn_nrn_pre_df["syn_depth"] = (
        syn_nrn_pre_df["ctr_pt_position"].apply(lambda x: x[1] * voxel_resolution[1])
        / 1000
    )

    syn_nrn_post_df = syn_df.query("post_pt_root_id == @oid").reset_index(drop=True)
    syn_nrn_post_df["syn_depth"] = (
        syn_nrn_post_df["ctr_pt_position"].apply(lambda x: x[1] * voxel_resolution[1])
        / 1000
    )

    syn_nrn_pre_df["side"] = "pre"
    syn_nrn_post_df["side"] = "post"

    syn_depth_df = pd.concat(
        [syn_nrn_pre_df[["side", "syn_depth"]], syn_nrn_post_df[["side", "syn_depth"]]]
    )
    syn_depth_df["dummy"] = 1

    if layers is not None:
        layer_x, layer_y = generate_layer_data(layers, [-1, 1], axis="x")
        layer_ticks = generate_layer_ticks(layers, depth_bounds)
        layer_tick_labels = ["L1", "L2/3", "L4", "L5", "L6", "WM"]
        ax.plot(layer_x, layer_y, color="k", linestyle=":", alpha=0.45, zorder=-1)

    syn_plot_order = ["post", "pre"]
    syn_hue_order = [[0.894, 0.103, 0.108], [0.214, 0.495, 0.721]]
    sns.violinplot(
        x="dummy",
        hue="side",
        y="syn_depth",
        hue_order=syn_plot_order,
        palette=syn_hue_order,
        linewidth=0,
        bw=bw,
        split=True,
        data=syn_depth_df,
        ax=ax,
        legend=None,
    )

    ax.set_xlim(-0.45, 0.45)
    ax.set_ylim(*depth_bounds)
    if layers is not None:
        ax.set_yticks(layer_ticks)
        ax.set_yticklabels(layer_tick_labels)

    if soma_depth is not None:
        ax.plot(
            [0],
            [soma_depth],
            color="w",
            marker="o",
            markersize=7,
            alpha=1,
            markeredgecolor="k",
        )

    ax.text(
        -0.225,
        depth_bounds[0] - 10,
        "post",
        color=syn_hue_order[0],
        fontdict={"fontsize": 12, "horizontalalignment": "center"},
    )
    ax.text(
        0.225,
        depth_bounds[0] - 10,
        "pre",
        color=syn_hue_order[1],
        fontdict={"fontsize": 12, "horizontalalignment": "center"},
    )

    ax.text(
        -0.225,
        depth_bounds[0] + 10,
        f"{len(syn_nrn_post_df)}",
        color=syn_hue_order[0],
        fontdict={
            "fontsize": 8,
            "verticalalignment": "top",
            "horizontalalignment": "center",
        },
    )
    ax.text(
        0.225,
        depth_bounds[0] + 10,
        f"{len(syn_nrn_pre_df)}",
        color=syn_hue_order[1],
        fontdict={
            "fontsize": 8,
            "verticalalignment": "top",
            "horizontalalignment": "center",
        },
    )

    ax.invert_yaxis()
    ax.legend().remove()
    ax.set_xticklabels((""))

    sns.despine(ax=ax, offset=5, bottom=True)
    ax.get_xaxis().set_visible(False)

    return ax


def generate_layer_data(layers, span, axis="x"):
    y_data = np.array(layers).repeat(3)
    x_data = np.array([span[0], span[1], np.nan] * len(layers))
    if axis == "x":
        return x_data, y_data
    elif axis == "y":
        return y_data, x_data
    else:
        raise ValueError(r'Axis must be "x" or "y"')


def generate_layer_ticks(layers, depth_bounds):
    all_bnds = np.sort(np.concatenate([depth_bounds, layers]))
    layer_ticks = all_bnds[:-1] + np.diff(all_bnds) / 2
    return layer_ticks


def get_lvl2_chunk_points(oid, seg_source, voxel_resolution=[4, 4, 40]):
    cv = cloudvolume.CloudVolume(seg_source, use_https=True)
    cvm = cv.mesh
    frag_labels = cvm.get_fragment_labels(segid=oid, level=2)
    xyzs3 = [np.array(cvm.meta.meta.decode_chunk_position(l)) for l in frag_labels]
    xyzs3 = np.vstack(
        xyzs3
    ) * cvm.meta.meta.graph_chunk_size + cvm.meta.meta.voxel_offset(0)
    mip_offset = np.array(cv.mip_resolution(0)) / voxel_resolution
    xyzs3 = xyzs3 * voxel_resolution * mip_offset
    return xyzs3


def plot_layers(
    layers,
    depth_bounds,
    span,
    ax=None,
    orientation="horizontal",
    labels=["L1", "L2/3", "L4", "L5", "L6", "WM"],
    zorder=-1,
    **kwargs,
):

    orientation_dict = {"horizontal": "x", "vertical": "y"}
    layer_x, layer_y = generate_layer_data(
        layers, span, axis=orientation_dict.get(orientation, None)
    )
    layer_ticks = generate_layer_ticks(layers, depth_bounds)
    ax.plot(layer_x, layer_y, zorder=zorder, **kwargs)

    if orientation_dict[orientation] == "x":
        ax.set_ylim(*depth_bounds)
        ax.invert_yaxis()
        ax.set_yticks(layer_ticks)
        ax.set_yticklabels(labels)
    elif orientation_dict[orientation] == "y":
        ax.set_xlim(*depth_bounds)
        ax.set_xticks(layer_ticks)
        ax.set_xticklabels(labels)
    return ax


def pcg_anatomy_projection(
    oid,
    seg_source,
    ax=None,
    layers=None,
    width_bounds=None,
    depth_bounds=None,
    soma_position=None,
    color="k",
    bar_length=250,
    bar_start=None,
    offset=0,
    alpha=0.2,
    voxel_resolution=[4, 4, 40],
    zorder=1,
):
    if ax is None:
        ax = plt.gca()

    xyzs3 = get_lvl2_chunk_points(oid, seg_source, voxel_resolution=voxel_resolution)

    if layers is not None:
        plot_layers(
            layers,
            depth_bounds,
            width_bounds,
            ax,
            color=(0.5, 0.5, 0.5),
            linestyle=":",
            alpha=0.45,
        )

    ax.scatter(
        xyzs3[:, 0] / 1000 + offset,
        xyzs3[:, 1] / 1000,
        s=0.25,
        alpha=alpha,
        color=color,
        zorder=zorder,
    )

    if soma_position is not None:
        ax.plot(
            soma_position[0] / 1000 + offset,
            soma_position[1] / 1000,
            color="w",
            marker="o",
            markersize=7,
            alpha=1,
            markeredgecolor="k",
            zorder=zorder + 0.5,
        )

    if bar_start is None:
        bar_start = (layers[-1] + 50, width_bounds[0])
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
            bar_start[0] + 60,
            f"{bar_length} $\mu$m",
            color="k",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_aspect("equal")
    sns.despine(ax=ax, offset=5, bottom=True)
    ax.set_xticks([])
    return ax


def background_fill(
    ax,
    spacing,
    max_dist,
    limits,
    colors,
    color_first=0.98,
    color_last=0.6,
    base_zorder=-1,
):
    n_inc = np.int(np.ceil(max_dist / spacing))
    increments = np.arange(1, n_inc + 1)
    shades = np.linspace(color_first, color_last, n_inc)
    limits = np.array(limits)
    for ii, (inc, shade) in enumerate(zip(increments, shades)):
        ax.fill_between(
            limits,
            limits - spacing * inc,
            limits + spacing * inc,
            color=(shade, shade, shade),
            zorder=base_zorder - ii - 1,
        )

    ax.fill_between(
        limits,
        limits,
        [limits[0], limits[0]],
        color=colors[0],
        alpha=0.05,
        zorder=base_zorder,
    )
    ax.fill_between(
        limits,
        limits,
        [limits[1], limits[1]],
        color=colors[1],
        alpha=0.05,
        zorder=base_zorder,
    )
    return ax


def synapse_location_plot(
    oid,
    syn_df,
    target_df,
    background_spacing,
    background_max,
    layers,
    depth_bounds,
    soma_df=None,
    ax=None,
    hue_column=None,
    palette=None,
    s=5,
    alpha=0.6,
    fill_background=True,
    background_colors=((0, 0, 1), (1, 0, 0)),
    soma_depth_column="soma_depth_um",
):
    if ax is None:
        ax = plt.gca()
    syn_nrn_pre_df = syn_df.query("pre_pt_root_id == @oid").reset_index(drop=True)
    syn_nrn_pre_df["syn_depth"] = (
        syn_nrn_pre_df["ctr_pt_position"].apply(lambda x: x[1] * 4) / 1000
    )

    keep_columns = ["pt_root_id", soma_depth_column]
    if hue_column is not None:
        keep_columns.append(hue_column)

    syn_nrn_pre_df = syn_nrn_pre_df.merge(
        target_df[keep_columns],
        left_on="post_pt_root_id",
        right_on="pt_root_id",
        how="inner",
    )

    if soma_df is None:
        soma_df = target_df
    soma_dat = soma_df.query("pt_root_id == @oid", engine="python")[
        "soma_depth_um"
    ].values
    if len(soma_dat) == 1:
        soma_depth = soma_dat[0]
    else:
        soma_depth = None

    ax = sns.scatterplot(
        y="syn_depth",
        x=soma_depth_column,
        hue=hue_column,
        edgecolor="k",
        palette=palette,
        data=syn_nrn_pre_df,
        s=s,
        alpha=alpha,
        ax=ax,
    )
    if fill_background:
        ax = background_fill(
            ax,
            background_spacing,
            background_max,
            depth_bounds,
            colors=background_colors,
        )
    else:
        ax.plot(depth_bounds, depth_bounds, color="k", linestyle=":", linewidth=0.5)

    ax.plot(
        [soma_depth],
        [soma_depth],
        "wo",
        markersize=6,
        markeredgecolor="k",
        linewidth=0.5,
        alpha=1,
    )

    layer_x, layer_y = generate_layer_data(layers, depth_bounds, axis="x")
    layer_ticks = generate_layer_ticks(layers, depth_bounds)
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

    ax.set_xlim(*depth_bounds)
    ax.set_ylim(*depth_bounds)

    ax.invert_yaxis()
    ax.legend().remove()
    ax.set_aspect("equal")
    ax.xaxis.set_label_position("top")
    sns.despine(ax=ax, offset=5, top=False, bottom=True)

    return ax


def target_summary_plot(
    syn_df_bin_count_long,
    cell_type_column,
    palette,
    order,
    min_bin_size=0,
    sizes=(8, 150),
    min_bg_count=0,
    linewidth=0.5,
    back_color=(0.6, 0.6, 0.6),
    ax=None,
):
    if min_bin_size > 0:
        syn_df_bin_count_long.loc[
            syn_df_bin_count_long.query("bin_total < @min_bin_size").index, "p"
        ] = pd.NA

    cell_type_order = {ct: ii for ii, ct in enumerate(order)}
    syn_df_bin_count_long["x_val"] = syn_df_bin_count_long["cell_type_comp"].apply(
        lambda x: cell_type_order[x]
    )

    if ax is None:
        ax = plt.gca()
    linewidth = 0.5
    back_color = (0.6, 0.6, 0.6)

    sns.scatterplot(
        y="syn_bin",
        x="x_val",
        size="p",
        size_norm=(0, 1),
        sizes=sizes,
        edgecolor="k",
        palette=palette,
        linewidth=linewidth,
        hue="category",
        data=syn_df_bin_count_long.query(
            'p>p_overall and num_syn_overall>@min_bg_count and category=="base"',
            engine="python",
        ),
        ax=ax,
        alpha=0.8,
    )
    sns.scatterplot(
        y="syn_bin",
        x="x_val",
        size="p_overall",
        size_norm=(0, 1),
        sizes=sizes,
        color="w",
        linewidth=0,
        data=syn_df_bin_count_long.query(
            'p>p_overall and num_syn_overall>@min_bg_count and category=="base"',
            engine="python",
        ),
        ax=ax,
        alpha=0.5,
    )

    sns.scatterplot(
        y="syn_bin",
        x="x_val",
        size="p_overall",
        size_norm=(0, 1),
        sizes=sizes,
        color=back_color,
        linewidth=0,
        data=syn_df_bin_count_long.query(
            'p<=p_overall and num_syn_overall>@min_bg_count and category=="base"',
            engine="python",
        ),
        ax=ax,
        alpha=0.3,
    )
    sns.scatterplot(
        y="syn_bin",
        x="x_val",
        size="p",
        size_norm=(0, 1),
        sizes=sizes,
        edgecolor="k",
        palette=palette,
        linewidth=linewidth,
        hue="category",
        data=syn_df_bin_count_long.query(
            'p<=p_overall and num_syn_overall>@min_bg_count and category=="base"',
            engine="python",
        ),
        ax=ax,
        alpha=0.8,
    )

    sns.scatterplot(
        y="syn_bin",
        x="x_val",
        size="p",
        size_norm=(0, 1),
        sizes=sizes,
        edgecolor="k",
        palette=palette,
        linewidth=linewidth,
        hue="category",
        data=syn_df_bin_count_long.query(
            'p>p_overall and num_syn_overall>@min_bg_count and category=="high"',
            engine="python",
        ),
        ax=ax,
        alpha=1,
    )
    sns.scatterplot(
        y="syn_bin",
        x="x_val",
        size="p_overall",
        size_norm=(0, 1),
        sizes=sizes,
        color="w",
        linewidth=0,
        data=syn_df_bin_count_long.query(
            'p>p_overall and num_syn_overall>@min_bg_count and category=="high"',
            engine="python",
        ),
        ax=ax,
        alpha=0.7,
    )

    sns.scatterplot(
        y="syn_bin",
        x="x_val",
        size="p_overall",
        size_norm=(0, 1),
        sizes=sizes,
        color=sns.light_palette(palette["low"], 8)[1],
        linewidth=0,
        data=syn_df_bin_count_long.query(
            'p<p_overall and num_syn_overall>@min_bg_count and category=="low"',
            engine="python",
        ),
        ax=ax,
        alpha=1,
    )
    sns.scatterplot(
        y="syn_bin",
        x="x_val",
        size="p",
        size_norm=(0, 1),
        sizes=sizes,
        edgecolor="k",
        palette=palette,
        linewidth=linewidth,
        hue="category",
        data=syn_df_bin_count_long.query(
            'p<p_overall and num_syn_overall>@min_bg_count and category=="low"',
            engine="python",
        ),
        ax=ax,
        alpha=1,
    )

    sns.scatterplot(
        y="syn_bin",
        x="x_val",
        s=sizes[0],
        marker="o",
        color="w",
        edgecolor="k",
        linewidth=0.25,
        data=syn_df_bin_count_long.query(
            "p.isnull() and num_syn_overall>@min_bg_count", engine="python"
        ),
        ax=ax,
        alpha=0.2,
    )

    try:
        ax.get_legend().remove()
    except:
        pass

    ax.invert_yaxis()
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    sns.despine(ax=ax, top=False, bottom=True, trim=True)

    return ax


def xaxis_with_gap(order, gap_ind):
    order_gap = order[0:gap_ind] + ["_gap_col_"] + order[gap_ind:]
    all_xticks = np.arange(len(order_gap))
    xticks_gap = all_xticks[all_xticks != gap_ind]
    return xticks_gap, order_gap


def build_title(subtype, sid, oid):
    if subtype is not None:
        prefix = f"{subtype} — "
    else:
        prefix = ""
    if sid is not None:
        mid = f"Cell ID: {sid} — "
    else:
        mid = ""
    return f"{prefix}{mid}OID: {oid}"


def compartment_specificity_card(
    pre_syn_df,
    column_df,
    order,
    order_gap_ind,
    sublabel,
    layer_bins,
    layer_names,
    baseline_count_long,
    spec_palette,
    ax=None,
    size_range=(10, 200),
    min_bg_count=10,
    min_bin_size=20,
    voxel_resolution=voxel_resolution,
):
    if ax is None:
        ax = plt.gca()

    pre_syn_df["syn_depth"] = syn_depth(pre_syn_df, voxel_resolution)
    syn_df_ct = synapse_compartment_profile(pre_syn_df, column_df)

    cell_type_column = "cell_type_comp"
    syn_df_bin_count_long = full_synapse_data(
        syn_df_ct, cell_type_column, layer_bins, baseline_count_long, order=order
    )

    xticks_gap, order_comp_gap = xaxis_with_gap(order, order_gap_ind)
    target_summary_plot(
        syn_df_bin_count_long,
        cell_type_column,
        spec_palette,
        order_comp_gap,
        sizes=size_range,
        min_bin_size=10,
        ax=ax,
    )

    ax.set_yticks(np.arange(len(layer_bins) - 1))
    _ = ax.set_yticklabels(layer_names, fontdict=dict(fontsize=8))

    ax.set_xticks(xticks_gap)
    _ = ax.set_xticklabels(sublabel, rotation=45, ha="left", fontsize=6)

    ax.tick_params(direction="in")
    vls = [2.5, 5.5, 8.5, 11.5, 14.5, 17.5, 18.5]
    for v in vls:
        ax.vlines(v, -0.25, len(layer_names) - 0.75, linewidth=0.5, linestyle=":")

    ects = ["P23", "P4", "P5 (IT)", "P5 (PT)", "P5 (NP)", "P6", "", "", "", "", "", ""]
    ect_cell_type_column = "cell_type"
    ect_cell_types = [
        "23P",
        "4P",
        "5P_IT",
        "5P_PT",
        "5P_NP",
        "6P",
        "Unsure E",
        "BC",
        "BPC",
        "MC",
        "NGC",
        "Unsure I",
    ]
    ect_locs = [1, 4, 7, 10, 13, 16, 18, 20, 21, 22, 23, 24]
    for t, l, ct in zip(ects, ect_locs, ect_cell_types):
        ax.text(l, -1.25, t, ha="center", fontsize=8)
        ax.text(
            l,
            -0.25,
            len(syn_df_ct.query(f"{ect_cell_type_column} == @ct")),
            ha="center",
            fontsize=8,
        )
    ax.set_aspect("auto")
    return ax


####


def output_count_bar(
    syn_df,
    num_rows=None,
    labels=None,
    cell_type="cell_type",
    hue="valence",
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


def _augment_table_results(res_df, cap=4, cap_extra=0.2):
    res_df = res_df.copy()
    res_df["l2or"] = np.log2(res_df["odds_ratio"])
    res_df["l2or_lb"] = np.log2(res_df["or_lb"])
    res_df["l2or_ub"] = np.log2(res_df["or_ub"])
    res_df["direction"] = np.where(res_df["odds_ratio"] > 1, "high", "low")

    res_df["l2or_cap"] = np.where(
        res_df["l2or"] < -cap, -cap - cap_extra, res_df["l2or"]
    )
    res_df["l2or_lb_cap"] = np.where(
        res_df["l2or_lb"] < -cap, -cap - cap_extra, res_df["l2or_lb"]
    )
    res_df["l2or_ub_cap"] = np.where(
        res_df["l2or_ub"] < -cap, -cap - cap_extra, res_df["l2or_ub"]
    )

    res_df["l2or_cap"] = np.where(
        res_df["l2or"] > cap, cap + cap_extra, res_df["l2or_cap"]
    )
    res_df["l2or_lb_cap"] = np.where(
        res_df["l2or_lb"] > cap, cap + cap_extra, res_df["l2or_lb_cap"]
    )
    res_df["l2or_ub_cap"] = np.where(
        res_df["l2or_ub"] > cap, cap + cap_extra, res_df["l2or_ub_cap"]
    )
    return res_df


def output_preference_index(
    res_df,
    groups,
    axes,
    palette,
    cap=4,
    cap_extra=0.2,
    extra_mult=2,
    cell_type_column="cell_type",
):
    res_df = _augment_table_results(res_df, cap=cap, cap_extra=cap_extra)

    for ax in axes:
        ax.set_visible(False)

    for ii, (ax, group) in enumerate(zip(axes, groups)):
        display_xticks = ii == 0
        _preference_panel(
            group,
            res_df,
            palette,
            cell_type_column=cell_type_column,
            display_xticks=display_xticks,
            cap=cap,
            cap_extra=cap_extra,
            extra_mult=extra_mult,
            ax=ax,
        )
    pass


def _pref_color(row, palette):
    if row["sig"]:
        return palette[row["direction"]]
    else:
        return palette["base"]


def _preference_panel(
    group,
    res_df,
    palette,
    cell_type_column="cell_type",
    display_xticks=False,
    color_function=_pref_color,
    cap=4,
    cap_extra=0.2,
    extra_mult=3,
    pref_label="Preference Factor",
    ax=None,
):
    if ax is None:
        ax = plt.gca()
    ax.set_visible(True)

    len_group = len(group["cts"])

    for ii, ct in enumerate(group["cts"]):
        row = res_df.query(f"{cell_type_column} == @ct")
        if len(row) == 0:
            plt.plot(
                -cap - cap_extra,
                ii,
                marker="o",
                color="k",
                markerfacecolor=None,
                markeredgecolor="k",
                markersize=4,
            )
            continue
        else:
            row = row.iloc[0]

        color = color_function(row, palette)

        ax.plot(
            [row["l2or_lb_cap"], row["l2or_ub_cap"]],
            [ii, ii],
            color=color,
            marker=None,
            linewidth=1.5,
            zorder=-1,
        )
        ax.plot(
            row["l2or_cap"],
            ii,
            color=color,
            marker="o",
            markeredgecolor=None,
            markeredgewidth=0.5,
            markersize=5,
            zorder=2,
        )
        ax.set_xlim(
            [-cap - (1 + extra_mult) * cap_extra, cap + (1 + extra_mult) * cap_extra]
        )

    ax.vlines(
        0,
        0 - extra_mult * cap_extra,
        len_group - 1 + extra_mult * cap_extra,
        color="k",
        linewidth=1,
        alpha=0.5,
        zorder=-10,
    )
    ax.hlines(
        np.arange(0, len_group),
        -cap,
        cap,
        color="k",
        alpha=0.25,
        linewidth=0.25,
        zorder=-12,
    )
    ax.fill_betweenx(
        [0 - extra_mult * cap_extra, len_group - 1 + extra_mult * cap_extra],
        [cap, cap],
        [cap + extra_mult * cap_extra, cap + extra_mult * cap_extra],
        color=(0.5, 0.5, 0.5),
        linewidth=0.5,
        alpha=0.1,
        zorder=-10,
    )
    ax.fill_betweenx(
        [0 - extra_mult * cap_extra, len_group - 1 + extra_mult * cap_extra],
        [-cap, -cap],
        [-cap - extra_mult * cap_extra, -cap - extra_mult * cap_extra],
        color=(0.5, 0.5, 0.5),
        linewidth=0.5,
        alpha=0.1,
        zorder=-10,
    )

    ax.set_yticks(np.arange(len_group))
    ax.set_ylim(-3 * cap_extra, len_group - 1 + 3 * cap_extra)
    ax.invert_yaxis()

    if display_xticks:
        ax.set_xticks(np.arange(-4, 4.1, 2))
        ax.set_xticks(np.arange(-3, 3.1, 2), minor=True)
        ax.set_xticklabels(["$2^{-4}$", "$2^{-2}$", "1", "$2^2$", "$2^4$"], fontsize=4)
        sns.despine(
            ax=ax, offset=3, trim=True, top=False, bottom=True, right=True, left=False
        )
        ax.set_xlabel(pref_label)
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")

    else:
        ax.xaxis.set_visible(False)
        sns.despine(
            ax=ax, offset=3, trim=True, top=True, bottom=True, right=True, left=False
        )

    ax.vlines(
        np.arange(-cap, cap + 0.01, 1),
        -extra_mult * cap_extra,
        len_group - 1 + extra_mult * cap_extra,
        linewidth=0.25,
        alpha=0.25,
        zorder=-13,
    )
    ax.set_yticks(np.arange(len_group))
    ax.set_yticklabels(group["labels"], fontsize=13 - len(group["labels"]))
    ax.set_ylim(-3 * cap_extra, len_group - 1 + 3 * cap_extra)
    ax.invert_yaxis()
    return ax


def _output_category(row):
    if row["targ_in_column"]:
        return "Col"
    elif row["num_soma"] == 0:
        return "Orph"
    elif row["num_soma"] == 1:
        return "NonCol"
    elif row["num_soma"] > 1:
        return "Multi"


def output_categories(pre_syn_df, column_df, soma_count_df):
    pre_syn_soma_df = pre_syn_df.merge(
        soma_count_df, how="left", left_on="post_pt_root_id", right_on="pt_root_id"
    ).fillna(0)
    pre_syn_soma_df["targ_in_column"] = np.isin(
        pre_syn_soma_df["post_pt_root_id"], column_df["pt_root_id"]
    )
    pre_syn_soma_df["targ_category"] = pre_syn_soma_df.apply(_output_category, axis=1)
    return pre_syn_soma_df.drop(columns=["targ_in_column", "pt_root_id"])


def output_category_budget(pre_syn_soma_df, categories, palette=None, ax=None):
    if ax is None:
        ax = plt.gca()

    pre_syn_soma_df["dummy"] = 1
    sns.histplot(
        x="dummy",
        hue="targ_category",
        hue_order=categories,
        palette=palette,
        data=pre_syn_soma_df,
        discrete=True,
        multiple="stack",
        stat="probability",
        linewidth=0.5,
        ax=ax,
    )

    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.5))
    ax.set_yticks(np.arange(0, 1.01, 0.1), minor=True)
    sns.despine(ax=ax, offset=5, bottom=True, trim=True)
    ax.xaxis.set_visible(False)
    ax.set_ylabel("Fraction output syn.")

    ax.legend_.set_bbox_to_anchor((0.35, 0.25))
    ax.legend_.set_title(None)
    ax.legend_.set_frame_on(False)
    return ax


axon_color = [0.214, 0.495, 0.721]
dendrite_color = [0.894, 0.103, 0.108]


def quick_plot(
    nrn,
    layer_bnds,
    height_bounds,
    width_bounds,
    axon_color=axon_color,
    dendrite_color=dendrite_color,
    colorize=True,
):

    if colorize == False:
        axon_color = "k"
        dendrite_color = "k"

    axon_pts = nrn.skeleton.vertices[nrn.anno.is_axon.mesh_index.to_skel_mask]
    dend_pts = nrn.skeleton.vertices[~nrn.anno.is_axon.mesh_index.to_skel_mask]
    soma_pt = nrn.skeleton.vertices[nrn.skeleton.root]

    fig, ax = plt.subplots(facecolor="w", dpi=150)
    ax.scatter(
        axon_pts[:, 0] / 1000,
        axon_pts[:, 1] / 1000,
        marker=".",
        s=0.5,
        color=axon_color,
    )
    ax.scatter(
        dend_pts[:, 0] / 1000,
        dend_pts[:, 1] / 1000,
        marker=".",
        s=0.5,
        color=dendrite_color,
    )
    ax.scatter(
        soma_pt[0] / 1000,
        soma_pt[1] / 1000,
        s=20,
        color="w",
        edgecolor="k",
        linewidth=0.5,
    )
    plot_layers(
        layer_bnds,
        height_bounds,
        width_bounds,
        color=(0.5, 0.5, 0.5),
        linewidth=0.5,
        linestyle=":",
        ax=ax,
    )
    return fig, ax


from matplotlib import colors


def path_line(path, sk, x_ind=0, y_ind=1, rescale=1):
    xs = sk.vertices[path, x_ind].squeeze()
    ys = sk.vertices[path, y_ind].squeeze()
    return xs / rescale, ys / rescale


def all_paths(sk, x_ind=0, y_ind=1, rescale=1):
    all_xs = []
    all_ys = []
    for path in sk.cover_paths_with_parent():
        xs, ys = path_line(path, sk, x_ind=x_ind, y_ind=y_ind, rescale=rescale)
        all_xs.append(xs)
        all_xs.append([np.nan])
        all_ys.append(ys)
        all_ys.append([np.nan])
    return np.concatenate(all_xs), np.concatenate(all_ys)


def darken_color(clr, delta=0.3):
    clr_hsv = colors.rgb_to_hsv(clr)
    vdiff = clr_hsv[2]
    return colors.hsv_to_rgb([x - delta * y for x, y in zip(clr_hsv, [0, 0, vdiff])])


def lighten_color(clr, delta=0.3):
    clr_hsv = colors.rgb_to_hsv(clr)
    vdiff = 1 - clr_hsv[2]
    return colors.hsv_to_rgb([x + delta * y for x, y in zip(clr_hsv, [0, 0, vdiff])])


def plot_neuron(
    nrn,
    color,
    x_offset=0,
    ax=None,
    dendrite_only=True,
    axon_color=None,
    rescale=1000,
    axon_anno=None,
    dendrite_anno=None,
    axon_kwargs={},
    dendrite_kwargs={},
    soma_kwargs={},
    min_x=None,
    return_extent=False,
):
    soma_pt = np.atleast_2d(nrn.skeleton.vertices[nrn.skeleton.root]) / rescale
    soma_pt_x = soma_pt[0, 0]

    if axon_anno is not None:
        is_axon = nrn.anno[axon_anno].mesh_mask
    elif dendrite_anno is not None:
        is_axon = ~nrn.anno[dendrite_anno].mesh_mask
    else:
        is_axon = np.full(len(nrn.mesh.vertice), False)

    with nrn.mask_context(~is_axon) as nrnf:
        x_dend, y_dend = all_paths(nrnf.skeleton, x_ind=0, y_ind=1, rescale=rescale)
        all_x = x_dend[~np.isnan(x_dend)]

    if not dendrite_only:
        with nrn.mask_context(is_axon) as nrnf:
            x_axon, y_axon = all_paths(nrnf.skeleton, x_ind=0, y_ind=1, rescale=rescale)
        all_x = np.concatenate((x_axon[~np.isnan(x_axon)], all_x))

    if min_x is not None:
        if np.min(all_x - soma_pt_x) + x_offset < min_x:
            x_offset = min_x + np.abs(np.min(all_x - soma_pt_x)) - soma_pt_x
    else:
        x_offset = x_offset - soma_pt_x

    if ax is None:
        ax = plt.gca()

    if axon_color is None:
        axon_color = lighten_color(color)

    if not dendrite_only:
        ax.plot(x_axon + x_offset, y_axon, color=axon_color, **axon_kwargs)
    ax.plot(x_dend + x_offset, y_dend, color=color, **dendrite_kwargs)

    ax.scatter(
        soma_pt[:, 0] + x_offset,
        soma_pt[:, 1],
        color=color,
        **soma_kwargs,
    )
    if ax.get_ylim()[0] < ax.get_ylim()[1]:
        ax.invert_yaxis()

    if return_extent:
        return min(all_x + x_offset), max(all_x + x_offset)
