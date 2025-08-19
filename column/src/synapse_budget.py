import numpy as np
import pandas as pd
import swifter
import statsmodels.api as sm
from statsmodels.stats import contingency_tables as cont
from scipy import stats
import matplotlib.pyplot as plt

voxel_resolution = [4, 4, 40]


def _delta_vec(row):
    return (
        np.array(np.array(row["pt_position"]) - row["ctr_pt_position"])
    ) * voxel_resolution


def _theta(v):
    return np.arccos(v[1] / np.linalg.norm(v))


def cell_typed_synapse_profile(syn_profile_df, column_df):
    syn_profile_df = syn_profile_df.merge(
        column_df[["pt_root_id", "pt_position", "cell_type"]],
        left_on="post_pt_root_id",
        right_on="pt_root_id",
    ).drop(columns="pt_root_id")
    syn_profile_df["soma_depth"] = syn_profile_df["pt_position"].apply(
        lambda x: x[1] / 1000
    )
    return syn_profile_df


def target_centric_synapse_coordinates(synapse_df, target_df):
    synapse_df["delta_vector"] = synapse_df.swifter.progress_bar(False).apply(
        _delta_vec, axis=1
    )
    synapse_df["syn_r"] = (
        synapse_df["delta_vector"]
        .swifter.progress_bar(False)
        .apply(lambda x: np.linalg.norm(x) / 1000)
    )
    synapse_df["syn_theta"] = (
        synapse_df["delta_vector"].swifter.progress_bar(False).apply(_theta)
    )
    return synapse_df.drop(columns="delta_vector")


def _simple_thresholds():
    l6_soma = 20
    l6_apical = np.pi * 0.20
    l5pt_soma = 25
    l5pt_apical = np.pi * 0.17
    l5np_soma = 20
    l5np_apical = np.pi * 0.15
    l5it_soma = 20
    l5it_apical = np.pi * 0.15
    l4_apical = np.pi * 0.1
    l4_soma = 20
    l23_soma = 20
    l23_apical = np.pi * 0.2

    soma_thresholds = {
        "23P": l23_soma,
        "4P": l4_soma,
        "5P_IT": l5it_soma,
        "5P_NP": l5np_soma,
        "5P_PT": l5pt_soma,
        "6P": l6_soma,
    }
    apical_thresholds = {
        "23P": l23_apical,
        "4P": l4_apical,
        "5P_IT": l5it_apical,
        "5P_NP": l5np_apical,
        "5P_PT": l5pt_apical,
        "6P": l6_apical,
    }

    return soma_thresholds, apical_thresholds


def _syn_category(row):
    soma_thresholds, apical_thresholds = _simple_thresholds()
    r_thresh = soma_thresholds.get(row["cell_type"], None)
    theta_thresh = apical_thresholds.get(row["cell_type"], -np.inf)

    if r_thresh is None:
        return pd.NA

    if row["syn_r"] < r_thresh:
        return "soma"
    elif row["syn_theta"] > theta_thresh:
        return "basal"
    else:
        return "apical"


def _ct_with_compartment(row):
    if pd.isna(row["compartment"]):
        return row["cell_type"]
    else:
        return f"{row['cell_type']}_{row['compartment']}"


def synapse_compartment_profile(syn_profile_df, target_df):
    syn_profile_df = cell_typed_synapse_profile(syn_profile_df, target_df)
    syn_profile_df = target_centric_synapse_coordinates(syn_profile_df, target_df)
    syn_profile_df["compartment"] = syn_profile_df.swifter.progress_bar(False).apply(
        _syn_category, axis=1
    )
    syn_profile_df["cell_type_comp"] = (
        syn_profile_df[["cell_type", "compartment"]]
        .swifter.progress_bar(False)
        .apply(_ct_with_compartment, axis=1)
    )
    return syn_profile_df


def generate_profile_bins(layer_bnds, height_bnds, bins_per_layer):
    if np.isscalar(bins_per_layer):
        bins_per_layer = np.full(len(layer_bnds), bins_per_layer)
    bins_per_layer = np.array(bins_per_layer) + 1
    all_bounds = np.concatenate([[height_bnds[0]], layer_bnds, [height_bnds[1]]])
    bins = []
    for lb, ub, bn in zip(all_bounds[:-1], all_bounds[1:], bins_per_layer):
        bins.append(np.linspace(lb, ub, bn))
    bins.append([height_bnds[1]])
    return np.unique(np.concatenate(bins))


def get_syn_profile_count(
    syn_profile_df, bins, cell_type_column, order=None, pbar=False, return_profile=False
):

    syn_profile_df = syn_profile_df.drop(
        syn_profile_df.query(
            "syn_depth_um <= @bins[0] or syn_depth_um >= @bins[-1]"
        ).index
    ).reset_index(drop=True)
    syn_profile_df["syn_bin"] = (
        syn_profile_df["syn_depth_um"]
        .swifter.progress_bar(pbar)
        .apply(
            lambda x: np.flatnonzero(np.logical_and(x >= bins[:-1], x < bins[1:]))[0]
        )
    )

    syn_profile_count = (
        syn_profile_df.groupby(["syn_bin", cell_type_column])
        .count()[["syn_depth_um"]]
        .rename(columns={"syn_depth_um": "syn_count"})
        .reset_index()
    )
    syn_profile_count = syn_profile_count.pivot_table(
        index="syn_bin", columns=cell_type_column, values="syn_count", fill_value=0
    )

    if order is not None:
        drop_cols = syn_profile_count.columns.values
        drop_cols = drop_cols[~np.isin(drop_cols, order)]
        if len(drop_cols) > 0:
            syn_profile_count = syn_profile_count.drop(columns=drop_cols)

    if return_profile:
        return syn_profile_count, syn_profile_df
    else:
        return syn_profile_count


def plot_bin_definition(bins, layers, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.hlines(bins, xmin=0, xmax=1, linewidth=0.5, alpha=0.4)
    ax.hlines(layers, xmin=0, xmax=1, linewidth=0.75)
    ax.invert_yaxis()
    ax.set_ylabel("Depth ($\mu m$)")
    ax.set_xticks([])
    return ax


def _transform_baseline_syn_count(syn_profile_count):
    syn_profile_count_long = syn_profile_count.melt(
        value_name="num_syn", ignore_index=False
    ).reset_index()
    syn_profile_count_long = syn_profile_count_long.merge(
        syn_profile_count_long.groupby("syn_bin").sum()["num_syn"].rename("bin_total"),
        left_on="syn_bin",
        right_index=True,
    )

    syn_profile_count_long["bin_other"] = (
        syn_profile_count_long["bin_total"] - syn_profile_count_long["num_syn"]
    )
    syn_profile_count_long["p"] = (
        syn_profile_count_long["num_syn"] / syn_profile_count_long["bin_total"]
    )
    return syn_profile_count_long


def baseline_counts(syn_df, cell_type_column, layers, order):
    syn_profile_count, syn_profile_bin_df = get_syn_profile_count(
        syn_df, layers, cell_type_column, order=order, return_profile=True
    )
    syn_profile_count_long = _transform_baseline_syn_count(syn_profile_count)
    return syn_profile_bin_df, syn_profile_count, syn_profile_count_long


def baseline_context(syn_df_long, baseline_long, cell_type_column):
    bin_lookup = {
        row["syn_bin"]: row["bin_total"]
        for _, row in syn_df_long.drop_duplicates("syn_bin").iterrows()
    }
    syn_df_long = syn_df_long.merge(
        baseline_long,
        on=["syn_bin", cell_type_column],
        how="right",
        suffixes=(None, "_overall"),
    )
    syn_df_long["bin_total"] = syn_df_long["syn_bin"].apply(
        lambda x: bin_lookup.get(x, 0)
    )
    syn_df_long["num_syn"] = syn_df_long["num_syn"].fillna(0)
    syn_df_long["bin_other"] = syn_df_long["bin_total"] - syn_df_long["num_syn"]
    syn_df_long["p"] = syn_df_long["num_syn"] / syn_df_long["bin_total"]
    syn_df_long["num_syn"] = syn_df_long["num_syn"].astype(int)
    syn_df_long["bin_other"] = syn_df_long["bin_other"].astype(int)
    return syn_df_long


def _categorize(row):
    if not row["sig"]:
        return "base"
    elif row["p"] < row["p_overall"]:
        return "low"
    else:
        return "high"


def binom_test_ci(row):
    if row["bin_total"] == 0:
        return np.nan, np.nan
    lb, ub = sm.stats.proportion_confint(
        row["num_syn"], row["bin_total"], method="agresti_coull"
    )
    return lb, ub


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


def add_stats(syn_bin_long, alpha=0.01):
    lbub = np.vstack(syn_bin_long.apply(binom_test_ci, axis=1))
    pval = syn_bin_long.apply(fisher_exact_rows, axis=1)
    syn_bin_long["lci"] = lbub[:, 0].squeeze()
    syn_bin_long["uci"] = lbub[:, 1].squeeze()
    sig, pval_adj, _, _ = sm.stats.multipletests(pval, method="holm-sidak")
    syn_bin_long["p_value"] = pval_adj
    syn_bin_long["sig"] = sig
    syn_bin_long["category"] = syn_bin_long.apply(_categorize, axis=1)
    return syn_bin_long


def full_synapse_data(
    syn_df_ct, cell_type_column, layer_bins, profile_df_long, order=None
):
    syn_df_bin_comp, syn_df_bin_count, syn_df_bin_count_long = baseline_counts(
        syn_df_ct, cell_type_column, layer_bins, order
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


def _table(row):
    return [
        [row["num_syn"], row["bin_other"]],
        [row["num_syn_overall"], row["bin_other_overall"]],
    ]


def _all_zero_upper_bound(ts, alpha=0.025):
    Nqis = []
    for t in ts:
        t = np.array(ts)
        qi = t[1, 0] / np.sum(t[1, :])
        Ni = t[0, 1]
        Nqis.append(Ni * qi)
    return -np.log(alpha) / np.sum(Nqis)


def stratified_tables(
    syn_count_long,
    cell_type_column,
    order=None,
    bin_threshold=10,
    global_bin_threshold=100,
    return_res=False,
):
    """Compute the stratified statistics

    Parameters
    ----------
    syn_count_long : _type_
        _description_
    cell_type_column : _type_
        _description_
    order : _type_, optional
        _description_, by default None
    bin_threshold : int, optional
        _description_, by default 10
    global_bin_threshold : int, optional
        _description_, by default 100
    return_res : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    if order is None:
        order = np.unique(syn_count_long[cell_type_column])
    cts = []
    lodds_ratios = []
    lor_lb = []
    lor_ub = []
    pvalues = []
    reses = []
    for ct in order:
        df = syn_count_long.query(
            f"{cell_type_column} == @ct and bin_total>@bin_threshold and bin_total_overall>@global_bin_threshold"
        )
        if len(df) == 0:
            cts.append(ct)
            lodds_ratios.append(np.nan)
            lor_lb.append(np.nan)
            lor_ub.append(np.nan)
            pvalues.append(np.nan)
        else:
            ts = []
            for _, row in df.iterrows():
                ts.append(_table(row))
            if len(ts) > 1:
                res = cont.StratifiedTable(ts)
                lodds_ratios.append(res.oddsratio_pooled)
                uc_zeros = [t[0][0] for t in ts]
                if np.all(uc_zeros == 0):
                    lor_lb.append(0)
                    lor_ub.append(_all_zero_upper_bound(ts))
                else:
                    lor_lb.append(res.oddsratio_pooled_confint()[0])
                    lor_ub.append(res.oddsratio_pooled_confint()[1])
                pvalues.append(res.test_null_odds().pvalue)
            elif len(ts) == 1:
                res = cont.Table2x2(ts[0])
                lodds_ratios.append(res.log_oddsratio)
                lor_lb.append(res.log_oddsratio_confint()[0])
                lor_ub.append(res.log_oddsratio_confint()[1])
                pvalues.append(res.oddsratio_pvalue())
            if return_res and len(ts) > 0:
                reses.append(res)
            cts.append(ct)

    sig, pvalues, _, _ = sm.stats.multipletests(pvalues, method="hs")
    if return_res:
        return (
            pd.DataFrame(
                {
                    cell_type_column: cts,
                    "odds_ratio": lodds_ratios,
                    "or_lb": lor_lb,
                    "or_ub": lor_ub,
                    "pvalue": pvalues,
                    "sig": sig,
                }
            ),
            reses,
        )
    else:
        return pd.DataFrame(
            {
                cell_type_column: cts,
                "odds_ratio": lodds_ratios,
                "or_lb": lor_lb,
                "or_ub": lor_ub,
                "pvalue": pvalues,
                "sig": sig,
            }
        )
