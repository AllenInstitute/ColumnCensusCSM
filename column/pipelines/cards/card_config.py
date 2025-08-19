from common_setup import project_paths
import seaborn as sns
import numpy as np
from synapse_budget import generate_profile_bins
from itertools import product

data_dir = project_paths.data
layer_bnds = np.load(f"{data_dir}/layer_bounds_v3.npy")
height_bounds = np.load(f"{data_dir}/height_bounds_v1.npy")
width_bounds = np.load(f"{data_dir}/width_bounds_v1.npy")

layer_bnds_orig = layer_bnds.copy()
height_bounds_orig = height_bounds.copy()

layer_bnds = layer_bnds - height_bounds[0]
height_bounds = height_bounds - height_bounds[0]

DEFAULT_E_TYPES = [
    "23P",
    "4P",
    "5P-IT",
    "5P-PT",
    "5P-NP",
    "6P-CT",
    "6P-IT",
    "6P-U",
    "WM-P",
    "Unsure E",
]

DEFAULT_I_TYPES = [
    "BC",
    "BPC",
    "MC",
    "NGC",
    "Unsure I",
]


class CategoryConfig(object):
    def __init__(self, e_types=None, i_types=None):
        if e_types is None:
            e_types = DEFAULT_E_TYPES
        if i_types is None:
            i_types = DEFAULT_I_TYPES

        self.e_types = e_types
        self.i_types = i_types

        self.e_component_data = {
            "dendrite": {"suffix": "basal", "label": "dB"},
            "prox": {"suffix": "prox", "label": "pB"},
            "soma": {"suffix": "soma", "label": "S"},
            "apical": {"suffix": "apical", "label": "A"},
        }

        self.i_component_data = {
            "dendrite": {"suffix": "basal", "label": "B"},
            "soma": {"suffix": "soma", "label": "S"},
        }

    @property
    def compartments(self):
        return ["soma", "prox", "basal", "apical", "other"]

    @property
    def e_type_comps(self):
        out = []
        for ct, comp_dict in product(self.e_types, self.e_component_data):
            comp = self.e_component_data[comp_dict]["suffix"]
            out.append(f"{ct}_{comp}")
        return out

    @property
    def i_type_comps(self):
        out = []
        for ct, comp_dict in product(self.i_types, self.i_component_data):
            comp = self.i_component_data[comp_dict]["suffix"]
            out.append(f"{ct}_{comp}")
        return out

    @property
    def order(self):
        return self.e_types + self.i_types

    @property
    def order_comp(self):
        return self.e_type_comps + self.i_type_comps

    @property
    def n_e_types(self):
        return len(self.e_types)

    @property
    def n_i_types(self):
        return len(self.i_types)

    @property
    def cell_type_comp_order(self):
        return {ct: ii for ii, ct in enumerate(self.order_comp)}

    @property
    def e_groups(self):
        out = []
        for ct in self.e_types:
            out.append(
                {
                    "cts": [
                        f'{ct}_{v["suffix"]}' for k, v in self.e_component_data.items()
                    ],
                    "labels": [v["label"] for k, v in self.e_component_data.items()],
                }
            )
        return out

    @property
    def i_groups(self):
        out = []
        for ct in self.i_types:
            out.append(
                {
                    "cts": [
                        f'{ct}_{v["suffix"]}' for k, v in self.i_component_data.items()
                    ],
                    "labels": [v["label"] for k, v in self.i_component_data.items()],
                }
            )
        return out


class VisualizationConfig(object):
    def __init__(self, e_palette="RdPu", i_palette="Greens"):
        self.axon_color = [0.214, 0.495, 0.721]
        self.dendrite_color = [0.894, 0.103, 0.108]
        self.other_color = [0.8, 0.8, 0.8]

        self.presyn_color = self.axon_color
        self.postsyn_color = self.dendrite_color

        self.e_palette_name = e_palette
        self.i_palette_name = i_palette
        self.e_colors = sns.color_palette(self.e_palette_name, n_colors=9)
        self.i_colors = sns.color_palette(self.i_palette_name, n_colors=9)

        self.base_ind = 5
        self.soma_ind = 8
        self.apical_ind = 3
        self.prox_ind = 6

        self.e_color = self.e_colors[self.base_ind]
        self.i_color = self.i_colors[self.base_ind]

        self.valence_palette = {"Exc": self.e_color, "Inh": self.i_color}

        self.e_basal_color = self.e_color
        self.e_proximal_color = self.e_colors[self.prox_ind]
        self.e_soma_color = self.e_colors[self.soma_ind]
        self.e_apical_color = self.e_colors[self.apical_ind]

        self.i_basal_color = self.i_color
        self.i_soma_color = self.i_colors[self.soma_ind]
        self.i_apical_color = self.i_colors[self.apical_ind]
        self.i_proximal_color = self.i_colors[self.prox_ind]

        self.e_component_palette = {
            "basal": self.e_basal_color,
            "soma": self.e_soma_color,
            "apical": self.e_apical_color,
            "prox": self.e_proximal_color,
            "other": self.other_color,
        }

        self.i_component_palette = {
            "basal": self.i_basal_color,
            "soma": self.i_soma_color,
            "apical": self.i_apical_color,
            "prox": self.i_proximal_color,
            "other": self.other_color,
        }

        self.apical_palette = {
            "Exc": self.e_apical_color,
            "Inh": self.i_apical_color,
        }

        self.soma_palette = {
            "Exc": self.e_soma_color,
            "Inh": self.i_soma_color,
        }

        self.depth_bandwidth = 0.1
        self.markers = ("o", "d", "X")
        self.min_dist_bin_max = 250
        self.dist_bin_spacing = 20

        self.spec_palette = {
            "base": (0.6, 0.6, 0.6),
            "high": (0.986, 0.692, 0.251),
            "low": (0.342, 0.859, 0.851),
        }


class LayerConfig(object):
    def __init__(self):
        self.layers = ["L1", "L2/3", "L4", "L5", "L6", "WM"]
        self.layer_bounds = layer_bnds
        self.layer_bounds_orig = layer_bnds_orig
        # self.layer_bounds = height_bounds
        self.height_bounds = height_bounds
        self.height_bounds_orig = height_bounds_orig

        self.width_bounds = width_bounds
        # self.bins_per_layer = [30]
        self.bins_per_layer = [2, 4, 3, 4, 4]
        # self.layer_bin_names = [
        #     "L1",
        #     "L23 U",
        #     "L23 M",
        #     "L23 D",
        #     "L4 U",
        #     "L4 D",
        #     "L5 U",
        #     "L5 D",
        #     "L6 U",
        #     "L6 D",
        #     "WM",
        # ]
        self.num_bins = 50
        self.layer_bin_names = [str(ii) for ii in range(self.num_bins)]
        self._layer_bins = None

    @property
    def layer_bins(self):
        if self._layer_bins is None:
            # self._layer_bins = generate_profile_bins(
            #     self.layer_bounds, self.height_bounds, self.bins_per_layer
            # )
            self._layer_bins = np.linspace(*height_bounds, self.num_bins)
        return self._layer_bins

    def assign_layer(self, depth_data):
        return [self.layers[x] for x in np.digitize(depth_data, self.layer_bounds)]

    def assign_layer_bin(self, depth_data):
        return np.digitize(depth_data, self.layer_bins[1:])


class remapConfig(object):
    def __init__(self, targets=[], functions=[]):
        self._remap_functions = functions
        self._target_columns = targets

    def add_remap(self, func, column_name):
        self._remap_functions.append(func)
        self._target_columns.append(column_name)

    def remap(self, df):
        df = df.copy()
        for col, func in zip(self._target_columns, self._remap_functions):
            df[col] = df.apply(func, axis=1)
        return df


def _remap_to_6p(row):
    if row["cell_type"] in ["6CT", "6IT"]:
        return "6P"
    else:
        return row["cell_type"]


def SimplifyLayer6Remap(catConfig):
    def _remap_comp_to_6p(row):
        if row["cell_type"] in catConfig.i_types:
            return row["cell_type"]
        else:
            return f'{row["cell_type"]}_{row["compartment"]}'

    return remapConfig(
        targets=["cell_type", "cell_type_comp"],
        functions=[_remap_to_6p, _remap_comp_to_6p],
    )


def BasicLayer6Remap():
    return remapConfig(targets=["cell_type"], functions=[_remap_to_6p])
