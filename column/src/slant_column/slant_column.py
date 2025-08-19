import pandas as pd
import numpy as np
import os

bbox_df = pd.read_pickle(f"{os.path.dirname(__file__)}/bbox_df.pkl")
param_df = pd.read_pickle(f"{os.path.dirname(__file__)}/slant_param_df.pkl")
unit_vec = param_df.iloc[0]["unit_vec"]
start_depth = param_df.iloc[0]["start_depth"]

def bounds_slant(y, unit_vec, bbox_df=bbox_df):
    """
    Get bounds for the slant box at a specific depth.


    """
    if y < start_depth:
        return bbox_df.bbox_a[0], bbox_df.bbox_b[0]
    else:
        bounds_a_0 = np.array(
            [bbox_df.bbox_a[0][0], start_depth, bbox_df.bbox_a[0][2]]
        ) * [4, 4, 40]
        bounds_b_0 = np.array(
            [bbox_df.bbox_b[0][0], start_depth, bbox_df.bbox_b[0][2]]
        ) * [4, 4, 40]

        unit_vec_theta = np.arctan(
            np.sqrt(unit_vec[0] ** 2 + unit_vec[2] ** 2) / unit_vec[1]
        )
        delta_x = 4 * (y - start_depth) / np.cos(unit_vec_theta) * unit_vec
        bounds_a = bounds_a_0 + delta_x
        bounds_b = bounds_b_0 + delta_x
    return bounds_a / [4, 4, 40], bounds_b / [4, 4, 40]


def in_slant_box(row, unit_vec, start_depth, bbox_df, deepest_point):
    y = row["point_y"]
    if y > deepest_point:
        return False
    bounds_a, bounds_b = bounds_slant(y, unit_vec, start_depth, bbox_df)
    true_x = np.logical_and(row["point_x"] > bounds_a[0], row["point_x"] <= bounds_b[0])
    true_z = np.logical_and(row["point_z"] > bounds_a[2], row["point_z"] <= bounds_b[2])
    return true_x & true_z


# with open(f'{os.path.dirname(__file__)}/bounds_slant.pkl', 'br') as f:
#     bounds_slant = pickle.load(f)

# with open(f'{os.path.dirname(__file__)}/in_slant_box.pkl', 'br') as f:
#     in_slant_box = pickle.load(f)


def point_in_slant(xyz):
    row = {
        "point_x": xyz[0],
        "point_y": xyz[1],
        "point_z": xyz[2],
    }
    return in_slant_box(
        row,
        unit_vec,
        param_df.iloc[0]["start_depth"],
        bbox_df,
        deepest_point=param_df.iloc[0]["deepest_point"],
    )


def dist_from_slant_line(xyz, anchor_xyz):
    y = xyz[:,1]
    y_above = y > 4 * start_depth
    del_xz = np.zeros((len(y_above),3))
    
    unit_vec_theta = np.arctan(
        np.sqrt(unit_vec[0] ** 2 + unit_vec[2] ** 2) / unit_vec[1]
    )

    slant_xz = np.atleast_2d((y - 4*start_depth)).T / np.cos(unit_vec_theta) * np.atleast_2d(unit_vec)
    
    if anchor_xyz[1] < 4 * start_depth:
        slant_offset = np.array([0,0,0])
    else:
        slant_offset = (anchor_xyz[1] - 4*start_depth) / np.cos(unit_vec_theta) * unit_vec

    del_xz[y_above] = slant_xz[y_above] 
    del_xz = anchor_xyz[[0,2]] + del_xz[:, [0,2]] - slant_offset[[0,2]]
    rad_xz = xyz[:, [0,2]] - del_xz
    return np.linalg.norm(rad_xz, axis=1)