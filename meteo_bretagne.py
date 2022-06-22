# !pip install contextily
# !pip install geopandas
# !pip install pygsp
# !pip install loadmydata

import re
from math import asin, cos, radians, sin, sqrt

import contextily as cx
import geopandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.dates import DateFormatter
from pygsp import graphs
from loadmydata.load_molene_meteo import load_molene_meteo_dataset
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform

##

plt.ion()
plt.show()

##

def get_geodesic_distance(point_1, point_2) -> float:
    """
    Calculate the great circle distance (in km) between two points
    on the earth (specified in decimal degrees)

    https://stackoverflow.com/a/4913653
    """

    lon1, lat1 = point_1
    lon2, lat2 = point_2

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def get_exponential_similarity(
    condensed_distance_matrix, bandwidth, threshold
):
    exp_similarity = np.exp(
        -(condensed_distance_matrix ** 2) / bandwidth / bandwidth
    )
    res_arr = np.where(exp_similarity > threshold, exp_similarity, 0.0)
    return res_arr

##

CRS = "EPSG:4326"

STATION_LIST = ["ARZAL","AURAY","BELLE ILE-LE TALUT","BIGNAN","BREST-GUIPAVAS","BRIGNOGAN","DINARD","GUERANDE","ILE DE GROIX","ILE-DE-BREHAT","KERPERT","LANDIVISIAU","LANNAERO","LANVEOC","LORIENT-LANN BIHOUE","LOUARGAT","MERDRIGNAC","NOIRMOUTIER EN","OUESSANT-STIFF","PLEUCADEUC","PLEYBER-CHRIST SA","PLOERMEL","PLOUDALMEZEAU","PLOUGUENAST","PLOUMANAC'H","POMMERIT-JAUDY","PONTIVY","PTE DE CHEMOULIN","PTE DE PENMARCH","PTE DU RAZ","QUIMPER","QUINTENIC","ROSTRENEN","SAINT-CAST-LE-G","SARZEAU SA","SIBIRIL S A","SIZUN","SPEZET","ST BRIEUC","ST NAZAIRE-MONTOIR","ST-SEGAL S A","THEIX","VANNES-SENE",]

##

data_df, stations_df, description = load_molene_meteo_dataset()

##

# only keep a subset of stations
keep_cond = stations_df.Nom.isin(STATION_LIST)
stations_df = stations_df[keep_cond]
keep_cond = data_df.station_name.isin(STATION_LIST)
data_df = data_df[keep_cond].reset_index().drop("index", axis="columns")

# convert temperature from Kelvin to Celsius
data_df["temp"] = data_df.t - 273.15  # temperature in Celsius

# convert pandas df to geopandas df
stations_gdf = geopandas.GeoDataFrame(
    stations_df,
    geometry=geopandas.points_from_xy(
        stations_df.Longitude, stations_df.Latitude
    ),
).set_crs(CRS)

temperature_df = data_df.pivot(
    index="date", values="temp", columns="station_name"
)

# drop the NaNs
temperature_df_no_nan = temperature_df.dropna(axis=0, how="any")

temperature_array = temperature_df_no_nan.to_numpy()

##

stations_np = stations_df[["Longitude", "Latitude"]].to_numpy()
dist_mat_condensed = pdist(stations_np, metric=get_geodesic_distance)
dist_mat_square = squareform(dist_mat_condensed)

##

sigma = np.median(dist_mat_condensed)  # median heuristic

def GrapheBretagne(threshold=0.85):
    adjacency_matrix_gaussian = squareform(
        get_exponential_similarity(dist_mat_condensed, sigma, threshold)
    )
    G_gaussian = graphs.Graph(adjacency_matrix_gaussian)
    print(
        f"The graph is {'not ' if not G_gaussian.is_connected(recompute=True) else ''}connected, with {G_gaussian.N} nodes, {G_gaussian.Ne} edges"
    )
    return G_gaussian

##

def plot_graphe_bretagne(G):
    ax = stations_gdf.geometry.plot(figsize=(10, 7))
    cx.add_basemap(ax, crs=stations_gdf.crs.to_string(), zoom=8)
    ax.set_axis_off()
    G.set_coordinates(stations_np)
    G.plot(ax=ax)

##

W = squareform(
    get_exponential_similarity(dist_mat_condensed, sigma, 0.85)
)
crs = stations_gdf.crs.to_string()

temp = temperature_array.copy()
for i in range(len(temp)):
    temp[i] -= np.average(temp[i])

C = np.corrcoef(temperature_array, rowvar=False) - np.eye(len(W))
t = 0.8
Wc = C * (C >= t)

H = np.zeros_like(W)
for i in range(len(W)):
    for j in range(len(W)):
        if abs(W[i,j]) > 1e-14:
            H[i,j] = C[i,j]

import os
os.makedirs("data", exist_ok=True)

np.savetxt("data/GraphBretagneNN.txt", W, fmt='%.6f')
np.savetxt("data/GraphCoords.txt", stations_np, fmt='%.6f')
np.savetxt("data/Temperature.txt", temperature_array, fmt='%.1f')
open("data/Map.txt", 'w').write(crs)
np.savetxt("data/GraphBretagneCorr.txt", Wc, fmt='%.6f')
np.savetxt("data/GraphBretagneHybr.txt", H, fmt='%.6f')


