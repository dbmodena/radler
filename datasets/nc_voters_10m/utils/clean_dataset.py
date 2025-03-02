import networkx as nx
import numpy as np
import pandas as pd
import polars as pl
import config
from collections import namedtuple
from tqdm import tqdm


def fusion(ds, cluster, id_index):
    """
    Produce the clean entity from a cluster of matching records
    :param ds: the dataset in the dataframe format
    :param cluster: the identifiers of the matching records
    :return: the clean entity as a dictionary
    """

    entity = dict()

    r_id = cluster.pop()
    record = dict(zip(ds.columns, ds.row(id_index[r_id])))
    for attribute, aggregation in config.er_features["nc_voters_10m"]["default_fusion"].items():
        entity[attribute] = record[attribute]

    entity["matches"] = cluster

    return entity


def main():
    ds = pl.read_csv("../dataset.csv")

    id_index = {row["_id"]: i for i, row in enumerate(ds.iter_rows(named=True))}

    matches = list(pd.read_csv("../matches.csv").itertuples(index=False, name=None))

    g = nx.Graph()
    g.add_nodes_from(list(ds["_id"]))
    g.add_edges_from(matches)

    clusters = [x for x in nx.connected_components(g)]

    unique = {c.pop() for c in clusters if len(c) == 1}

    cols = list(config.er_features["nc_voters_10m"]["default_fusion"].keys()) + ["matches"]

    unique_records = ds.filter(pl.col("_id").is_in(unique))
    unique_records = unique_records.with_columns(pl.col("_id").apply(lambda x: {x}).alias("matches")).select(cols)

    non_unique = [c for c in clusters if len(c) > 1]

    entities = [fusion(ds, c, id_index) for c in tqdm(non_unique, desc="Processing clusters")]

    clean_ds = pl.concat([pl.DataFrame(entities), unique_records])
    clean_ds = clean_ds.with_columns(pl.col("matches").apply(lambda x: str(x)).alias("matches")).to_pandas()

    clean_ds.to_csv("../../../clean_datasets/nc_voters_10m.csv", index=False)


if __name__ == "__main__":
    main()
