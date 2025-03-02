import itertools as it
import lzma
import pickle as pkl
import polars as pl
import random
from collections import Counter


def to_sql(attribute, value):
    return "%s == '%s'" % (attribute, value)


def blocking(blocker, path_candidates, record_ids):
    """
    Load the candidate matching pairs of records (i.e., candidates) obtained using the selected blocker
    :param blocker: the selected blocking function (i.e., blocker)
    :param path_candidates: the path of the Pickle (LZMA) file containing the candidates for that blocker
    :param record_ids: the list of all record identifiers (to compute the Cartesian product)
    :return: the set of the candidates to be classified by the matcher
    """

    if blocker == "None (Cartesian Product)":
        candidates = set(list(it.combinations(record_ids, 2)))
    else:
        candidates = set(pkl.load(lzma.LZMAFile(path_candidates, "rb")))

    return candidates


def get_neighbors(ds, candidates):
    """
    Collect from the candidates the neighbors of every record
    :param ds: the dataset in the dataframe format
    :param candidates: the list of candidates obtained using the selected blocker
    :return: the dictionaries of neighbors and edge weights
    """

    record_ids = list(ds["_id"])  # all records in the dataset
    neighbors = {record_id: {record_id} for record_id in record_ids}
    edge_weights = {record_id: {record_id: 1.0} for record_id in record_ids}
    weighed = True if len(random.choice(list(candidates))) == 3 else False

    for candidate in candidates:
        neighbors[candidate[0]].add(candidate[1])
        neighbors[candidate[1]].add(candidate[0])
        edge_weights[candidate[0]][candidate[1]] = candidate[2] if weighed else 1.0
        edge_weights[candidate[1]][candidate[0]] = candidate[2] if weighed else 1.0

    return neighbors, edge_weights


def compute_distribution(num_group_entities):
    """
    Compute the distribution from a given count of entities per group
    :param num_group_entities: the number of entities from each group
    :return: the distribution of the groups as a list of ratios that sum to 1
    """

    scale_factor = 1 / sum(num_group_entities)

    return [g * scale_factor for g in num_group_entities]


def detect_distribution(ds, sample_attributes, distribution_type, value_filter, min_support=0.01, top_groups=10):
    """
    Automatically detect the distribution of the groups in the sample attributes
    :param ds: the dataset in the dataframe format
    :param sample_attributes: the attributes used to define the groups
    :param distribution_type: the type of target distribution
    :param value_filter: the dictionary of values to ignore for each attribute
    :param min_support: the minimum support required to take a group into account
    :param top_groups: the maximum number of groups to take into account
    :return: the groups and their distribution, the maximum size for the sample in case of early stopping
    """

    if isinstance(ds, pl.DataFrame):
        values = list(ds[sample_attributes].drop_nulls().iter_rows(named=False))
    else:
        values = list(ds[sample_attributes].dropna().itertuples(index=False, name=None))

    distinct_values = list(set(values))
    num_records = len(ds)

    for v in list(value_filter.keys()):
        i = sample_attributes.index(v)
        distinct_values = [x for x in distinct_values if x[i] not in value_filter[v]]

    value_counts = Counter(values)
    candidate_groups = [(value, count) for value, count in value_counts.items() if count / num_records >= min_support]
    candidate_groups.sort(key=lambda x: x[1], reverse=True)
    if len(candidate_groups) > top_groups:
        candidate_groups = candidate_groups[:top_groups]
    group_occurrences = [x[1] for x in candidate_groups]

    groups = [x[0] for x in candidate_groups]
    distribution = compute_distribution([1 for _ in range(0, len(groups))]) \
        if distribution_type == "equal_representation" else compute_distribution(group_occurrences)
    max_sample_size = (min(group_occurrences) * len(groups)) \
        if distribution_type == "equal_representation" else sum(group_occurrences)

    return groups, distribution, max_sample_size
