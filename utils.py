import itertools as it
import lzma
import numpy as np
import pickle as pkl
import random
import statistics


def to_sql(sample_attributes, groups):
    return [[sample_attributes[i] + " == '" + group[i]  + "'" for i in range(0, len(group))] for group in groups]


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
    Get the neighbors (i.e., candidate matching records) for every record
    :param ds: the dataset in the dataframe format
    :param candidates: the list of candidate matching pairs of records
    :return: the dictionary of the neighbors of every record and the one of the edge weights
    """

    record_ids = list(ds["_id"])  # all records in the dataset
    neighbors = {record_id: {record_id} for record_id in record_ids}
    edge_weights = {record_id: {record_id: 1.0} for record_id in record_ids}
    weighed = True if len(random.choice(list(candidates))) == 3 else False

    for candidate in candidates:
        for i in range(0, 2):
            pivot = candidate[0] if i == 0 else candidate[1]
            other = candidate[1] if i == 0 else candidate[0]
            neighbors[pivot].add(other)
            edge_weights[pivot][other] = candidate[2] if weighed else 1.0

    return neighbors, edge_weights


def matching(left_id, right_id, gold):
    """
    Check if the pair of records is present in the list of matches obtained using the selected matcher
    :param left_id: the identifier of the left record
    :param right_id: the identifier of the right record
    :param gold: the list of matches obtained using the selected matcher
    :return: a Boolean value denoting if the pair of records is a match
    """

    return (left_id, right_id) in gold or (right_id, left_id) in gold


def find_matching_neighbors(record_id, neighbors, matches, comparisons, num_comparisons, records, gold):
    """
    Find all matches of the current record (proceed recursively by following the matches)
    :param record_id: the identifier of the current record
    :param neighbors: the set of neighbors of the current record
    :param matches: the set of matches of the current record
    :param comparisons: the dictionary to keep track of the performed comparisons
    :param num_comparisons: the number of performed comparisons
    :param records: the sketches of the records
    :param gold: the list of matches obtained using the selected matcher
    :return: the updated versions of matches, comparisons, and num_comparisons
    """

    for neighbor in list(neighbors):
        if neighbor not in matches and neighbor not in comparisons[record_id]:
            num_comparisons += 1
            comparisons[record_id].add(neighbor)
            if neighbor in comparisons.keys():
                comparisons[neighbor].add(record_id)
            else:
                comparisons[neighbor] = {neighbor, record_id}
            if matching(record_id, neighbor, gold):
                matches.add(neighbor)
                matches, comparisons, num_comparisons = find_matching_neighbors(neighbor,
                                                                                records[neighbor]["neighbors"],
                                                                                matches,
                                                                                comparisons,
                                                                                num_comparisons,
                                                                                records,
                                                                                gold)

    return matches, comparisons, num_comparisons


def fusion(ds, cluster, aggregations, time_attribute=None, default_aggregation="vote"):
    """
    Obtain the clean entity from a cluster of matching records
    :param ds: the dataset in the dataframe format
    :param cluster: the identifiers of the matching records
    :param aggregations: the dictionary defining the aggregation function for every attribute
    :param time_attribute: the attribute representing the temporal dimension of the data
    :param default_aggregation: the default aggregation function
    :return: the clean entity as a dictionary
    """

    entity = dict()
    matching_records = ds.loc[ds["_id"].isin(cluster)]

    for attribute, aggregation in aggregations.items():
        if aggregation == "latest" and time_attribute is None:
            aggregation = default_aggregation
        if aggregation == "min":
            entity[attribute] = matching_records[attribute].min()
        elif aggregation == "max":
            entity[attribute] = matching_records[attribute].max()
        elif aggregation == "avg":
            entity[attribute] = round(matching_records[attribute].mean(), 2)
        elif aggregation == "sum":
            entity[attribute] = round(matching_records[attribute].sum(), 2)
        elif aggregation == "vote":
            modes = list(matching_records[attribute].mode())
            entity[attribute] = modes[0] if len(modes) > 0 else np.nan
        elif aggregation == "random":
            entity[attribute] = np.random.choice(matching_records[attribute])
        elif aggregation == "concat":
            entity[attribute] = " ; ".join(matching_records[attribute])
        elif aggregation == "latest":
            attribute_values = list(matching_records.sort_values(by=time_attribute, ascending=False)[attribute])
            entity[attribute] = next((x for x in attribute_values if x is not None), None)

    return entity


def compute_distribution(num_group_entities):
    """
    Compute the distribution from a given count of entities per group
    :param num_group_entities: the number of entities for each group
    :return: the distribution of the groups as a list of ratios that sum to 1
    """

    scale_factor = 1 / sum(num_group_entities)

    return [group * scale_factor for group in num_group_entities]


def detect_distribution(ds, sample_attributes, distribution_type, value_filter, min_support=0.01, top_groups=10):
    """
    Automatically detect the distribution of the groups in the sample attributes
    :param ds: the dataset in the dataframe format
    :param sample_attributes: the attributes used to define the groups
    :param distribution_type: the type of target distribution (i.e., statistical/demographic parity)
    :param value_filter: the dictionary of the values to ignore for each attribute
    :param min_support: the minimum support required to take a group into account
    :param top_groups: the maximum number of groups to take into account
    :return: the groups and their distribution, the maximum size for the sample in case of early stopping
    """

    candidate_groups = list()
    values = list(ds[sample_attributes].dropna().itertuples(index=False, name=None))  # all columns to string
    distinct_values = list(set(values))
    num_records = len(ds)

    for v in list(value_filter.keys()):
        i = sample_attributes.index(v)
        distinct_values = [x for x in distinct_values if x[i] not in value_filter[v]]

    for value in distinct_values:
        num_occurrences = values.count(value)
        support = num_occurrences / num_records
        if support >= min_support:
            candidate_groups.append((value, num_occurrences))
    candidate_groups.sort(key=lambda x: x[1], reverse=True)
    if len(candidate_groups) > top_groups:
        candidate_groups = candidate_groups[:top_groups]
    group_occurrences = [x[1] for x in candidate_groups]

    groups = [x[0] for x in candidate_groups]
    distribution = compute_distribution([1 for _ in range(0, len(groups))]) \
        if distribution_type == "statistical_parity" else compute_distribution(group_occurrences)
    max_sample_size = (min(group_occurrences) * len(groups)) \
        if distribution_type == "statistical_parity" else sum(group_occurrences)

    return groups, distribution, max_sample_size
