import config
import csv
import math
import pandas as pd
import polars as pl
import radler
from utils import compute_distribution, detect_distribution


class Task:
    """
    The object describing the task at hand
    """

    def __init__(self, task_id, run_id, dataset, algorithm, mode, sample_attributes, groups, target_distribution,
                 sample_size=math.inf):

        self.task_id = task_id
        self.run_id = run_id

        # Dataset
        self.ds = dataset
        self.path_ds = config.er_features[self.ds]["path_ds"]

        # Blocking function (i.e., blocker)
        self.blocker = config.er_features[self.ds]["default_blocker"]
        self.path_candidates = config.er_features[self.ds]["blockers"][self.blocker]["path_candidates"]
        self.mem_neighbors = config.mem_neighbors
        self.neighbors = None

        # Matching function (i.e., matcher)
        self.matcher = config.er_features[self.ds]["default_matcher"]
        self.path_gold = config.er_features[self.ds]["matchers"][self.matcher]["path_gold"]
        self.mem_matches = config.mem_matches
        self.matches = None

        # Aggregation functions (i.e., aggregations)
        self.default_aggregation = config.er_features[self.ds]["default_aggregation"]
        self.aggregations = config.er_features[self.ds]["default_fusion"]

        # Attribute projection
        self.attributes = list(self.aggregations.keys())

        # Sampling
        self.algorithm = algorithm
        self.mode = mode
        self.sample_attributes = sample_attributes
        self.num_attributes = len(self.sample_attributes)
        self.groups = groups
        self.num_groups = len(self.groups)
        self.target_distribution = compute_distribution(target_distribution)
        self.sample_size = sample_size
        self.stochastic_acceptance_size = config.stochastic_acceptance_size
        self.stochastic_acceptance_timeout = config.stochastic_acceptance_timeout

    def set_neighbors(self, neighbors):
        self.neighbors = neighbors

    def set_matches(self, matches):
        self.matches = matches


class Run:
    """
    The object used to collect the parameters and the metrics of the current run
    """

    def __init__(self, task_id, run_id, algorithm, mode, dataset, sample_attributes, groups, min_support, top_groups,
                 distribution_type, target_distribution, sample_size):
        self.task_id = task_id  # identifier of the task at hand
        self.run_id = run_id  # identifier of the current run of the task at hand
        self.algorithm = algorithm
        self.mode = mode
        self.dataset = dataset
        self.sample_attributes = sample_attributes
        self.num_sample_attributes = len(self.sample_attributes)
        self.groups = groups
        self.num_groups = len(self.groups)
        self.min_support = min_support  # criteria used to determine the groups
        self.top_groups = top_groups  # criteria used to determine the groups
        self.distribution_type = distribution_type
        self.target_distribution = target_distribution
        self.sample_size = sample_size  # sample size (as the ratio over the maximum number of entities)
        self.num_entities = None
        self.num_group_entities = None
        self.num_comparisons = None
        self.num_iterations = None
        self.setup_time = None
        self.cleaning_time = None
        self.tot_time = None
        self.avg_cluster_size = None
        self.num_cleaned_entities = None
        self.progressive_recall = None


def load_matches(task, ds, id_mapping):

    reader = csv.reader(open(task.path_gold, mode="r"))
    next(reader)  # skip the header row

    if task.mem_matches == "dictionary":
        task.matches = [set() for _ in range(len(ds["_id"]))]

    for row in reader:
        l_id, r_id = row
        (l_id, r_id) = (id_mapping[l_id], id_mapping[r_id])

        if task.mem_matches == "dictionary":
            task.matches[l_id].add(r_id)
            task.matches[r_id].add(l_id)


def load_neighbors(task, ds, id_mapping):

    reader = csv.reader(open(task.path_candidates, mode="r"))
    next(reader)  # skip the header row

    if task.mem_neighbors == "dictionary":
        task.neighbors = [{i} for i in range(len(ds["_id"]))]

    for row in reader:
        l_id, r_id = row
        (l_id, r_id) = (id_mapping[l_id], id_mapping[r_id])

        if task.mem_neighbors == "dictionary":
            task.neighbors[l_id].add(r_id)
            task.neighbors[r_id].add(l_id)


def run(algorithm, mode, dataset, sample_attributes, min_support, top_groups, distribution_type, sample_size,
        value_filter=dict(), task_id=None, run_id=None):

    # Select the groups from the clean dataset and define their distribution
    if config.df_lib[dataset] == "polars":
        ds = pl.read_csv(f"clean_datasets/{dataset}.csv")
        ds = ds.rename({c: c.lower() for c in ds.columns}) \
             .with_columns([pl.col(c).str.to_lowercase() for c in ds.columns if ds[c].dtype == pl.Utf8])
    else:
        ds = pd.read_csv(f"clean_datasets/{dataset}.csv", low_memory=False)
        ds.columns = ds.columns.str.lower()
        for column in ds.columns:
            if ds[column].dtype == "object":
                ds[column] = ds[column].str.lower()
    groups, target_distribution, max_sample_size = detect_distribution(ds, sample_attributes, distribution_type,
                                                                       value_filter, min_support, top_groups)

    # Initialize the Task object
    sample_size = math.floor(sample_size * max_sample_size) if 0 <= sample_size <= 1 else sample_size
    task = Task(task_id, run_id, dataset, algorithm, mode, sample_attributes, groups, target_distribution,
                sample_size=sample_size)
    run = Run(task_id, run_id, algorithm, mode, dataset, sample_attributes, groups, min_support, top_groups,
              distribution_type, target_distribution, sample_size)

    # Load the dataset in the dataframe format
    if config.df_lib[dataset] == "polars":
        ds = pl.read_csv(task.path_ds)
        id_mapping = dict(zip(ds["_id"].to_list(), range(len(ds))))
        ds = ds.with_columns(pl.Series("_id", range(len(ds))))
        ds = ds.rename({c: c.lower() for c in ds.columns}) \
             .with_columns([pl.col(c).str.to_lowercase() for c in ds.columns if ds[c].dtype == pl.Utf8])
    else:
        ds = pd.read_csv(task.path_ds, low_memory=False)
        id_mapping = dict(zip(ds["_id"], range(len(ds))))
        ds["_id"] = range(len(ds))
        ds.columns = ds.columns.str.lower()
        for column in ds.columns:
            if ds[column].dtype == "object":
                ds[column] = ds[column].str.lower()

    # Load the neighbors and the matches
    load_matches(task, ds, id_mapping)
    load_neighbors(task, ds, id_mapping)

    # Perform entity resolution
    results, run_stats = radler.run(task, ds, run_stats=run)

    # Print the dataframe containing the resulting entities
    if len(results.index) > 0:
        attributes = task.attributes + ["matches", "num_comparisons", "time"]
        print("\n")
        print(results[attributes])
        results[attributes].to_csv("results.csv", index=False)

    return results, run_stats


def main():
    task_id = 0
    stats = list()
    for dataset in config.datasets:
        for distribution_type in config.distribution_types:
            for algorithm in config.algorithms:
                mode = config.radler_modes[algorithm] if algorithm in config.radler_modes else None
                algorithm = "radler" if algorithm in config.radler_modes else algorithm
                for sample_size in config.sample_sizes:
                    sample_size = sample_size / 10
                    for run_id in range(0, config.num_runs):
                        print({"task_id": task_id, "algorithm": algorithm, "mode": mode, "dataset": dataset,
                               "distribution_type": distribution_type, "sample_size": sample_size, "run_id": run_id})
                        results, run_stats = run(task_id=task_id, run_id=run_id, algorithm=algorithm, mode=mode,
                                                 dataset=dataset, sample_attributes=config.sample_attributes[dataset],
                                                 min_support=config.min_support, top_groups=config.top_groups,
                                                 distribution_type=distribution_type, sample_size=sample_size,
                                                 value_filter=config.value_filter[dataset])
                        stats.append(run_stats.__dict__)
                    task_id += 1
    stats = pd.DataFrame(stats)
    stats.to_csv("run_stats.csv", index=False)


if __name__ == "__main__":
    main()
