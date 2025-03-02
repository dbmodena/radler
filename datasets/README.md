In this folder, we store the datasets used in our experimental evaluation, together with their ground truth and the candidate sets generated as the output of the blocking phase.

For alaska_cameras, nc_voters, and nyc_funding, the respective folder contains:
- "dataset.csv", i.e., the dataset in the CSV format.
- "matches.csv", i.e., the ground truth in the CSV format.
- "blockers", i.e., the folder containing the candidate sets in the CSV format for the blockers described in Section 4.1.1.

For alaska_cameras, in the utils folder we also provide the script for the generation of 25 random sythetic candidate sets with the same recall and cardinality as the ones used in Figure 7.
Note that, in order to use them, their path should be properly added to the "config.py" file.

For nc_voters_10m, we do not report the entire dataset and candidate set due to its size, but only the following scripts:
- "data_preparation.ipynb", to produce the dataset described in Section 4.1.1 from the original dataset with synthetic errors, available at https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution (the dataset should be saved in the utils folder as "dataset_original.csv").
- "blocking_soundex.ipynb", to produce the candidate set described in Section 4.1.1 on the produced dataset.
- "clean_dataset.py", to produce the clean version of the dataset and save it in the "clean_datasets" folder.
