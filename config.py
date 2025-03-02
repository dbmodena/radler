datasets = ["alaska_cameras", "nc_voters", "nyc_funding"]  # ["alaska_cameras", "nc_voters", "nyc_funding", "nc_voters_10m"]
distribution_types = ["equal_representation"]  # ["equal_representation", "demographic_parity"]
algorithms = ["radler"]  # ["radler", "cost", "benefit", "random"]
sample_sizes = [i for i in range(1, 11)]
num_runs = 10
min_support = 0.01
top_groups = 10
mem_neighbors = "dictionary"
mem_matches = "dictionary"
stochastic_acceptance_size = 100_000
stochastic_acceptance_timeout = 0.1

df_lib = {
    "alaska_cameras": "pandas",
    "nc_voters": "pandas",
    "nc_voters_10m": "polars",
    "nyc_funding": "pandas"
}

sample_attributes = {
    "alaska_cameras": ["brand"],
    "nc_voters": ["sex", "race"],
    "nc_voters_10m": ["suburb"],
    "nyc_funding": ["source"]
}
value_filter = {
    "alaska_cameras": dict(),
    "nc_voters": dict(),
    "nc_voters_10m": dict(),
    "nyc_funding": dict()
}

radler_modes = {
    "radler": "complete",
    "cost": "cost",
    "benefit": "benefit",
    "random": "unweighed"
}

path_dir_ds = "./datasets/"
file_ds = "dataset.csv"
file_gold = "matches.csv"
file_candidates = "blockers/candidates_"
file_blocks = "blockers/blocks_"

er_features = {
    "alaska_cameras": {
        "path_ds": path_dir_ds + "alaska_cameras/" + file_ds,
        "attributes": ["_id", "description", "brand", "model", "type", "mp",
                       "optical_zoom", "digital_zoom", "screen_size", "price"],
        "default_aggregation": "vote",
        "default_fusion": {
            "brand": "vote",
            "model": "vote",
            "type": "vote",
            "mp": "vote",
            "optical_zoom": "vote",
            "digital_zoom": "vote",
            "screen_size": "vote",
            "price": "avg"
        },
        "blockers": {
            "SparkER (Meta-Blocking)": {
                "path_candidates": path_dir_ds + "alaska_cameras/" + file_candidates + "sparker.csv"
            }
        },
        "default_blocker": "SparkER (Meta-Blocking)",
        "matchers": {
            "Ground Truth": {
                "path_gold": path_dir_ds + "alaska_cameras/" + file_gold
            }
        },
        "default_matcher": "Ground Truth"
    },
    "nc_voters": {
        "path_ds": path_dir_ds + "nc_voters/" + file_ds,
        "attributes": ["_id", "first_name", "last_name", "age", "birth_place", "sex", "race", "address",
                       "city", "zip_code", "street_name", "house_number", "party", "registration_date"],
        "default_aggregation": "vote",
        "default_fusion": {
            "first_name": "vote",
            "last_name": "vote",
            "age": "max",
            "birth_place": "vote",
            "sex": "vote",
            "race": "vote",
            "city": "vote",
            "zip_code": "vote",
            "party": "vote",
            "registration_date": "min"
        },
        "blockers": {
            "PyJedAI (Similarity Join)": {
                "path_candidates": path_dir_ds + "nc_voters/" + file_candidates + "pyjedai.csv",
            }
        },
        "default_blocker": "PyJedAI (Similarity Join)",
        "matchers": {
            "Ground Truth": {
                "path_gold": path_dir_ds + "nc_voters/" + file_gold
            }
        },
        "default_matcher": "Ground Truth"
    },
    "nc_voters_10m": {
        "path_ds": path_dir_ds + "nc_voters_10m/" + file_ds,
        "attributes": ["_id", "given_name", "surname", "suburb", "postcode"],
        "default_aggregation": "vote",
        "default_fusion": {
            "given_name": "vote",
            "surname": "vote",
            "suburb": "vote",
            "postcode": "vote"
        },
        "blockers": {
            "Soundex": {
                "path_candidates": path_dir_ds + "nc_voters_10m/" + file_candidates + "soundex.csv",
            }
        },
        "default_blocker": "Soundex",
        "matchers": {
            "Ground Truth": {
                "path_gold": path_dir_ds + "nc_voters_10m/" + file_gold
            }
        },
        "default_matcher": "Ground Truth"
    },
    "nyc_funding" : {
        "path_ds": path_dir_ds + "nyc_funding/" + file_ds,
        "attributes": ["_id", "name", "address", "year", "agency", "source", "counselor", "amount", "status"],
        "default_aggregation": "vote",
        "default_fusion": {
            "name": "vote",
            "address": "vote",
            "source": "vote",
            "amount": "sum"
        },
        "blockers": {
            "SparkER (Meta-Blocking)": {
                "path_candidates": path_dir_ds + "nyc_funding/" + file_candidates + "sparker.csv",
            }
        },
        "default_blocker": "SparkER (Meta-Blocking)",
        "matchers": {
            "Ground Truth": {
                "path_gold": path_dir_ds + "nyc_funding/" + file_gold
            }
        },
        "default_matcher": "Ground Truth"
    }
}
