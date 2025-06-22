import ir_datasets

def load_datasets():
    """
    Loads the specified datasets and stores them in the database.
    """
    datasets_to_load = [
        "trec-tot/2023/train",
        "antique/train",
    ]

    # map of datasets
    # {"trec-tot/2023/train": ds, "antique/train": ds}
    datasets = {}
    for dataset in datasets_to_load:
        print(f"Loading {dataset}...")
        ds = ir_datasets.load(dataset)
        print(f"{dataset} loaded")
        datasets[dataset] = ds

    return datasets

if __name__ == "__main__":
    datasets = load_datasets()
