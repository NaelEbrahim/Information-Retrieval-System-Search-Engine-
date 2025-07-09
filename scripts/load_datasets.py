import ir_datasets

def load_datasets():
    """
    Loads the specified datasets and stores them in the database.
    """
    datasets_to_load = [
        "trec-tot/2023/train",
        "antique/train",
    ]

    datasets = {}
    for dataset in datasets_to_load:
        print(f"Loading {dataset}...")
        if dataset == "antique/train":
            # تحميل يدوي لملف antique لتفادي مشكلة الترميز
            file_path = r"C:\Users\NAEL PC\.ir_datasets\antique\collection.tsv"
            docs = []
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line_number, line in enumerate(f):
                    try:
                        doc_id, text = line.strip().split('\t')
                        docs.append({'doc_id': doc_id, 'text': text})
                    except ValueError:
                        print(f"Skipping malformed line {line_number + 1}")
            print(f"{dataset} loaded manually with {len(docs)} documents.")
            datasets[dataset] = docs  # نضيفها كمصفوفة وثائق
        else:
            # تحميل عادي باستخدام ir_datasets
            ds = ir_datasets.load(dataset)
            print(f"{dataset} loaded")
            datasets[dataset] = ds

    return datasets

if __name__ == "__main__":
    datasets = load_datasets()
