import random


class SimpleDataset:
    def __init__(self, records):
        self.records = list(records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return [r[idx] for r in self.records]
        if isinstance(idx, slice):
            return self.records[idx]
        if isinstance(idx, list):
            return {k: [self.records[i][k] for i in idx] for k in self.records[0].keys()} if self.records else {}
        return self.records[idx]

    def shuffle(self, seed=42):
        rng = random.Random(seed)
        copied = self.records[:]
        rng.shuffle(copied)
        return SimpleDataset(copied)

    @classmethod
    def from_dict(cls, data):
        keys = list(data.keys())
        n = len(data[keys[0]]) if keys else 0
        recs = [{k: data[k][i] for k in keys} for i in range(n)]
        return cls(recs)


def concatenate_datasets(datasets):
    records = []
    for ds in datasets:
        records.extend(ds.records)
    return SimpleDataset(records)


def _synthetic_dataset(name, size=3000):
    records = []
    half = size // 2
    for i in range(size):
        label = 1 if i < half else 0
        records.append(
            {
                "input": f"{name} sample #{i} {'seen' if label else 'unseen'}",
                "label": label,
            }
        )
    return SimpleDataset(records)


def load_data(args, tokenizer, model_name, split):
    # Lite fallback dataset that keeps script runnable in dependency-constrained environments.
    return _synthetic_dataset(args.data)


def get_data_subsets(args, dataset):
    dataset = dataset.shuffle(seed=args.seed)
    labels = dataset['label']
    in_idxs = [idx for idx, x in enumerate(labels) if x == 1]
    out_idxs = [idx for idx, x in enumerate(labels) if x == 0]

    in_num = int(args.target_num * args.contamination)
    out_num = args.target_num - in_num

    in_subset = SimpleDataset.from_dict(dataset[in_idxs[:in_num]]) if in_num > 0 else SimpleDataset([])
    out_subset = SimpleDataset.from_dict(dataset[out_idxs[:out_num]]) if out_num > 0 else SimpleDataset([])
    in_labels = [1] * in_num
    out_labels = [0] * out_num
    return in_subset, out_subset, in_labels, out_labels
