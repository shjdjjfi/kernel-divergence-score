import math
from datetime import datetime


def _l2(x, y):
    return math.sqrt(sum((a - b) * (a - b) for a, b in zip(x, y)))


def estimate_gamma(X, quant=0.5):
    dists = []
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            dists.append(_l2(X[i], X[j]))
    if not dists:
        return 1.0
    dists.sort()
    idx = min(len(dists) - 1, int(quant * (len(dists) - 1)))
    q = dists[idx] or 1e-8
    return 1.0 / q


def rbf_kernel(X, gamma):
    K = []
    for i in range(len(X)):
        row = []
        for j in range(len(X)):
            d = _l2(X[i], X[j])
            row.append(math.exp(-gamma * d * d))
        K.append(row)
    return K


def _normalize_rows(X):
    out = []
    for row in X:
        n = math.sqrt(sum(v * v for v in row)) or 1.0
        out.append([v / n for v in row])
    return out


def _kl_like(pre_K, post_K):
    eps = 1e-12
    total = 0.0
    pre_sum = 0.0
    for i in range(len(pre_K)):
        for j in range(len(pre_K[i])):
            p = max(pre_K[i][j], eps)
            q = max(post_K[i][j], eps)
            total += abs(p * (math.log(p) - math.log(q)))
            pre_sum += p
    return -(total / (math.sqrt(pre_sum) or 1.0))


class Profiler:
    def __init__(self, args):
        self.args = args

    def get_embeddings(self, args, model, dataset):
        emb_dict = {i: [] for i in range(33)}
        bs = max(1, int(args.inference_batch_size))
        for start in range(0, len(dataset), bs):
            rows = dataset[start:start + bs]
            texts = [r['input'] for r in rows]
            batch = {'raw_texts': texts}
            hidden_states = model(batch, output_hidden_states=True, hidden_states_layers_to_output=[-1])[0]
            for lidx in range(33):
                emb_dict[lidx].extend(hidden_states[lidx])
        return emb_dict

    def profile(self, args, model, dataset, pre_embs, post_embs):
        pre_emb = _normalize_rows(pre_embs[32])
        post_emb = _normalize_rows(post_embs[32])

        # keep runtime bounded in lightweight mode
        max_points = 120
        if len(pre_emb) > max_points:
            pre_emb = pre_emb[:max_points]
            post_emb = post_emb[:max_points]

        gamma1 = args.gamma if args.gamma is not None else estimate_gamma(pre_emb)
        gamma2 = args.gamma if args.gamma is not None else estimate_gamma(post_emb)

        pre_K = rbf_kernel(pre_emb, gamma1)
        post_K = rbf_kernel(post_emb, gamma2)
        kernel_divergence_score = _kl_like(pre_K, post_K)

        if args.answer_level_shuffling:
            prefix = f"{args.timestamp}\t{args.model}_{args.data}_{args.sub_data}{args.target_num}_{args.split}_{args.contamination}_ALS{args.perturbation}"
        else:
            prefix = f"{args.timestamp}\t{args.model}_{args.data}_{args.sub_data}{args.target_num}_{args.split}_{args.contamination}"

        if args.gamma is not None:
            prefix += f"_gamma={args.gamma}"
        if args.epochs != 1:
            prefix += f"_epoch={args.epochs}"
        if args.sgd:
            prefix += "_sgd"

        with open('out/results.tsv', 'a') as f:
            f.write(f"{prefix}_seed{args.seed}_KDS\t{kernel_divergence_score}\t{str(args)}\n")
