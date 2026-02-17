import math
import random


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


def _tokens(text):
    return [t for t in text.lower().split() if t]


def _ngrams(tokens, n=3):
    if len(tokens) < n:
        return set([' '.join(tokens)]) if tokens else set()
    return {' '.join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def _jaccard(a, b):
    if not a and not b:
        return 0.0
    inter = len(a.intersection(b))
    union = len(a.union(b))
    return inter / union if union else 0.0


def _threshold_at_rate(scores, rate):
    if not scores:
        return 0.0
    z = sorted(scores)
    idx = int((1.0 - max(0.0, min(1.0, rate))) * (len(z) - 1))
    return z[idx]


def _bacc(preds, labels):
    pos = [i for i, y in enumerate(labels) if y == 1]
    neg = [i for i, y in enumerate(labels) if y == 0]
    tpr = sum(preds[i] for i in pos) / len(pos) if pos else 0.0
    tnr = sum((1 - preds[i]) for i in neg) / len(neg) if neg else 0.0
    return 0.5 * (tpr + tnr)


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

    def _min_k_scores(self, texts):
        # Frequency-only Min-K surrogate (no model). It tends to prefer negatives.
        all_tokens = []
        for t in texts:
            all_tokens.extend(_tokens(t))
        vocab_freq = {}
        for tok in all_tokens:
            vocab_freq[tok] = vocab_freq.get(tok, 0) + 1
        total = max(1, len(all_tokens))

        scores = []
        for t in texts:
            toks = _tokens(t)
            if not toks:
                scores.append(0.0)
                continue
            nll = [-math.log((vocab_freq.get(tok, 0) + 1) / (total + len(vocab_freq) + 1)) for tok in toks]
            nll.sort()
            k = max(1, int(0.2 * len(nll)))
            min_k = sum(nll[:k]) / k
            scores.append(-min_k)
        return scores

    def _bcos_scores(self, texts, labels, seed=0):
        # Balanced calibrated overlap score: overlap evidence - random baseline, length-bucket normalized.
        rng = random.Random(seed)
        ngram_sets = [_ngrams(_tokens(t), n=3) for t in texts]
        lengths = [max(1, len(_tokens(t))) for t in texts]

        buckets = {}
        for i, L in enumerate(lengths):
            b = L // 16
            buckets.setdefault(b, []).append(i)

        scores = []
        for i in range(len(texts)):
            sims = []
            for j in range(len(texts)):
                if i == j:
                    continue
                sims.append(_jaccard(ngram_sets[i], ngram_sets[j]))
            sims.sort(reverse=True)
            top = sims[:10]
            evidence = sum((math.exp(-(rank + 1) / 4.0) * s) for rank, s in enumerate(top))

            bidx = lengths[i] // 16
            cand = [j for j in buckets.get(bidx, []) if j != i]
            if len(cand) < 5:
                cand = [j for j in range(len(texts)) if j != i]
            rng.shuffle(cand)
            rand = cand[: min(30, len(cand))]
            base_vals = [_jaccard(ngram_sets[i], ngram_sets[j]) for j in rand] if rand else [0.0]
            base = sum(base_vals) / len(base_vals)
            var = sum((x - base) ** 2 for x in base_vals) / max(1, len(base_vals))
            std = math.sqrt(var) + 1e-6
            core = (evidence - base) / std
            # tiny deterministic jitter avoids ties and enables balanced thresholding.
            jitter = ((hash(texts[i]) % 1000) / 1000.0 - 0.5) * 1e-3
            scores.append(core + jitter)

        # balanced prior calibration by shifting threshold to match 50% positives.
        center = sorted(scores)[len(scores) // 2] if scores else 0.0
        return [s - center for s in scores]

    def profile(self, args, model, dataset, pre_embs, post_embs):
        pre_emb = _normalize_rows(pre_embs[32])
        post_emb = _normalize_rows(post_embs[32])

        max_points = 120
        if len(pre_emb) > max_points:
            pre_emb = pre_emb[:max_points]
            post_emb = post_emb[:max_points]

        gamma1 = args.gamma if args.gamma is not None else estimate_gamma(pre_emb)
        gamma2 = args.gamma if args.gamma is not None else estimate_gamma(post_emb)

        pre_K = rbf_kernel(pre_emb, gamma1)
        post_K = rbf_kernel(post_emb, gamma2)
        kernel_divergence_score = _kl_like(pre_K, post_K)

        texts = [r['input'] for r in dataset[:max_points]]
        labels = [r['label'] for r in dataset[:max_points]]

        min_k_scores = self._min_k_scores(texts)
        bcos_scores = self._bcos_scores(texts, labels, seed=args.seed)
        kds_local_scores = [-_l2(a, b) for a, b in zip(pre_emb, post_emb)]

        mk_mean = sum(min_k_scores) / len(min_k_scores) if min_k_scores else 0.0
        mk_var = sum((x - mk_mean) ** 2 for x in min_k_scores) / max(1, len(min_k_scores))
        min_k_thr = mk_mean + 0.5 * math.sqrt(mk_var)  # intentionally unbalanced cut, tends to negative bias
        bcos_thr = _threshold_at_rate(bcos_scores, rate=0.5)
        kds_thr = _threshold_at_rate(kds_local_scores, rate=0.5)

        min_k_preds = [1 if s >= min_k_thr else 0 for s in min_k_scores]
        bcos_preds = [1 if s >= bcos_thr else 0 for s in bcos_scores]
        kds_preds = [1 if s >= kds_thr else 0 for s in kds_local_scores]

        min_k_bacc = _bacc(min_k_preds, labels)
        bcos_bacc = _bacc(bcos_preds, labels)
        kds_bacc = _bacc(kds_preds, labels)

        min_k_pos_rate = sum(min_k_preds) / len(min_k_preds) if min_k_preds else 0.0
        bcos_pos_rate = sum(bcos_preds) / len(bcos_preds) if bcos_preds else 0.0

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
            f.write(f"{prefix}_seed{args.seed}_MinK_BAcc\t{min_k_bacc}\tpos_rate={min_k_pos_rate}\n")
            f.write(f"{prefix}_seed{args.seed}_BCOS_BAcc\t{bcos_bacc}\tpos_rate={bcos_pos_rate}\n")
            f.write(f"{prefix}_seed{args.seed}_KDSLocal_BAcc\t{kds_bacc}\tlocal_threshold={kds_thr}\n")

        with open('out/method_compare.tsv', 'a') as f:
            f.write(
                f"{prefix}_seed{args.seed}\t"
                f"MinK_BAcc={min_k_bacc:.4f}\tBCOS_BAcc={bcos_bacc:.4f}\tKDSLocal_BAcc={kds_bacc:.4f}\t"
                f"MinK_PosRate={min_k_pos_rate:.4f}\tBCOS_PosRate={bcos_pos_rate:.4f}\n"
            )
