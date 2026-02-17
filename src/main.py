import argparse
import os
import random
from datetime import datetime

from data.data_utils import concatenate_datasets, get_data_subsets, load_data
from model.model_utils import load_model
from profiler import Profiler


def train_model(args, model, dataset):
    # Lite training step to preserve pipeline semantics without heavy ML deps.
    if hasattr(model, "fit"):
        model.fit(dataset, args.contamination)
    return model


def main(args):
    model = load_model(args)

    full_dataset = load_data(args, model.tokenizer, args.model, split=args.split)
    in_dataset, out_dataset, _, _ = get_data_subsets(args, full_dataset)
    dataset = concatenate_datasets([in_dataset, out_dataset])

    profiler = Profiler(args)
    pre_embs = profiler.get_embeddings(args, model, dataset)
    model = train_model(args, model, dataset)
    post_embs = profiler.get_embeddings(args, model, dataset)
    profiler.profile(args, model, dataset, pre_embs, post_embs)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--out_dir', type=str, default="")

    parser.add_argument('--model', type=str, default='llama3')
    parser.add_argument('--model_dir', type=str, default="/nobackup2/froilan/datasets/")
    parser.add_argument('--memory_for_model_activations_in_gb', type=int, default=4)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dim', type=int, default=8)
    parser.add_argument('--verbose', action='store_true')

    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--inference_batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--sgd', action='store_true')

    parser.add_argument('--data_dir', type=str, default="/nobackup2/froilan/datasets/")
    parser.add_argument('--data', type=str, default='beavertails')
    parser.add_argument('--sub_data', type=str, default='')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--target_num', type=int, default=1000)
    parser.add_argument('--contamination', type=float, default=-1)
    parser.add_argument('--cpu_profiler', action='store_true')

    parser.add_argument('--perturbation', type=float, default=0.05)
    parser.add_argument('--answer_level_shuffling', action='store_true')
    parser.add_argument('--synonym_replacement', action='store_true')
    parser.add_argument('--random_deletion', action='store_true')
    parser.add_argument('--word_level_shuffling', action='store_true')

    parser.add_argument('--gamma', type=float)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    random.seed(args.seed)

    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    args.timestamp = timestamp

    if os.path.exists('token'):
        with open('token', 'r') as f:
            args.token = f.read().strip()
    else:
        args.token = ''

    os.makedirs('out', exist_ok=True)
    main(args)
