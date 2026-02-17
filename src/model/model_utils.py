from model.lite_model import LiteModelWrapper


model_dirs = {
    'mistral': 'mistralai/Mistral-7B-Instruct-v0.2',
    'llama3.1': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'phi3-small': 'microsoft/Phi-3-small-128k-instruct',
}


def load_model(args, model_name=None, peft_path=None):
    model_name = args.model if model_name is None else model_name
    # Dependency-safe fallback model for restricted Codex runtime.
    return LiteModelWrapper(args, model_dirs.get(model_name, model_name))
