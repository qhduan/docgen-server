import os
import time

import numpy as np
from tqdm import tqdm
from onnxruntime import (
    GraphOptimizationLevel, InferenceSession,
    SessionOptions, get_all_providers
)
from transformers import AutoTokenizer


def create_model_for_provider(
    model_path: str,
    provider: str = 'CPUExecutionProvider'
) -> InferenceSession:
    assert provider in get_all_providers(), \
        f"provider {provider} not found, {get_all_providers()}"
    # Few properties that might have an impact on performances (provided by MS)
    options = SessionOptions()
    options.intra_op_num_threads = int(os.environ.get('NUM_THREADS', 4))
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    # Load the model as a graph and prepare the CPU backend
    session = InferenceSession(model_path, options, providers=[provider])
    session.disable_fallback()
    return session


def predict(model, pkv_init, context, max_length=128):
    tokenized = tokenizer(context)
    input_ids = tokenized['input_ids']
    pkv = pkv_init
    attention_mask = [1.0] * (len(input_ids) + pkv.shape[4])
    output_tokens = []
    for _ in range(max_length):
        input_ = np.array([input_ids]).astype(np.int64)
        out, pkv = model.run(['output', 'pkv_output'], {
            'input': input_,
            'attention_mask': np.array([attention_mask]).astype(np.float32),
            'pkv': pkv
        })
        token = out[:, -1, :].argmax()
        input_ids = [token]
        output_tokens.append(token)
        attention_mask.append(1.0)
        if tokenizer.decode(output_tokens).endswith('"""'):
            break
    return tokenizer.decode(output_tokens)


def infer(context, pkv_name=None):
    pkv_init = pkvs.get(pkv_name, pkvs['google'])
    return predict(onnx_model, pkv_init, context)


current_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
tokenizer = AutoTokenizer.from_pretrained(os.path.join(current_dir, 'models/tokenizer'))
onnx_model = create_model_for_provider(os.path.join(current_dir, 'models/python_docgen.quant.onnx'))
pkvs = {
    'google': np.load(os.path.join(current_dir, 'models/python_pkv_google.npy')),
    'google_cn': np.load(os.path.join(current_dir, 'models/python_pkv_google.npy')),
    'default': np.load(os.path.join(current_dir, 'models/python_pkv_endoftext.npy')),
}
