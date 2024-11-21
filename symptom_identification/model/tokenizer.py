from transformers import AutoTokenizer

def load_tokenizer(model_name):
    # This function loads a pretrained tokenizer based on the provided model name.
    return AutoTokenizer.from_pretrained(model_name)