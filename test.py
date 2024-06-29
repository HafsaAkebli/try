import timm

# List all available models in timm
available_models = timm.list_models()
print([model for model in available_models if 'xcit' in model])