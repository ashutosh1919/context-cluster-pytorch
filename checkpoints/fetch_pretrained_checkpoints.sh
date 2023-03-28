pip install -U --no-cache-dir gdown --pre

cd checkpoints

gdown 1HMEsDt65vKUnf6UVTcCBMMJ4T9XjDIJi -O coc_tiny_plain.pth.tar


# Below checkpoints for other models doesn't support the latest version of code.
# But the original repo authors will upload (might have uploaded) latest checkpoints
# If yes, then you can mention ID below for corresponding models to fetch.

# gdown 1O3On2K9TX4bl4jZoP1SxHM0dNYRBtbA3 -O coc_tiny.pth.tar

# gdown 1e_y9FnP62FxbiNIYv0EgJqQ4C7XYKWB6 -O coc_small.pth.tar

# gdown 1FX5op33LoARcP_ZNhDp4yRUZbdJRFcA0 -O coc_medium.pth.tar