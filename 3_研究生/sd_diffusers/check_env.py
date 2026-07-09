import torch
import diffusers


print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print("vram GB:", round(props.total_memory / 1024**3, 2))
print("diffusers:", diffusers.__version__)
