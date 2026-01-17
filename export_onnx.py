import torch

from model import EMG3DCNNRegressor

model = EMG3DCNNRegressor(in_ch=12, base=16, dropout=0.15)
data = torch.load("runs/emg_20260117_140547/best.pt", map_location="cpu")
model.load_state_dict(data["model_state"])

model.eval()

# batch, time, channels, height, width
dummy_input = torch.randn(1, 100, 12, 8, 8)

onnx_path = "model.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch"},
        "output": {0: "batch"},
    },
)

print("Saved:", onnx_path)
