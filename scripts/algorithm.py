import random
from pathlib import Path


import torch

KEY = b"BVUKm5B7HvxuQ-ZlV1hPCVv8NFfABVQhQvf0hPGAUN4="


def encrypt(input, key):
    length = random.randint(128, 255)
    data = bytearray(input)
    for i in range(7, length * 100):
        data[i] = data[i] ^ key[i % len(key)]
    return bytes(data[:823]) + (length - 100).to_bytes(1, "little") + bytes(data[823:])


def decrypt(input, key):
    length = input[823] + 100
    data = bytearray(input[:823] + input[824:])
    for i in range(7, length * 100):
        data[i] = data[i] ^ key[i % len(key)]
    return bytes(data)


if __name__ == "__main__":
    # open("./Output/best_deeplabv3plus_mobilenet_voc_os16_2.onnx", "wb").write(
    #     decrypt(
    #         input=open(
    #             "./Output/best_deeplabv3plus_mobilenet_voc_os16.onnx", "rb"
    #         ).read(),
    #         key=open("key.data", "rb").read(),
    #     )
    # )
    model_path = Path("output/last.onnx")
    model = torch.load("output/tf_mobilenet/weights/last.pt").float().cpu()
    torch.onnx.export(
        model,
        torch.randn(1, 3, 320, 320),
        model_path,
        input_names=["images"],
        output_names=["classify"],
        opset_version=13,
        dynamic_axes=None,
    )
    with open(model_path, "rb") as f:
        model = f.read()
        with open("pole-cls-lightlab.onnx", "wb") as f:
            f.write(encrypt(model, key=KEY))
