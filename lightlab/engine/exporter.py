from typing import Any
import warnings
import torch
from pathlib import Path

from lightlab.utils.torch_utils import smart_inference_mode, get_latest_opset
from lightlab.utils import LOGGER
from lightlab.cfg import Settings, get_cfg
from lightlab.models.detect import DetectionModel
from lightlab.models.modules import DetectHead, C2f


class Exporter:
    def __init__(self, cfg: Settings = Settings(), overrides=None) -> None:
        self.args = get_cfg(cfg, overrides)

    @smart_inference_mode()
    def __call__(self, model_path, output_dir, device="cpu", format="onnx") -> Any:
        if format in ("tensorrt", "trt"):  # 'engine' aliases
            fmt = "engine"

        if self.args.half and format == "onnx" and device == "cpu":
            LOGGER.warning(
                "WARNING ⚠️ half=True only compatible with GPU export, i.e. use device=0"
            )
            self.args.half = False
            assert (
                not self.args.dynamic
            ), "half=True not compatible with dynamic=True, i.e. use only one."
        # check image size

        # Update model
        model = torch.load(model_path)

        model.to(device)
        for p in model.parameters():
            p.requires_grad = False

        model.eval()
        model.float()
        model = model.fuse()

        for m in model.modules():
            if isinstance(m, (DetectHead)):  # Segment and Pose use Detect base class
                m.dynamic = self.args.dynamic
                m.export = True
                m.format = format
            elif isinstance(m, C2f) and format != "tflite":
                # EdgeTPU does not support FlexSplitV while split provides cleaner ONNX graph
                m.forward = m.forward_split

        imgsz = (
            [self.args.imgsz, self.args.imgsz]
            if isinstance(self.args.imgsz, int)
            else self.args.imgsz
        )
        if self.args.dynamic:
            self.args.batch = 1
        im = torch.zeros(self.args.batch, 3, *imgsz).to(device)
        y = None
        for _ in range(2):
            y = model(im)  # dry runs
        if (
            self.args.half
            and (format == "onnx" or format == "engine")
            and device != "cpu"
        ):
            im, model = im.half(), model.half()  # to FP16
        # suppress TracerWarning
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        # suppress shape prim::Constant missing ONNX warning
        warnings.filterwarnings("ignore", category=UserWarning)
        # suppress CoreML np.bool deprecation warning
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        if format == "onnx":
            import onnx

            opset_version = self.args.opset or get_latest_opset()
            LOGGER.info(
                f"Starting export with onnx {onnx.__version__} opset {opset_version}..."
            )
            file = Path(output_dir).joinpath(Path(model_path).name).with_suffix(".onnx")

            output_names = ["output0"]

            dynamic = self.args.dynamic
            if dynamic:
                dynamic = {"images": {0: "batch"}}  # shape(1,3,640,640)
                if isinstance(model, DetectionModel):
                    dynamic["output0"] = {
                        0: "batch",
                        2: "anchors",
                    }  # shape(1, 84, 8400)
            torch.onnx.export(
                model.cpu() if dynamic else model,
                im.cpu() if dynamic else im,
                file,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=["images"],
                output_names=output_names,
                dynamic_axes=dynamic or None,
            )
            model_onnx = onnx.load(file)  # load onnx model

            # Simplify
            if self.args.simplify:
                try:
                    import onnxsim

                    LOGGER.info(f"Simplifying with onnxsim {onnxsim.__version__}...")
                    # subprocess.run(f'onnxsim "{f}" "{f}"', shell=True)
                    model_onnx, check = onnxsim.simplify(model_onnx)
                    assert check, "Simplified ONNX model could not be validated"
                except Exception as e:
                    LOGGER.info(f"Simplifier failure: {e}")

            onnx.save(model_onnx, file)
        elif format == "engine":
            pass


if __name__ == "__main__":
    ex = Exporter(overrides=dict(imgsz=320))
    # ex("output/segment/weights/best.pt", "output/segment/weights")
    # ex("output/coco128/weights/best.pt", "output/coco128/weights")
    ex("output/tf_mobilenet/weights/best.pt", "output/tf_mobilenet/weights")
