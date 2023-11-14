import timm
import torch.utils.model_zoo as model_zoo

model = timm.create_model(
    "tf_mobilenetv3_large_075",
    features_only=True,
    scriptable=True,
    exportable=True,
    # out_indices=(1, 2, 3, 4),
)
print(model.feature_info.channels())
print(model._stage_out_idx)
model._stage_out_idx = {2: 0, 3: 1, 5: 2, 6: 3}
import torch

im = torch.zeros((1, 3, 128, 128))
preds = model(im)
# for pred in preds:
#     print(pred.shape[1])
print(preds[-1].shape[1])
