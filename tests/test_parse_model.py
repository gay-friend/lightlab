from lightlab.models.utils import parse_model

if __name__ == "__main__":
    model, save = parse_model(model_name="lightlab_mobilenetv3", scale="large", nc=10)
    print(save)
    print([m.i for m in model])

    # print(model)
