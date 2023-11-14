class Station:
    def __init__(self) -> None:
        self._obj_dict = {}

    def register(self, obj: object):
        name = obj.name if hasattr(obj, "name") else obj.__name__
        self._obj_dict[name] = obj

    def __getitem__(self, name: str):
        return self._obj_dict[name]()


MODELS = Station()
