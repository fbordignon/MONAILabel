# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import json


def is_subclass(n, o, base_c):
    if inspect.isclass(o) and n != base_c:
        b = [cls.__name__ for cls in o.__bases__]
        if base_c in b:
            return True
    return False


def get_class_of_subclass(module, base_classes) -> dict:
    print(f"{module} => {base_classes}")
    res = dict()
    for n, o in inspect.getmembers(module):
        if not inspect.isclass(o) or inspect.isabstract(o):
            continue

        # print(f"{n} => {o}")
        for base_c in base_classes:
            if is_subclass(n, o, base_c):
                cp = f"{o.__module__}.{o.__name__}"
                if res.get(cp):
                    res[cp]["alias"].append(n)
                else:
                    res[cp] = {
                        "name": o.__name__,
                        "alias": [n],
                        "class": cp,
                        "module": o.__module__,
                        "dictionary": o.__module__.endswith("dictionary"),
                        "base_class": base_c,
                        "category": ".".join(o.__module__.split(".")[:3]),
                    }
                break

    sorted_d = dict()
    for k in sorted(res.keys()):
        v = res[k]
        v["alias"] = sorted(v["alias"])
        sorted_d[k] = v

    return sorted_d


class MonaiUtils:
    @staticmethod
    def version():
        try:
            import monai

            return monai.__version__
        except ImportError:
            return ""

    @staticmethod
    def list_transforms():
        import monai.transforms as mt

        return get_class_of_subclass(mt, ["Transform", "MapTransform"])


def main():
    mutil = MonaiUtils()
    transforms = mutil.list_transforms()

    for t in transforms:
        print(f"{t} => {transforms[t]['category']}")

    categories = list({v["category"] for v in transforms.values()})
    print(json.dumps(categories, indent=2))


if __name__ == "__main__":
    main()
