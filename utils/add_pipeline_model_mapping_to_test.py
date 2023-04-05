# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A script to add and/or update the attribute `pipeline_model_mapping` in model test files.

This script will be (mostly) used in the following 2 situations:

  - run within a (scheduled) CI job to:
    - check if model test files in the library have updated `pipeline_model_mapping`,
    - and/or update test files and (possibly) open a GitHub pull request automatically
  - being run by a `transformers` member to quickly check and update some particular test file(s)

This script is **NOT** intended to be run (manually) by community contributors.
"""


import argparse
import glob
import inspect
import os
import sys
import unittest


# This is required to make the module import works (when the python process is running from the root of the repo)
sys.path.append(".")

from get_test_info import get_test_classes  # noqa E402
from tests.test_pipeline_mixin import pipeline_test_mapping  # noqa E402


PIPELINE_TEST_MAPPING = {}
for task, _ in pipeline_test_mapping.items():
    PIPELINE_TEST_MAPPING[task] = {"pt": None, "tf": None}


def get_framework(test_class):
    """Infer the framework from the test class `test_class`."""

    if "ModelTesterMixin" in [x.__name__ for x in test_class.__bases__]:
        return "pt"
    elif "TFModelTesterMixin" in [x.__name__ for x in test_class.__bases__]:
        return "tf"
    elif "FlaxModelTesterMixin" in [x.__name__ for x in test_class.__bases__]:
        return "flax"
    else:
        return None


def get_mapping_for_task(task, framework):
    """Get mappings defined in `XXXPipelineTests` for the task `task`."""
    # Use the cached results
    if PIPELINE_TEST_MAPPING[task][framework] is not None:
        return PIPELINE_TEST_MAPPING[task][framework]

    pipeline_test_class = pipeline_test_mapping[task]["test"]
    mapping = None

    if framework == "pt":
        mapping = getattr(pipeline_test_class, "model_mapping", None)
    elif framework == "tf":
        mapping = getattr(pipeline_test_class, "tf_model_mapping", None)

    if mapping is not None:
        mapping = dict(mapping.items())

    # cache the results
    PIPELINE_TEST_MAPPING[task][framework] = mapping
    return mapping


def get_model_for_pipeline_test(test_class, task):
    """Get the model architecture(s) related to the test class `test_class` for a pipeline `task`."""
    framework = get_framework(test_class)
    if framework is None:
        return None
    mapping = get_mapping_for_task(task, framework)
    if mapping is None:
        return None

    config_classes = list({model_class.config_class for model_class in test_class.all_model_classes})
    if len(config_classes) != 1:
        raise ValueError("There should be exactly one configuration class from `test_class.all_model_classes`.")

    # This could be a list/tuple of model classes, but it's rare.
    model_class = mapping.get(config_classes[0], None)
    return model_class


def get_pipeline_model_mapping(test_class):
    """Get `pipeline_model_mapping` for `test_class`."""
    mapping = [(task, get_model_for_pipeline_test(test_class, task)) for task in pipeline_test_mapping]
    mapping = sorted([(task, model) for task, model in mapping if model is not None], key=lambda x: x[0])

    return dict(mapping)


def get_pipeline_model_mapping_string(test_class):
    """Get `pipeline_model_mapping` for `test_class` as a string (to be added to the test file).

    This will be a 1-line string. After this is added to a test file, `make style` will format it beautifully.
    """
    framework = get_framework(test_class)
    if framework == "pt":
        framework = "torch"
    default_value = "{}"

    mapping = get_pipeline_model_mapping(test_class)
    if len(mapping) == 0:
        return ""

    texts = []
    for k, v in mapping.items():
        texts.append(f'"{k}": {v.__name__}')
    text = "{" + ", ".join(texts) + "}"
    text = f"pipeline_model_mapping = {text} if is_{framework}_available() else {default_value}"

    return text


def is_valid_test_class(test_class):
    """Restrict to `XXXModelTesterMixin` and should be a subclass of `unittest.TestCase`."""
    base_class_names = {"ModelTesterMixin", "TFModelTesterMixin", "FlaxModelTesterMixin"}
    if not issubclass(test_class, unittest.TestCase):
        return False
    return len(base_class_names.intersection([x.__name__ for x in test_class.__bases__])) > 0


def find_test_class(test_file):
    """Find a test class in `test_file` to which we will add `pipeline_model_mapping`."""
    test_classes = [x for x in get_test_classes(test_file) if is_valid_test_class(x)]

    target_test_class = None
    for test_class in test_classes:
        # If a test class has defined `pipeline_model_mapping`, let's take it
        if getattr(test_class, "pipeline_model_mapping", None) is not None:
            target_test_class = test_class
            break
    # Take the test class with the shortest name (just a heuristic)
    if target_test_class is None and len(test_classes) > 0:
        target_test_class = sorted(test_classes, key=lambda x: x.__name__)[0]

    return target_test_class


def add_pipeline_model_mapping(test_class, overwrite=False):
    """Add `pipeline_model_mapping` to `test_class`."""
    if getattr(test_class, "pipeline_model_mapping", None) is not None:
        if not overwrite:
            return "", -1

    line_to_add = get_pipeline_model_mapping_string(test_class)
    if len(line_to_add) == 0:
        return "", -1
    line_to_add = line_to_add + "\n"

    # The code defined the class `test_class`
    class_lines, class_start_line_no = inspect.getsourcelines(test_class)
    # `inspect` gives the code for an object, including decorator(s) if any.
    # We (only) need the exact line of the class definition.
    for idx, line in enumerate(class_lines):
        if line.lstrip().startswith("class "):
            break
    class_lines = class_lines[idx:]
    class_start_line_no += idx
    class_end_line_no = class_start_line_no + len(class_lines) - 1

    target_indent = 0
    # The index in `class_lines` that is immediately before the place to which we will add `pipeline_model_mapping`
    target_idx = None
    # If `pipeline_model_mapping` is found in `test_class`.
    def_line = None
    for idx, line in enumerate(class_lines):
        if line.strip().startswith("all_model_classes = "):
            target_indent = len(line) - len(line.lstrip())
            target_idx = idx
        elif line.strip().startswith("all_generative_model_classes = "):
            target_indent = len(line) - len(line.lstrip())
            target_idx = idx
        elif line.strip().startswith("pipeline_model_mapping = "):
            target_indent = len(line) - len(line.lstrip())
            target_idx = idx - 1
            def_line = line

    if target_idx is None:
        return "", -1

    # Remove existing `pipeline_model_mapping`
    if def_line is not None:
        # `target_idx + 1` is the index for the line defining `target_idx + 1`
        for idx, line in enumerate(class_lines[target_idx + 1 :]):
            indent = len(line) - len(line.lstrip())
            if idx == 0 or indent > target_indent or (indent == target_indent and line.strip() == ")"):
                # These lines are going to be removed before writing to the test file.
                class_lines[target_idx + 1 + idx] = None
            elif idx > 0 and indent <= target_indent:
                # Outside the definition block of `pipeline_model_mapping`
                break

    # Add indentation
    line_to_add = " " * target_indent + line_to_add
    # Insert `pipeline_model_mapping` to `class_lines`
    # (The line at `target_idx` should be kept by definition!)
    class_lines = class_lines[: target_idx + 1] + [line_to_add] + class_lines[target_idx + 1 :]
    # Remove the lines that are marked to be removed
    class_lines = [x for x in class_lines if x is not None]

    # Move from test class to module (in order to write to the test file)
    module_lines = inspect.getsourcelines(inspect.getmodule(test_class))[0]
    # Be careful with the 1-off between line numbers and array indices
    module_lines = module_lines[: class_start_line_no - 1] + class_lines + module_lines[class_end_line_no:]
    code = "".join(module_lines)

    moddule_file = inspect.getsourcefile(test_class)
    with open(moddule_file, "w", encoding="UTF-8", newline="\n") as fp:
        fp.write(code)

    return line_to_add


def add_pipeline_model_mapping_to_test_file(test_file, overwrite=False):
    """Add `pipeline_model_mapping` to `test_file`."""
    test_class = find_test_class(test_file)
    if test_class:
        add_pipeline_model_mapping(test_class, overwrite=overwrite)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_file", type=str, help="A path to the test file, starting with the repository's `tests` directory."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="If to check and modify all test files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If to overwrite a test class if it has already defined `pipeline_model_mapping`.",
    )
    args = parser.parse_args()

    if not args.all and not args.test_file:
        raise ValueError("Please specify either `test_file` or pass `--all` to check/modify all test files.")
    elif args.all and args.test_file:
        raise ValueError("Only one of `--test_file` and `--all` could be specified.")

    test_files = []
    if args.test_file:
        test_files = [args.test_file]
    else:
        pattern = os.path.join("tests", "models", "**", "test_modeling_*.py")
        for test_file in glob.glob(pattern):
            # `Flax` is not concerned at this moment
            if not test_file.startswith("test_modeling_flax_"):
                test_files.append(test_file)

    for test_file in test_files:
        add_pipeline_model_mapping_to_test_file(test_file, overwrite=args.overwrite)
