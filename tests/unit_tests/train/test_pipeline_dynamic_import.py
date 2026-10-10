# Copyright (c) 2025 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Regression tests for the one import the closure checker cannot see.

``DataProcessorPipeline._resolve_step_class`` resolves a processor step in two
ways: by registry name, or -- when the step was never registered -- by
``importlib.import_module()`` on a path read out of a saved JSON config. The
second path is invisible to static analysis, so ``check_import_closure.py``
reports it as the single ``opaque`` reference and moves on. Whatever guarantees
it holds have to come from an executed test instead.

Two guarantees matter after the FlagScale -> FlagOS-Robo rename:

1. A pipeline saved by *this* package reloads. The save side writes
   ``cls.__module__``, so the rename changed the string that lands in the JSON;
   if the module layout and the saved path ever disagree, round-trip breaks.
2. A pipeline saved by *FlagScale* fails loudly. Those configs carry
   ``flagscale.*`` paths that no longer exist. An ``ImportError`` naming the
   stale module is the correct outcome -- silently substituting a different
   class would corrupt training data with no symptom.
"""

import json
from pathlib import Path

import pytest

from flagos_robo.train.processor.pipeline import (
    DataProcessorPipeline,
    IdentityProcessorStep,
    ProcessorStepRegistry,
)

# The only in-repo ProcessorStep subclass without a registry name, hence the
# only one whose round-trip goes through the dynamic-import path.
UNREGISTERED_STEP = IdentityProcessorStep


class TestSavedPathTracksTheRename:
    def test_save_writes_the_flagos_robo_module_path(self, tmp_path):
        DataProcessorPipeline(steps=[UNREGISTERED_STEP()], name="p").save_pretrained(tmp_path)
        entry = json.loads((tmp_path / "p.json").read_text())["steps"][0]

        assert "registry_name" not in entry, (
            f"{UNREGISTERED_STEP.__name__} gained a registry name; this test no "
            "longer exercises the dynamic-import path -- point it at another "
            "unregistered step or drop it"
        )
        assert entry["class"].startswith("flagos_robo."), (
            f"saved class path {entry['class']!r} is not under flagos_robo; the "
            "rename did not reach ProcessorStep.__module__"
        )
        assert "flagscale" not in entry["class"]

    def test_round_trip_reconstructs_the_same_class(self, tmp_path):
        DataProcessorPipeline(steps=[UNREGISTERED_STEP()], name="p").save_pretrained(tmp_path)
        back = DataProcessorPipeline.from_pretrained(tmp_path, config_filename="p.json")
        assert type(back.steps[0]) is UNREGISTERED_STEP

    def test_the_saved_path_is_actually_importable(self, tmp_path):
        """Asserting the string starts with ``flagos_robo.`` is not enough --
        a path can be correctly prefixed and still point nowhere."""
        import importlib

        DataProcessorPipeline(steps=[UNREGISTERED_STEP()], name="p").save_pretrained(tmp_path)
        full = json.loads((tmp_path / "p.json").read_text())["steps"][0]["class"]
        module_path, class_name = full.rsplit(".", 1)
        assert getattr(importlib.import_module(module_path), class_name) is UNREGISTERED_STEP


class TestLegacyConfigsFailLoudly:
    @staticmethod
    def _write(tmp_path, class_path):
        (tmp_path / "legacy.json").write_text(
            json.dumps({"name": "legacy", "steps": [{"class": class_path}]})
        )

    def test_flagscale_class_path_raises_import_error(self, tmp_path):
        stale = "flagscale.train.processor.pipeline.IdentityProcessorStep"
        self._write(tmp_path, stale)
        with pytest.raises(ImportError) as exc:
            DataProcessorPipeline.from_pretrained(tmp_path, config_filename="legacy.json")
        assert "flagscale.train.processor.pipeline" in str(exc.value), (
            "the error must name the stale module, otherwise a user staring at a "
            "pre-migration checkpoint has nothing to go on"
        )

    def test_unknown_registry_name_also_raises_import_error(self, tmp_path):
        """The registry tier converts KeyError to ImportError; a caller that
        only catches ImportError must still see registry failures."""
        (tmp_path / "legacy.json").write_text(
            json.dumps({"name": "legacy", "steps": [{"registry_name": "no_such_step_xyz"}]})
        )
        with pytest.raises(ImportError):
            DataProcessorPipeline.from_pretrained(tmp_path, config_filename="legacy.json")


class TestRegisteredStepsBypassTheDynamicImport:
    def test_registered_steps_are_saved_by_name_not_module_path(self, tmp_path):
        """This is why only one step is exposed to the rename: every other
        in-repo step is registered, and registry names are rename-proof."""
        name = next(iter(ProcessorStepRegistry.list()))
        step = ProcessorStepRegistry.get(name)
        assert getattr(step, "_registry_name", None) == name

    def test_every_in_repo_step_except_the_known_one_is_registered(self):
        """Guards against a future unregistered step quietly acquiring a
        rename-fragile saved path."""
        import inspect

        from flagos_robo.train.processor import pipeline as pipeline_mod

        # Abstract bases (ObservationProcessorStep and the other eight
        # ProcessorStep+ABC mixins) are excluded: they cannot be instantiated,
        # so they never reach a saved config and the rename cannot bite them.
        unregistered = {
            obj.__name__
            for _, obj in inspect.getmembers(pipeline_mod, inspect.isclass)
            if issubclass(obj, pipeline_mod.ProcessorStep)
            and obj is not pipeline_mod.ProcessorStep
            and not inspect.isabstract(obj)
            and getattr(obj, "_registry_name", None) is None
        }
        assert unregistered == {UNREGISTERED_STEP.__name__}, (
            f"unregistered ProcessorStep subclasses changed: {sorted(unregistered)}. "
            "Each one saves a module path that the next rename will break -- either "
            "register it or add it to this test's expected set deliberately."
        )
