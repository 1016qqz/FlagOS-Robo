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

Three guarantees matter after the FlagScale -> FlagOS-Robo rename:

1. A pipeline saved by *this* package reloads. The save side writes
   ``cls.__module__``, so the rename changed the string that lands in the JSON;
   if the module layout and the saved path ever disagree, round-trip breaks.
2. A pipeline saved by *FlagScale* fails loudly. Those configs carry
   ``flagscale.*`` paths that no longer exist. An ``ImportError`` naming the
   stale module is the correct outcome -- silently substituting a different
   class would corrupt training data with no symptom.
3. A multi-stage config instantiates *every* stage. One stage resolving and
   the rest being dropped would look like success to a shape-only check.

Note on the config format: the plan document called for a "minimal pipeline
YAML" fed to ``build_pipeline``. Neither exists -- the configuration format is
the JSON written by ``save_pretrained``, and the entry point is
``from_pretrained``. These tests use the real pair.
"""

import importlib
import inspect
import json

import pytest

from flagos_robo.train.processor import pipeline as pipeline_mod
from flagos_robo.train.processor.pipeline import (
    DataProcessorPipeline,
    IdentityProcessorStep,
    ProcessorStepRegistry,
)

# The only instantiable in-repo ProcessorStep subclass without a registry
# name, hence the only one whose round-trip goes through dynamic import.
UNREGISTERED_STEP = IdentityProcessorStep

# A subclass of a *registered* step, itself never registered -- defined with a
# real class statement at module level so that ``__module__`` points at this
# test module, the way a user-defined subclass behaves. (Building it with
# ``type()`` inside a method sets ``__module__`` to ``abc``, courtesy of
# ABCMeta's calling frame, which made the saved path unresolvable for reasons
# unrelated to what is under test.)
class UnregisteredSubclass(ProcessorStepRegistry.get("device_processor")):  # type: ignore[misc]
    pass


def _save(tmp_path, steps, name="p"):
    DataProcessorPipeline(steps=steps, name=name).save_pretrained(tmp_path)
    return json.loads((tmp_path / f"{name}.json").read_text())


class TestSavedPathTracksTheRename:
    def test_save_writes_the_flagos_robo_module_path(self, tmp_path):
        entry = _save(tmp_path, [UNREGISTERED_STEP()])["steps"][0]

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
        _save(tmp_path, [UNREGISTERED_STEP()])
        back = DataProcessorPipeline.from_pretrained(tmp_path, config_filename="p.json")
        assert type(back.steps[0]) is UNREGISTERED_STEP

    def test_the_saved_path_is_actually_importable(self, tmp_path):
        """Asserting the string starts with ``flagos_robo.`` is not enough --
        a path can be correctly prefixed and still point nowhere."""
        full = _save(tmp_path, [UNREGISTERED_STEP()])["steps"][0]["class"]
        module_path, class_name = full.rsplit(".", 1)
        assert getattr(importlib.import_module(module_path), class_name) is UNREGISTERED_STEP


class TestMultiStageConfigInstantiatesEveryStage:
    """A config with several stages, spanning both resolution tiers and three
    source modules. The failure this catches is partial construction: stage 0
    resolves, the rest are silently dropped or collapsed into one type."""

    STAGE_NAMES = [
        "rename_observations_processor",
        "to_batch_processor_observation",
        "device_processor",
    ]

    def test_every_declared_stage_is_instantiated_in_order(self, tmp_path):
        expected = [ProcessorStepRegistry.get(n) for n in self.STAGE_NAMES]
        # Append the unregistered step so one stage goes through dynamic import
        # and the rest through the registry, in a single config.
        steps = [cls() for cls in expected] + [UNREGISTERED_STEP()]
        cfg = _save(tmp_path, steps, name="multi")

        assert len(cfg["steps"]) == 4
        assert [s.get("registry_name") for s in cfg["steps"][:3]] == self.STAGE_NAMES
        assert "class" in cfg["steps"][3], "the 4th stage must use the import tier"

        back = DataProcessorPipeline.from_pretrained(tmp_path, config_filename="multi.json")
        assert [type(s) for s in back.steps] == expected + [
            UNREGISTERED_STEP
        ], "reloaded stage types or their order do not match the saved config"

    def test_the_stages_span_more_than_one_source_module(self):
        """If every stage came from pipeline.py the test would not exercise
        cross-module resolution, which is where the rename actually bit."""
        modules = {ProcessorStepRegistry.get(n).__module__ for n in self.STAGE_NAMES}
        assert len(modules) >= 3, modules
        assert all(m.startswith("flagos_robo.train.processor.") for m in modules), modules


class TestLegacyConfigsFailLoudly:
    @staticmethod
    def _write(tmp_path, entry):
        (tmp_path / "legacy.json").write_text(json.dumps({"name": "legacy", "steps": [entry]}))

    def test_flagscale_class_path_raises_import_error(self, tmp_path):
        stale = "flagscale.train.processor.pipeline.IdentityProcessorStep"
        self._write(tmp_path, {"class": stale})
        with pytest.raises(ImportError) as exc:
            DataProcessorPipeline.from_pretrained(tmp_path, config_filename="legacy.json")
        assert "flagscale.train.processor.pipeline" in str(exc.value), (
            "the error must name the stale module, otherwise a user staring at a "
            "pre-migration checkpoint has nothing to go on"
        )

    def test_unknown_registry_name_also_raises_import_error(self, tmp_path):
        """The registry tier converts KeyError to ImportError; a caller that
        only catches ImportError must still see registry failures."""
        self._write(tmp_path, {"registry_name": "no_such_step_xyz"})
        with pytest.raises(ImportError):
            DataProcessorPipeline.from_pretrained(tmp_path, config_filename="legacy.json")

    def test_a_real_module_with_a_missing_class_also_raises(self, tmp_path):
        """AttributeError is caught and re-raised as ImportError, so a config
        naming a class that was deleted during migration fails the same way."""
        self._write(tmp_path, {"class": "flagos_robo.train.processor.pipeline.NoSuchStepClass"})
        with pytest.raises(ImportError):
            DataProcessorPipeline.from_pretrained(tmp_path, config_filename="legacy.json")


class TestSubclassOfARegisteredStepRoundTripsAsItself:
    """``register`` sets ``_registry_name`` on the class it decorates, so a
    subclass *inherits* it. Reading that attribute with ``getattr`` made the
    save side treat an unregistered subclass as registered: it wrote the
    parent's registry name, and reload silently produced the parent class.

    Observed before the fix, on a throwaway subclass of ``device_processor``::

        SAVED ENTRY: {'registry_name': 'device_processor', ...}
        SAVED type : SneakySubclass
        RELOAD type: DeviceProcessorStep

    No exception, no warning -- a pipeline that quietly drops a subclass's
    behaviour. The same inheritance blind spot is why the audit below reads
    ``__dict__`` rather than using ``getattr``.
    """

    def test_the_subclass_inherits_the_parents_registry_name(self):
        """The precondition for the bug. If this ever fails the mechanism
        changed and the two tests below are no longer meaningful."""
        assert getattr(UnregisteredSubclass, "_registry_name", None) == "device_processor"
        assert "_registry_name" not in UnregisteredSubclass.__dict__

    def test_it_is_saved_by_class_path_not_the_inherited_name(self, tmp_path):
        entry = _save(tmp_path, [UnregisteredSubclass()], name="sub")["steps"][0]
        assert "registry_name" not in entry, (
            f"saved the inherited registry name {entry.get('registry_name')!r}; "
            "reload will substitute the parent class"
        )
        assert entry["class"].endswith(".UnregisteredSubclass")

    def test_reload_does_not_silently_substitute_the_parent_class(self, tmp_path):
        _save(tmp_path, [UnregisteredSubclass()], name="sub")
        back = DataProcessorPipeline.from_pretrained(tmp_path, config_filename="sub.json")
        assert type(back.steps[0]) is UnregisteredSubclass, (
            f"reloaded as {type(back.steps[0]).__name__}, not the subclass that "
            "was saved -- the parent class was substituted silently"
        )


class TestRegisteredStepsBypassTheDynamicImport:
    def test_registered_steps_carry_their_registry_name(self):
        """This is why only one step is exposed to the rename: every other
        in-repo step is registered, and registry names are rename-proof."""
        name = next(iter(ProcessorStepRegistry.list()))
        assert ProcessorStepRegistry.get(name).__dict__.get("_registry_name") == name

    def test_every_in_repo_step_except_the_known_one_is_registered(self):
        """Guards against a future unregistered step quietly acquiring a
        rename-fragile saved path."""
        # Abstract bases (ObservationProcessorStep and the other eight
        # ProcessorStep+ABC mixins) are excluded: they cannot be instantiated,
        # so they never reach a saved config and the rename cannot bite them.
        unregistered = {
            obj.__name__
            for _, obj in inspect.getmembers(pipeline_mod, inspect.isclass)
            if issubclass(obj, pipeline_mod.ProcessorStep)
            and obj is not pipeline_mod.ProcessorStep
            and not inspect.isabstract(obj)
            # __dict__, not getattr: see TestSubclassOfARegisteredStep... above.
            # getattr would report a subclass of a registered step as
            # registered, hiding the very class this audit exists to surface.
            and obj.__dict__.get("_registry_name") is None
        }
        assert unregistered == {UNREGISTERED_STEP.__name__}, (
            f"unregistered ProcessorStep subclasses changed: {sorted(unregistered)}. "
            "Each one saves a module path that the next rename will break -- either "
            "register it or add it to this test's expected set deliberately."
        )
