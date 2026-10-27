#!/usr/bin/env python
"""D7 milestone check: the three policies are instantiable.

Three levels, each printed with real output (plan doc lines 319-324):
  L0  the three classes import
  L1  the three Configs build from examples/*/conf/train/*.yaml
  L2  __init__ runs through (random weights, no pretrained load)
  L2+ from_pretrained, only when the checkpoint is actually on disk

Design notes
------------
* L1 feeds the YAML through the SAME path train_pi.py:753-760 uses, i.e.
  TrainConfig.from_hydra_config({train: <yaml>, experiment: {...}}), because the
  YAML carries one ${experiment.exp_dir} interpolation and resolves only inside
  that wrapper. Loading the YAML bare would silently miss that.
* L2 for pi0/pi05 is offline by construction: modeling_pi0.py:341 builds the VLM
  architecture via transformers CONFIG_MAPPING["paligemma"](), so no weights and
  no gated-repo token are needed. Therefore L2 does NOT depend on paligemma.
* L2 for Gr00tN15 sets load_pretrained=False, which routes to
  GR00TN15Config.from_pretrained(base_model_path) + random weights -- real
  architecture config, no weight download.
* Instantiation runs on CPU/meta-free real tensors; device is forced to cpu to
  avoid colliding with the ~38GB/GPU already in use by other tenants.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

RESULTS = []


def record(level, name, ok, detail):
    RESULTS.append({"level": level, "name": name, "ok": ok, "detail": detail})
    print(f"[{level}] {'PASS' if ok else 'FAIL'} {name} :: {detail}", flush=True)


def l0(quiet=False):
    """Resolve the three policy classes.

    quiet=True is used when L2/L2+ were requested without L0: the classes are
    still needed as inputs, but reporting an L0 row the caller did not ask for
    would inflate the pass count.
    """
    specs = [
        ("PI0Policy", "flagos_robo.models.pi0.modeling_pi0", "PI0Policy"),
        ("PI05Policy", "flagos_robo.models.pi05.modeling_pi05", "PI05Policy"),
        ("Gr00tN15", "flagos_robo.models.vla", "Gr00tN15"),
    ]
    out = {}
    import importlib

    for label, mod, attr in specs:
        try:
            m = importlib.import_module(mod)
            cls = getattr(m, attr)
            out[label] = cls
            if not quiet:
                record("L0", label, True, f"{cls.__module__}.{cls.__qualname__}")
        except Exception as e:
            if not quiet:
                record("L0", label, False, f"{type(e).__name__}: {e}")
            else:
                print(f"[L0:prereq] {label} import failed: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
    return out


def l1(examples_root):
    from omegaconf import OmegaConf

    from flagos_robo.train.train_config import TrainConfig

    cases = [
        ("pi0", "pi0/conf/train/pi0.yaml"),
        ("pi0.5", "pi0_5/conf/train/pi0_5.yaml"),
        ("gr00t_n1_5", "gr00t_n1_5/conf/train/gr00t_n1_5.yaml"),
    ]
    out = {}
    for label, rel in cases:
        p = Path(examples_root) / rel
        try:
            train_yaml = OmegaConf.load(p)
            # Same wrapper shape train_pi.py:753-760 receives from the runner.
            full = OmegaConf.create(
                {
                    "train": train_yaml,
                    "experiment": {"exp_dir": "/tmp/d7_exp", "seed": 42},
                }
            )
            tc = TrainConfig.from_hydra_config(full)
            assert tc.model.model_name == label, f"{tc.model.model_name} != {label}"
            # prove the interpolation really resolved, not left as a literal
            od = tc.system.checkpoint.output_directory
            assert "${" not in str(od), f"unresolved interpolation: {od}"
            out[label] = tc
            record(
                "L1",
                label,
                True,
                f"{p.name} model_name={tc.model.model_name} ckpt={tc.model.checkpoint_dir} "
                f"output_directory={od} batch_size={tc.system.batch_size}",
            )
        except Exception as e:
            record("L1", label, False, f"{type(e).__name__}: {e}")
            traceback.print_exc()
    return out


def _features(state_dim, action_dim, cams):
    """Minimal but REAL feature set, matching lerobot/pi0_base config.json shapes.

    Both PI0Config.validate_features and Gr00tN15Config.validate_features need
    input_features populated; normally make_policy derives them from the dataset
    via dataset_to_policy_features(ds_meta.features). D7 is an instantiation
    check, not a data check, so the features are declared directly with the same
    shapes the real checkpoint config.json carries.
    """
    from flagos_robo.models.configs.types import FeatureType, PolicyFeature

    inp = {
        f"observation.images.{c}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))
        for c in cams
    }
    inp["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))
    out = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))}
    return inp, out


def _nparams(m):
    n = sum(p.numel() for p in m.parameters())
    return f"{n / 1e9:.3f}B params" if n >= 1e9 else f"{n / 1e6:.1f}M params"


def l2(classes, groot_base):
    import torch

    from flagos_robo.models.pi0.configuration_pi0 import PI0Config
    from flagos_robo.models.pi05.configuration_pi05 import PI05Config
    from flagos_robo.models.vla.gr00t_n1_5.configuration_gr00t_n1_5 import Gr00tN15Config

    cams = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]

    # --- pi0 / pi05: fully offline. modeling_pi0.py:341 builds the VLM
    # architecture from transformers CONFIG_MAPPING["paligemma"](), so the gated
    # google/paligemma-3b-pt-224 repo is NOT on the instantiation path.
    for label, cfg_cls, cls_key in [
        ("pi0", PI0Config, "PI0Policy"),
        ("pi0.5", PI05Config, "PI05Policy"),
    ]:
        if cls_key not in classes:
            record("L2", label, False, "class unavailable (L0 failed)")
            continue
        try:
            cfg = cfg_cls()
            cfg.device = "cpu"
            cfg.dtype = "float32"
            inp, out = _features(cfg.max_state_dim, cfg.max_action_dim, cams)
            cfg.input_features, cfg.output_features = inp, out
            policy = classes[cls_key](cfg)
            assert isinstance(policy, torch.nn.Module)
            record(
                "L2",
                label,
                True,
                f"{type(policy).__name__} random-init OK, {_nparams(policy)}, "
                f"device={cfg.device} paligemma_variant={cfg.paligemma_variant}",
            )
            del policy
        except Exception as e:
            record("L2", label, False, f"{type(e).__name__}: {e}")
            traceback.print_exc()

    # --- Gr00tN15: load_pretrained=False -> GR00TN15Config.from_pretrained(path)
    # for the real architecture, weights random. Needs the local snapshot dir.
    label, cls_key = "gr00t_n1_5", "Gr00tN15"
    if cls_key not in classes:
        record("L2", label, False, "class unavailable (L0 failed)")
    elif not Path(groot_base).is_dir():
        record("L2", label, False, f"base_model_path missing: {groot_base}")
    else:
        try:
            cfg = Gr00tN15Config(base_model_path=groot_base, load_pretrained=False)
            cfg.device = "cpu"
            inp, out = _features(cfg.max_state_dim, cfg.max_action_dim, cams)
            cfg.input_features, cfg.output_features = inp, out
            policy = classes[cls_key](cfg)
            assert isinstance(policy, torch.nn.Module)
            record(
                "L2",
                label,
                True,
                f"{type(policy).__name__} random-init OK, {_nparams(policy)}, "
                f"load_pretrained=False base={groot_base}",
            )
            del policy
        except Exception as e:
            record("L2", label, False, f"{type(e).__name__}: {e}")
            traceback.print_exc()


def l2plus(classes, ckpt_dirs):
    """L2+ : from_pretrained with the REAL downloaded checkpoint.

    Plan doc line 322: only attempted when the weights are actually on disk.
    A missing dir is reported SKIP, never PASS -- an absent checkpoint must not
    be able to make this level look satisfied.
    """
    from flagos_robo.models.pi0.configuration_pi0 import PI0Config
    from flagos_robo.models.pi05.configuration_pi05 import PI05Config

    for label, cfg_cls, cls_key in [
        ("pi0", PI0Config, "PI0Policy"),
        ("pi0.5", PI05Config, "PI05Policy"),
    ]:
        d = ckpt_dirs.get(label)
        if cls_key not in classes:
            record("L2+", label, False, "class unavailable (L0 failed)")
            continue
        if not d or not (Path(d) / "model.safetensors").is_file():
            RESULTS.append(
                {
                    "level": "L2+",
                    "name": label,
                    "ok": None,
                    "detail": f"SKIP weights not on disk: {d}",
                }
            )
            print(f"[L2+] SKIP {label} :: weights not on disk: {d}", flush=True)
            continue
        try:
            cfg = cfg_cls.from_pretrained(d)
            cfg.device = "cpu"
            cfg.dtype = "float32"
            cfg.pretrained_path = d
            policy = classes[cls_key].from_pretrained(d, config=cfg)
            record(
                "L2+",
                label,
                True,
                f"{type(policy).__name__} from_pretrained({d}) OK, {_nparams(policy)}",
            )
            del policy
        except Exception as e:
            record("L2+", label, False, f"{type(e).__name__}: {e}")
            traceback.print_exc()


def l2plus_groot(classes, groot_base, cams=("base_0_rgb",)):
    """L2+ for Gr00tN15: load_pretrained=True against the real snapshot.

    Separate from l2plus() because GR00T does not take the
    `Policy.from_pretrained(dir, config=cfg)` route -- weights come in through
    the config flag, which makes gr00t_n1.py:711 call the HF
    `super().from_pretrained` on base_model_path.

    This level asserts VALUES, not just "no exception". Two probe tensors are
    compared bit-for-bit against the safetensors shards, and the same probes
    are read off a load_pretrained=False build to prove the comparison can
    fail: without that second half, an all-random model would also report
    "loaded" as long as nothing raised.

    Note on the parameter count: the instance holds 2.414B while the shards
    hold 2.724B. The whole 310,640,640 difference is one tensor,
    backbone.eagle_model.language_model.lm_head.weight (2048 x 151680). GR00T
    uses the VLM as a feature extractor and never projects to vocab, so no
    lm_head is built, and gr00t_n1.py:163 additionally pops LLM layers past
    select_layer. Both are upstream FlagScale behaviour -- diffing the whole
    gr00t_n1_5 subtree against the source tree yields package-rename lines
    only. So the gap is expected, and this check pins it down explicitly
    rather than leaving the number unexplained.
    """
    import glob

    import torch

    label, cls_key = "gr00t_n1_5", "Gr00tN15"
    if cls_key not in classes:
        record("L2+", label, False, "class unavailable (L0 failed)")
        return
    shards = sorted(glob.glob(str(Path(groot_base) / "*.safetensors")))
    if not shards:
        RESULTS.append(
            {
                "level": "L2+",
                "name": label,
                "ok": None,
                "detail": f"SKIP weights not on disk: {groot_base}",
            }
        )
        print(f"[L2+] SKIP {label} :: weights not on disk: {groot_base}", flush=True)
        return
    try:
        from safetensors import safe_open

        from flagos_robo.models.vla.gr00t_n1_5.configuration_gr00t_n1_5 import (
            Gr00tN15Config,
        )

        def _build(load_pretrained):
            cfg = Gr00tN15Config(base_model_path=groot_base, load_pretrained=load_pretrained)
            cfg.device = "cpu"
            inp, out = _features(cfg.max_state_dim, cfg.max_action_dim, list(cams))
            cfg.input_features, cfg.output_features = inp, out
            return classes[cls_key](cfg)

        # One probe from each half of the model, so a backbone-only or
        # action-head-only load would be caught.
        probes = [
            "action_head.model.transformer_blocks.0.attn1.to_q.weight",
            "backbone.eagle_model.language_model.model.layers.0.self_attn.q_proj.weight",
        ]
        where = {}
        for f in shards:
            with safe_open(f, framework="pt") as s:
                for k in s.keys():
                    if k in probes:
                        where[k] = f
        missing = [p for p in probes if p not in where]
        if missing:
            record("L2+", label, False, f"probe tensors absent from shards: {missing}")
            return

        loaded = dict(_build(True).named_parameters())
        random = dict(_build(False).named_parameters())
        n_loaded = sum(p.numel() for p in loaded.values())

        for p in probes:
            key = "_groot_model." + p
            if key not in loaded:
                record("L2+", label, False, f"probe absent from model: {p}")
                return
            with safe_open(where[p], framework="pt") as s:
                want_t = s.get_tensor(p)
            got = loaded[key].detach().to(want_t.dtype)
            rnd = random[key].detach().to(want_t.dtype)
            if not torch.equal(got, want_t):
                record("L2+", label, False, f"probe {p} does not match checkpoint")
                return
            if torch.equal(rnd, want_t):
                record(
                    "L2+", label, False, f"probe {p} matches a random build too -- check is vacuous"
                )
                return

        record(
            "L2+",
            label,
            True,
            f"Gr00tN15 load_pretrained=True({groot_base}) OK, "
            f"{n_loaded / 1e9:.3f}B params, {len(probes)} probe tensors "
            f"bit-identical to shards (and differ under random init)",
        )
    except Exception as e:
        record("L2+", label, False, f"{type(e).__name__}: {e}")
        traceback.print_exc()


def main():
    ap = argparse.ArgumentParser()
    # examples/ is not migrated yet (it lands in batch B7), so L1 still reads the
    # YAML out of the FlagScale reference tree. Verified byte-identical to the
    # host copies. Repoint with --examples-root once B7 moves them in-repo.
    ap.add_argument(
        "--examples-root",
        default="/workspace/flagscale_ref/FlagScale/examples",
        help="dir holding pi0/ pi0_5/ gr00t_n1_5/ with conf/train/*.yaml",
    )
    ap.add_argument("--groot-base", default="/workspace/models/nvidia/GR00T-N1.5-3B")
    ap.add_argument("--pi0-ckpt", default="/workspace/models/lerobot/pi0_base")
    ap.add_argument("--pi05-ckpt", default="/workspace/models/lerobot/pi05_libero_base")
    ap.add_argument("--levels", default="L0,L1,L2,L2+")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    want = {x.strip() for x in args.levels.split(",") if x.strip()}
    print(f"=== D7 instantiation check | levels={sorted(want)} ===", flush=True)

    # L2/L2+ consume the classes L0 resolves. Requesting them without L0 must
    # still resolve the classes (as a silent prerequisite), not report every
    # case as "class unavailable".
    needs_classes = want & {"L0", "L2", "L2+"}
    classes = l0(quiet="L0" not in want) if needs_classes else {}
    if "L1" in want:
        l1(args.examples_root)
    if "L2" in want:
        l2(classes, args.groot_base)
    if "L2+" in want:
        l2plus(classes, {"pi0": args.pi0_ckpt, "pi0.5": args.pi05_ckpt})
        l2plus_groot(classes, args.groot_base)

    passed = sum(1 for r in RESULTS if r["ok"] is True)
    failed = sum(1 for r in RESULTS if r["ok"] is False)
    skipped = sum(1 for r in RESULTS if r["ok"] is None)
    print(f"=== D7 SUMMARY pass={passed} fail={failed} skip={skipped} ===", flush=True)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(RESULTS, indent=2))
        print(f"wrote {args.json_out}", flush=True)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
