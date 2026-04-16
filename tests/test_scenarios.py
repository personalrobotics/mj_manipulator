# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for the scenario loader and runner."""

from __future__ import annotations

import textwrap

import pytest

from mj_manipulator import scenarios


@pytest.fixture
def scenarios_dir(tmp_path):
    """Write a few scenario + non-scenario files into a temp dir."""
    # A valid scenario
    (tmp_path / "alpha.py").write_text(
        textwrap.dedent(
            '''
            """Alpha scenario."""

            scene = {"objects": {"can": 1}, "fixtures": {}}

            def do_alpha(robot):
                return "alpha"
            '''
        ).strip()
    )
    # Another valid scenario — no docstring so describe() falls back to stem
    (tmp_path / "beta.py").write_text(
        textwrap.dedent(
            """
            scene = {"objects": {"box": 2}}

            def do_beta(robot):
                return robot
            """
        ).strip()
    )
    # Helper with no `scene =` — should be filtered out by discover()
    (tmp_path / "helpers.py").write_text(
        textwrap.dedent(
            '''
            """Helper module, not a scenario."""

            def build_robot():
                return None
            '''
        ).strip()
    )
    # Dunder file — ignored
    (tmp_path / "__init__.py").write_text("")
    # Underscore-prefixed file — ignored
    (tmp_path / "_private.py").write_text('"""Private."""\n\nscene = {}\n')
    return tmp_path


class TestDiscover:
    def test_finds_scenarios_with_scene_assignment(self, scenarios_dir):
        found = scenarios.discover([scenarios_dir])
        assert set(found.keys()) == {"alpha", "beta"}

    def test_skips_helper_module_without_scene(self, scenarios_dir):
        found = scenarios.discover([scenarios_dir])
        assert "helpers" not in found

    def test_skips_underscore_prefixed(self, scenarios_dir):
        found = scenarios.discover([scenarios_dir])
        assert "_private" not in found

    def test_missing_dir_silently_skipped(self, tmp_path):
        missing = tmp_path / "nope"
        found = scenarios.discover([missing])
        assert found == {}

    def test_multiple_dirs_first_hit_wins(self, tmp_path):
        d1 = tmp_path / "d1"
        d2 = tmp_path / "d2"
        d1.mkdir()
        d2.mkdir()
        (d1 / "alpha.py").write_text('"""First."""\n\nscene = {}\n')
        (d2 / "alpha.py").write_text('"""Second."""\n\nscene = {}\n')
        found = scenarios.discover([d1, d2])
        # First dir's alpha wins
        assert found["alpha"] == d1 / "alpha.py"


class TestDescribe:
    def test_returns_docstring_first_line(self, scenarios_dir):
        desc = scenarios.describe(scenarios_dir / "alpha.py")
        assert desc == "Alpha scenario."

    def test_no_docstring_returns_stem(self, scenarios_dir):
        desc = scenarios.describe(scenarios_dir / "beta.py")
        assert desc == "beta"


class TestLoad:
    def test_load_by_name(self, scenarios_dir):
        mod = scenarios.load("alpha", search_dirs=[scenarios_dir])
        assert mod.scene["objects"] == {"can": 1}

    def test_load_by_path(self, scenarios_dir):
        mod = scenarios.load(str(scenarios_dir / "beta.py"))
        assert mod.scene["objects"] == {"box": 2}

    def test_unknown_raises(self, scenarios_dir):
        with pytest.raises(ValueError, match="not found"):
            scenarios.load("nope", search_dirs=[scenarios_dir])


class TestGetUserFunctions:
    def test_binds_robot_as_first_arg(self, scenarios_dir):
        mod = scenarios.load("alpha", search_dirs=[scenarios_dir])
        fns = scenarios.get_user_functions(mod, robot="ROBOT")
        # do_alpha's first param is named "robot", so it should be bound
        assert "do_alpha" in fns
        assert fns["do_alpha"]() == "alpha"

    def test_binding_preserves_arg(self, scenarios_dir):
        mod = scenarios.load("beta", search_dirs=[scenarios_dir])
        fns = scenarios.get_user_functions(mod, robot="the-robot")
        assert fns["do_beta"]() == "the-robot"

    def test_skips_underscore_prefixed(self, tmp_path):
        p = tmp_path / "s.py"
        p.write_text('"""S."""\n\nscene = {}\n\ndef _hidden(robot):\n    pass\n\ndef visible(robot):\n    pass\n')
        mod = scenarios.load(str(p))
        fns = scenarios.get_user_functions(mod, robot=None)
        assert "_hidden" not in fns
        assert "visible" in fns


class TestResolveSpawnCount:
    def test_returns_spawn_count(self, tmp_path):
        p = tmp_path / "s.py"
        p.write_text('"""S."""\n\nscene = {"objects": {"can": 3}, "spawn_count": 2}\n')
        mod = scenarios.load(str(p))
        assert scenarios.resolve_spawn_count(mod) == 2

    def test_missing_returns_none(self, scenarios_dir):
        mod = scenarios.load("alpha", search_dirs=[scenarios_dir])
        assert scenarios.resolve_spawn_count(mod) is None

    def test_none_scenario_returns_none(self):
        assert scenarios.resolve_spawn_count(None) is None
