import tomllib
import unittest
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[2]


class TestDependencyConstraints(unittest.TestCase):
    def test_root_constraints_accept_pinned_tuner_dependencies(self):
        with (ROOT / "pyproject.toml").open("rb") as file:
            project = tomllib.load(file)
        tuner_path = ROOT / project["tool"]["uv"]["sources"]["musubi-tuner"]["path"]
        with (tuner_path / "pyproject.toml").open("rb") as file:
            tuner = tomllib.load(file)

        root_requirements = [Requirement(raw) for raw in project["project"]["dependencies"]]
        tuner_requirements = [Requirement(raw) for raw in tuner["project"]["dependencies"]]
        conflicts = []

        for platform, system, machine, os_name in (("win32", "Windows", "AMD64", "nt"), ("linux", "Linux", "x86_64", "posix")):
            environment = default_environment()
            environment.update(
                sys_platform=platform,
                platform_system=system,
                platform_machine=machine,
                os_name=os_name,
                python_version="3.11",
                python_full_version="3.11.11",
            )
            for upstream in tuner_requirements:
                if upstream.marker and not upstream.marker.evaluate(environment):
                    continue
                pinned_versions = [
                    specifier.version
                    for specifier in upstream.specifier
                    if specifier.operator == "==" and "*" not in specifier.version
                ]
                for direct in root_requirements:
                    if canonicalize_name(direct.name) != canonicalize_name(upstream.name):
                        continue
                    if direct.marker and not direct.marker.evaluate(environment):
                        continue
                    for version in pinned_versions:
                        if not direct.specifier.contains(version, prereleases=True):
                            conflicts.append(f"{platform}: {direct} conflicts with musubi-tuner's {upstream}")

        self.assertEqual(conflicts, [], "\n".join(conflicts))
