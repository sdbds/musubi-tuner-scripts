import ast
import re
import sys
import unittest
from collections import defaultdict
from pathlib import Path
from unittest import mock

import toml


class TestI18nCompleteness(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[2]
    GUI_ROOT = ROOT / "gui"
    SOURCE_ROOTS = (
        GUI_ROOT / "main.py",
        GUI_ROOT / "launch.py",
        GUI_ROOT / "theme.py",
        GUI_ROOT / "components",
        GUI_ROOT / "wizard",
        GUI_ROOT / "utils",
    )

    @classmethod
    def setUpClass(cls):
        if str(cls.GUI_ROOT) not in sys.path:
            sys.path.insert(0, str(cls.GUI_ROOT))
        from utils.i18n import TRANSLATIONS

        cls.translations = TRANSLATIONS

    @classmethod
    def _source_paths(cls):
        for root in cls.SOURCE_ROOTS:
            if root.is_file():
                yield root
                continue
            for path in root.rglob("*.py"):
                if path.name == "i18n.py" or path.name.startswith("demo_"):
                    continue
                yield path

    def test_every_literal_translation_key_exists_for_all_languages(self):
        usages = {}
        for path in self._source_paths():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if not isinstance(node.func, ast.Name) or node.func.id != "t" or not node.args:
                    continue
                key_node = node.args[0]
                if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                    usages.setdefault(key_node.value, []).append(
                        f"{path.relative_to(self.ROOT)}:{node.lineno}"
                    )

        missing = []
        for key, locations in sorted(usages.items()):
            missing_languages = [
                lang for lang, values in self.translations.items() if not values.get(key)
            ]
            if missing_languages:
                missing.append(
                    f"{key}: {', '.join(missing_languages)} ({', '.join(locations)})"
                )

        self.assertEqual(missing, [], "Missing translation entries:\n" + "\n".join(missing))

    def test_visible_translation_text_maps_unambiguously_between_languages(self):
        from utils.i18n import _flatten_translation_strings

        flattened = {
            lang: _flatten_translation_strings(values)
            for lang, values in self.translations.items()
        }
        collisions = []

        for source_lang, source_values in flattened.items():
            for target_lang, target_values in flattened.items():
                if source_lang == target_lang:
                    continue
                targets_by_source = defaultdict(set)
                paths_by_source = defaultdict(list)
                for path, source_text in source_values.items():
                    target_text = target_values.get(path)
                    if not source_text or not target_text or source_text == target_text:
                        continue
                    targets_by_source[source_text].add(target_text)
                    paths_by_source[source_text].append(".".join(path))

                for source_text, targets in targets_by_source.items():
                    if len(targets) > 1:
                        collisions.append(
                            f"{source_lang}->{target_lang} {source_text!r} -> "
                            f"{sorted(targets)!r} ({', '.join(paths_by_source[source_text])})"
                        )

        self.assertEqual(
            collisions,
            [],
            "DOM language sync cannot disambiguate these translations:\n"
            + "\n".join(collisions),
        )

    def test_target_language_pairs_cover_late_content_without_chaining(self):
        from utils.i18n import get_translation_pairs, get_translation_pairs_to_language

        for target_lang in self.translations:
            combined = get_translation_pairs_to_language(target_lang)
            for source_lang in self.translations:
                if source_lang == target_lang:
                    continue
                for source_text, target_text in get_translation_pairs(source_lang, target_lang).items():
                    self.assertEqual(combined[source_text], target_text)

            chained = {
                source_text: (target_text, combined[target_text])
                for source_text, target_text in combined.items()
                if target_text in combined and combined[target_text] != target_text
            }
            self.assertEqual(chained, {}, f"{target_lang} translations must be idempotent")

    def test_components_and_wizard_have_no_untranslated_chinese_ui_literals(self):
        han = re.compile(r"[\u3400-\u9fff]")
        violations = []

        for source_root in (self.GUI_ROOT / "components", self.GUI_ROOT / "wizard"):
            for path in source_root.rglob("*.py"):
                if path.name.startswith("demo_"):
                    continue
                tree = ast.parse(path.read_text(encoding="utf-8"))
                parents = {}
                for parent in ast.walk(tree):
                    for child in ast.iter_child_nodes(parent):
                        parents[child] = parent

                for node in ast.walk(tree):
                    if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                        continue
                    if not han.search(node.value) or self._is_docstring(node, parents):
                        continue
                    if self._is_inside_translation_call(node, parents):
                        continue
                    preview = node.value.replace("\n", " ")[:60]
                    violations.append(
                        f"{path.relative_to(self.ROOT)}:{node.lineno}: {preview}"
                    )

        self.assertEqual(
            violations,
            [],
            "Chinese UI literals must be routed through t():\n" + "\n".join(violations),
        )

    def test_preset_source_labels_follow_the_active_language(self):
        from components import preset_manager
        from utils.i18n import I18n

        manager = preset_manager.PresetManager.__new__(preset_manager.PresetManager)
        manager.scope = "train"
        entries = [
            {"name": "builtin", "label": "Starter", "source": "builtin"},
            {
                "name": "lens_finetune",
                "label": "Lens 微调",
                "label_key": "preset_label_lens_finetune",
                "source": "builtin",
            },
            {"name": "custom", "label": "Custom", "source": "user"},
        ]

        with mock.patch.object(preset_manager.config_manager, "list_config_entries", return_value=entries):
            with mock.patch.object(preset_manager, "t", side_effect=I18n("en").t):
                self.assertEqual(
                    manager._get_preset_options(),
                    {
                        "builtin": "Starter (Built-in)",
                        "lens_finetune": "Lens Fine-tune (Built-in)",
                        "custom": "Custom (User)",
                    },
                )
            with mock.patch.object(preset_manager, "t", side_effect=I18n("zh").t):
                self.assertEqual(
                    manager._get_preset_options(),
                    {
                        "builtin": "Starter (内置)",
                        "lens_finetune": "Lens 微调 (内置)",
                        "custom": "Custom (用户)",
                    },
                )

    def test_chinese_toggle_statuses_remain_localized(self):
        self.assertEqual(self.translations["zh"]["status_on"], "开启")
        self.assertEqual(self.translations["zh"]["status_off"], "已关闭")

    def test_builtin_preset_label_keys_exist_for_all_languages(self):
        missing = []
        for path in (self.GUI_ROOT / "presets").rglob("*.toml"):
            label_key = toml.loads(path.read_text(encoding="utf-8")).get("_label_key")
            if not label_key:
                continue
            missing_languages = [
                lang for lang, values in self.translations.items() if not values.get(label_key)
            ]
            if missing_languages:
                missing.append(
                    f"{path.relative_to(self.ROOT)}: {label_key}: {', '.join(missing_languages)}"
                )

        self.assertEqual(missing, [], "Missing preset label translations:\n" + "\n".join(missing))

    @staticmethod
    def _is_docstring(node, parents):
        parent = parents.get(node)
        if not isinstance(parent, ast.Expr):
            return False
        owner = parents.get(parent)
        return isinstance(owner, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and bool(
            owner.body and owner.body[0] is parent
        )

    @staticmethod
    def _is_inside_translation_call(node, parents):
        current = node
        while current in parents:
            current = parents[current]
            if isinstance(current, ast.Call):
                return isinstance(current.func, ast.Name) and current.func.id == "t"
        return False


if __name__ == "__main__":
    unittest.main()
