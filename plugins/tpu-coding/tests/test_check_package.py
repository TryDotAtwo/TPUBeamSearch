import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "check_package.py"
SPEC = importlib.util.spec_from_file_location("check_package", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_package(root: Path) -> Path:
    (root / ".codex-plugin").mkdir(parents=True)
    (root / "skills" / "code-tpu" / "references").mkdir(parents=True)
    (root / ".codex-plugin" / "plugin.json").write_text(json.dumps({
        "name": "tpu-coding", "version": "1.0.0", "description": "TPU helper",
        "skills": ["skills/code-tpu"],
    }), encoding="utf-8")
    (root / "skills" / "code-tpu" / "SKILL.md").write_text(
        "# TPU\n\n[Evidence](references/evidence.json)\n", encoding="utf-8")
    (root / "skills" / "code-tpu" / "references" / "evidence.json").write_text(
        json.dumps({"schema_version": 1, "entries": [{
            "id": "baseline", "status": "documented", "claim": "A claim",
            "scope": "Test scope", "checked_on": "2026-08-31",
            "recheck_when": "On update", "regression_case": "Smoke test",
            "source_urls": ["https://example.com/source"],
        }]}), encoding="utf-8")
    return root


class CheckPackageTests(unittest.TestCase):
    def check(self, edit=None):
        with tempfile.TemporaryDirectory() as temp:
            root = make_package(Path(temp) / "package")
            if edit:
                edit(root)
            return MODULE.check_package(root)

    def test_valid_package_has_no_errors(self):
        self.assertEqual(self.check(), [])

    def test_valid_manifest_skills_string_is_accepted(self):
        def edit(root):
            path = root / ".codex-plugin" / "plugin.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            data["skills"] = "skills/code-tpu"
            path.write_text(json.dumps(data), encoding="utf-8")
        self.assertEqual(self.check(edit), [])

    def test_missing_relative_markdown_link_is_rejected(self):
        errors = self.check(lambda root: (root / "skills" / "code-tpu" / "SKILL.md").write_text(
            "[Missing](references/nope.json)", encoding="utf-8"))
        self.assertTrue(any("missing local link" in error for error in errors), errors)

    def test_escaping_markdown_link_is_rejected(self):
        errors = self.check(lambda root: (root / "skills" / "code-tpu" / "SKILL.md").write_text(
            "[Escape](../../../outside.md)", encoding="utf-8"))
        self.assertTrue(any("escapes package" in error for error in errors), errors)

    def test_duplicate_evidence_id_is_rejected(self):
        def edit(root):
            path = root / "skills" / "code-tpu" / "references" / "evidence.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            data["entries"].append(data["entries"][0].copy())
            path.write_text(json.dumps(data), encoding="utf-8")
        self.assertTrue(any("duplicate evidence id" in e for e in self.check(edit)))

    def test_missing_evidence_source_is_rejected(self):
        def edit(root):
            path = root / "skills" / "code-tpu" / "references" / "evidence.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            data["entries"][0]["source_urls"] = []
            path.write_text(json.dumps(data), encoding="utf-8")
        self.assertTrue(any("source_urls" in e for e in self.check(edit)))

    def test_invalid_evidence_date_and_status_are_rejected(self):
        def edit(root):
            path = root / "skills" / "code-tpu" / "references" / "evidence.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            data["entries"][0].update(checked_on="2026-02-30", status="proved")
            path.write_text(json.dumps(data), encoding="utf-8")
        errors = self.check(edit)
        self.assertTrue(any("checked_on" in e for e in errors), errors)
        self.assertTrue(any("status" in e for e in errors), errors)

    def test_wrong_schema_is_rejected(self):
        def edit(root):
            path = root / "skills" / "code-tpu" / "references" / "evidence.json"
            data = json.loads(path.read_text(encoding="utf-8")); data["schema_version"] = 2
            path.write_text(json.dumps(data), encoding="utf-8")
        self.assertTrue(any("schema_version" in e for e in self.check(edit)))

    def test_malformed_evidence_json_is_rejected(self):
        errors = self.check(lambda root: (root / "skills" / "code-tpu" / "references" / "evidence.json").write_text(
            "{not json", encoding="utf-8"))
        self.assertTrue(any("invalid JSON" in error for error in errors), errors)

    def test_non_object_manifest_json_is_rejected(self):
        errors = self.check(lambda root: (root / ".codex-plugin" / "plugin.json").write_text(
            "[]", encoding="utf-8"))
        self.assertTrue(any("manifest" in error and "object" in error for error in errors), errors)

    def test_null_manifest_json_is_rejected(self):
        errors = self.check(lambda root: (root / ".codex-plugin" / "plugin.json").write_text(
            "null", encoding="utf-8"))
        self.assertTrue(any("manifest" in error and "object" in error for error in errors), errors)

    def test_non_object_evidence_json_is_rejected(self):
        errors = self.check(lambda root: (root / "skills" / "code-tpu" / "references" / "evidence.json").write_text(
            "[]", encoding="utf-8"))
        self.assertTrue(any("evidence" in error and "object" in error for error in errors), errors)

    def test_null_evidence_json_is_rejected(self):
        errors = self.check(lambda root: (root / "skills" / "code-tpu" / "references" / "evidence.json").write_text(
            "null", encoding="utf-8"))
        self.assertTrue(any("evidence" in error and "object" in error for error in errors), errors)

    def test_boolean_schema_version_is_rejected(self):
        def edit(root):
            path = root / "skills" / "code-tpu" / "references" / "evidence.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            data["schema_version"] = True
            path.write_text(json.dumps(data), encoding="utf-8")
        self.assertTrue(any("schema_version" in error for error in self.check(edit)))

    def test_nonstring_evidence_status_is_rejected_without_crashing(self):
        def edit(root):
            path = root / "skills" / "code-tpu" / "references" / "evidence.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            data["entries"][0]["status"] = []
            path.write_text(json.dumps(data), encoding="utf-8")
        errors = self.check(edit)
        self.assertTrue(any("status" in error for error in errors), errors)

    def test_malformed_https_url_is_rejected_without_crashing(self):
        def edit(root):
            path = root / "skills" / "code-tpu" / "references" / "evidence.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            data["entries"][0]["source_urls"] = ["https://["]
            path.write_text(json.dumps(data), encoding="utf-8")
        errors = self.check(edit)
        self.assertTrue(any("source_urls" in error for error in errors), errors)

    def test_missing_required_files_are_rejected(self):
        def edit(root):
            (root / "skills" / "code-tpu" / "SKILL.md").unlink()
            (root / "skills" / "code-tpu" / "references" / "evidence.json").unlink()
        errors = self.check(edit)
        self.assertEqual(sum("missing required file" in error for error in errors), 2)

    def test_manifest_skills_escape_and_missing_paths_are_rejected(self):
        def edit(root):
            path = root / ".codex-plugin" / "plugin.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            data["skills"] = ["../../../outside", "skills/absent"]
            path.write_text(json.dumps(data), encoding="utf-8")
        errors = self.check(edit)
        self.assertTrue(any("escapes package" in error for error in errors), errors)
        self.assertTrue(any("path is missing" in error for error in errors), errors)

    def test_angle_space_anchor_and_external_inline_links_are_accepted(self):
        def edit(root):
            references = root / "skills" / "code-tpu" / "references"
            (references / "with space.md").write_text("# Local", encoding="utf-8")
            (root / "skills" / "code-tpu" / "SKILL.md").write_text(
                "[Local](<references/with space.md>) [Anchor](#local) "
                "[Web](https://example.com) [Mail](mailto:test@example.com) [Codex](codex://skill)",
                encoding="utf-8")
        self.assertEqual(self.check(edit), [])

    def test_cli_returns_zero_for_valid_and_one_for_invalid_package(self):
        with tempfile.TemporaryDirectory() as temp:
            root = make_package(Path(temp) / "package")
            valid = subprocess.run([sys.executable, str(SCRIPT), str(root)], capture_output=True, text=True)
            (root / "skills" / "code-tpu" / "SKILL.md").unlink()
            invalid = subprocess.run([sys.executable, str(SCRIPT), str(root)], capture_output=True, text=True)
        self.assertEqual(valid.returncode, 0, valid.stderr + valid.stdout)
        self.assertEqual(invalid.returncode, 1, invalid.stderr + invalid.stdout)
        self.assertIn("ERROR: missing required file", invalid.stdout)


if __name__ == "__main__":
    unittest.main()
