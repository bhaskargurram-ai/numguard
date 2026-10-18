"""Security invariants of the GitHub workflows and the action (skipped when PyYAML is not installed).

These encode docs/SECURITY_MODEL.md as checks, so a later edit cannot quietly
break the trust-zone split:

* no workflow is triggered by ``pull_request_target``;
* no ``run:`` script contains a ``${{ }}`` expression (values reach scripts
  through ``env:`` only, preventing script injection);
* every checkout sets ``persist-credentials: false``;
* jobs that run PR code (check.yml, agent.yml's verify) reference no secrets
  and get a read-only token; numguard is installed before any PR code runs;
* comment.yml never checks out the pull request and never installs a project;
* agent.yml: ``propose`` (holds the key) never installs or runs the PR, which
  it checks out into ``pr/`` as data; ``gate`` checks nothing out; no caches;
* permissions are minimal and callers set ``permissions: {}`` for workflow_run.
"""

from __future__ import annotations

import json
import re
import unittest

from tests import helpers as h

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML is optional for the test suite
    yaml = None

WF_DIR = h.ROOT / ".github" / "workflows"
EXAMPLES = h.ROOT / "examples" / "workflows"


def load(path):
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["on"] = data.pop(True, data.get("on"))  # YAML 1.1 parses the key `on` as True
    return data


def steps(job):
    return job.get("steps") or []


def checkouts(job):
    return [s for s in steps(job) if str(s.get("uses", "")).startswith("actions/checkout@")]


def names(job):
    return [s.get("name") or s.get("uses") for s in steps(job)]


@unittest.skipIf(yaml is None, "PyYAML not installed")
class WorkflowSecurityTest(unittest.TestCase):
    def all_workflows(self):
        files = sorted(WF_DIR.glob("*.yml")) + sorted(EXAMPLES.glob("*.yml"))
        self.assertGreaterEqual(len(files), 12)
        return [(f, load(f)) for f in files]

    def test_no_pull_request_target_and_no_expressions_in_scripts(self):
        for path, wf in self.all_workflows():
            on = wf["on"]
            triggers = on if isinstance(on, (dict, list)) else [on]
            self.assertNotIn("pull_request_target", triggers, path.name)
            self.assertNotIn("pull_request_target", path.read_text(encoding="utf-8").split("# ")[0])
            for name, job in (wf.get("jobs") or {}).items():
                for s in steps(job):
                    if "run" in s:
                        self.assertNotIn("${{", s["run"], f"{path.name}:{name}:{s.get('name')}")

    def test_every_checkout_drops_credentials(self):
        for path, wf in self.all_workflows():
            for name, job in (wf.get("jobs") or {}).items():
                for s in checkouts(job):
                    self.assertIs((s.get("with") or {}).get("persist-credentials"), False, f"{path.name}:{name}")

    def test_top_level_permissions_are_minimal(self):
        for path, wf in self.all_workflows():
            perms = wf.get("permissions")
            self.assertIn(perms, ({}, {"contents": "read"}), path.name)
            for name, job in (wf.get("jobs") or {}).items():
                jp = job.get("permissions")
                if jp is None:
                    continue
                writes = {k for k, v in jp.items() if v == "write"}
                allowed = {"pull-requests", "contents", "pages", "id-token"}
                self.assertLessEqual(writes, allowed, f"{path.name}:{name}")
                if "pull-requests" in writes:
                    # only the privileged comment job may write to PRs
                    self.assertTrue(path.name.endswith("comment.yml") or name == "comment", f"{path.name}:{name}")

    def test_check_workflow_is_unprivileged(self):
        wf = load(WF_DIR / "check.yml")
        self.assertIn("workflow_call", wf["on"])
        job = wf["jobs"]["check"]
        self.assertEqual(job["if"], "github.event_name == 'pull_request'")
        self.assertEqual(job["permissions"], {"contents": "read"})
        text = json.dumps(job)
        self.assertNotIn("secrets.", text)
        self.assertNotIn("github.token", text)
        self.assertNotIn("actions/cache", text)
        n = names(job)
        install_pr = n.index("Install the project under test (runs PR code)")
        self.assertLess(n.index("Install numguard (trusted source, before any PR code)"), install_pr)
        self.assertLess(n.index("Select the configuration (base commit first)"), install_pr)
        self.assertLess(install_pr, n.index("Run numguard (runs PR code; no token)"))
        upload = next(s for s in steps(job) if str(s.get("uses", "")).startswith("actions/upload-artifact"))
        self.assertEqual(upload["with"]["name"], "numguard-result")
        tool = [s for s in checkouts(job) if "repository" in s.get("with", {})]
        self.assertEqual(tool[0]["with"]["repository"], "${{ inputs.numguard-repository }}")

    def test_comment_workflow_never_touches_pr_code(self):
        wf = load(WF_DIR / "comment.yml")
        self.assertIn("workflow_call", wf["on"])
        job = wf["jobs"]["comment"]
        self.assertEqual(job["permissions"], {"contents": "read", "actions": "read", "pull-requests": "write"})
        for s in checkouts(job):
            self.assertEqual(s["with"]["repository"], "${{ inputs.numguard-repository }}")  # the tool, never the PR
            self.assertNotIn("head", json.dumps(s))
        text = json.dumps(job)
        for forbidden in ("install-command", "INSTALL_COMMAND", "pull_request.head", "pip install -e", "secrets.",
                          "python -m numguard check", "actions/cache"):
            self.assertNotIn(forbidden, text)
        post = next(s for s in steps(job) if s.get("name") == "Post or update the numguard card")
        self.assertIn("--expect-head-sha", post["run"])
        self.assertEqual(post["env"]["HEAD_SHA"], "${{ steps.pr.outputs.head-sha }}")

    def test_agent_zones(self):
        wf = load(WF_DIR / "agent.yml")
        jobs = wf["jobs"]
        self.assertEqual(set(jobs), {"gate", "propose", "verify"})
        self.assertEqual(checkouts(jobs["gate"]), [])

        propose = jobs["propose"]
        ptext = json.dumps(propose)
        for forbidden in ("install-command", "INSTALL_COMMAND", "eval ", "pip install -e", "python -m numguard",
                          "agent verify"):
            self.assertNotIn(forbidden, ptext)
        pr_checkouts = [s for s in checkouts(propose) if "repository" not in s.get("with", {})]
        self.assertEqual(len(pr_checkouts), 1)
        self.assertEqual(pr_checkouts[0]["with"]["path"], "pr")
        self.assertEqual(propose["permissions"], {"contents": "read"})
        key_steps = [s for s in steps(propose) if "secrets." in json.dumps(s)]
        self.assertEqual([s.get("id") for s in key_steps], ["propose"])
        self.assertIn("agent propose", key_steps[0]["run"])
        self.assertIn("NUMGUARD_BIN", key_steps[0]["run"])  # the isolated venv, never the job's Python

        verify = jobs["verify"]
        vtext = json.dumps(verify)
        self.assertNotIn("secrets", vtext)
        self.assertNotIn("actions/cache", vtext)
        self.assertNotIn('"cache"', vtext)
        self.assertEqual(verify["permissions"], {"contents": "read"})
        self.assertEqual(set(verify["needs"]), {"gate", "propose"})
        self.assertEqual([j for j in jobs if "secrets." in json.dumps(jobs[j])], ["gate", "propose"])
        gate_secret = re.findall(r"\$\{\{[^}]*secrets\.[^}]*\}\}", json.dumps(jobs["gate"]))
        self.assertEqual(gate_secret, ["${{ secrets.llm-api-key != '' }}"])
        self.assertEqual(wf["on"]["workflow_call"]["secrets"]["llm-api-key"]["required"], False)

    def test_callers(self):
        for d in (WF_DIR, EXAMPLES):
            ev = load(d / "numguard.yml")
            self.assertEqual(list(ev["on"]), ["pull_request"])
            self.assertEqual(ev["name"], "numguard")
            self.assertEqual(ev["permissions"], {"contents": "read"})
            self.assertNotIn("secrets", json.dumps(ev))
            for name in ("numguard-comment.yml", "numguard-agent.yml"):
                wf = load(d / name)
                self.assertEqual(wf["on"]["workflow_run"]["workflows"], ["numguard"])
                self.assertEqual(wf["permissions"], {})
                for job in wf["jobs"].values():
                    self.assertNotIn("steps", job)  # only calls reusable workflows
            agent = load(d / "numguard-agent.yml")
            self.assertEqual(agent["jobs"]["comment"]["with"]["pr-number"], "${{ needs.agent.outputs.pr-number }}")
            self.assertEqual(agent["jobs"]["agent"]["secrets"], {"llm-api-key": "${{ secrets.NUMGUARD_LLM_API_KEY }}"})

    def test_callers_only_pass_declared_inputs(self):
        declared = {name: set(load(WF_DIR / name)["on"]["workflow_call"]["inputs"])
                    for name in ("check.yml", "comment.yml", "agent.yml")}
        for d in (WF_DIR, EXAMPLES):
            for path in d.glob("numguard*.yml"):
                for jname, job in load(path)["jobs"].items():
                    target = str(job.get("uses", "")).split("@")[0].rsplit("/", 1)[-1]
                    with self.subTest(file=str(path), job=jname):
                        self.assertIn(target, declared)
                        self.assertLessEqual(set(job.get("with") or {}), declared[target])

    def test_release_ci_and_pages(self):
        rel = load(WF_DIR / "release.yml")
        self.assertEqual(rel["on"]["push"], {"tags": ["v*"]})
        self.assertEqual(set(rel["on"]["workflow_dispatch"]["inputs"]), {"version", "pypi"})
        # "Run workflow" creates the tag itself (only that job may write contents besides the release)
        self.assertEqual(rel["jobs"]["tag"]["if"], "github.event_name == 'workflow_dispatch'")
        self.assertEqual(rel["jobs"]["tag"]["permissions"], {"contents": "write"})
        for s in rel["jobs"]["build"]["steps"]:  # untrusted input reaches scripts via env only
            self.assertNotIn("${{", s.get("run", ""))
        pub = rel["jobs"]["publish"]
        self.assertEqual(pub["steps"][-1]["with"]["password"], "${{ secrets.PYPI_API_TOKEN }}")
        self.assertEqual(rel["jobs"]["publish"]["permissions"], {"id-token": "write"})
        self.assertEqual(rel["jobs"]["publish"]["environment"]["name"], "pypi")
        self.assertEqual(rel["jobs"]["github-release"]["needs"], ["build", "tag"])  # not gated on PyPI
        self.assertIn("--clobber", json.dumps(rel["jobs"]["github-release"]))
        pages = load(WF_DIR / "pages.yml")
        self.assertEqual(pages["jobs"]["deploy"]["if"], "github.event.repository.private == false")
        self.assertIn("workflow_dispatch", pages["on"])
        ci = load(WF_DIR / "ci.yml")
        self.assertEqual(ci["jobs"]["test"]["strategy"]["matrix"]["python-version"], ["3.10", "3.11", "3.12", "3.13"])
        pkg = json.dumps(ci["jobs"]["package"])
        for needle in ("python -m build", "twine check --strict", "venv", "init --repo", "--config"):
            self.assertIn(needle, pkg)

    def test_action_yml_inputs_reach_scripts_via_env_only(self):
        action = load(h.ROOT / "action.yml")
        self.assertEqual(action["runs"]["using"], "composite")
        for s in action["runs"]["steps"]:
            self.assertNotIn("${{", s.get("run", ""), s.get("name"))
        commands = {"check", "summary", "comment", "agent-propose", "agent-verify"}
        conditions = set()
        for s in action["runs"]["steps"]:
            conditions |= set(re.findall(r"inputs.command == '([a-z-]+)'", s.get("if", "")))
        self.assertEqual(conditions, commands)
        key_steps = [s for s in action["runs"]["steps"] if "llm-api-key" in json.dumps(s)]
        self.assertEqual([s["if"] for s in key_steps], ["inputs.command == 'agent-propose'"])


class RefDefaultsTest(unittest.TestCase):
    def test_reusable_workflows_default_to_this_release(self):
        """A caller that omits `numguard-ref` must install the release its `@ref` names, not an older one."""
        import numguard

        want = f"v{numguard.__version__}"
        seen = 0
        for path in sorted(WF_DIR.glob("*.yml")):
            inputs = ((load(path).get("on") or {}).get("workflow_call") or {}).get("inputs") or {}
            if "numguard-ref" in inputs:
                seen += 1
                self.assertEqual(inputs["numguard-ref"].get("default"), want, path.name)
        self.assertGreater(seen, 0)


if __name__ == "__main__":
    unittest.main()
