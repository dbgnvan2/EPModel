"""LX.4–LX.7: what the prompt carries, in every arm.

Purpose: prove the renderer keeps the theory out (LX.4), that A1 shows no plain description and permutes its tokens
         per seed (LX.5), that A3's shuffle is a derangement inside each legality group (LX.6), and that A4 shows no
         tension (LX.7); each check is shown to bite on an input that should fail it.
Spec:    docs/plan_v3_llm_agent_experiment.md#3.4, #10
Tests:   this file
"""

import re

import pytest
from conftest import stub_run

from experiments.llm_agents.arms import SPECS, permutations
from experiments.llm_agents.config import GROUPS, Label, load_banned, load_labels, load_text
from experiments.llm_agents.prompts import Labeller, OptionSpec, RoundInput, Seen, banned_hits, render
from experiments.llm_agents.source import family_terms

WEEKS = 208  # the experiment's horizon (llm.md's horizon_weeks): a real-length run, P9
LLM_ARMS = ("A1", "A2", "A3", "A4")


def _terms(run):
    return [t for t, _ in load_banned()] + list(family_terms(run))


def _fragments(template: str) -> list[str]:
    """The fixed words of a label, split at its {fields}: what a plain description would leave in a prompt."""
    return [f.strip(" ,") for f in re.split(r"\{[a-z]+\}", template) if len(f.strip(" ,")) > 3]


@pytest.mark.parametrize("arm", LLM_ARMS)
def test_lx4_no_banned_term_in_any_arm(arm):
    run, chooser = stub_run(arm, 1, WEEKS)
    assert len({w for w, _ in chooser.prompts}) == WEEKS, "a prompt every week"
    people = {p.value for p, person in run.state.people.items() if person.role.name != "EXTERNAL"}
    assert {c.person for c in run.source.choices} == people, "every family member was prompted"
    terms = _terms(run)
    for (week, token), prompt in chooser.prompts.items():
        hits = banned_hits(prompt.system + "\n" + prompt.user, terms)
        assert not hits, f"{arm} week {week} {token}: {hits}"
    assert not banned_hits("\n".join(load_text().values()), terms)


def test_lx4_a_banned_term_is_caught():
    run, _ = stub_run("A2", 1, WEEKS)
    terms = _terms(run)
    # Text that reads as neutral but carries a term: other case, punctuation, a family name, a role word.
    for text in ("Your TENSION rises. Triangle!", "turn to P3 about Ravi", "keep in touch with P2, (spouse)",
                 "the Family meets", "P4 is acute."):
        assert banned_hits(text, terms), text
    # Whole-word: a term inside a longer word is not a hit (no false alarm on "triangular" for "triangle").
    assert banned_hits("a triangular table", ["triangle"]) == []
    # A label mutated to carry a theory word is caught in the rendered prompt.
    labels = dict(load_labels())
    labels["PURSUE"] = Label("PURSUE", "tie", "pursue fusion with {target}", "sought fusion with {target}")
    labeller = Labeller(SPECS["A2"], permutations("A2", 1, ["a", "b"], labels), labels, load_text())
    prompt = render(labeller, RoundInput(1, 10.0, None, (("P2", 3.0, "open"),), (), (), (),
                                         (OptionSpec("PURSUE", "P2"),)))
    assert "fusion" in banned_hits(prompt.user, terms)


def test_lx5_opaque_arm_has_no_description():
    fragments = {f for label in load_labels().values() for f in _fragments(label.option) + _fragments(label.seen)}
    a1, chooser = stub_run("A1", 1, WEEKS)
    for key, prompt in chooser.prompts.items():
        leaked = [f for f in fragments if f in prompt.user]
        assert not leaked, f"A1 {key}: {leaked}"
    assert any(c.flag is None for c in a1.source.choices)
    # The check bites: the same search finds plain descriptions in A2's prompts.
    _, plain = stub_run("A2", 1, WEEKS)
    assert any(f in p.user for p in plain.prompts.values() for f in fragments)


def test_lx5_labels_differ_between_seeds():
    labels = load_labels()
    people = ["ana", "bruno", "halim", "marta", "nadia", "pia", "ravi", "sofia"]
    perms = [permutations("A1", seed, people, labels) for seed in range(1, 6)]
    for i in range(len(perms)):
        for j in range(i + 1, len(perms)):
            assert perms[i].act_token != perms[j].act_token
            assert perms[i].person_order != perms[j].person_order
    for p in perms:
        assert sorted(p.act_token.values()) == sorted(f"K{i}" for i in range(1, len(labels) + 1))
    # The same seed gives the same permutation (it is fixed per seed, not per process).
    assert permutations("A1", 3, people, labels) == perms[2]


@pytest.mark.parametrize("seed", range(1, 21))
def test_lx6_shuffle_is_a_derangement_within_groups(seed):
    labels = load_labels()
    shown = permutations("A3", seed, ["a", "b"], labels).label_of
    assert sorted(shown.values()) == sorted(labels), "a bijection on the kinds"
    for group in GROUPS:
        members = [k for k in labels if labels[k].group == group]
        assert {shown[k] for k in members} == set(members), f"{group}: a label left its group"
        if len(members) > 1:
            assert all(shown[k] != k for k in members), f"{group}: a fixed point"
        else:
            assert shown[members[0]] == members[0]
    assert permutations("A2", seed, ["a", "b"], labels).label_of == {k: k for k in labels}


def test_lx7_no_feedback_arm_shows_no_tension():
    text = load_text()
    marks = [_fragments(text[k])[0] for k in ("tension_first", "tension_line")]
    marks.append(_fragments(text["history_line"])[-1])
    _, a4 = stub_run("A4", 1, WEEKS)
    for key, prompt in a4.prompts.items():
        assert not [m for m in marks if m in prompt.user], f"A4 {key}"
    # The check bites: A2's prompts carry both the tension line and the change in the history.
    _, a2 = stub_run("A2", 1, WEEKS)
    users = [p.user for p in a2.prompts.values()]
    assert all(any(m in u for u in users) for m in marks)


def test_lx7_render_without_feedback_drops_every_number_of_tension():
    labels = load_labels()
    r = RoundInput(5, 37.0, -4.0, (("P2", 3.0, "open"),), (), (Seen(None, None, None, False),),
                   ((4, "pull back from P2", -4.0),), (OptionSpec("DISTANCE", "P2"),))
    for arm, shows in (("A4", False), ("A2", True)):
        lab = Labeller(SPECS[arm], permutations(arm, 1, ["a", "b"], labels), labels, load_text())
        user = render(lab, r).user
        assert ("37" in user) is shows and ("-4" in user) is shows
