# Proposal — the Phase D definitions the owner asked me to draft

> **Status: draft for owner approval, 2026-10-10.** Plan decisions P7 and P12 (`docs/implementation_plan_phase_d.md`
> §1a). Each definition is `[I]`. None is built until approved, and each goes into spec revision 15 with its
> criterion.

## 1. C.14 — what a "management technique" is

**The criterion** (`M11.C.14`): management technique has zero independent effect while marital distance is high.

**The source.** Bowen on parental closeness (`_LEDGER.md` L04; `docs/theory/ch04.md:50`): when the parents were
emotionally close, "they could do no wrong" — firmness, permissiveness, punishment, "talking it out" all worked; when
they were emotionally divorced, "any and all 'management approaches' were equally unsuccessful". The gate is `M5.C.1`'s
row: while marital distance is high, any move targeting the symptom-bearer produces no improvement.

**Proposed definition.**
- A **technique** is a parent's way of handling the symptomatic child, scripted as the same act toward the child every
  week. Bowen names four, mapped to the model's acts:
  - **firmness:** `OVERFUNCTION`;
  - **permissiveness:** `UNDERFUNCTION`;
  - **punishment:** `CONFLICT`;
  - **talking it out:** `STAY-IN-CONTACT`.
- **Marital distance high:** the spouses' own tie is held at distance (both members below their contact optimum, the
  `distance` readout), declared at the arm's start.
- **The test, a null under `M11.4`:** under high marital distance, a firmness arm and a talking-it-out arm do not differ
  in the child's symptom load by more than the declared equivalence margin. Margin, minimum detectable effect and power
  are declared (`M11.4a`).
- **Mutation:** remove `M5.C.1`'s marital-distance gate. The two techniques then differ, and the null fails.

## 2. C.23 — the expressed-emotion term and the suppression term

**The criterion** (`M11.C.23`): high expressed emotion with suppression relapses more than low expressed emotion
without it.

**The source.** Kerr, KS25.9, citing the expressed-emotion literature: hostility, over-involvement and critical
comments from relatives raise relapse "even if the person is on medication"; without them relapse is much less likely
"even with patients not on any medications".

**Proposed definitions.**
- **The EE term:** how much of the family's climate reaches the patient. It is the sum of three things the log already
  records:
  - **hostility:** the impingement of `CONFLICT` acts the patient receives;
  - **critical comments:** blame-valenced acts toward the patient (`M1.A.18a`'s blame readout);
  - **over-involvement:** the mother's investment in her tie to the patient (`M1.B.8`), weighting the impingement she
    delivers.

  Nothing new is written. The term is the existing appraisal path for these acts.
- **The suppression term ("medication"):** a declared factor that scales down the patient's symptom accumulation,
  0.5 `[I]`. It does not touch appraisal or acute anxiety, so the relational climate still reaches the patient.
- **Arms:**
  - **high EE with suppression:** two relatives script hostility and criticism toward the patient, and the mother
    over-invests;
  - **low EE without suppression:** none of that.
- **Readout: relapse.** A symptom onset after the patient's symptom has remitted (`M7.D.2`), counted per seed.
- **Mutation:** zero the EE term, so these acts deliver no impingement to the patient. Suppression then wins, and the
  criterion fails.

## 3. C.28 — the allocation-mobility term

**The criterion** (`M11.C.28`): sink mobility is protective. A fixed allocation gives fewer, more severe outcomes; a
rotating one gives more, milder ones.

**The source.**
- Kerr, FE07.6: "when the family emotional process does not always focus on the same person or on the same
  relationship, it is less likely that the functioning of any one person will be chronically impaired".
- FE10.3: two families at the same level, one with a fixed focus (one catastrophic outcome), one mobile (several
  moderate ones).
- The pass notes say what makes an allocation fixed: `M1.C.4`'s activation memory, so its decay rate is consequential.

**Proposed definition.**
- **The mobility term:** the decay rate of the activation memory that holds the family's focus where it last was:
  - on a triangle (`M1.C.4`);
  - on the sink allocation's target (`M1.D.1`).
- **Arms:**
  - **fixed:** the memory does not decay, so the focus stays on the child it settled on;
  - **rotating:** it decays at the declared rate, so the focus can move between children.

  Both arms have the same family, level and stress, so their adaptiveness is equal (`M11.G.2`).
- **Readouts:**
  - **count:** symptom episodes across the children;
  - **severity:** the largest single symptom load reached.

  The fixed arm must show fewer episodes and a higher maximum.
- **Mutation:** pin the mobility term, so both arms are fixed. The arms then do not differ, and the criterion fails.

## 4. P12 — a criterion that proves the triangle change

**What it must prove.** The triangle change of 2026-10-09: a `TRIANGLE` relieves its seeker in proportion to the
third's help, `third_help`. Help is calm (the third's own excess low) and lean (the third closer to the seeker than to
the partner). Today no passing criterion reads it, and the mutant `triangle-help-ignored` survives
(`docs/phase_c_completion_report.md` §13 item 5).

**Proposed criterion (`M11.C.46`):**
- **Arms:** the seeker turns to a third (scripted, held open as in `M11.C.3`), and the arms differ only in the third:
  - **helpful:** a calm third, closer to the seeker;
  - **unhelpful:** an anxious third, or one closer to the partner.
- **Direction:** the seeker's relief is larger with the helpful third.
- **Class:** premise. The rule writes the relief, so a pass confirms the code and is never reported as a finding
  (`M11.5`).
- **Mutation:** `triangle-help-ignored` (relief no longer scaled by help) makes the arms equal, and the criterion
  fails. This is the run that proves the change.

## What the owner decides

Approve, correct or reject each of §1–§4. Approved items go into spec revision 15 and are built with their criteria in
D-I, step 7 onward.
