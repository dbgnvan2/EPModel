# Proposal — ASR quotations (SP-1)

> **Status: DRAFT 2026-10-21, for the owner's review. Nothing in it is applied.** The owner answered Q1–Q8 on
> 2026-10-22, all as recommended (§6); the §4 texts still wait for the owner's approval, and the build (§7) follows
> it. No spec, explainer, ledger or test edit is made until the owner approves it (decision D1, `CHANGELOG.md`). It answers the stage 1 milestone review's
> systemic pattern SP-1 and the review's owner questions Q1–Q3 (`docs/cycles/2026-10-21_milestone-review_v2-stage-1.md`).
> Each paraphrase in §4 was drafted from its transcript passage and then checked against the passage by a second
> reader who had not drafted it; the entries that check corrected are marked. Line numbers are at commit `c893109`.
> Nothing here changes what a run computes, so no record is rerun.

## 1. Why

**The rule.** Two of the six corpus sources are automatic-speech-recognition (ASR) transcripts: the 1979 Basic Video
Series (Tapes 1–6) and the Kerr–Bowen interviews (`KB01`–`KB15`). The rule is to cite them and never quote them. It
was committed in `docs/theory/_LECTURES_1979.md` on 2026-08-23 at 11:50 (`528bb50`) and in
`docs/theory/_KERR_INTERVIEWS.md` at 13:09 (`e14d3d7`), and restated in `docs/theory/_SOURCE_QUALITY.md` at 17:53
(`6a56bb2`). An ASR transcript is a machine's reading of speech. Tape 1, the one tape with two independent
transcriptions, agrees between them on 97.2% of words (`_SOURCE_QUALITY.md`), so the two differ on about one word in
thirty-six. A quotation presents such a reading as the speaker's own words.

What each source allows (`_SOURCE_QUALITY.md`, the citation legend in `docs/model_explainer.md`):

| Source | Name the speaker? | Quote? |
|---|---|---|
| Tapes 1–6 (one speaker) | Yes: the substance may be attributed to Bowen | No |
| `KB01`, Otter transcript (clean speaker labels) | Yes, by label: Speaker 1 is Kerr, Speaker 2 is Bowen | No |
| `KB02`–`KB15`, Hijack transcripts (no speaker labels) | No: cite the interview ("the KB04 interview"). Question-shaped text is recorded as Kerr's framing | No |
| The 22 papers (`ChNN` · `LNN.N`), *Family Evaluation* (`FEnn`, the Epilogue `FE11` is Bowen's own), Kerr's book (`KSnn`) | Yes | Yes |

**What the review found.** SP-1: the rule has two parts, cite the interview and do not quote, and the project applies
the first unevenly and the second hardly at all (MR-1, MR-3, MR-4, MR-5, MR-8). The `kb/*.md` extraction notes state
the rule in their headers and quote in their bodies; the spec, the explainer, the ledger and the DECISIONS records
copied from them. The 2026-08-28 corpus-fidelity sweep checked that each quotation was verbatim in the corpus, which
counted an exact ASR match as a pass instead of a breach.

**What the owner decided (D1, 2026-10-21).** This proposal comes first: for each quotation in the spec, the explainer
and the ledger, a paraphrase and its citation, reviewed before any spec edit. Approved DECISIONS records stay frozen,
with a note. The `kb/*.md` notes keep their quotes, and their headers say the quotes only locate passages. A test
fails on any new quotation of four or more words. The spec edits go into revision 34 with MR-10, MR-12, MR-13, MR-15
and D3, so the spec changes once.

**The repository is public** (`dbgnvan2/EPModel`), so every quotation in it is published.

## 2. The scan

The triage's scan was a session script and was not committed. This proposal reran it with a new script (kept in the
session's scratch folder; §7 SP1.2 turns its logic into a test).

**Method.**

- Every tracked `.md` and `.html` file was read. A quoted span is text between straight or curly double quotes, up to
  800 characters, with no blank line inside.
- Each span is split at ellipses, bracketed insertions and dashes. Each fragment of four or more words is normalised
  (lower case, punctuation and emphasis marks removed) and looked up in every transcript: the six tapes, the second
  Tape 1 transcription, the fifteen interviews (Hijack set), `KB01`'s Otter transcript, and the TMBAP archive
  transcripts.
- A row is **verbatim** (the whole span is in one transcript), **partial** (a fragment is), **near** (at least 60% of
  a fragment's word triples are in one transcript), or **cited, not found** (no match, but an ASR citation on a nearby
  line).
- Each row was then classified by hand against the written extractions (`docs/theory/chNN.md`,
  `family_evaluation/feNN.md`, `kerr_book/ksNN.md`), because Bowen repeats his written phrases in speech and a
  transcript match can be coincidence.

**Two further scans** over the documents in §4's scope:

- **Unquoted copies:** runs of seven or more transcript words with no quote marks. This found two new items, both
  bold text copied without quote marks: `model_explainer.md:596` (SP1-E05) and `_LEDGER.md:594` (SP1-L01). The other hits were quotations already listed, or ordinary or theory wording that also occurs in
  a written source.
- **Attribution:** "Bowen says", "Bowen names" and similar beside a `KB02`–`KB15` citation. This found two not already
  in the quotation rows: `model_explainer.md:382` (MR-3, SP1-E02) and `_LEDGER.md:1393` (SP1-L04).

**Matches per file** (rows; a row is one quoted span):

| File | Verbatim | Partial | Near | Cited, not found |
|---|---|---|---|---|
| `docs/bowen_agent_model_spec_v2.md` | 52 | 6 | 9 | 27 |
| `docs/model_explainer.md` | 9 | 0 | 7 | 7 |
| `docs/theory/_LEDGER.md` | 10 | 1 | 14 | 4 |
| `docs/theory/_CONVERGENCES.md` | 0 | 0 | 2 | 0 |
| `docs/agent_model_proposal.html` | 2 | 0 | 1 | 0 |
| `docs/DECISIONS FOR APPROVAL.md` | 19 | 1 | 8 | 5 |
| `docs/theory/_LECTURES_1979.md` | 33 | 8 | 4 | 2 |
| `docs/theory/_KERR_INTERVIEWS.md` | 20 | 1 | 0 | 0 |
| `docs/theory/kb/_KB_PASS2.md` | 25 | 2 | 0 | 1 |
| `docs/theory/Extractions to be human validated.md` | 10 | 0 | 1 | 1 |
| `docs/theory/_PASS2_CORRECTIONS.md` | 1 | 0 | 4 | 4 |
| `docs/theory/_SOURCE_QUALITY.md` | 2 | 0 | 0 | 1 |
| `docs/theory/kb/kb01.md`–`kb15.md` (14 files) | 272 matched quotes in all | | | |
| `TODO.md` | 2 quotations (lines 1046 and 1098) and one term | | | |
| `docs/proposal_rev11_emergence.md` | 1 (line 84, `KB10`, 28 words) | | | |
| `docs/spec_coverage.md` | 1 (line 299, generated from the spec) | | | |

Review records (`docs/cycles/`, `docs/REVIEW_spec_rev10_2026-09-23.md`), `CHANGELOG.md` and `LEARNINGS.md` have
"cited, not found" rows only: they quote the project's own documents, not the transcripts. The frozen grid spec
(`docs/bowen_individual_family_model_spec.md`) has two matches, both a term or a title. `papers/` holds the owner's
files: four of its reports have seven rows, one of them a verbatim five-word match and two near.

**Against the triage's count.** The triage found 103 quotations: 61 in the spec, 25 in DECISIONS, 9 in the ledger and
8 in the explainer, and 240 in the `kb/*.md` notes. This scan reports more rows because it counts near matches and
splits spans at dashes as well as ellipses; the hand classification then removes the written-source quotations and
the text that is not a quotation. In §4's five documents the scan found 151 rows. 87 are
ASR quotations, two of them misquotations (65 in the spec, 9 in the explainer, 10 in the ledger, 3 in the proposal
html); §4 replaces them in 62 entries, one per sentence, and adds the three items from the further scans, 65 entries in
all. The other 64 are kept: 45 written-source quotations and 19 that are not quotations (a term, the project's own
wording, the owner's words). The kept rows are listed after the tables with their source.

**Limits of the scan.** A misplaced straight quote can pair the wrong marks and hide a span; the unquoted-run scan is
the backstop for that. A paraphrase that stays too close to its passage would pass a quote-mark scan, so every
proposed text in §4 was checked for runs of four or more transcript words; the runs found are ordinary English
(*so this is a*) or theory vocabulary, and are listed in §8.

## 3. Treatment per document

| Document | Treatment | Why |
|---|---|---|
| The spec, the explainer, the ledger, `_CONVERGENCES.md`, `agent_model_proposal.html` | Each ASR quotation replaced by the paraphrase in §4 | D1. These are the documents the owner and reviewers read as the model's contract and explanation. |
| `TODO.md` (lines 1046 and 1098, in the 2026-08-28 backlog) | Paraphrased in the same commit | A working list, not a frozen record. |
| `docs/spec_coverage.md` | Regenerated from the spec | It is generated. |
| `docs/DECISIONS FOR APPROVAL.md` | Left unedited, with the note below under its status line | D1: approved records stay frozen. |
| `docs/proposal_rev11_emergence.md` | Left unedited, with the same note | An approved working record (status APPROVED 2026-10-06), so it is treated as DECISIONS is (Q2). |
| `kb/*.md`, `kb/_KB_PASS2.md`, `_LECTURES_1979.md`, `_KERR_INTERVIEWS.md`, `Extractions to be human validated.md`, `_PASS2_CORRECTIONS.md`, `_SOURCE_QUALITY.md` | Quotes kept; the header note below added or extended | D1: these are the extraction notes, and a quote there locates a passage for anyone checking a finding. |
| Six `kb` headings that credit Bowen by name | Retitled (table below) | They sit in Hijack-set notes, where the rule is to cite the interview. |
| Review records, `CHANGELOG.md`, `LEARNINGS.md`, the frozen grid spec | No change | Dated records, and none quotes a transcript. |
| `papers/` | Out of scope; exempt from the test | The owner's files. Seven rows in four reports, one a verbatim five-word match; left to the owner. |

**The note for frozen records** (proposed wording):

> *Quotations of the 1979 tapes and the Kerr–Bowen interviews in this record are ASR text. They were written after
> the no-quotation rule (2026-08-23) and break it. They are kept unedited as the record of what was approved; they
> are not the speakers' words and are not copied into other documents. The current wording is in the spec under the
> same requirement IDs (SP-1, 2026-10-21).*

**The header note for extraction notes** (proposed wording, added after each file's citation rule):

> *The quotations below are ASR text. They only locate passages in the transcript for a reader checking a finding;
> they are not the speakers' words and are not copied into any other document (SP-1, 2026-10-21).*

**The six headings:**

| Heading | Proposed |
|---|---|
| `kb02.md:37` K02.3 | The interview states its own epistemic standard, and it explains Q-VALIDATION |
| `kb09.md:12` K09.1 | The interview's instruction not to take the theory on its author's authority |
| `kb10.md:15` K10.1 | The interview names THREE variables for societal process, and they are not our three dials |
| `kb12.md:47` K12.5 | A late differentiating exercise described in the interview, and it is not a weekend `[T]` |
| `kb13.md:33` K13.2 | The interview does not take general systems theory as the foundation |
| `kb15.md:39` K15.4 | Paul MacLean, and where the interview defers `[T]` |

`kb14.md` K14.7's "the Bowen Theory" is a name, not an attribution, and stays. The ⚠ marks stay.

**The extraction notes keep 272 matched quotes in a public repository.** D1 decided this, and this proposal does
not reopen it; the number is stated so that the decision is made knowing it.

## 4. The paraphrases

Each entry gives the proposed text as it would read after the change, the source, and a note. The current text is not
reproduced here, because it is the quotation; it is at the given line of `c893109`. **Every proposed text keeps the
requirement's MUST unchanged except M11.F.3's alternative B (Q3).** Where the current text claims more than its
passage, the paraphrase keeps to the passage and the note says so; those requirements are listed in Q7.

Entries marked *Corrected in the check* were changed by the second reader, most often to restore a hedge or a
condition the current text had dropped (the 2026-08-28 sweep found the same fault in `M1.A.13a`, `M1.D.7b`,
`M1.A.18d`, `M8.6a` and `M1.A.5d`).

### The spec (`docs/bowen_agent_model_spec_v2.md`)

| # | Where | Source | Proposed text | Note |
|---|---|---|---|---|
| SP1-S01 | M1.A.0, line 77 | KB02 (Hijack); KB13 (Hijack) | The KB02 and KB13 interviews say this twice. KB02 puts the emotional process on a par with instinct: the force behind bird migration, the force that leads a salmon home to its spawning river, and, asked whether this reaches down to the cellular level, the force that draws an amoeba toward food. KB13 says that others took up the term *emotional* as a synonym for feeling, which was never the intended meaning. → `kb/kb02.md` · K02.1, `kb/kb13.md` · K13.1 | 'Bowen, twice' becomes the KB02 and KB13 interviews. In KB02 the extension to the cell is the interviewer's question, which the answer accepts. |
| SP1-S02 | M1.A.3a, line 98 | KB14 (Hijack) | **M1.A.3a** — *the marriage-ceremony break, resolved.* The KB14 interview gives it as a usual pattern with no explanation: couples who got on well as friends before marrying find the relationship goes wrong from the wedding on, and the speaker has often wondered why but takes it as a fact. → `kb/kb14.md` · K14.3; user mechanism, E2, 2026-08-24; formerly open | 'Bowen states' becomes the KB14 interview. The current text says the speaker never worked out why; the passage says the speaker has often wondered and takes it as a fact. The check restored that the pattern is usual, not universal. K14.3 added. *Corrected in the check.* |
| SP1-S03 | M1.A.4a, Corroboration, line 107 | KB05 (Hijack) | **Corroboration:** the KB05 interview describes years of work on a separate relationship with each parent, which went fairly well, yet the old process still drew the person back in when visiting home — real functional improvement that failed in one situation, and so was not yet basic. → `kb/kb05.md` · K05.3; `kb/kb07.md` · K07.2 | The check removed an overstatement: the passage describes a habit, not every visit. K05.3 added. *Corrected in the check.* |
| SP1-S04 | § This unifies with the solid-self definition (lines 169-170), line 169 | Tape 2 | Solid self is what a person believes and stands for, the position held whatever happens; pseudo-self can be negotiated away, giving way under outside pressure, which solid self never does (1979 Tape 2, M6.I.4). | The one-word term *negotiable* loses its quote marks. The check widened the pressure back to the passage's (from the surroundings, not only from people). The replacement spans lines 168–170. *Corrected in the check.* |
| SP1-S05 | M1.A.9a, line 236 | KB12 (Hijack) | The KB12 interview states the same structure in the model's terms: a poorly differentiated self can also take an I-position, but a self-centred, dogmatic and forceful one, while a well-differentiated self takes one without attacking or defending against the other, so the way an I-position is taken gives a fair indication of how well the person taking it functions. → `kb/kb12.md` · K12.3; the same two-sidedness on the external agent is `kb/kb07.md` · K07.1. | 'The corpus states' becomes the KB12 interview. The check restored the passage's hedge on what the manner of an I-position shows. *Corrected in the check.* |
| SP1-S06 | M1.A.13a (first sentence), line 281 | Tape 6 | **M1.A.13a** The tier **MUST** be **derived, not assigned**, from two inputs the model already carries: whether the person holds a **functional-doer position** (the head of the clan, a role the family confers and the person accepts; or a child carrying emotional endowment — an only or oldest child, or a gifted one), and **how suddenly** they pass from functioning fully to loss. | Also rewords an unquoted four-word transcript run in the same sentence. 'Head of the clan' stays: it is Bowen's written term in Ch15. |
| SP1-S07 | M1.A.13a (second sentence), line 281 | Tape 6 | 1979 Tape 6 (`theory/_LECTURES_1979.md` · T6.1): Bowen says he believes the most telling cue to how strongly a family will react is the functional role the dying person filled in it, with the potential for reaction highest when someone fully functioning dies within a short time. | The check restored the passage's hedge (Bowen gives it as his belief), which is the 2026-08-28 backlog item for M1.A.13a. T6.1 added. *Corrected in the check.* |
| SP1-S08 | M1.A.15 (Tape 4 sentence), line 294 | Tape 4 | 1979 Tape 4 (`theory/_LECTURES_1979.md` · C1): Bowen says a child still dependent on the parents for money and emotional support and living with them cannot be the one who does this work; it takes someone in a position to risk it, because once someone starts to differentiate, the family will threaten to disinherit them. | The current quotation joins two parts of the passage without marking the words left out. The gate and the no-material-stock rule are unchanged. |
| SP1-S09 | M1.A.18, What it is, line 301 | KB10 (Hijack) | **What it is.** The KB10 interview names **three** variables for societal process: a person's **way of thinking** (what the world looks like depends on the head observing it, and systems thinking gives the broader view), **differentiation of self**, then **how intense anxiety is**. | 'Bowen names' becomes the KB10 interview: the same passage as MR-3 in the explainer. |
| SP1-S10 | M1.A.18, Why it is needed, line 302 | KB01 (Otter; Speaker 2 = Bowen) | That is the misconception Bowen calls grotesque in the KB01 interview: taking differentiation to mean pulling a self away from the family by shouting the others down and making a show of being different. Pseudo-differentiation is therefore **not a separate error to be modelled**; it is what the move degrades into at low `systems_perspective` (M5.F.4). → `theory/_KERR_INTERVIEWS.md` · Correction 2 | KB01 (Otter) labels the turn as Bowen's, so naming him stays. The example that prompts it is Kerr's turn. The one-word term loses its quote marks. |
| SP1-S11 | M1.A.18a, line 305 | KB07 (Hijack) | **M1.A.18a** The **observable readout** for `systems_perspective` **MUST** be **two-sided**: emitting blame **or** praise is a loss, per the neutrality gauge — the KB07 interview holds that a person cannot fully see people in systems terms until they can be emotionally neutral, neither blaming nor praising. | The current text states as fact what the passage gives as an opinion; the paraphrase states it as a view. The requirement is unchanged. |
| SP1-S12 | M1.A.18b, line 316 | KB07 (Hijack) | Its **rate of change MUST** be coupled to `basic_level` — the KB07 interview describes undifferentiation as something like the cement or hardener that fixes a person's thinking in place, and says that where it can be broken up, the thinking can change more quickly. *Corrected 2026-08-28: the requirement made `basic_level` a **ceiling** on the attainable value. The source makes a **rate** claim, and hedges it — lower undifferentiation lets fixed thinking change faster. A ceiling may still be the right design, but it is `[I]`, not this passage. The speaker is also uncertain: the line falls between a Bowen-anchored clause and a Kerr-anchored one in an unmarked transcript, so cite the interview, not the man.* → `kb/kb07.md` · K07.3; M10.A.1 | Replaces the second sentence and the italic note; the note names the hedge instead of quoting it. The first sentence, the MUST NOT, is unchanged. |
| SP1-S13 | M1.B.8, line 362 | Tape 1 (two transcripts: Hijack-style and Otter) | 1979 Tape 1 states the systems rationale: Bowen treats negative and positive feeling alike here, since a negative feeling holds a family system together as well as a positive one; either kind invests psychic energy in someone else, and the sign changes only how people react. Valence changes the reaction, not the investment. | Tape 1, where both transcriptions agree (owner question Q4). |
| SP1-S14 | M1.B.10, line 374 | Tape 6 | **M1.B.10** A `Relationship` **MUST** hold `taboo_set`, which grows by default as each party learns what makes the other anxious; 1979 Tape 6 calls this the start of the communication cutoff in a marriage (`theory/_LECTURES_1979.md` · T6.4). → §4.6 | The quotation had no citation; Tape 6 · T6.4 added. Ch10 states the same mechanism in writing and could be cited instead. |
| SP1-S15 | M1.B.10a, line 376 | Tape 6 | **M1.B.10a** The set **MUST NOT** be implemented as monotone. A subject **MUST** be returnable by **purposeful mention under self-control**: 1979 Tape 6 says that raising the taboo subject on purpose, provided one can keep one's own anxious reaction to the other in check, can desensitize the reflex as a whole, and wishes people knew it; it adds that the response works like any other reflex, and that by the time a couple seeks help, repairing it is beyond what they can do alone. This is the differentiating move applied to a single topic, and it **MUST** carry the same gates (`M5.C`). → `theory/_LECTURES_1979.md` · T6.4 | Owner question Q7. The check restored the passage's form (a wish), its modal (can, not would) and its next point (a couple seldom repairs it alone). The MUSTs are unchanged. *Corrected in the check.* |
| SP1-S16 | M1.C.3a (the 'Verbatim:' sentence), line 392 | Tape 5 | 1979 Tape 5 (`theory/_LECTURES_1979.md` · C2), on the symptom patterns in which a family's anxiety is absorbed (distance, conflict between the spouses, illness in one of them, and the like): how intense and how fixed they are shows the level of anxiety; in well-differentiated people they are mild, appear when anxiety rises and then pass; lower on the scale they are more intense; and their intensity depends on differentiation and, most of all, on anxiety. The passage is about these patterns, not triangles; applying it to a triangle's routing capacity is the spec's inference (`kb/kb04.md` · K04.15). | Owner question Q7. The current text is labelled verbatim but changes a word and reverses the order of two pieces. The passage concerns symptom patterns, not triangles; the paraphrase says so and names the triangle application as the spec's inference (K04.15). *Corrected in the check.* |
| SP1-S17 | M1.C.3b (italic sentence), line 394 | KB04 (Hijack) | *This is why the KB04 interview reports, as a recent change, doing little with triangles any more and treating the number of triangles present as making no real difference (`kb/kb04.md` · K04.15): a statement about **coaching technique**, not about whether triangles operate.* | 'Bowen says late in life' becomes the KB04 interview; the timing is the transcript's own. K04.15 added. |
| SP1-S18 | M1.C.3c, detriangle self, line 397 | KB06 (Hijack) | - **detriangle self** — the actor withdraws their own emotional participation while staying in contact (`KB06`: after repeated visits home have revived the old family process, the aim is to observe it calmly and objectively and, using what one knows of triangles, to step calmly out of the primary triangle formed with both parents, and out of the wider family of origin). This is the outside position and it is what `M8` counts. | The check restored the passage's form (a stated aim, after repeated visits home) and the wider family of origin. No finding ID covers the passage, so the bare KB06 citation stays. *Corrected in the check.* |
| SP1-S19 | M1.D.4a (second sentence), line 424 | KB08 (Hijack) | The KB08 interview describes reserving *schizophrenia* for families where no member has it and calling the rest functional psychoses, and a private-practice test of treatability and of differentiation — whether anyone could be a self despite the others' objections — in which, if one person could, the work stayed with that person and a good result was expected. → `kb/kb08.md` · K08.2 | 'Bowen reserved … and ran' becomes the KB08 interview. The requirement's first sentence repeats a four-word transcript phrase without quote marks (adjacent issues). |
| SP1-S20 | M1.D.7a1, line 435 | KB10 (Hijack) | **M1.D.7a1** The three dials are **drivers of anxiety**, which is one of three variables the KB10 interview names for societal process — the person's way of thinking, differentiation of self, then **how intense anxiety is**. The dials therefore sit **one level below** that decomposition and **MUST NOT** be presented as the societal model. The interview's first variable, *mode of thinking*, has no representation here and is not currently implementable; that gap **MUST** be stated wherever societal results are reported. → `kb/kb10.md` · K10.1 | The whole requirement is given because the personal attribution recurs in two later sentences; each becomes the KB10 interview. The MUSTs are unchanged. |
| SP1-S21 | M1.E.2a (last clause), line 472 | KB03 (Hijack) | and, in the KB03 interview, an account of interviewing cancer families with no intent to fix anything reports that within a few months they were producing more ideas per session than private-practice families and were clearly doing better.<br><br>*The italic note in the same requirement:* *Corrected 2026-08-28: these had been called "an experiment and an independent replication". Neither is. The residents are a training exercise recounted from memory in the KB03 interview; the cancer families are a self-reported comparison against the speaker's own private practice, introduced in the same interview as confirming what the other speaker had said — the opposite of independent. The directive is well supported; the evidentiary vocabulary was not, and it failed this document's own standard (`§0.5`).* | Owner question Q5. 'Kerr, interviewing cancer families' becomes the KB03 interview. The check found that the italic note in the same requirement names both men for the KB03 accounts; it is reworded too. *Corrected in the check.* |
| SP1-S22 | M1.E.7, What a landed contact delivers, line 483 | KB07 (Hijack) | **What a landed contact delivers** — not instruction, but two things together: in the KB07 interview's account, leaving the session aware both that another way of responding existed and that one had been emotionally caught up in the situation. → `kb/kb07.md` · K07.4 | The current text names no one, and that is kept. The account adds that nothing the coach said caused the change. |
| SP1-S23 | M1.E.7, The rate MUST be low, line 484 | KB07 (Hijack) | **The rate MUST be low.** As recalled from the recipient's side, over two years of roughly six to eight contacts a year: the KB07 interview puts the contacts from which something useful came at probably four, perhaps five, in the two years. → `kb/kb07.md` · K07.4 `[#]` | The check restored the hedge on the count and gives the contact rate as recalled, not measured. The MUST and the [#] grade are unchanged. *Corrected in the check.* |
| SP1-S24 | M1.E.7, The binder-failure condition, line 485 | KB10 (Hijack) | **The binder-failure condition** is capacity exhaustion, carried over by analogy from the societal scale: the KB10 interview says a society's attitudes change once it has no other way left to deal with what is happening, and counts that change as part of regression, not insight. Pain is necessary and **not sufficient** — without frame supply, exhaustion **MUST** produce symptom escalation, not insight. → `kb/kb10.md` · K10.6 | Owner question Q7. The passage is about a society adapting, which the interview counts as regression; the paraphrase says the condition is carried over by analogy. The check kept the MUST sentence, which the draft had dropped. *Corrected in the check.* |
| SP1-S25 | M1.F.4a, line 514 | Tape 6 | **M1.F.4a** The **register** of a message **MUST** constrain the register of the reply, independently of content: 1979 Tape 6 reports that when Bowen raises the subject in tangential words, people reply in kind, against the deliberate use of plain ones — *death, die, bury* rather than *passed, gone, deceased*. → `theory/_LECTURES_1979.md` · T6.6 | Ch15, Bowen's published paper on death in the family, says the same in writing and may be quoted instead. |
| SP1-S26 | M2.A.0c, line 567 | Tape 3; KB14 (Hijack) | **M2.A.0c** Spouses **MUST** be paired at comparable **`basic_level`**, never `functional_level`. 1979 Tape 3 specifies the field: Bowen says that when spouses are said to match on differentiation, the base level is meant, not functioning, which shifts up and down easily. The KB14 interview gives the matching as one of the fairly constant facts from the family research — people choose spouses whose differentiation of self is nearly the same as their own — and as the usual case, including in couples who live together. → `kb/kb14.md` · K14.2; `theory/_LECTURES_1979.md` · B9 | The current KB14 quotation puts a bracketed word where the transcript has a noise tag. The check restored the hedge on the research finding. B9 and K14.2 kept. *Corrected in the check.* |
| SP1-S27 | M4.C.8 / M4.C.8a blockquote (first sentence), line 720 | KB12 (Hijack) | > The KB12 interview (`kb/kb12.md` · K12.6): focusing on feelings is almost like digging into a volcano — the more is exposed, the more keeps pouring out, and the nearest thing to a resolution comes only once the feelings are spent and the volcano tires. | The blockquote had no citation and sits under an FE03.6 citation, so it read as *Family Evaluation*; K12.6 added. The check restored two hedges. *Corrected in the check.* |
| SP1-S28 | M4.C.8 / M4.C.8a blockquote (second sentence), line 720 | KB12 (Hijack) | If the effort goes into the intellect instead, the feeling world stops being a volcano and becomes something small and orderly, which the person attends to unprompted, putting the feelings into words by way of the intellect and integrating them. | Misquote: the current phrase joins a word and a correction that the transcript keeps apart, and part of it is the interviewer's question. The check restored the passage's condition and its landing point. *Corrected in the check.* |
| SP1-S29 | M4.C.4, line 723 | KB07 (Hijack); KB12 (Hijack) | **M4.C.4** `systems_perspective` (M1.A.18) **MUST** fall with acute anxiety. Polarity capture is a **loss** in someone who had the perspective, not merely a correlate of never having had it: the KB07 interview holds that systems thinking has no place for polarized debate, and that anyone locked into a polarity has either not yet reached systems thinking or reached it and lost it; and the KB12 interview, that the better a person functions, the further they can see past polarities, and the worse, the more they are caught in them. → `kb/kb07.md` · K07.6, `kb/kb12.md` · K12.7 | Two quotations replaced. The KB07 passage does not itself tie the loss to acute anxiety; that link is the requirement's (K07.6, K12.7). |
| SP1-S30 | M4.C.5, line 732 | KB05 (Hijack) | The KB05 interview describes people taking years to reach the point of being with their mother without taking what she says as criticism, and reads hearing a parent as critical as a sign that the hearer is probably being critical or defensive. → `kb/kb05.md` · K05.8. This readout is cheaper and harder to game than one taken from the emitted move. | The transcript's pronoun is ambiguous; the paraphrase follows K05.8's reading, which the sentence order supports. K05.8 added. |
| SP1-S31 | M4.D (Consequence paragraph, last sentence), line 758 | KB10 (Hijack) | Anxiety-relieving action genuinely relieves anxiety — the KB10 interview describes band-aid laws passed to ease immediate anxiety and says that the more of them are passed, the more they feed the very problem they were meant to fix — so an agent selecting a binder is not making an error a longer horizon would correct. → `kb/kb10.md` · K10.7; M1.A.0 (the moves are instinct-level, not feeling-states) | 'Bowen on band-aid legislation' becomes the KB10 interview, and the paragraph's three references to the quotation become the passage. The *Family Evaluation* quotation in the same note is owner question Q8. |
| SP1-S32 | M5.B.3a, line 844 | Tape 6; KB05 (Hijack); KB08 (Hijack) | **M5.B.3a** `REDUCE_CUTOFF` **MUST** have a **floor as well as no ceiling**, and the target is **one open relationship, not universality** — 1979 Tape 6 (T6.7): no one can have open relationships with everyone, but there are signs that having one with those who matter most, even one important other, is a healthier way to live. Cutoff is load-bearing for the person maintaining it — the KB05 interview calls cutting off a lifeline that lets a person live and adjust — so it may be reduced only as fast as that person can absorb, and the far side of a cutoff has its own state and its own willingness. Total openness is also not the target: the KB08 interview, while recommending near-complete disclosure between spouses as a workable exercise, says that keeping a self seems to require holding some of it back from the other, and that if everyone in a family told everything, one person would be de-selfed in favour of the group self. With `L12.4`'s two-sided closeness band, four sources agree that **contact is not monotonically good**. → `kb/kb05.md` · K05.2, `kb/kb08.md` · K08.4 | Owner question Q7. The check added that the same interview recommends near-full disclosure between spouses, and restored the family limit and the hedge on holding back. The floor rests on Tape 6, KB05 and L12.4. *Corrected in the check.* |
| SP1-S33 | M5.C.1a, line 868 | KB06 (Hijack) | For a **reactive, explosive** family the technique **inverts**: calm self first, and the success criterion is **relative** — the KB06 interview says one need not control one's reactiveness completely; being less reactive than everyone else present is already a step toward a calmer family (`kb/kb06.md` · K06.4). | K06.4 added. The requirement is unchanged. |
| SP1-S34 | M5.C.1a (second paragraph), line 869 | KB08 (Hijack) | Applying the peace-agree technique to a reactive family is not merely ineffective. `KB08` records three outcomes of forcing conjoint contact on families already at capacity: acute psychosis; a severely disturbed young woman who, by a second-hand account from the early days, blinded herself with scissors after a young therapist required joint family meetings; and a patient who, two weeks into a research protocol of joint meetings that the family had been told it probably could not stand, left the hospital on a pass and killed herself. → `kb/kb05.md` · K05.1, `kb/kb06.md` · K06.4, `kb/kb08.md` · K08.1 | Owner question Q7. 'A protocol Bowen had warned against' becomes the interview's account. The check restored that the warning was an opinion given to that family as it entered the research, and that the second case is reported second-hand. *Corrected in the check.* |
| SP1-S35 | M5.D.7a, line 896 | KB01 (Otter: Speaker 2 = Bowen; Hijack also) | Bowen names the contrary reading as the field's characteristic misconception: in the KB01 interview he calls it grotesque that people treat differentiation as something finished in a weekend, or even an hour. *Corrected 2026-08-28; paraphrased at SP-1 (ASR rule). KB01's Otter transcript labels the speaker, so the attribution to Bowen stands.* | Owner question Q5. KB01 (Otter) labels the turn as Bowen's, so he may be named; the 2026-08-28 note had applied the Hijack rule. The two transcriptions differ on one word, which the paraphrase does not depend on. |
| SP1-S36 | M5.F.2a, line 943 | KB04 (Hijack), KB03 (Hijack), KB06 (Hijack), KB12 (Hijack), KB07 (Hijack) | Five independent forms across five interviews: the KB04 interview treats telling someone that one will keep out of their problem as itself a sign of being involved (`KB04` · K04.10); the KB03 interview calls a therapist's announcement that the trouble is the family's alone largely denial, since the therapist is involved (`KB03` · K03.3); the KB06 interview takes a claim to have settled one's attachment to one's parents as evidence of that attachment, probably the more so the more insistently it is made (`KB06` · K06.7); the KB12 interview describes a low-level I-position as selfish, dogmatic and forceful (`KB12` · K12.3); and the KB07 interview counts catching oneself diagnosing, criticising or praising someone as a lapse in systems perspective (`KB07` · K07.1). → `kb/_KB_PASS2.md` | Three quotations (KB04, KB03, KB06) replaced, and two unquoted bold phrases that are transcript wording reworded. The MUST statement and the Consequence line are unchanged. |
| SP1-S37 | M6.I.4 (§ M6 invariants table), line 965 | Tape 2 | Dyadic exchange conserves **pseudo-self** — defined in 1979 Tape 2 as what is *negotiable*: basic self is what a person believes in and stands by, the position they keep **whatever the pressure**; pseudo-self is belief taken on through schooling, reading and the like, and pressure from others can change it, where it cannot change basic self (`_LECTURES_1979.md` §A5). | Two Tape 2 quotations replaced. The negotiability test and the revision 27, 28 and 31 text are unchanged. |
| SP1-S38 | M10.A.2, line 1167 | KB10 (Hijack) | Deriving from `basic_level` is preferred wherever it is defensible, and the KB10 interview gives the principle, framed as a personal view: a theory consists of the fewest parts that still fit together into one account, not of everything that could be put in (`KB10` · K10.9). | The check restored the passage's framing as a personal view, and its form as a description of what a theory is, not a rule. 'The source states' becomes the KB10 interview. *Corrected in the check.* |
| SP1-S39 | M10.C.2, line 1206 | KB11 (Hijack) | and — **withdrawn at pass 3** — the eight-to-ten-generation figure, which `KB11` · K11.1 shows was rhetorical: the interview says the ten-generation idea, first raised at a 1977 conference on the double bind, was not put forward as the number of generations schizophrenia takes; three had been the earlier figure, and ten was, roughly, a way of saying that neither number is a fixed count and that the process runs over many generations. | Owner question Q1. The check removed a claim the passage does not make (that the earlier figure of three was also only a gesture) and restored the passage's hedge. The 1977 conference is in the passage. *Corrected in the check.* |
| SP1-S40 | M11.C.11, line 1323 | Tape 6 | — 1979 Tape 6 ranks a long-absent member's return, its example a member back from a long stay in an institution, as likely the strongest of the additions that disturb a family, because the family has adapted to the absence; the same passage says a family generally reacts more to a loss than to an addition, so it ranks the return among additions only and does not by itself support the return exceeding the removal (`_LECTURES_1979.md` T6.8) | Owner question Q7. The check added the passage's next point: the lecture ranks the return among additions only, and says a family generally reacts more to a loss than to an addition. This passage alone does not support the return exceeding the removal. *Corrected in the check.* |
| SP1-S41 | M11.F.1, line 1467 | KB09 (Hijack) | Among the principal aims the KB09 interview gives is an open theory, one whose new knowledge comes from research and from science, not from its author's statements. The interview identifies discipleship, which treats the theory as one man's creation, as the way a theory turns into a dogma and loses touch with science: the more people treat the theory as fact or creed, the more surely it dies out as a dogma. The model is a thing to be tested. → `kb/kb09.md` · K09.1 | Two KB09 passages. 'His stated goal' becomes the KB09 interview. The check restored that the open theory is one of several aims, and turned a certainty back into a matter of degree. *Corrected in the check.* |
| SP1-S42 | M11.F.2, line 1469 | KB02 (Hijack), KB10 (Hijack), KB13 (Hijack) | Three independent statements: the KB02 interview holds that drawing an analogy does not extend a theory (`KB02` · K02.4); the KB10 interview calls the family triangle seen again at work and in society an analogy, not a sound connection, since the two do not link up and differ in places (`KB10` · K10.8); and the KB13 interview says society rises and falls much as a family does, not identically, on the same principles (`KB13` · K13.4). → `kb/_KB_PASS2.md` | Three quotations replaced. The KB10 exchange corrects itself; the corrected form is used. The requirement is unchanged and no code or test reads this wording. |
| SP1-S43 | M11.F.3, line 1471 | KB13 (Hijack); also KB09 (Hijack) for the second statement | **A:** Asked about general systems being claimed as the basis of family theory, the KB13 interview does not see **general systems theory** as the answer, allowing that it may work within limits, and looks instead to **natural systems**, a frame that comes before every discipline (`KB13` · K13.2). In a later passage on what a theory is, the interview holds that a theory worth the name must be tied to nature — earth, seasons, tides — and places general systems, with mathematics, among what comes from human thought (`KB13` · K13.3). The KB09 interview, separately, says neither side fully understood the other's system but that the approaches are compatible up to a point (`KB09` · K09.3). A simulation is a general-systems artifact, so this is a real named objection. Both statements **MUST** be recorded; neither collapses into the other. → `kb/kb13.md` · K13.2, `kb/kb09.md` · K09.3<br><br>**B:** In the Epilogue to *Family Evaluation* Bowen states the objection in his own writing. Of the systems theories he names from the 1940s, one was "general systems theory applied to the human by von Bertalanffy", which "contained concepts from mathematics"; and "Existing systems theory did not fit well with the idea of the human as an evolutionary being… To get beyond mathematics and technology, I fashioned a natural systems theory" (`family_evaluation/fe11.md` · FE11.2). He also allows a limited fit: "certain aspects of human relationships can be described by any form of systems thinking, just as any mechanical machine can be a set of systems within systems. The analogy breaks down in the human when each family member has a brain that can control its part in the drama" (FE11.2). He declines one task rather than forbidding it: he is "no longer prepared to debate all the issues that go into connecting living matter with a technological extension of the brain. Anyone is welcome to that field if they wish" (FE11.2). A simulation is a general-systems artifact, so this is a real named objection. Both statements — the objection and the limited fit — **MUST** be recorded; neither collapses into the other. The KB13 and KB09 interviews make related points in speech and are cited, not quoted (`kb/kb13.md` · K13.2, `kb/kb09.md` · K09.3). → `family_evaluation/fe11.md` · FE11.2, `kb/kb13.md` · K13.2, `kb/kb09.md` · K09.3 | Owner question Q3. A keeps the interviews; the check restored KB13's allowance that general systems may work within limits and KB09's qualifier, and separated two KB13 passages that the current text joins into one argument. B restates the objection from the Epilogue (FE11.2, written, quotable) and changes what is recorded: the second statement becomes the Epilogue's limited fit. B was checked twice; the second check put all of the Epilogue's wording inside quote marks, kept the clause its analogy refers back to, and used the spec's citation form. *Corrected in the check.* |
| SP1-S44 | M15.A.5 (lines 1767–1768), line 1767 | Tape 3 | Functioning is what an observer sees and it shifts easily, rising with a compliment and falling with a criticism (1979 Tape 3; M2.A.0c); basic level is not observable in a single view. | Tape 3 quotation replaced; '1979 Tape 3' added as the citation (the sentence named only M2.A.0c). |
| SP1-S45 | M15.C.1 (lines 1816–1817), line 1816 | Tape 1 (both transcriptions) | `M1.B.8` requires `investment` to be **valence-blind**: conflict-laden preoccupation is *high* investment — 1979 Tape 1 holds that, for keeping a family system together, a negative feeling serves as well as a positive one. | Tape 1 quotation replaced, though both transcriptions agree (owner question Q4); '1979 Tape 1' added. |
| SP1-S46 | Revision 9 — the corpus-fidelity sweep (modal-inflation table), line 2444 | KB07 (Hijack) | \| `M1.A.18b` \| KB07, **hedged**: undifferentiation is roughly what sets thinking hard, and loosening it would let the thinking **change faster** (K07.3) \| a **ceiling** on the attainable value \| | In the revision 9 history record (owner question Q2). The paraphrase names the hedge instead of quoting it. |
| SP1-S47 | Revision 8 — what the estimator may read (lines 2489–2490), line 2489 | Tape 2 | Solid self is the position a person keeps **whatever the pressure**; pseudo-self is the **negotiable** part, which pressure from others can change (1979 Tape 2). | In the revision 8 history record (owner question Q2). Same passage as M6.I.4; '1979 Tape 2' added. |

### The explainer (`docs/model_explainer.md`)

| # | Where | Source | Proposed text | Note |
|---|---|---|---|---|
| SP1-E01 | §5 outside_ness two axes, bullet after the impingement table, line 367 | KB12 (Hijack) | - `[T]` `KB12` · K12.3 — in the KB12 interview, a highly differentiated person takes an I-position without going on the attack or on the defensive toward the other, so the manner in which an I-position is taken gives a fairly good indication of how well the person taking it functions. | Citation corrected: the line cites KS12.3 (Kerr's book, an unrelated finding); the words are KB12 · K12.3, as spec M1.A.9a cites them. The check restored the passage's hedge. *Corrected in the check.* |
| SP1-E02 | §2.x systems_perspective, 'Why it exists.' (MR-3), line 382 | KB10 (Hijack) | **Why it exists.** The KB10 interview names **three** main variables for societal process: first, how people think (what one sees varies with the mind doing the observing, and systems thinking gives the wider view); second, differentiation of self; third, how intense anxiety runs, with more anxiety bringing more societal problems and regression (`KB10` · K10.1). The three societal dials are drivers of the **third** only. This is the individual-level representation of the **first**. | Misquote: two passages joined as one list. 'Bowen names' becomes the KB10 interview (MR-3). The check restored that these are the main variables, not a complete list, and the passage's societal scope. *Corrected in the check.* |
| SP1-E03 | §2.x systems_perspective, 'How it moves' — rate of change, line 387 | KB07 (Hijack) | - **Rate of change** coupled to `basic_level` — the KB07 interview loosely likens undifferentiation to a setting agent that keeps a person's thinking fixed; if that can be broken up, the thinking can change faster (`KB07` · K07.3). `M1.A.18b` (a rate since revision 9; this line said ceiling until revision 12) | Hedge kept; the check restored the passage's condition. K07.3 added. *Corrected in the check.* |
| SP1-E04 | §4.6 taboo_set, 'Does:', line 594 | Tape 6 | **Does:** subjects are withdrawn from a tie, and growth is the **default** — each party learns what makes the other anxious and stops raising it, which is how communication between spouses starts to shut down (1979 Tape 6). | Seven-word Tape 6 quotation replaced; an inline Tape 6 citation added. Ch10 and Ch15 state the same mechanism in other words. |
| SP1-E05 | §4.6 taboo_set, 'Does not', line 596 | Tape 6 | **Does not:** ⚠ **it is not monotone.** Growth is a default, not a law. **In Tape 6 Bowen says that raising a taboo subject on purpose, provided one can keep one's own anxious reaction to the other in check, can desensitize the reflex as a whole**, and wishes people knew it; he adds that the response works like any other reflex, and that by the time a couple seeks help, repairing it is beyond what they can do alone. It is the differentiating move applied to a single topic, carrying the same gates as any other. `M1.B.10a` | Found by the unquoted-run scan: a bold ten-word transcript copy without quote marks. The check restored the passage's modal (can, not could), its form as a wish, and its next point (a couple seldom repairs it alone), as in M1.B.10a. *Corrected in the check.* |
| SP1-E06 | §6.1 Triangle table, `routing_capacity` row, Source cell (MR-8), line 801 | Tape 5 | `[T]` 1979 Tape 5 — Bowen says how intense the symptoms and relationship patterns become depends on differentiation and, most of all, on anxiety. | The check restored the strength of the passage's claim. The passage concerns symptoms and relationship patterns in the nuclear family; applying it to triangles is the project's inference (K04.15), and the text now says so. *Corrected in the check.* |
| SP1-E07 | §6.1 note 'The coach never works the triangles' (MR-4), line 805 | KB04 (Hijack) | This is how to read the late remark in the KB04 interview (`KB04` · K04.15): triangles now get little attention in the work, their number makes no real difference, this is described as a recent change, and the method, in an outline the interview admits is too simple to be the whole story, comes down to two steps — lower the anxiety, then pursue differentiation of self. It is a claim about **coaching technique**, not about whether triangles operate. | MR-4: the current ellipsis spans a line that may be the other speaker, so the paraphrase uses only what both readings share. 'Bowen means' becomes the KB04 interview. The check restored the passage's caveat on the method and its stronger statement about the number of triangles. *Corrected in the check.* |
| SP1-E08 | §14, paragraph 'And the wider framing was wrong' (MR-5), line 1452 | KB01 (Otter) | In a late interview with Michael Kerr (`KB01`), Bowen gives it as his view that a theory, to deserve the name, must be grounded in nature — in natural phenomena, the same forces that make grass grow and keep the earth turning — and more than once looks ahead to measurement: he is fairly sure the emotional bond between people has **measurable biochemical markers**, if researchers learn to identify them, and he says that with enough funding and enough subjects a biochemical evaluation could replace observing people. | Same passages as SP1-P01. KB01 (Otter) attribution kept; the view framing restored; a KB01 citation added. *Corrected in the check.* |

### The ledger (`docs/theory/_LEDGER.md`)

| # | Where | Source | Proposed text | Note |
|---|---|---|---|---|
| SP1-L01 | L06.2, paragraph 'KB03 supplies the mechanism', line 594 | KB03 (Hijack) | **`KB03` supplies the mechanism and the best evidence in the project.** "Therapeutic-itis": **the interview holds, as a belief with fairly good evidence behind it, that the problem began with the family's own efforts to help the patient**, so helping is the same operation. | Found by the unquoted-run scan: a bold eighteen-word transcript copy without quote marks. The paraphrase restores the hedge; the check reworded it so that it reads plainly. *Corrected in the check.* |
| SP1-L02 | L15.3, 'What survives', line 1314 | Tape 6 | the three tiers, and the rule that generates them — Bowen says he believes the **functional position** the dead person had held is the best indicator of how strongly the family will react, and puts the potential for reaction **highest when death comes quickly to someone who was still functioning fully** (Tape 6 · T6.1). The ordering is by **functional position plus suddenness**, not by role label. | Tape 6 quotation replaced, and with it the current text's silent repair of the ASR wording. The check restored the passage's hedge. T6.1 added. *Corrected in the check.* |
| SP1-L03 | L15.3, 'The grandmother pair is settled', line 1319 | Tape 6 | **The grandmother pair is settled, and neither earlier reading was right.** Tape 6: the patriarch's death is fairly likely to set off a shock wave, more so where the family is a closed emotional system, and his wife's death afterwards draws a **smaller** reaction — unless **she was the family's matriarch**, the respected one who carried responsibility, in which case the shock wave attaches to **her** death more than to her husband's (Tape 6 · T6.1). Whoever held the functional-doer position, either sex. | The quotation and a four-word transcript run in its lead-in replaced. The check restored the passage's condition (a closed family emotional system). *Corrected in the check.* |
| SP1-L04 | L16.2, paragraph 'KB04 adds the other half', line 1393 | KB04 (Hijack) | **`KB04` adds the other half.** The KB04 interview lists **four** ways of *absorbing* anxiety, distance first among them, where the 1979 lecture lists **three** that *end in symptoms*. | Attribution only: 'Bowen enumerates' becomes the KB04 interview. |
| SP1-L05 | L16.3 Calibration — WITHDRAWN at pass 3 (owner question 1), line 1404 | KB11 (Hijack) | `KB11` · K11.1 says otherwise: the interview explains that ten generations was put forward without meaning that schizophrenia needs ten generations to develop, that the earlier statement had been three, and that the figure was, loosely, a way of saying that the process is fixed at neither three generations nor ten but is **multigenerational**. | Owner question Q1, the same quotation as M10.C.2. The check restored the hedge and removed a claim stronger than the passage. *Corrected in the check.* |
| SP1-L06 | L17.2, paragraph 'KB04 corroborates and supplies the mechanism', line 1465 | KB04 (Hijack) | **`KB04` corroborates and supplies the mechanism** (K04.2). The interview defines reactivity as one way anxiety shows itself; in theory the two framings are both correct and amount to much the same, but, in the interview's view, reactivity is **easier to observe and get hold of** than anxiety, a vague notion by comparison. A drum analogy, put as a question in the interview — tension in the skin is the anxiety, the noise when struck is the reactivity — is accepted as accurate, with a stated dislike of the analogy itself. This answers the item the 1979 lectures left silent. | Two further short quotations in the same paragraph, missed by the scan, also replaced. The check restored a hedge and replaced 'Kerr's drum analogy' with an unattributed form (owner question Q5). *Corrected in the check.* |
| SP1-L07 | L17.5, 'CORRECTED at the Kerr interview', line 1508 | KB01 (Otter) | Bowen is **not** indifferent to formal or quantitative approaches: in his view nothing earns the name of theory unless it is **grounded in natural phenomena — the same forces that make grass grow and keep the earth turning**. And he actively anticipates measurement — he is fairly sure the emotional bond between people has **measurable biochemical markers**, if researchers learn to identify them, and he says that with enough funding and enough subjects **a biochemical evaluation could replace observing people** (`KB01`; `_KERR_INTERVIEWS.md` Correction 1). | KB01 (Otter) attribution kept. Three quotations replaced; the scope of the measurement remarks is stated. |
| SP1-L08 | L20.2, 'Two further detectors from the interviews', line 1699 | KB04 (Hijack) | **Two further detectors from the interviews.** `KB04` · K04.11: catching one's head inside the other's problem — at that point the therapist is **no longer of use** to the client — with a recovery move (a factual relationship question) and the return of humour as the signal. | Five-word quotation replaced; K04.11 added. |
| SP1-L09 | U1 — The marriage-ceremony break, Verdict, line 2397 | KB14 (Hijack) | **Verdict:** UNMODELLED, now closed. The KB14 interview states the phenomenon and offers no explanation for it: couples who got on well as friends before marriage commonly find the relationship going wrong from the wedding on, and the interview records this as an observed fact whose cause remains a puzzle (`KB14` · K14.3). | 'Bowen states' becomes the KB14 interview. The current text says the interview declines to explain; the passage only leaves the question open. The quotation also altered the transcript. K14.3 added. |

### `docs/agent_model_proposal.html`

| # | Where | Source | Proposed text | Note |
|---|---|---|---|---|
| SP1-P01 | §10 corpus reconciliation, framing note, paragraph 'And the wider framing this supported was wrong' (lines 1611-1619), line 1613 | KB01 (Otter) | In a late interview with Michael Kerr (<span class="m">KB01</span>), Bowen gives it as his view that a theory, to deserve the name, must be grounded in nature &mdash; in natural phenomena, the same forces that make grass grow and keep the earth turning &mdash; and more than once looks ahead to <em>measurement</em>: he is fairly sure the emotional bond between people has <b>measurable biochemical markers</b>, if researchers learn to identify them, and he says that with enough funding and enough subjects a biochemical evaluation could replace observing people. | KB01 (Otter) speaker labels allow attribution to Bowen; kept. The check restored the passage's framing as his view, in place of 'insists'. *Corrected in the check.* |


### Kept: written-source quotations and text that is not a quotation

| File, line | Class | Why kept |
|---|---|---|
| `bowen_agent_model_spec_v2.md`:96 | written-quote | Written source: FE02 (Family Evaluation, written). |
| `bowen_agent_model_spec_v2.md`:222 | written-quote | Written source: KS05.1 (Kerr). Its citation and its splice are adjacent issues. |
| `bowen_agent_model_spec_v2.md`:292 | written-quote | Written source: KS12 (Kerr, written; Bowen's ward-fire account as Kerr quotes it). |
| `bowen_agent_model_spec_v2.md`:294 | written-quote | Written source: Ch10 / Ch21 (the 22 papers, written). |
| `bowen_agent_model_spec_v2.md`:314 | not-a-quotation | Quotes the spec's own M7.A.2. |
| `bowen_agent_model_spec_v2.md`:416 | written-quote | Written source: KS04 (Kerr, written). |
| `bowen_agent_model_spec_v2.md`:472 | not-a-quotation | Quotes the project's earlier wording (a kb03.md heading). |
| `bowen_agent_model_spec_v2.md`:482 | not-a-quotation | Quotes the owner's decisions of 2026-10-11. |
| `bowen_agent_model_spec_v2.md`:573 | written-quote | Written source: KS07, KS12 (Kerr, written). |
| `bowen_agent_model_spec_v2.md`:715 | written-quote | Written source: KS23 (Kerr, written). |
| `bowen_agent_model_spec_v2.md`:758 | written-quote | In no extraction note and no transcript; traced 2026-10-22 to the book's Chapter 3 source text (Kerr, written). Owner question Q8. |
| `bowen_agent_model_spec_v2.md`:758 | not-a-quotation | The spec's own summary in quote marks. |
| `bowen_agent_model_spec_v2.md`:965 | not-a-quotation | The spec's own summary of M1.A.5c. |
| `bowen_agent_model_spec_v2.md`:1165 | not-a-quotation | Quotes the ledger's U9 entry, not a transcript. |
| `bowen_agent_model_spec_v2.md`:1321 | written-quote | Written source: KS12 (Kerr, written). |
| `bowen_agent_model_spec_v2.md`:1322 | not-a-quotation | The owner's own words (decisions of 2026-10-13 to 2026-10-15). |
| `bowen_agent_model_spec_v2.md`:2173 | written-quote | Written source: FE05 (Kerr, written). |
| `bowen_agent_model_spec_v2.md`:2173 | not-a-quotation | Quotes the project's extraction note on FE05.4. |
| `model_explainer.md`:191 | written-quote | Written source: KS05.3. |
| `model_explainer.md`:296 | written-quote | Written source: KS05.1. |
| `model_explainer.md`:335 | written-quote | Written source: Ch17 · L17.2. |
| `model_explainer.md`:598 | written-quote | Written source: KS04.10. |
| `model_explainer.md`:800 | written-quote | Written source: Ch17 · L17.1. |
| `model_explainer.md`:828 | written-quote | Written source: Ch20 · L20.1. |
| `model_explainer.md`:940 | written-quote | Written source: Ch11 · L11.3. |
| `model_explainer.md`:970 | written-quote | Written source: Ch11 · L11.3. |
| `model_explainer.md`:1446 | written-quote | Written source: Ch17 · L17.5. |
| `model_explainer.md`:2001 | written-quote | Written source: KS04.14. |
| `_CONVERGENCES.md`:222 | written-quote | Written source: Ch11 · L11.3. |
| `_CONVERGENCES.md`:317 | written-quote | Written source: Ch05. |
| `_LEDGER.md`:138 | written-quote | Written source: KS05.3. |
| `_LEDGER.md`:439 | written-quote | Written source: Ch04. |
| `_LEDGER.md`:509 | not-a-quotation | A theory term (also in Ch05). |
| `_LEDGER.md`:511 | written-quote | Written source: Ch05. |
| `_LEDGER.md`:695 | not-a-quotation | A theory term (Ch08). |
| `_LEDGER.md`:796 | written-quote | Written source: Ch09 · L09.2. |
| `_LEDGER.md`:1039 | written-quote | Written source: Ch11 · L11.3. |
| `_LEDGER.md`:1288 | written-quote | Written source: Ch15. |
| `_LEDGER.md`:1402 | not-a-quotation | Quotes the ledger's own pass-1 wording. |
| `_LEDGER.md`:1451 | written-quote | Written source: Ch17 · L17.2. |
| `_LEDGER.md`:1453 | written-quote | Written source: Ch17 · L17.2. |
| `_LEDGER.md`:1479 | not-a-quotation | A theory term (Ch17). |
| `_LEDGER.md`:1498 | written-quote | Written source: Ch17 · L17.5. |
| `_LEDGER.md`:1680 | written-quote | Written source: Ch20 · L20.1. |
| `_LEDGER.md`:1684 | written-quote | Written source: Ch20 · L20.1. |
| `_LEDGER.md`:1745 | written-quote | Written source: Ch21 · L21.1. |
| `_LEDGER.md`:2683 | written-quote | Written source: FE03.1. |

## 5. The test against new quotations

D1 asks for a test that fails on any new quotation of four or more words. The transcripts are not in the repository,
so the default suite, which CI runs on a blank machine, cannot read them. Three forms were measured against §4's
classified rows: 87 ASR quotations and 64 kept quotations (written sources and text that is not a
quotation), as the drafters classified them. "Caught" is how many of today's ASR quotations the test would have failed on; "flagged" is how many kept
quotations it fails on and that would need an allowlist entry.

**A. A citation test in the default suite.** For each quoted span of four or more words in a tracked `.md` or
`.html` file, fail if an ASR citation is near it, unless the file is exempt or the span is allowlisted. It needs no
transcripts, so it runs in CI and on both Macs. The citation pattern was measured in two forms: the scan's
(`KBnn`, `Knn.n`, `Tape N`, "Kerr–Bowen", "interview"), and a broadened one that adds `kb/kbNN`, `_KB_PASS2`,
`_LECTURES_1979`, `_KERR_INTERVIEWS`, "1979 lecture" and "Kerr–Bowen interview".

| Where the citation is looked for | Pattern | Caught | Flagged |
|---|---|---|---|
| Same line | scan's | 46 of 87 | 7 |
| Same line | broadened | 60 of 87 | 7 |
| Same paragraph | scan's | 52 of 87 | 23 |
| Same paragraph | broadened | 68 of 87 | 23 |

It misses a quotation whose citation sits in another paragraph or that has none (`M15.C.1` had none).

**B. A transcript fingerprint in the repository.** Commit hashed word runs of the transcripts, so the default suite
can match quoted spans directly.

| Run length | Caught | Flagged |
|---|---|---|
| 4 words | 87 of 87 | 29 |
| 5 words | 85 of 87 | 6 |
| 6 words | 82 of 87 | 1 |

A list of hashed word runs can be reversed: guess a run from a vocabulary, hash it, look it up, and chain the runs
that overlap. In a public repository that comes close to publishing the transcripts. Not recommended.

**C. A in the default suite, and the full transcript match as a local test (recommended).**

- **In the default suite:** A, with the broadened pattern and the same-paragraph scope. It catches 68 of 87; the 23
  kept quotations it flags go on the allowlist once, each with its written source.
- **Locally:** a test marked `transcripts`, deselected by default as `ensemble` is (`pytest.ini`), run with
  `python3 -m pytest -m transcripts`. It reads the transcript folder named by the environment variable
  `EPMODEL_TRANSCRIPTS` and skips, with the reason printed, when the folder is absent. It fails on (1) a quoted span in
  a non-exempt document that shares a run of four words with a transcript, and (2) an unquoted run of seven or more
  transcript words in §4's documents; each unless allowlisted. Measured: (1) catches 87 of 87 and flags 29 kept
  quotations, all written sources that Bowen repeats in speech; (2) flags four coincidences once §4's two unquoted
  copies are paraphrased.
- **When it runs:** before each milestone review (one line added to `docs/milestone_review_brief.md`'s checklist) and
  before any spec revision that changes a citation.

**The rules file.** The exempt list and the allowlist are editorial content, so they go in a JSON file beside
`tests/doc_claims.json`: `tests/asr_quotations.json`. Each exempt entry is a path or folder with its reason (§3's
table); each allowlist entry is a file, the span's text and its written source. A stale entry fails a test.

**What the default suite would flag today**, measured over all tracked documents with the same-line broadened form:
196 spans in 32 files. 73 are in documents this proposal changes (60 spec, 5 explainer, 4 ledger, 3 html, 1 `TODO.md`);
the rest are in documents §3 leaves as they are and the test exempts (extraction notes in `kb/`, `kerr_book/` and `family_evaluation/`, DECISIONS, the
revision 11 proposal, and the review records, which quote the spec's quotations). This proposal is exempt too: its
tables put written quotations beside ASR citations.

## 6. Questions for the owner

**The owner's answers, 2026-10-22.** All eight as recommended, each after a walk-through: Q1, part of SP-1; Q2, (a)
to (c) as recommended; Q3, alternative B, approved as a change to what `M11.F.3` records; Q4, paraphrase both Tape 1
passages; Q5, as recommended; Q6, form C; Q7, each mismatch to `TODO.md` as its own decision; Q8, keep the quotation
and trace it (the recommendation as revised on 2026-10-22, below). The §4 texts themselves are not yet approved.

**Q1. `KB11` in `M10.C.2`, `_LEDGER.md` L16.3 and DECISIONS A1 (the review's Q1; SP1-S39, SP1-L05).** Is the quotation an accepted
exception, or part of SP-1? Did approving A1 approve the quotation's form, or only the withdrawal of the
eight-to-ten-generation target?
*Recommendation: part of SP-1.* Paraphrase it in the spec and the ledger (§4); A1 stays unedited under the frozen-record
note. This proposal reads the 2026-08-25 approval as approving the withdrawal. Checking the paraphrase against the
passage restored a hedge and removed a claim the passage does not make (§4). If the owner approved the form as an
exception, the two current texts stay and go on the allowlist.

**Q2. Closed records (the review's Q2).** D1 decided that DECISIONS stays frozen with a note. Left open:
(a) the note's wording (§3); (b) whether `docs/proposal_rev11_emergence.md`, also an approved working record, is
treated the same way; (c) the spec's own revision history, which quotes a tape in the revision 8 entry and an
interview in the revision 9 entry (SP1-S47, SP1-S46).
*Recommendation:* (a) the wording in §3; (b) yes, the same note; (c) paraphrase the two history entries, because the
spec is the live contract and is read as a whole. The alternative for (c) is to keep them with the note.

**Q3. `M11.F.3` (the review's Q3, MR-1; SP1-S43).** Alternative A paraphrases the interviews. Alternative B restates the
objection from the Epilogue to *Family Evaluation* (`FE11` · FE11.2), which is Bowen's own writing and may be quoted.
*Recommendation: B.* The Epilogue is the latest primary Bowen text in the project and states the objection more
exactly than the interviews; A rests on unlabelled ASR. **B changes the requirement:** the second statement it records
becomes the Epilogue's limited fit, with the clause on where the analogy breaks down, in place of the interview's
compatibility up to a point; and it records that Bowen declined one line of work rather than forbidding it. It needs
the owner's approval as a requirement change.

**Q4. Tape 1 passages where both transcriptions agree** (`M1.B.8`, `M15.C.1`; SP1-S13, SP1-S45). `_LECTURES_1979.md` rule 2 lets a
second transcript license a quotation, and Tape 1 has one.
*Recommendation: paraphrase anyway.* Two ASR readings that agree are still ASR, one rule for every recorded source is
simpler to apply and to test, and the paraphrase loses nothing the requirement uses. The alternative is to keep both
under rule 2 and allowlist them.

**Q5. Naming a person.** `_SOURCE_QUALITY.md` allows `KB01` (Otter) to name the speaker by label, and records
question-shaped text in the Hijack set as Kerr's framing. Three places test this: `M5.D.7a`, where a 2026-08-28 note
applied the Hijack rule to `KB01` (SP1-S35); `M1.E.2a`'s italic note, which names Bowen and Kerr for two `KB03`
accounts by context (SP1-S21); and `_LEDGER.md` L17.2, which calls a question-shaped analogy in `KB04` Kerr's
(SP1-L06).
*Recommendation: follow `_SOURCE_QUALITY.md` as written, with one tightening.* Name Bowen in `M5.D.7a` (the Otter
label allows it). In `M1.E.2a`, cite the interview: attribution by context is what the rule excludes. In L17.2, use the
unattributed form the check proposed ("put as a question in the interview"), which is true whichever reading of the
rule is taken.

**Q6. The test's form (§5).** *Recommendation: C*, with A in the default suite at paragraph scope and the local test
at four-word runs.

**Q7. Requirements whose passage supports less than they say.** The paraphrase keeps to the passage; the requirement's
MUST is not changed in this revision. Each mismatch goes to `TODO.md` as its own item for a decision on the
requirement, because changing what a requirement requires is a rule change, not a citation fix:

| Requirement | What the passage supports |
|---|---|
| `M11.C.11` (SP1-S40) | Tape 6 ranks a long absentee's return as probably the strongest of the additions, and says a family generally reacts more to a loss. It does not by itself support the return exceeding the removal. `_LECTURES_1979.md` T6.8 makes the same step. |
| `M1.C.3a` (SP1-S16) | Tape 5 speaks of the intensity of symptoms and relationship patterns, not of triangles. |
| `M1.E.7` (SP1-S24) | The `KB10` passage is about a society coming to accept what it can no longer deal with; its use as the condition for a rise in `systems_perspective` is the spec's analogy. |
| `M5.B.3a` (SP1-S32) | The `KB08` statement on de-selfing is limited to a family where everyone does it, and the same interview recommends near-full disclosure between spouses as an exercise. |
| `M5.C.1a` (SP1-S34) | The `KB08` warning was given to one family before it entered the research, not as a general warning against the protocol. |
| `M1.B.10a` (SP1-S15) | Tape 6 states a closed subject's reopening as a wish, and says couples usually cannot reopen it themselves by the time they seek help. It supports the MUST as a possibility, not as a common outcome. |

*Recommendation:* as above, and the hedges the check restored (§4) are applied with the paraphrases, since they change
the evidence a requirement cites and not what it requires.

**Q8. A *Family Evaluation* quotation that no extraction note records.** `M4.D`'s italic note (spec line 758, the
paragraph of SP1-S31), the spec's revision history (line 2449) and `model_explainer.md:691` quote words credited to
*Family Evaluation* Ch3. The check found them in no extraction note and in no transcript. It is a written source, so
outside the ASR rule, but a quotation that cannot be traced should not stand as one. **Traced 2026-10-22:** the words
are in the book's Chapter 3 source text (`Family Evaluation_Chapter03_Individuality_and_Togetherness.txt`, in the
folder `family_evaluation/_INDEX.md` names), word for word once whitespace is normalised; the bold is the spec's.
Chapter 3 is Kerr's (`fe03.md`), so the grade is `[K]`; none of the three places credits it to Bowen. The actions the
passage describes are those aimed at the basic relationship problem that can make a system more flexible;
"differentiated actions" is the spec's own term for them.
*Recommendation (revised 2026-10-22, after the trace):* keep the quotation. Add the passage to `fe03.md` as a
finding, cite that finding in all three places, and name the book's subject in place of "differentiated actions".
*The earlier recommendation, withdrawn:* if the book lacked the words, replace the quotation in all three places
with two traceable quotations that make the same point. For line 758 the clause would read: *the corpus arguably cuts
the other way — Family Evaluation (Kerr's chapter) defines regression as actions taken "to relieve the anxiety of the
moment rather than being determined by a long-term view" (`family_evaluation/fe05.md` · FE05.4), and Kerr writes that
"A well-thought-out principle and long-term view can guide decision making in preference to acting on the feelings
and anxiety of the moment and a short-term view" (`kerr_book/ks00.md` · KS00.3).*

## 7. Build plan, after approval

Order: the rules file and the default-suite test first, written so they fail on today's tree (that failure is the
proof the test can fail); then the paraphrases until it passes; then the local test, the notes and headers, and the
spec revision. No change under `src/bowen/` or `config/bowen/`, so the engine hash does not move and no record reruns.

| ID | Acceptance criterion | Verified by |
|---|---|---|
| SP1.1 | Every §4 entry, as the owner approved it, replaces the current text at its line, and Q8's trace is applied: the passage is added to `fe03.md` as a finding and cited in all three places. | `python3 -m pytest -m transcripts` reports no finding outside the allowlist (SP1.3); SP1.2's test passes. |
| SP1.2 | A quoted span of four or more words beside an ASR citation, outside an exempt file and not allowlisted, fails the default suite. | `tests/test_asr_quotations.py::test_sp1_2_no_quotation_beside_asr_citation`, proved failing by `test_sp1_2_mutant_new_quotation_fails`, which feeds a made-up document with a new quoted span beside a `KB04` citation to the same checker. |
| SP1.3 | The local test fails on a quoted four-word transcript run and on an unquoted seven-word run, and skips with its reason when the transcripts are absent. | `tests/test_asr_quotations_local.py::test_sp1_3_no_transcript_runs` (marker `transcripts`); `test_sp1_3_mutant_transcript_run_fails` builds its mutant from the transcript at run time, so no transcript words are committed; `test_sp1_3_skips_without_transcripts`. |
| SP1.4 | Every exempt path and allowlisted span in `tests/asr_quotations.json` exists, and every entry has a reason. | `tests/test_asr_quotations.py::test_sp1_4_rules_entries_are_live` |
| SP1.5 | `DECISIONS FOR APPROVAL.md` and `proposal_rev11_emergence.md` carry the frozen-record note. | `tests/test_asr_quotations.py::test_sp1_5_frozen_records_carry_note` |
| SP1.6 | Every extraction note in §3 carries the header note, and no heading in `kb02.md`–`kb15.md` names Bowen as the speaker. | `test_sp1_6_extraction_notes_carry_header`, `test_sp1_6_hijack_headings_cite_the_interview` |
| SP1.7 | Spec revision 34 carries the §4 spec entries with MR-10, MR-12, MR-13, MR-15 and D3, and a revision history entry; `docs/spec_coverage.md` is regenerated, with the `KB06` phrase in `tools/spec_coverage_overrides.json`'s `M5.F.2a` note reworded. | `tests/test_doc_claims.py`, the coverage page's currency test, and the full suite. |
| SP1.8 | The documents that describe this are updated in the same commit: this proposal's status line, `CHANGELOG.md`, `TODO.md` (SP-1 closed, Q7 items added), `docs/theory/_STATUS.md`, `docs/REVIEWER_READING_LIST.md`, `docs/milestone_review_brief.md` (the local test's line), `docs/SPEC_REVIEW_CHECKLIST.md`. | `tests/test_doc_checklist.py`, `tests/test_doc_claims.py`; the status line is checked by hand at review. |

**Not code-testable:** whether each paraphrase says what its passage says. The check is human: the second reader's
pass recorded in §4, and the owner's review of this proposal. Before the build, `tests/` is searched for any test that
reads a sentence §4 replaces.

## 8. Limits

- The paraphrases were drafted and checked by Claude, one reader drafting and a second checking, not by the owner. The
  passage each one was checked against is itself ASR, so a paraphrase is only as good as the transcript's reading of
  the speech. Where two transcriptions exist (Tape 1, `KB01`), both were read.
- Every proposed text was checked for runs of four or more words shared with any transcript. 19 of the 66 texts
  (65 entries and M11.F.3's alternative B) have one. In all but two the run comes from a different transcript than the
  passage cited, and is ordinary English (*as one of the*, *caught up in the*) or theory vocabulary (*differentiation
  of self*). The two exceptions: *the head of the clan* in SP1-S06, which is Bowen's written term in Ch15, and a run
  in SP1-S18 that straddles the paraphrase's last words and the spec's own next sentence. Alternative B's runs are
  inside its Epilogue quotations or in the spec's own kept sentence.
- The classification of near matches as written sources is by hand and could be wrong either way. The kept table in
  §4 lists each with its source so a reviewer can check it.
- The scan read `.md` and `.html` files only. `tools/spec_coverage_overrides.json` holds one `KB06` phrase, found
  through the page generated from it; other JSON and Python files were not scanned.
- Option A catches about four in five of today's quotations. The rest are caught only by the local test, which runs
  only where the transcripts are.

## Adjacent issues found, not fixed

- `docs/model_explainer.md:598` says Bowen names only four contexts, citing `KS04.10`, Kerr's book. It is a written
  source, so outside SP-1, but the sentence may claim more than the passage; it needs its own check.
- `M1.A.5d` (spec line 222) quotes Kerr's written words from `ks05.md` KS05.1 but cites `ks14.md` · KS14.2, which
  holds only the clause after the quotation; KS05.1 should be added. The quotation also joins two KS05.1 passages and
  stops before the clause that states the harmful case (the 2026-08-28 backlog item). A written-source fix for
  revision 34.
- `TODO.md`'s 2026-08-28 backlog (lines 1035–1053: `M1.A.13a`, `M1.D.7b`, `M1.A.18d`, `M8.6a`, `M1.A.5d`,
  `M11.F.3a`) overlaps SP-1. §4 settles the quotation part of each; the build closes or narrows those items.
- `M1.D.4a`'s first sentence repeats a four-word transcript phrase without quote marks. It is ordinary English and is
  left; the owner may reword it without changing the requirement.
