# Stage 3 step 1 — T-130: the event sizes against Holmes and Rahe (1967)

**Status, 2026-10-24: checked. All thirteen values agree with the paper's Table 3. Two item names differ from the
paper's wording, and two differ by an article; no value needs correcting.** Following the plan
(`docs/implementation_plan_version_2_stage_3.md` §3.6, step 6), "unchecked" comes out of `config/bowen/event_sizes.md`,
the spec (`M1.F.6a`) and the explainer in the batch's first commit, together with the item wording below. T-130 is
ticked in that commit. Nothing here moves the engine hash.

The source is Holmes, T. H. and Rahe, R. H. (1967), "The Social Readjustment Rating Scale", *Journal of Psychosomatic
Research* 11(2): 213–218, doi:10.1016/0022-3999(67)90010-4. The owner supplied a scan of the article on 2026-10-24 as
a PDF (`Holmes1967SRRS.pdf`, 6 pages, journal pages 213–218). Its embedded title, `PII: 0022-3999(67)90010-4`,
matches the article's DOI. The values were read from the page image of Table 3, "Social readjustment rating scale",
on journal page 216 (PDF page 4), and checked against the scan's text layer. The PDF is the owner's copy and is not
added to the repository.

What the values are, according to the paper (p. 215): each item's mean score across the 394 raters, divided by 10, in
rank order. Marriage was the anchor: raters were told it was worth 500 (p. 213), so it appears as 50.

## Earlier attempts (2026-10-24)

| Source | Result |
|---|---|
| `https://doi.org/10.1016/0022-3999(67)90010-4` | redirected to `linkinghub.elsevier.com`, which returned a one-word "Redirecting" page with no article |
| `https://www.sciencedirect.com/science/article/pii/0022399967900104` | HTTP 403: the publisher refuses automated access |
| a web search for a PDF of the article | found the PubMed record and reproductions of the scale only. None is the article, so none was read for values |

## The comparison

Each row is a row of `config/bowen/event_sizes.md` that takes a value from the scale. "Rank" is the paper's rank in
Table 3. All rows are on p. 216.

| kind | who | config's item | config's `lcu` | paper's item | rank | paper's value | value agrees | wording |
|---|---|---|---|---|---|---|---|---|
| `GRIEF` | spouse | Death of spouse | 100 | Death of spouse | 1 | 100 | yes | same |
| `GRIEF` | relative | Death of a close family member | 63 | Death of close family member | 5 | 63 | yes | config adds "a" |
| `JOB_LOSS` | self | Dismissal from work | 47 | Fired at work | 8 | 47 | yes | **differs** |
| `JOB_CHANGE` | self | Change to different line of work | 36 | Change to different line of work | 18 | 36 | yes | same |
| `MOVE` | self | Change in residence | 20 | Change in residence | 32 | 20 | yes | same |
| `SERIOUS_ILLNESS` | self | Personal injury or illness | 53 | Personal injury or illness | 6 | 53 | yes | same |
| `OWN_ILLNESS` | self | Personal injury or illness | 53 | Personal injury or illness | 6 | 53 | yes | same |
| `FAMILY_ILLNESS` | tie | Change in health of family member | 44 | Change in health of family member | 11 | 44 | yes | same |
| `PARTNERING_LOAD` | self | Marriage | 50 | Marriage | 7 | 50 | yes | same |
| `RETIREMENT_LOAD` | tie | Retirement | 45 | Retirement | 10 | 45 | yes | same |
| `BIRTH_LOAD` | self | Gain of a new family member | 39 | Gain of new family member | 14 | 39 | yes | config adds "a" |
| `LAUNCH_LOAD` | self | Son or daughter leaving home | 29 | Son or daughter leaving home | 23 | 29 | yes | same |
| `FIND_WORK_LOAD` | self | Spouse begins or stops work | 26 | Wife begin or stop work | 26 | 26 | yes | **differs** |

`INCOME_LOSS` is outside the scale (the owner, Q8) and keeps `income_loss_intensity`, so it has nothing to check.

## The wording differences

- **`JOB_LOSS`.** The paper's Table 3 says "Fired at work"; "Dismissal from work" is the wording of later
  reproductions. The value is the same.
- **`FIND_WORK_LOAD`.** The paper's Table 3 says "Wife begin or stop work". The config's "Spouse begins or stops work"
  makes the item gender-neutral. In the paper, the rater is the one readjusting to a wife's change; the model gives
  the load to the person who finds work (`who` = self). That mapping is the model's editorial choice (`[I]`, the
  owner's Q6) and is not something the paper states.

Decided as recommended under the owner's instruction of 2026-10-24, pending the owner's review: at step 6 the config's
`item` column quotes the paper's wording for these two rows, and the `FIND_WORK_LOAD` row says that the
gender-neutral reading and the `who` mapping are the model's. The two missing articles are corrected to the paper's
wording at the same time.
