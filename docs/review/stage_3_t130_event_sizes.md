# Stage 3 step 1 — T-130: the event sizes against Holmes and Rahe (1967)

**Status, 2026-10-24: not checked. The paper has not been read, so no value is confirmed or corrected and T-130
stays open.** `config/bowen/event_sizes.md`, the spec (`M1.F.6a`) and the explainer still say the values are
unchecked, which is true. Nothing here moves the engine hash.

The source to check against is Holmes, T. H. and Rahe, R. H. (1967), "The Social Readjustment Rating Scale",
*Journal of Psychosomatic Research* 11(2): 213–218, doi:10.1016/0022-3999(67)90010-4: the paper's own table of life
events with their mean values, not a reproduction of it.

## What was tried (2026-10-24)

| Source | Result |
|---|---|
| `https://doi.org/10.1016/0022-3999(67)90010-4` | redirected to `linkinghub.elsevier.com`, which returned a one-word "Redirecting" page with no article |
| `https://www.sciencedirect.com/science/article/pii/0022399967900104` | HTTP 403: the publisher refuses automated access |
| a web search for a PDF of the article | found the PubMed record (which confirms the citation, 1967 Aug; 11(2): 213–8) and reproductions of the scale only: teaching handouts and summary pages (testandcalc.com, simplypsychology.org, carepatron.com, brandeis.edu, mindtools.com, kresserinstitute.com). None is the article, so none was read for values |

The owner was asked for the PDF the same day and said they would provide it. When it is in hand, the comparison below
is filled from the paper's table, value by value with its page, and any correction or the removal of "unchecked"
waits for the batch's first commit (`docs/implementation_plan_version_2_stage_3.md` §3.6).

## The values to check

Each row is a row of `config/bowen/event_sizes.md` that takes a value from the scale. The paper's columns stay empty
until the paper is read. The item wording is to be checked too: the config's `item` column is not known to be the
paper's wording.

| kind | who | config's item | config's `lcu` | paper's item | paper's value | page | agrees |
|---|---|---|---|---|---|---|---|
| `GRIEF` | spouse | Death of spouse | 100 | not read | not read | — | — |
| `GRIEF` | relative | Death of a close family member | 63 | not read | not read | — | — |
| `JOB_LOSS` | self | Dismissal from work | 47 | not read | not read | — | — |
| `JOB_CHANGE` | self | Change to different line of work | 36 | not read | not read | — | — |
| `MOVE` | self | Change in residence | 20 | not read | not read | — | — |
| `SERIOUS_ILLNESS` | self | Personal injury or illness | 53 | not read | not read | — | — |
| `OWN_ILLNESS` | self | Personal injury or illness | 53 | not read | not read | — | — |
| `FAMILY_ILLNESS` | tie | Change in health of family member | 44 | not read | not read | — | — |
| `PARTNERING_LOAD` | self | Marriage | 50 | not read | not read | — | — |
| `RETIREMENT_LOAD` | tie | Retirement | 45 | not read | not read | — | — |
| `BIRTH_LOAD` | self | Gain of a new family member | 39 | not read | not read | — | — |
| `LAUNCH_LOAD` | self | Son or daughter leaving home | 29 | not read | not read | — | — |
| `FIND_WORK_LOAD` | self | Spouse begins or stops work | 26 | not read | not read | — | — |

`INCOME_LOSS` is outside the scale (the owner, Q8) and keeps `income_loss_intensity`, so it has nothing to check.
