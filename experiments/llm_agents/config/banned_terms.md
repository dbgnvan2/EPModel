# LLM-agent experiment — terms no prompt may contain

> The vocabulary lint (`docs/plan_v3_llm_agent_experiment.md` §3.4, LX.4). Every prompt the renderer produces, in
> every arm, for every person, is checked against every term here and against every person's name in the family file
> (`config/bowen/family_phase_c.md`, its `name` and `id` columns, read at check time so a renamed person is still
> caught). A build whose prompts contain one does not run.
>
> Matching is case-insensitive and whole-word: a term matches when no letter stands directly before or after it
> (`kin` does not match `taking`). A term with a space or a hyphen matches only as written; the other spellings are
> listed as rows of their own.
>
> * `class` — why the term is here: `theory` (the theory's own terms), `family` (family and role words), `act` (the
>   act kinds' names in `config/bowen/event_kinds.md`), `clinical` (clinical and help-seeking words).

| term | class |
|---|---|
| Bowen | theory |
| Kerr | theory |
| differentiation | theory |
| differentiated | theory |
| undifferentiated | theory |
| undifferentiation | theory |
| fusion | theory |
| fused | theory |
| triangle | theory |
| triangles | theory |
| triangling | theory |
| triangulation | theory |
| detriangle | theory |
| detriangling | theory |
| cutoff | theory |
| cut off | theory |
| cut-off | theory |
| emotional | theory |
| emotionally | theory |
| projection | theory |
| multigenerational | theory |
| sibling position | theory |
| nuclear | theory |
| chronic | theory |
| acute | theory |
| reactivity | theory |
| reactive | theory |
| overfunction | theory |
| overfunctioning | theory |
| over-functioning | theory |
| underfunction | theory |
| underfunctioning | theory |
| under-functioning | theory |
| pursuer | theory |
| pursuit | theory |
| distancer | theory |
| distancing | theory |
| self-directed | theory |
| automatic | theory |
| functional level | theory |
| basic level | theory |
| pseudo-self | theory |
| togetherness | theory |
| individuality | theory |
| nodal | theory |
| relief | theory |
| family | family |
| families | family |
| mother | family |
| father | family |
| mom | family |
| dad | family |
| parent | family |
| parents | family |
| child | family |
| children | family |
| kid | family |
| kids | family |
| son | family |
| daughter | family |
| sister | family |
| brother | family |
| sibling | family |
| siblings | family |
| husband | family |
| wife | family |
| spouse | family |
| partner | family |
| marriage | family |
| married | family |
| divorce | family |
| grandparent | family |
| grandmother | family |
| grandfather | family |
| grandchild | family |
| aunt | family |
| uncle | family |
| cousin | family |
| baby | family |
| teenager | family |
| relative | family |
| relatives | family |
| kin | family |
| household | family |
| generation | family |
| generations | family |
| coach | family |
| coaching | family |
| professional | family |
| PURSUE | act |
| DISTANCE | act |
| CONFLICT | act |
| OVERFUNCTION | act |
| UNDERFUNCTION | act |
| I-POSITION | act |
| STAY-IN-CONTACT | act |
| stay in contact | act |
| REDUCE_CUTOFF | act |
| PREVENT_ALIGNMENT | act |
| FRAME_AMBIGUITY | act |
| SPLIT | act |
| DISPLACE | act |
| PROVOKE | act |
| WITHHOLD | act |
| therapy | clinical |
| therapist | clinical |
| therapeutic | clinical |
| counselor | clinical |
| counsellor | clinical |
| counseling | clinical |
| counselling | clinical |
| clinician | clinical |
| clinical | clinical |
| doctor | clinical |
| Dr | clinical |
| patient | clinical |
| anxiety | clinical |
| anxious | clinical |
| stress | clinical |
| symptom | clinical |
| symptoms | clinical |
| diagnosis | clinical |
| disorder | clinical |
| depression | clinical |
| trauma | clinical |
| mental | clinical |
| illness | clinical |
