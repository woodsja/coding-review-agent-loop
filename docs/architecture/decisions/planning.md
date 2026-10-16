# Planning and implementation — decisions

Decision log for [Planning and implementation](../planning.md). Entries are append-only: a
replaced entry gets a `Superseded by #M: <clause>.` note appended and is
never rewritten or deleted. The detail doc describes the current design.

- Implementation claims may declare `test_level` and coverage gaps, and one post-acceptance gate checks the coverage map against the pushed head with at most one same-session re-ask, never through repair or the retry budget (#1290).
- Row-ID defects in matrix claims are claim defects, dropped with one degradation record instead of rejecting the envelope, and every copy of a duplicate is dropped so no arbitrary copy becomes coverage (#920, #926).
- The remaining claim-scope defects also drop the claim with one record, while authority violations and hard bounds stay fatal and are checked first (#927).
- An `execution_refs` list of more than eight well-typed refs is de-duplicated and trimmed to its first eight distinct selectors with a caveat, rather than being fatal (#1300).
- Catalog-authority rejections never route to structured repair, since reformatting cannot change them; the run stops once as a `semantic-evidence-rejection` (#990).
- The approved-plan implementation hand-off gets exactly one coder re-ask for non-passing or non-authoritative selector rejections, and a local `npx playwright test` spelling is recognized only when the local binary exists, which is then launched directly (#1240).
- Package-script resolution reads `package.json` statically and adopts only a single simple command the recognizer already accepts; package.json logic is never executed (#1294).
- Hierarchical execution, where a planning child recommends another staged topology, is owned by #720, so such a child stops before decision persistence or decomposition (#720).
