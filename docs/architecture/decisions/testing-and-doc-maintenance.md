# Testing and document maintenance — decisions

Decision log for [Testing and document maintenance](../testing-and-doc-maintenance.md). Entries are append-only: a
replaced entry gets a `Superseded by #M: <clause>.` note appended and is
never rewritten or deleted. The detail doc describes the current design.

- The architecture path names the root of an architecture set, the root plus the `docs/architecture/*.md` files it links to one level deep, so an oversized single document can be split while root-only identities stay unchanged (#1454).
- The `legacy` architecture-impact parser mode keeps the earlier synonym normalization, selected only where stored or historical text is decoded (#916).
- A reviewer's own missing or invalid `architecture_impact.status` is re-asked once with the frozen prompt and session instead of being repaired, because repair cannot supply the reviewer's decision (#1185).
