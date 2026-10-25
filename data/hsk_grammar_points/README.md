# Canonical grammar format

Store UTF-8 JSON arrays with `grammar_id` (stable identifier), `pattern`
(grammar point), `min_hsk_level` (integer 1–9), `description`, `examples`
(array of Chinese strings), `standard` (exact version), `source` (citation
or URL), and `license` (redistribution terms). Optional `notes` preserves
normalization decisions. `schema.example.json` is illustrative data, not an
official HSK grammar inventory or a JSON Schema.

Normalize strings to NFC and trim outer whitespace. IDs must be unique in a
source standard; audit duplicate patterns and contradictory levels manually.
Do not combine standards without explicit provenance. Preserve licensed source
wording and document any edits to descriptions or examples.

No licensed source is bundled. Manifest grammar points are visibly marked
development placeholders and make no HSK-level claim. Replace them with verified
source-backed assignments before study collection. The `dev-1` manifest version
and 300-character reading/exam lengths are provisional engineering defaults.
