# Canonical vocabulary format

Store UTF-8 JSON arrays of records with `word` (Chinese lexical form),
`min_hsk_level` (integer 1–9), `standard` (exact source standard/version),
`source` (citation or URL), and `license` (redistribution terms).
Optional `notes` records normalization decisions. `schema.example.json` is
illustrative data, not an official HSK vocabulary list or a JSON Schema.

Select one licensed source before study collection. Preserve source spelling;
normalize Unicode to NFC and trim outer whitespace. Within one standard, use
one record per normalized word and retain its minimum attested level. Audit
duplicates and conflicting levels against the source rather than silently
merging incompatible standards. Preserve provenance and license for each row.

No licensed source is bundled. The prompt manifest is a development fixture:
its reading vocabulary and grammar requirements are empty, reading and exam
lengths are provisionally 300 characters, and grammar explanations use an
explicit ungraded placeholder. It is not study-ready. Replace these defaults
with source-backed, expert-validated requirements and freeze the prompt version
and lengths before generation. No manifest is written during import.
