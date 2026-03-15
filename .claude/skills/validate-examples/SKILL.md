---
name: validate-examples
description: Validate all example data files are cross-consistent and schema-valid
disable-model-invocation: true
---

# Validate Examples

Run all cross-consistency checks on the examples directory. Do NOT skip any step.

## Steps

1. **Check all creative files exist** — every `file_name` in `creative_metadata_samples.json` must exist in `examples/creatives/`

2. **Check creative names match** — the set of `creative_name` values in `creative_metadata_samples.json` must equal the set of `creative_name` values in `campaign_performance.csv`

3. **Validate JSON against Pydantic schemas**:
   ```python
   from omni_proof.brand_extraction.models import BrandProfile
   from omni_proof.ingestion.schemas import CreativeMetadata
   ```
   - `brand_profile.json` must load as `BrandProfile(**data)`
   - Each record in `creative_metadata_samples.json` must validate against `CreativeMetadata` (excluding extra fields like `creative_name`, `file_name`, `variant`, `treatment`)

4. **Validate compliance samples** — each entry in `compliance_samples.json` must have required fields: `asset_id`, `asset_name`, `passed`, `score`, `violations`, `evidence_sources`

5. **Run the demo** — execute `python examples/demo.py` and confirm it completes without errors

6. **Report results** — output a summary table:
   ```
   | Check                        | Status |
   |------------------------------|--------|
   | Creative files exist         | PASS/FAIL |
   | Creative names match CSV     | PASS/FAIL |
   | BrandProfile schema          | PASS/FAIL |
   | CreativeMetadata schema      | PASS/FAIL |
   | Compliance samples valid     | PASS/FAIL |
   | Demo runs clean              | PASS/FAIL |
   ```
