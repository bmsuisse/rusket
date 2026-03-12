You are a focused subagent reviewer for a single holistic investigation batch.

Repository root: /Users/dominikpeter/DevOps/rusket
Blind packet: /Users/dominikpeter/DevOps/rusket/.desloppify/review_packet_blind.json
Batch index: 16
Batch name: design_coherence
Batch rationale: seed files for design_coherence review

DIMENSION TO EVALUATE:

## design_coherence
Are structural design decisions sound — functions focused, abstractions earned, patterns consistent?
Look for:
- Functions doing too many things — multiple distinct responsibilities in one body
- Parameter lists that should be config/context objects — many related params passed together
- Files accumulating issues across many dimensions — likely mixing unrelated concerns
- Deep nesting that could be flattened with early returns or extraction
- Repeated structural patterns that should be data-driven
Skip:
- Functions that are long but have a single coherent responsibility
- Parameter lists where grouping would obscure meaning — do NOT recommend config/context objects or dependency injection wrappers just to reduce parameter count; only group when the grouping has independent semantic meaning
- Files that are large because their domain is genuinely complex, not because they mix concerns
- Nesting that is inherent to the problem (e.g., recursive tree processing)
- Do NOT recommend extracting callable parameters or injecting dependencies for 'testability' — direct function calls are simpler and preferred unless there is a concrete decoupling need

YOUR TASK: Read the code for this batch's dimension. Judge how well the codebase serves a developer from that perspective. The dimension rubric above defines what good looks like. Cite specific observations that explain your judgment.

Mechanical scan evidence — navigation aid, not scoring evidence:
The blind packet contains `holistic_context.scan_evidence` with aggregated signals from all mechanical detectors — including complexity hotspots, error hotspots, signal density index, boundary violations, and systemic patterns. Use these as starting points for where to look beyond the seed files.

Seed files (start here):
- rusket/mlflow.py
- scripts/gen_api_reference.py
- examples/11_instacart_recommender.py
- rusket/model.py
- rusket/als.py
- rusket/transactions.py
- rusket/spark.py
- rusket/model_selection.py
- rusket/vector_export.py
- rusket/pipeline.py
- examples/09_online_retail_basket_analysis.py
- rusket/grouped.py
- rusket/bert4rec.py
- rusket/eclat.py
- examples/10_online_retail_hupm_profit.py
- examples/07_als_pca_visualization.py
- rusket/_embedding_mixin.py
- examples/05_large_scale.py
- examples/08_spark_guide_translation.py
- examples/15_time_aware_sequences.py
- fix_pyright.py
- rusket/__init__.py
- rusket/_config.py
- rusket/_core.py
- rusket/_validation.py
- rusket/association_rules.py
- rusket/content_based.py
- rusket/evaluation.py
- rusket/faiss_ann.py
- rusket/fm.py
- rusket/gpu.py
- rusket/optuna.py
- rusket/svd.py
- scripts/eval_doc_snippets.py
- scripts/gen_llm_txt.py
- scripts/sync_changelog.py
- rusket/bpr.py
- rusket/ease.py
- rusket/fpgrowth.py
- rusket/fpmc.py
- rusket/lcm.py
- rusket/lightgcn.py
- rusket/nmf.py
- rusket/pacmap.py
- rusket/rules.py
- patch_mlflow.py
- rusket/cuda.py
- rusket/item_knn.py

Mechanical concern signals — investigate and adjudicate:
Overview (48 signals):
  design_concern: 25 — examples/05_large_scale.py, examples/07_als_pca_visualization.py, ...
  mixed_responsibilities: 12 — examples/09_online_retail_basket_analysis.py, examples/11_instacart_recommender.py, ...
  duplication_design: 9 — rusket/bpr.py, rusket/ease.py, ...
  coupling_design: 1 — rusket/_embedding_mixin.py
  structural_complexity: 1 — rusket/transactions.py

For each concern, read the source code and report your verdict in issues[]:
  - Confirm → full issue object with concern_verdict: "confirmed"
  - Dismiss → minimal object: {concern_verdict: "dismissed", concern_fingerprint: "<hash>"}
    (only these 2 fields required — add optional reasoning/concern_type/concern_file)
  - Unsure → skip it (will be re-evaluated next review)

  - [coupling_design] rusket/_embedding_mixin.py
    summary: Coupling pattern — assess if boundaries need adjustment
    question: Is the coupling intentional or does it indicate a missing abstraction boundary?
    evidence: Flagged by: coupling, smells
    evidence: [coupling] Implicit host contract: EmbeddingMixin depends on 3 undeclared self attrs (_item_labels, item_factors, pacmap)
    fingerprint: 5757d17d70f80a30
  - [design_concern] examples/05_large_scale.py
    summary: Design signals from orphaned, smells
    question: Is this file truly dead, or is it used via a non-import mechanism (dynamic import, CLI entry point, plugin)?
    evidence: Flagged by: orphaned, smells
    evidence: [orphaned] Orphaned file (79 LOC): zero importers, not an entry point
    fingerprint: 2a2f90094ad5d6ca
  - [design_concern] examples/07_als_pca_visualization.py
    summary: Design signals from orphaned, smells
    question: Is this file truly dead, or is it used via a non-import mechanism (dynamic import, CLI entry point, plugin)?
    evidence: Flagged by: orphaned, smells
    evidence: [orphaned] Orphaned file (111 LOC): zero importers, not an entry point
    fingerprint: 0fd56e2d61af2e57
  - [design_concern] examples/08_spark_guide_translation.py
    summary: Design signals from orphaned, smells
    question: Is this file truly dead, or is it used via a non-import mechanism (dynamic import, CLI entry point, plugin)?
    evidence: Flagged by: orphaned, smells
    evidence: [orphaned] Orphaned file (115 LOC): zero importers, not an entry point
    fingerprint: f4a6a77145b8b80f
  - [design_concern] examples/10_online_retail_hupm_profit.py
    summary: Design signals from orphaned, smells
    question: Is this file truly dead, or is it used via a non-import mechanism (dynamic import, CLI entry point, plugin)?
    evidence: Flagged by: orphaned, smells
    evidence: [orphaned] Orphaned file (162 LOC): zero importers, not an entry point
    fingerprint: a76bab3ce25ff7e6
  - [design_concern] examples/15_time_aware_sequences.py
    summary: Design signals from orphaned, smells
    question: Is this file truly dead, or is it used via a non-import mechanism (dynamic import, CLI entry point, plugin)?
    evidence: Flagged by: orphaned, smells
    evidence: [orphaned] Orphaned file (96 LOC): zero importers, not an entry point
    fingerprint: 2d32b6d032628f63
  - [design_concern] fix_pyright.py
    summary: Design signals from orphaned, smells
    question: Is this file truly dead, or is it used via a non-import mechanism (dynamic import, CLI entry point, plugin)?
    evidence: Flagged by: orphaned, smells
    evidence: [orphaned] Orphaned file (33 LOC): zero importers, not an entry point
    fingerprint: a88b0c90521b4a29
  - [design_concern] rusket/__init__.py
    summary: Design signals from cycles, smells
    question: Review the flagged patterns — are they design problems that need addressing, or acceptable given the file's role?
    evidence: Flagged by: cycles, smells
    evidence: [cycles] Import cycle (21 files): rusket/__init__.py -> rusket/als.py -> rusket/association_rules.py -> rusket/bpr.py -> rusket/content_based.py -> +16
    fingerprint: 9ebc222b78d686c5
  - [design_concern] rusket/_config.py
    summary: Design signals from smells
    question: Review the flagged patterns — are they design problems that need addressing, or acceptable given the file's role?
    evidence: Flagged by: smells
    evidence: [smells] 2x Except handler silently suppresses error (pass/continue, no log)
    fingerprint: b01b6d503a994865
  - [design_concern] rusket/_core.py
    summary: Design signals from smells
    question: Review the flagged patterns — are they design problems that need addressing, or acceptable given the file's role?
    evidence: Flagged by: smells
    evidence: [smells] 1x High cyclomatic complexity (>12 decision points)
    fingerprint: 4f8bdbd10db4a2b0
  - [design_concern] rusket/_validation.py
    summary: Design signals from smells
    question: Review the flagged patterns — are they design problems that need addressing, or acceptable given the file's role?
    evidence: Flagged by: smells
    evidence: [smells] 1x High cyclomatic complexity (>12 decision points)
    fingerprint: e77c64c1977d6de3
  - [design_concern] rusket/association_rules.py
    summary: Design signals from smells
    question: Review the flagged patterns — are they design problems that need addressing, or acceptable given the file's role?
    evidence: Flagged by: smells
    evidence: [smells] 1x Too many optional params — consider a config object
    fingerprint: 8f7ff573b6477f00
  - [design_concern] rusket/content_based.py
    summary: Design signals from smells
    question: Review the flagged patterns — are they design problems that need addressing, or acceptable given the file's role?
    evidence: Flagged by: smells
    evidence: [smells] 1x Function-level import (possible circular import workaround)
    fingerprint: 70f1d9ab0ca4df5d
  - [design_concern] rusket/evaluation.py
    summary: Design signals from smells
    question: Review the flagged patterns — are they design problems that need addressing, or acceptable given the file's role?
    evidence: Flagged by: smells
    evidence: [smells] 2x Except handler silently suppresses error (pass/continue, no log)
    fingerprint: ff2a27d333fc9946
  - [design_concern] rusket/faiss_ann.py
    summary: Design signals from smells
    question: Review the flagged patterns — are they design problems that need addressing, or acceptable given the file's role?
    evidence: Flagged by: smells
    evidence: [smells] 1x Too many optional params — consider a config object
    fingerprint: 6341f561342e67c0
  - [design_concern] rusket/fm.py
    summary: Design signals from smells
    question: Review the flagged patterns — are they design problems that need addressing, or acceptable given the file's role?
    evidence: Flagged by: smells
    evidence: [smells] 1x Too many optional params — consider a config object
    fingerprint: 16901445746a00cc
  - [design_concern] rusket/gpu.py
    summary: Design signals from facade, orphaned
    question: Is this file truly dead, or is it used via a non-import mechanism (dynamic import, CLI entry point, plugin)?
    evidence: Flagged by: facade, orphaned
    evidence: [orphaned] Orphaned file (38 LOC): zero importers, not an entry point
    fingerprint: a652d78ddf7dcb81
  - [design_concern] rusket/mlflow.py
    summary: Design signals from global_mutable_config, smells
    question: Review the flagged patterns — are they design problems that need addressing, or acceptable given the file's role?
    evidence: Flagged by: global_mutable_config, smells
    evidence: [smells] 1x Loose type annotation — use specific types
    fingerprint: 4c71b74ee7c33711
  - [design_concern] rusket/optuna.py
    summary: Design signals from smells
    question: Review the flagged patterns — are they design problems that need addressing, or acceptable given the file's role?
    evidence: Flagged by: smells
    evidence: [smells] 1x Except handler silently suppresses error (pass/continue, no log)
    fingerprint: cc305223bc8c60e9
  - [design_concern] rusket/pipeline.py
    summary: Design signals from smells, structural
    question: Review the flagged patterns — are they design problems that need addressing, or acceptable given the file's role?
    evidence: Flagged by: smells, structural
    evidence: File size: 538 lines
    fingerprint: 9a5a155db446729f
  - [design_concern] rusket/spark.py
    summary: Design signals from smells, structural
    question: Review the flagged patterns — are they design problems that need addressing, or acceptable given the file's role?
    evidence: Flagged by: smells, structural
    evidence: File size: 830 lines
    fingerprint: 31e4325c25d869b3
  - [design_concern] rusket/svd.py
    summary: Design signals from smells
    question: Review the flagged patterns — are they design problems that need addressing, or acceptable given the file's role?
    evidence: Flagged by: smells
    evidence: [smells] 1x Except handler silently suppresses error (pass/continue, no log)
    fingerprint: 3161d33fd7d21c0a
  - [design_concern] scripts/eval_doc_snippets.py
    summary: Design signals from smells
    question: Review the flagged patterns — are they design problems that need addressing, or acceptable given the file's role?
    evidence: Flagged by: smells
    evidence: [smells] 2x sys.exit() outside CLI entry point — use exceptions
    fingerprint: dda8b6757d1dd247
  - [design_concern] scripts/gen_api_reference.py
    summary: Design signals from smells, structural
    question: Review the flagged patterns — are they design problems that need addressing, or acceptable given the file's role?
    evidence: Flagged by: smells, structural
    evidence: File size: 522 lines
    fingerprint: 4ac6b44ae828c786
  - [design_concern] scripts/gen_llm_txt.py
    summary: Design signals from smells
    question: Review the flagged patterns — are they design problems that need addressing, or acceptable given the file's role?
    evidence: Flagged by: smells
    evidence: [smells] 1x Constant defined identically in multiple modules
    fingerprint: def44a69151c7447
  - [design_concern] scripts/sync_changelog.py
    summary: Design signals from smells
    question: Review the flagged patterns — are they design problems that need addressing, or acceptable given the file's role?
    evidence: Flagged by: smells
    evidence: [smells] 2x sys.exit() outside CLI entry point — use exceptions
    fingerprint: 907b391c0412fcbb
  - [duplication_design] rusket/bpr.py
    summary: Duplication pattern — assess if extraction is warranted
    question: Is the duplication worth extracting into a shared utility, or is it intentional variation?
    evidence: Flagged by: boilerplate_duplication, smells
    evidence: [smells] 1x Too many optional params — consider a config object
    fingerprint: 3852c24e63c91d8a
  - [duplication_design] rusket/ease.py
    summary: Duplication pattern — assess if extraction is warranted
    question: Is the duplication worth extracting into a shared utility, or is it intentional variation?
    evidence: Flagged by: boilerplate_duplication, smells
    evidence: [smells] 3x Function-level import (possible circular import workaround)
    fingerprint: 0ef4f2706fd16a2f
  - [duplication_design] rusket/fpgrowth.py
    summary: Duplication pattern — assess if extraction is warranted
    question: Is the duplication worth extracting into a shared utility, or is it intentional variation?
    evidence: Flagged by: boilerplate_duplication, smells
    evidence: [smells] 2x Too many optional params — consider a config object
    fingerprint: 0dac1acedb244c98
  - [duplication_design] rusket/fpmc.py
    summary: Duplication pattern — assess if extraction is warranted
    question: Is the duplication worth extracting into a shared utility, or is it intentional variation?
    evidence: Flagged by: boilerplate_duplication, smells
    evidence: [smells] 2x Too many optional params — consider a config object
    fingerprint: a4a38b316bdc7bfe
  (+18 more — use `desloppify show <detector> --no-budget` to explore)

RELEVANT FINDINGS — explore with CLI:
These detectors found patterns related to this dimension. Explore the findings,
then read the actual source code.

  desloppify show boilerplate_duplication --no-budget      # 47 findings
  desloppify show coupling --no-budget      # 2 findings
  desloppify show cycles --no-budget      # 1 findings
  desloppify show dupes --no-budget      # 7 findings
  desloppify show facade --no-budget      # 1 findings
  desloppify show global_mutable_config --no-budget      # 1 findings
  desloppify show orphaned --no-budget      # 11 findings
  desloppify show responsibility_cohesion --no-budget      # 2 findings
  desloppify show signature --no-budget      # 7 findings
  desloppify show smells --no-budget      # 133 findings
  desloppify show structural --no-budget      # 9 findings

Report actionable issues in issues[]. Use concern_verdict and concern_fingerprint
for findings you want to confirm or dismiss.

Task requirements:
1. Read the blind packet's `system_prompt` — it contains scoring rules and calibration.
2. Start from the seed files, then freely explore the repository to build your understanding.
3. Keep issues and scoring scoped to this batch's dimension.
4. Respect scope controls: do not include files/directories marked by `exclude`, `suppress`, or non-production zone overrides.
5. Return 0-10 issues for this batch (empty array allowed).
6. For design_coherence, use evidence from `holistic_context.scan_evidence.signal_density` — files where multiple mechanical detectors fired. Investigate what design change would address multiple signals simultaneously. Check `scan_evidence.complexity_hotspots` for files with high responsibility cluster counts.
7. Workflow integrity checks: when reviewing orchestration/queue/review flows,
8. xplicitly look for loop-prone patterns and blind spots:
9. - repeated stale/reopen churn without clear exit criteria or gating,
10. - packet/batch data being generated but dropped before prompt execution,
11. - ranking/triage logic that can starve target-improving work,
12. - reruns happening before existing open review work is drained.
13. If found, propose concrete guardrails and where to implement them.
14. Complete `dimension_judgment` for your dimension — all three fields (strengths, issue_character, score_rationale) are required. Write the judgment BEFORE setting the score.
15. Do not edit repository files.
16. Return ONLY valid JSON, no markdown fences.

Scope enums:
- impact_scope: "local" | "module" | "subsystem" | "codebase"
- fix_scope: "single_edit" | "multi_file_refactor" | "architectural_change"

Output schema:
{
  "batch": "design_coherence",
  "batch_index": 16,
  "assessments": {"<dimension>": <0-100 with one decimal place>},
  "dimension_notes": {
    "<dimension>": {
      "evidence": ["specific code observations"],
      "impact_scope": "local|module|subsystem|codebase",
      "fix_scope": "single_edit|multi_file_refactor|architectural_change",
      "confidence": "high|medium|low",
      "issues_preventing_higher_score": "required when score >85.0",
      "sub_axes": {"abstraction_leverage": 0-100, "indirection_cost": 0-100, "interface_honesty": 0-100, "delegation_density": 0-100, "definition_directness": 0-100, "type_discipline": 0-100}  // required for abstraction_fitness when evidence supports it; all one decimal place
    }
  },
  "dimension_judgment": {
    "<dimension>": {
      "strengths": ["0-5 specific things the codebase does well from this dimension's perspective"],
      "issue_character": "one sentence characterizing the nature/pattern of issues from this dimension's perspective",
      "score_rationale": "2-3 sentences explaining the score from this dimension's perspective, referencing global anchors"
    }  // required for every assessed dimension; do not omit
  },
  "issues": [{
    "dimension": "<dimension>",
    "identifier": "short_id",
    "summary": "one-line defect summary",
    "related_files": ["relative/path.py"],
    "evidence": ["specific code observation"],
    "suggestion": "concrete fix recommendation",
    "confidence": "high|medium|low",
    "impact_scope": "local|module|subsystem|codebase",
    "fix_scope": "single_edit|multi_file_refactor|architectural_change",
    "root_cause_cluster": "optional_cluster_name_when_supported_by_history",
    "concern_verdict": "confirmed|dismissed  // for concern signals only",
    "concern_fingerprint": "abc123  // required when dismissed; copy from signal fingerprint",
    "reasoning": "why dismissed  // optional, for dismissed only"
  }],
  "retrospective": {
    "root_causes": ["optional: concise root-cause hypotheses"],
    "likely_symptoms": ["optional: identifiers that look symptom-level"],
    "possible_false_positives": ["optional: prior concept keys likely mis-scoped"]
  }
}
