You are a focused subagent reviewer for a single holistic investigation batch.

Repository root: /Users/dominikpeter/DevOps/rusket
Blind packet: /Users/dominikpeter/DevOps/rusket/.desloppify/review_packet_blind.json
Batch index: 6
Batch name: abstraction_fitness
Batch rationale: seed files for abstraction_fitness review

DIMENSION TO EVALUATE:

## abstraction_fitness
Python abstraction fitness: favor direct modules, explicit domain APIs, and bounded packages over indirection and generic helper surfaces.
Look for:
- Functions that only forward args/kwargs to another function without policy or translation
- Protocol/base-class abstractions with one concrete implementation and no extension pressure
- Cross-module wrapper chains where calls hop through helper layers before reaching real logic
- Project-wide reliance on generic helper modules instead of bounded domain packages
- Over-broad dict/config/context parameters used as implicit parameter bags
Skip:
- Django/FastAPI/SQLAlchemy framework boundaries that require adapters or dependency hooks
- Wrappers that add retries, metrics, auth checks, caching, or tracing
- Intentional package facades used to stabilize public import paths
- Migration shims with active callers and clear sunset plan

YOUR TASK: Read the code for this batch's dimension. Judge how well the codebase serves a developer from that perspective. The dimension rubric above defines what good looks like. Cite specific observations that explain your judgment.

Mechanical scan evidence — navigation aid, not scoring evidence:
The blind packet contains `holistic_context.scan_evidence` with aggregated signals from all mechanical detectors — including complexity hotspots, error hotspots, signal density index, boundary violations, and systemic patterns. Use these as starting points for where to look beyond the seed files.

Seed files (start here):
- rusket/transactions.py
- rusket/optuna.py
- rusket/pca.py
- rusket/association_rules.py
- rusket/pacmap.py
- rusket/faiss_ann.py
- rusket/cuda.py
- examples/09_online_retail_basket_analysis.py
- rusket/_core.py
- rusket/_validation.py
- rusket/_config.py
- rusket/spark.py
- rusket/svd.py
- rusket/model.py
- rusket/recommend.py
- rusket/vector_export.py
- rusket/lightgcn.py
- rusket/item_knn.py
- rusket/viz.py
- rusket/hybrid_embedding.py
- rusket/mlflow.py
- rusket/streaming.py
- rusket/user_knn.py
- rusket/model_selection.py
- rusket/grouped.py
- rusket/eclat.py
- rusket/fpgrowth.py
- rusket/fm.py
- scripts/gen_api_reference.py
- rusket/als.py
- rusket/bert4rec.py
- rusket/bpr.py
- rusket/fin.py
- rusket/fpmc.py
- rusket/hupm.py
- rusket/__init__.py
- rusket/gpu.py
- rusket/lcm.py
- rusket/nmf.py
- rusket/pipeline.py
- rusket/popularity.py
- rusket/sasrec.py
- rusket/_compat.py
- rusket/content_based.py
- rusket/ease.py
- rusket/negfin.py
- rusket/rules.py
- examples/11_instacart_recommender.py

RELEVANT FINDINGS — explore with CLI:
These detectors found patterns related to this dimension. Explore the findings,
then read the actual source code.

  desloppify show facade --no-budget      # 1 findings
  desloppify show responsibility_cohesion --no-budget      # 6 findings
  desloppify show single_use --no-budget      # 1 findings
  desloppify show structural --no-budget      # 12 findings

Report actionable issues in issues[]. Use concern_verdict and concern_fingerprint
for findings you want to confirm or dismiss.

Task requirements:
1. Read the blind packet's `system_prompt` — it contains scoring rules and calibration.
2. Start from the seed files, then freely explore the repository to build your understanding.
3. Keep issues and scoring scoped to this batch's dimension.
4. Respect scope controls: do not include files/directories marked by `exclude`, `suppress`, or non-production zone overrides.
5. Return 0-10 issues for this batch (empty array allowed).
6. For abstraction_fitness, use evidence from `holistic_context.abstractions`:
7. - `delegation_heavy_classes`: classes where most methods forward to an inner object — entries include class_name, delegate_target, sample_methods, and line number.
8. - `facade_modules`: re-export-only modules with high re_export_ratio — entries include samples (re-exported names) and loc.
9. - `typed_dict_violations`: TypedDict fields accessed via .get()/.setdefault()/.pop() — entries include typed_dict_name, violation_type, field, and line number.
10. - `complexity_hotspots`: files where mechanical analysis found extreme parameter counts, deep nesting, or disconnected responsibility clusters.
11. Include `delegation_density`, `definition_directness`, and `type_discipline` alongside existing sub-axes in dimension_notes when evidence supports it.
12. Complete `dimension_judgment` for your dimension — all three fields (strengths, issue_character, score_rationale) are required. Write the judgment BEFORE setting the score.
13. Do not edit repository files.
14. Return ONLY valid JSON, no markdown fences.

Scope enums:
- impact_scope: "local" | "module" | "subsystem" | "codebase"
- fix_scope: "single_edit" | "multi_file_refactor" | "architectural_change"

Output schema:
{
  "batch": "abstraction_fitness",
  "batch_index": 6,
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
