# External Blind Review Session

Session id: ext_20260312_194859_84ed4d2b
Session token: 02e0e5cedee2977d4b2122daf02c07e7
Blind packet: /Users/dominikpeter/DevOps/rusket/.desloppify/review_packet_blind.json
Template output: /Users/dominikpeter/DevOps/rusket/.desloppify/external_review_sessions/ext_20260312_194859_84ed4d2b/review_result.template.json
Claude launch prompt: /Users/dominikpeter/DevOps/rusket/.desloppify/external_review_sessions/ext_20260312_194859_84ed4d2b/claude_launch_prompt.md
Expected reviewer output: /Users/dominikpeter/DevOps/rusket/.desloppify/external_review_sessions/ext_20260312_194859_84ed4d2b/review_result.json

Happy path:
1. Open the Claude launch prompt file and paste it into a context-isolated subagent task.
2. Reviewer writes JSON output to the expected reviewer output path.
3. Submit with the printed --external-submit command.

Reviewer output requirements:
1. Return JSON with top-level keys: session, assessments, issues.
2. session.id must be `ext_20260312_194859_84ed4d2b`.
3. session.token must be `02e0e5cedee2977d4b2122daf02c07e7`.
4. Include issues with required schema fields (dimension/identifier/summary/related_files/evidence/suggestion/confidence).
5. Use the blind packet only (no score targets or prior context).
