# Update the agent after changing a scorecard

The published scorecard governs applicability and verdicts. The agent's internal questions help it find and assess evidence. They should reflect the rubric without adding conditions of their own.

## Export the published scorecard

In **Scorecard → Edit scorecard**, choose **Save version**. Then export the `card` field from the settings endpoint:

```sh
curl -fsS http://localhost:8080/api/settings |
  python3 -c 'import json, sys; print(json.dumps(json.load(sys.stdin)["card"], ensure_ascii=False, indent=2))' \
  > /tmp/check-scorecard.json
```

When using access control, include `-H "Authorization: Bearer ${QA_TOKEN}"` with `curl`. The export above contains the scorecard only.

## Give the exported JSON to your coding agent

Open the CHECK checkout in Claude Code or Codex and pass the following prompt with the export path. With a chat-only assistant, attach the exported JSON and the current YAML profile and ask for the new file and registry diff.

```text
Update CHECK's internal review questions using the published scorecard in
/tmp/check-scorecard.json. Follow the criterion IDs and rubric in that file.

Find the active agent profile through AGENT_VERSION in backend/qa/settings.py
and REVIEW_PROFILES in backend/qa/agent.py. Preserve the existing YAML structure,
including sequence, fallback, criteria, and any stage-specific instructions.

For each llm or hybrid criterion, keep questions that still match the rubric,
revise ones with changed verdict boundaries, and add questions for new criteria.
Remove entries for criteria absent from the export. Skip code checks and timing targets.

Write two to four short, industry-neutral questions about observable evidence
per criterion. Consider source, chronology, policy applicability, and incomplete
history where relevant. Keep applicability and verdict boundaries aligned to the
published rubric.

Write backend/cards/agent-checks-v<N>.yaml with the next agent version, register
it, and advance AGENT_VERSION. Keep earlier profiles available for saved jobs.
Verify the new profile against the exported criterion IDs. Run the agent tests
and report the changes. Keep the published scorecard and job queue untouched.
```

Review the diff against the scorecard before deploying. To update an installation, rebuild the API and worker image with the Compose files used on that installation. Refresh both services before accepting new jobs so they share the same agent code and profiles. Existing assessments keep their original snapshots.
