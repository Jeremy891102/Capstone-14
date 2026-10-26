# Five-video smoke experiments

All exp006 smoke configurations use data/hd_epic_smoke5_loc8_v1: five original video windows,
eight existing localization questions each. Configs contain credential environment-variable
names only, never actual API keys.

The exp006_smoke5_{provider}_{method} folders are full five-video templates. Provider choices
are openai, openrouter and vertex; methods are whole_report and per_field. Only Vertex Gemini
has been validated by the five-video live smoke test. OpenAI/OpenRouter have not been run on
these videos and their pricing remains unset.

The live Gemini execution was staged without duplicating requests:

- exp006_step3_vertex_p09_whole_report: P09, 1 request.
- exp006_step4_vertex_p09_per_field: P09, 8 requests.
- exp006_remaining4_vertex_whole_report: other four videos, 4 requests.
- exp006_remaining4_vertex_per_field: other four videos, 32 requests.

Use either the full templates OR the staged configs to reproduce the test; running both
would duplicate inference and charges. Run commands perform paid inference. Validation,
scoring and log export are offline. Require local videos, set the corresponding API key,
verify current pricing, and agree on the run scope before execution.

Each execution writes runs/<run_id>/smoke_test.log. Each evaluation writes its own
runs/<run_id>/evaluations/<eval_id>/smoke_test.log without changing inference evidence.
Source MP4s, temporary clips, credentials and run directories are excluded from Git.
