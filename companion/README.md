# APEX Companion

The companion turns a bounded request into an allow-listed local benchmark,
validation, or environment operation. It uses only the Python standard library.
The model can propose a plan, but `companion.ops` validates every field before
the CLI constructs fixed script arguments. Child processes run without a shell.

Run an offline recorded example without a built runner:

```sh
python -m companion --offline --replay-results --yes "benchmark a small square multiply"
```

Run a local operation using a recorded interpretation:

```sh
python -m companion --offline "validate edge cases"
```

For live interpretation, set `GEMINI_API_KEY` in the environment and run
`python -m companion "benchmark a 128 by 128 multiply"`. The validated command
is shown and requires confirmation unless `--yes` is supplied. The companion
extracts facts from the shell's JSON output before requesting an explanation;
numeric claims in an explanation are checked against those facts.

Develop the live request/response path without a key by starting the local
mock server in one terminal:

```sh
python -m companion.tests.mock.mock_gemini_server --port 8765
```

Then use a synthetic key and local endpoint in another terminal:

```sh
GEMINI_API_KEY=local-placeholder \
APEX_GEMINI_ENDPOINT=http://127.0.0.1:8765 \
python -m companion --yes "benchmark a small matrix"
```

The mock listens on loopback by default and does not log request prompts or
headers. `--response` can provide fixed model text to exercise plan validation
and error handling.
