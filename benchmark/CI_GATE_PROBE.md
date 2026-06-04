# CI gate probe

Throwaway file. Its only purpose is a **benchmark-only** diff to verify that
the `detect_changes` job (added in #295) emits `run_integration=false` and the
live `integration_tests` suite is **skipped** while `All checks passed` stays
green. Safe to delete; this PR is not meant to be merged.
