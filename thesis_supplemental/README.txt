SUPPLEMENTAL FILES
The Right Pause: Severity-Aware Gate Enforcement for Voice Agents Before Irreversible Tool Calls
Mahima Sachan, M.S. thesis, California State University, Long Beach, 2026
Code repository: https://github.com/mahima110298/sage-voice-agents

01_preregistration_plan.md
  The analysis plan written before any test run (Chapter 4, Pre-Registration and Deviations).
  The header records the commit and date of the `prereg` and `sage-frozen` tags in the
  development repository and the SHA-256 of the plan text.

02_GATE_POLICY.md
  The full written gate policy: the 27 rules reproduced in Appendix A, plus the instructions,
  severity convention, and label conventions that were never shown to an agent.

03_benchmark_cases.zip
  All benchmark cases as JSON (metadata.json per case), without audio:
    development_original/   the 88 cases the pilot ran on (Chapter 2)
    development_corrected/  the corrected copy SAGE was built and tuned on
    test_locked/            the 119-case locked test set, its lock file (LOCK.json), and GATE_POLICY.md
  Also: test_subset_v3.json (the 80 cases that were run) and CASE_VALIDATION_REPORT.md.

04_results_scored_rows_test.csv
  One row per judged turn on the test set (primary judge): agent, configuration, pass, case,
  turn, required gate, detected gate, verdict, the agent's text and tool calls, guard latency,
  and the judge rules hash. Rows with a non-empty sut_error are agent errors and are excluded
  from every metric, as the thesis describes. The partial gpt-realtime-2.1-mini B0 pass 2 is
  included for completeness but is not used in any analysis. Every table in Chapter 5 can be
  recomputed from this file.

05_results_pilot.csv
  One row per judged turn of the pilot study (Chapter 2): arm, required and detected gate, verdict.

06_analysis_code.zip
  The author's code: the SAGE package (rp/), the hybrid judge, the analysis scripts, the
  test-set generator, and the case validator. Third-party libraries are not included; see
  requirements-rp.txt. API keys are read from environment variables and are not included.

07_test_audio_80cases.zip
  The synthesized speech for the 80 test cases that were run, one folder per case
  (input_turnN.wav). The voices are synthetic, generated with OpenAI gpt-4o-mini-tts with one
  synthetic speaker profile per case. They are not recordings of real people, and all names,
  accounts, and numbers in the cases are fictitious.
