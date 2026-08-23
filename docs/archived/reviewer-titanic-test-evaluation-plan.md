# Titanic Reviewer Benchmark Plan

## Goal

This is a lightweight first-round benchmark plan for the Titanic dataset.

The goal is simple:

- use a small set of benchmark metrics inspired by the AutoKaggle paper
- keep evaluation focused on `accuracy`-style task success
- add only a small set of reviewer-related unit tests

This version is intentionally much simpler than a full QA framework.

## Benchmark Scope

Dataset:

- `data/titanic.zip`

Focus of this first benchmark:

- can the workflow produce a valid Titanic submission
- is the result usable and not obviously broken
- is the prediction quality reasonable under a simple normalized score

Out of scope for now:

- complicated quality scorecards
- many manual review gates
- extensive failure taxonomy
- large benchmark matrix across many datasets

## Metrics We Will Use

Based on the AutoKaggle paper, the most useful lightweight metrics for us are:

### 1. MS

`MS = Made Submission`

Meaning:

- whether the system successfully generated a `submission.csv`

For repeated runs:

- `MS = number_of_runs_with_submission_csv / total_runs`

### 2. VS

`VS = Valid Submission`

Meaning:

- whether the generated `submission.csv` is valid

For Titanic, a valid submission means at least:

- file exists
- columns are exactly `PassengerId,Survived`
- row count matches the test set
- `Survived` values are valid binary predictions
- file can be treated as ready for Kaggle submission

For repeated runs:

- `VS = number_of_valid_submissions / total_runs`

### 3. NPS

`NPS = Normalized Performance Score`

The paper uses NPS to normalize performance scores.
For bounded metrics like Titanic `accuracy`, we can use the score directly:

- `NPS = accuracy`

If a metric is "smaller is better", the paper uses:

- `NPS = 1 / (1 + score)`

But for Titanic, since the competition metric is `accuracy`, we only need:

- `NPS = accuracy`

### 4. ANPS

`ANPS = Average Normalized Performance Score`

Meaning:

- average `NPS` over successful runs

For repeated runs:

- `ANPS = average NPS across valid successful runs`

If we only run once, then:

- `ANPS = NPS`

### 5. CS

`CS = Comprehensive Score`

Following AutoKaggle, we can keep:

- `CS = 0.5 * VS + 0.5 * ANPS`

This is useful because it balances:

- whether the system can finish correctly
- how good the prediction result is

## About "MPS"

I checked the AutoKaggle PDF you gave me.
I found:

- `MS`
- `VS`
- `NPS`
- `ANPS`
- `CS`

I did not find a metric named `MPS` in that paper.
So for now I recommend we do **not** add `MPS` unless your team has another source that defines it.

## Minimal Benchmark Protocol

For the Titanic first version, we can use this very small protocol:

1. run the pipeline on the Titanic dataset
2. check whether `submission.csv` was generated
3. check whether the submission format is valid
4. record the validation or leaderboard `accuracy`
5. convert that score to `NPS`
6. compute `MS`, `VS`, `ANPS`, and `CS`

If we only do one run for now, then the benchmark record is very simple:

- `MS`: 0 or 1
- `VS`: 0 or 1
- `NPS`: accuracy
- `ANPS`: same as NPS
- `CS`: `0.5 * VS + 0.5 * ANPS`

Example:

- submission generated: yes
- submission valid: yes
- accuracy: `0.79`

Then:

- `MS = 1.0`
- `VS = 1.0`
- `NPS = 0.79`
- `ANPS = 0.79`
- `CS = 0.895`

## Minimal Acceptance Standard

For the first Titanic milestone, we can keep the standard very simple:

- `MS = 1`
- `VS = 1`
- `accuracy` is recorded
- `CS` is recorded

Optional first-pass target:

- `accuracy >= baseline accuracy`

If you want an even simpler rule, then the first benchmark can be considered a pass as long as:

- submission is generated
- submission is valid
- score is measurable

## Minimal Unit Tests

We only keep a few reviewer-related unit tests.

Recommended initial set:

### 1. submission file created

Check:

- reviewer or benchmark helper can detect that `submission.csv` exists

### 2. submission format valid

Check:

- header is correct
- row count is correct
- prediction column values are valid

### 3. missing submission fails

Check:

- if `submission.csv` is missing, benchmark marks `MS = 0` and `VS = 0`

### 4. malformed submission fails

Check:

- wrong columns or wrong row count causes invalid submission

### 5. score-to-NPS conversion

Check:

- for Titanic `accuracy`, `NPS` equals the recorded accuracy

### 6. CS calculation

Check:

- `CS = 0.5 * VS + 0.5 * ANPS` is computed correctly

## Suggested Benchmark Record Format

For now, one markdown table is enough.

| Run ID | MS | VS | Accuracy | NPS | ANPS | CS | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| run-001 | 1 | 1 | 0.79 | 0.79 | 0.79 | 0.895 | first valid run |

## GitHub Project Recommendation

If you want the project to stay lightweight, keep only 4 item types:

1. benchmark run
2. unit test
3. bug
4. doc

Suggested columns:

- Backlog
- In Progress
- Review
- Done

Suggested first items:

1. Define Titanic benchmark record template
2. Implement `MS` and `VS` checks
3. Implement `NPS` and `CS` calculation helper
4. Add unit test for valid submission
5. Add unit test for missing submission
6. Add unit test for malformed submission
7. Run first Titanic benchmark and record results

## Recommended First Step

The best first implementation order is:

1. define the benchmark record format
2. implement `MS` and `VS`
3. implement `NPS` and `CS`
4. add the 4 to 6 small unit tests above
5. run the first Titanic benchmark
