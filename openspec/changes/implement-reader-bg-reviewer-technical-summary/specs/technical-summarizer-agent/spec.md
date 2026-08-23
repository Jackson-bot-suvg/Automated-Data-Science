## ADDED Requirements

### Requirement: Technical Summarizer writes final technical report
The system SHALL expose Technical Summarizer through the existing `summarizer_node(state)` and `run_technical_summary(state)` entry points, with `app/agents/summarizer.py` owning a TechnicalSummarizerAgent layer that converts workflow evidence into a technical report artifact.

#### Scenario: Model building is approved
- **WHEN** `current_phase` is `model_building` and the latest review status is `approved`
- **THEN** Technical Summarizer MUST write a Markdown report to `artifacts.technical_report`
- **THEN** it MUST update `report_state.technical_report_path`
- **THEN** it MUST append `technical_summary` to `phase_history`
- **THEN** it MUST set `current_phase` to `technical_summary`

### Requirement: Technical Summarizer preserves non-final routing
The system SHALL avoid producing the final technical report before all execution phases are complete.

#### Scenario: Non-final approved phase reaches summarizer
- **WHEN** `summarizer_node(state)` is invoked after an approved phase other than `model_building`
- **THEN** Technical Summarizer MUST NOT write a technical report artifact
- **THEN** it MUST leave `current_phase` unchanged so the router can return to planner

### Requirement: Technical report includes workflow evidence
The system SHALL include enough evidence for the Business Translator to produce a grounded business report.

#### Scenario: Report is rendered
- **WHEN** Technical Summarizer writes a report
- **THEN** the report MUST include task context, data summary, EDA findings, cleaning summary, feature engineering summary, model results, reviewer history, artifact index, and limitations
- **THEN** it MUST remain valid if optional LLM assistance is disabled
