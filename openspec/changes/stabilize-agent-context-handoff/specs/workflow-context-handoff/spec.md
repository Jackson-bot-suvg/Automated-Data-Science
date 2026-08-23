## ADDED Requirements

### Requirement: Workflow phases SHALL have explicit authoritative state ownership
The system SHALL define, per workflow phase, which state fields are authoritative outputs and which fields are compatibility mirrors only. Downstream agents MUST consume authoritative fields first and MUST NOT allow compatibility mirrors to override them.

#### Scenario: Target contract handoff after Reader
- **WHEN** Reader resolves the task target contract
- **THEN** `target_spec` MUST be the authoritative source for target columns, target mode, task family, submission schema, and evaluation semantics
- **THEN** any legacy target mirrors MAY be populated for compatibility
- **THEN** downstream phases MUST consume `target_spec` first and MUST NOT let legacy mirrors override `target_spec`

#### Scenario: Phase-owned state is handed to the next phase
- **WHEN** a workflow phase completes successfully
- **THEN** the phase MUST write its authoritative outputs into its owned state bucket and artifact keys
- **THEN** downstream phases MUST read those authoritative outputs before consulting report text or compatibility mirrors

### Requirement: Each run SHALL persist structured handoff artifacts
The system SHALL persist compact structured artifacts that describe what each important phase actually consumed and produced so task diagnosis does not require replaying logs.

#### Scenario: Run-level manifest is available
- **WHEN** a workflow run writes or updates artifacts
- **THEN** the run MUST maintain a structured `run_manifest` artifact under the run artifact root
- **THEN** the manifest MUST include the logical thread identity, the unique run identity, the resolved target contract, current artifact paths, and the latest phase outcomes

#### Scenario: Phase review outcomes are inspectable
- **WHEN** a reviewer evaluates a workflow phase
- **THEN** the workflow MUST persist a structured review artifact or equivalent structured record for that phase
- **THEN** the structured review record MUST capture status, blocking failures, warnings, and the checked contract context

### Requirement: Downstream reporting SHALL prefer structured artifacts over markdown inference
Technical and business reporting components MUST prefer authoritative structured state and artifacts over markdown-only inference when summarizing workflow outcomes.

#### Scenario: Technical summary consumes workflow outputs
- **WHEN** the technical summarizer prepares the final report
- **THEN** it MUST derive task definition, metrics, artifacts, and limitations from authoritative structured state or structured artifacts first
- **THEN** it MUST NOT depend on previous markdown reports as the primary machine-readable input

#### Scenario: Business translation consumes workflow outputs
- **WHEN** the business translator prepares the business report
- **THEN** it MUST prefer structured technical outputs such as validation metrics, reference baseline metrics, feature-importance summaries, model explanations, and run artifacts
- **THEN** it MAY read the technical report for presentation context, but it MUST NOT treat markdown text as the authoritative source of workflow state
