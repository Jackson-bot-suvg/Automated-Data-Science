"""Explicit reviewer-only ablation. The reader, routing and structural checks remain."""
SEMANTIC_PREFIXES=('target unresolved:', 'target leakage:')
def suppress_intake_semantics(check):
    result=check.model_copy(deep=True)
    result.blocking_failures=[s for s in result.blocking_failures if not s.startswith(SEMANTIC_PREFIXES)]
    return result

def install(method,observer):
    if method not in {'F0_ORIGINAL','M2_INTAKE_ONLY_STRONG','M0_NO_SEMANTIC_ENFORCEMENT'}:raise ValueError(method)
    # F0 later reviewers implement current-schema/shape checks, not frozen-intent comparisons.
    # Those structural checks are retained for all methods, hence F0 and M2 overlap exactly.
    if method!='M0_NO_SEMANTIC_ENFORCEMENT':return
    from app.agents.bg_reviewer import BackgroundReviewerAgent
    original=BackgroundReviewerAgent._check_reader_output
    def check(agent,state):
        result,fatal=original(agent,state)
        filtered=suppress_intake_semantics(result)
        observer.emit('intake_ablation',raw_failures=result.blocking_failures,remaining_failures=filtered.blocking_failures)
        return filtered,fatal
    BackgroundReviewerAgent._check_reader_output=check
