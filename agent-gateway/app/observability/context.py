from contextvars import ContextVar

trace_id_var: ContextVar[str | None] = ContextVar(
    "trace_id",
    default=None
)

run_id_var: ContextVar[str | None] = ContextVar(
    "run_id",
    default=None
)

span_id_var: ContextVar[str | None] = ContextVar(
    "span_id",
    default=None
)


def set_execution_context(trace_id, run_id):
    trace_token = trace_id_var.set(trace_id)
    run_token = run_id_var.set(run_id)

    return trace_token, run_token


def reset_execution_context(tokens):
    trace_token, run_token = tokens

    trace_id_var.reset(trace_token)
    run_id_var.reset(run_token)