"""Authorized task routing into the existing asynchronous control state machine."""

from .async_agent import AsyncAgentConfig, ChunkWorker, _execute_authorized_task
from .multitask_language import validate_permit


def execute_task(
    request,
    permit,
    port_factory,
    predictor,
    emit,
    config=AsyncAgentConfig(),
    *,
    worker_factory=ChunkWorker,
):
    task = validate_permit(request, permit)
    predictor.validate_binding(task)
    # The text vector may contain thousands of values. Bind it once at startup,
    # not by reserializing/hashing it inside every 50 ms journal event.
    task_hash = task.sha256
    condition_hash = predictor.condition.sha256

    def make_port():
        # Pass the host-validated task; a caller cannot preselect a different task.
        port = port_factory(task)
        if port.task != task:
            port.close()
            raise ValueError("Native port is bound to another task")
        return port

    def record(event, **values):
        emit(
            event,
            task_key=task.key,
            task_sha256=task_hash,
            condition_sha256=condition_hash,
            **values,
        )

    outcome = _execute_authorized_task(
        request,
        permit,
        task,
        make_port,
        predictor,
        record,
        config,
        worker_factory=worker_factory,
        checkpoint_sha256=predictor.checkpoint_sha256,
    )
    return dict(
        outcome,
        task_key=task.key,
        task_sha256=task_hash,
        condition_sha256=condition_hash,
    )
