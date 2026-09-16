"""Native routing and private recorder for the three-task integration candidate."""

from dataclasses import asdict
from pathlib import Path

from .async_native import IsolatedAsyncSimulationPort
from .finite_native import digest, save
from .task_registry import require_development
from .task_truth import PROFILES, StableTaskTruth, simulator_facts


def backend_factory(assets, seed, task):
    from actionstream.lerobot_backend import LeRobotBackend

    require_development(task)
    return LeRobotBackend(
        task_ids=[task.task_id],
        suite=task.suite,
        seed=seed,
        episode_length=800,
        model_id=str(Path(assets) / "xvla"),
        model_revision="12e8783e996944f5c97e490d37d4c145484ed70a",
        device="cuda",
        tokenizer_path=str(Path(assets) / "bart"),
    )


class TaskSimulationPort(IsolatedAsyncSimulationPort):
    new_truth = staticmethod(StableTaskTruth)

    def __init__(
        self, backend, seed, output, task, excluded_layouts=None, *, forced_open_until=0
    ):
        self.task = require_development(task)
        self.task_id = task.task_id
        self.canonical_instruction = task.canonical_instruction
        if type(forced_open_until) is not int or forced_open_until < 0:
            raise ValueError("Invalid physical intervention length")
        self.forced_open_until = forced_open_until
        super().__init__(backend, seed, output, excluded_layouts)

    def read_facts(self, env):
        return simulator_facts(env, self.task)

    def inference_process_extra_args(self):
        return (self.task.key,)

    def start(self, request_id, revision):
        from libero.libero import get_libero_path

        path = (
            Path(get_libero_path("bddl_files"))
            / self.task.suite
            / (self.canonical_instruction.replace(" ", "_") + ".bddl")
        )
        if digest(path) != PROFILES[self.task.key]["bddl_sha256"]:
            raise ValueError("Pinned task definition changed")
        save(
            self.output / "task.json",
            dict(
                task=asdict(self.task),
                sha256=self.task.sha256,
                bddl_sha256=digest(path),
            ),
        )
        save(
            self.output / "intervention.json",
            dict(forced_open_until=self.forced_open_until),
        )
        return super().start(request_id, revision)

    def step(self, action, request_id, revision, control):
        command = action.copy()
        if self.control < self.forced_open_until:
            command[-1] = -1.0
        return super().step(command, request_id, revision, control)
