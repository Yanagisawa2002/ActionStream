"""Public task semantics, independent of simulator bodies and evaluation labels."""

from dataclasses import asdict, dataclass

from .grounding import contract_hash


@dataclass(frozen=True)
class Task:
    key: str
    suite: str
    task_id: int
    canonical_instruction: str
    target_phrase: str
    relation: str
    destination_phrase: str

    @property
    def sha256(self):
        return contract_hash(asdict(self))


DEVELOPMENT = (
    Task(
        "object_tomato_basket",
        "libero_object",
        5,
        "pick up the tomato sauce and place it in the basket",
        "tomato sauce",
        "in",
        "basket",
    ),
    Task(
        "spatial_bowl_stove_plate",
        "libero_spatial",
        7,
        "pick up the black bowl on the stove and place it on the plate",
        "black bowl on the stove",
        "on",
        "plate",
    ),
    Task(
        "goal_wine_cabinet",
        "libero_goal",
        2,
        "put the wine bottle on top of the cabinet",
        "wine bottle",
        "on top of",
        "cabinet",
    ),
)


def registry_sha256():
    return contract_hash([asdict(task) for task in DEVELOPMENT])


def development_task(key):
    for task in DEVELOPMENT:
        if task.key == key:
            return task
    raise ValueError("Task is outside the three-task development registry")


def require_development(task):
    if type(task) is not Task or development_task(task.key) != task:
        raise ValueError("Task definition differs from the registered semantics")
    return task
