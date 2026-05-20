from pathlib import Path

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


class LIBEROWrapper:
    """
    Wraps LIBERO environment for our training loop.

    Supports selecting a task either by integer task_id or by matching task_name.
    """

    def __init__(
        self,
        task_suite_name: str = "libero_spatial",
        task_id: int | None = None,
        task_name: str | None = None,
        img_size: int = 224,
    ):
        self.img_size = img_size

        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[task_suite_name]()

        if task_id is None and task_name is None:
            task_id = 0

        if task_id is None:
            task_id = self._find_task_id(task_suite, task_name)

        self.task_id = task_id
        self.task = task_suite.get_task(task_id)

        bddl_file = (
            Path(get_libero_path("bddl_files"))
            / task_suite_name
            / f"{task_name}.bddl"
        )

        if not bddl_file.exists():
            raise FileNotFoundError(f"BDDL file not found: {bddl_file}")

        self.env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_file),
            camera_heights=img_size,
            camera_widths=img_size,
        )

        task = task_suite.get_task(task_id)

        if hasattr(task, "language"):
            self.instruction = task.language
        elif hasattr(task, "instruction"):
            self.instruction = task.instruction
        else:
            self.instruction = task_name.replace("_", " ")

    def _find_task_id(self, task_suite, task_name: str) -> int:
        for i, task in enumerate(task_suite.tasks):
            if task.name == task_name:
                return i

        available = [task.name for task in task_suite.tasks]
        raise ValueError(
            f"Unknown LIBERO task_name: {task_name}. "
            f"Available tasks are: {available}"
        )

    def reset(self) -> tuple[np.ndarray, str]:
        obs = self.env.reset()
        return obs["agentview_image"], self.instruction

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        obs, reward, done, info = self.env.step(action)
        return obs["agentview_image"], reward, done, info