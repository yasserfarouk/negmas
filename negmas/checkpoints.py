"""Implements Checkpoint functionality for easy dumping and restoration of any `NamedObject` in negmas."""

from __future__ import annotations

import shutil
from os import PathLike
from pathlib import Path
from typing import Any, Callable

import numpy as np

from negmas.helpers.inout import load
from negmas.types import NamedObject


class CheckpointMixin:
    """Adds the ability to save checkpoints to a `NamedObject`"""

    def checkpoint_init(
        self,
        step_attrib: str = "current_step",
        every: int = 1,
        folder: PathLike | None = None,
        filename: str | None = None,
        info: dict[str, Any] | None = None,
        exist_ok: bool = True,
        single: bool = True,
    ):
        """
        Initializes the object to automatically save a checkpoint

        Args:
            step_attrib: The attribute that defines the current step. If None, there is no step concept
            every: Number of steps per checkpoint. If < 1 no checkpoints will be saved
            folder: The directory to store checkpoints under
            filename: Name of the file to save the checkpoint under. If None, a unique name will be chosen.
                                 If `single_checkpoint` was False, then multiple files will be used prefixed with the
                                 step number
            info: Any extra information to save in the json file associated with each checkpoint
            exist_ok: Override existing files if any
            single: If True, only the most recent checkpoint will be kept

        Remarks:

            - single_checkpoint implies exist_ok

        """
        self.__checkpoint_every = -1 if folder is None else every
        self.__checkpoint_folder = folder
        self.__checkpoint_extra_info = info
        self.__checkpoint_exist_ok = exist_ok
        self.__checkpoint_single = single
        self.__step_atrrib = step_attrib
        self.__checkpoint_filename = filename

    def checkpoint_on_step_started(self) -> Path | None:
        """Should be called on every step to save checkpoints as needed.

        Returns:
            The path on which the checkpoint is stored if one is stored. None otherwise.

        Remarks:

            - Should be called at the BEGINNING of every step before any processing takes place
        """
        if self.__checkpoint_every < 1 or self.__checkpoint_folder is None:
            return None
        step = getattr(self, self.__step_atrrib)
        if step % self.__checkpoint_every == 0 or self.__checkpoint_every == 1:
            me: NamedObject = self  # type: ignore
            return me.checkpoint(
                path=self.__checkpoint_folder,
                file_name=self.__checkpoint_filename,
                info=self.__checkpoint_extra_info,
                exist_ok=self.__checkpoint_exist_ok or self.__checkpoint_single,
                single_checkpoint=self.__checkpoint_single,
                step_attribs=(self.__step_atrrib,),
            )

    def checkpoint_final_step(self) -> Path | None:
        """Should be called at the end of the simulation to save the final state

        Remarks:
            - Should be called after all processing of the final step is conducted.
        """
        if self.__checkpoint_every < 1 or self.__checkpoint_folder is None:
            return None
        me: NamedObject = self  # type: ignore
        return me.checkpoint(
            path=self.__checkpoint_folder,
            file_name=self.__checkpoint_filename,
            info=self.__checkpoint_extra_info,
            exist_ok=True,
            single_checkpoint=self.__checkpoint_single,
            step_attribs=(self.__step_atrrib,),
        )


class CheckpointRunner:
    """Runs an object based on its checkpoints saved in an earlier run"""

    def register_callback(self, callback: Callable[[NamedObject, int], None]) -> None:
        """Registers a callback to be called whenever a new step is loaded

        Args:
            callback: A callable that takes the named object (after it is loaded and an integer specifying the step
                      number and returns None.

        """
        self.__callbacks.append(callback)

    def __init__(
        self,
        folder: str | Path,
        id: str | None = None,
        callback: Callable[[NamedObject, int], None] | None = None,
        watch: bool = False,
        object_type: type[NamedObject] = NamedObject,
    ):
        self.__folder = Path(folder).absolute()
        if id is None:
            pattern = "*.json"
        else:
            pattern = "*id*.json"
        self.__infos = [load(_) for _ in self.__folder.glob(pattern)]
        self.__files = dict(
            zip(
                (_["step"] for _ in self.__infos), (_["filename"] for _ in self.__infos)
            )
        )
        self.__sorted_steps = sorted(list(self.__files.keys()))
        self._step_index = -1
        self.__object: NamedObject | None = None
        self.__object_type = object_type
        self.__callbacks = []
        if callback is not None:
            self.register_callback(callback)
        self.__watch = watch
        if watch:
            raise NotImplementedError("File watching is not implemented yet")

    @property
    def current_step(self) -> int:
        """Gets the current step number"""
        if self._step_index < 0:
            return -1
        return self.__sorted_steps[self._step_index]

    def goto(self, step: int, exact=False) -> int | None:
        """Goes to the nearest step for the given one returning the exact step number.

        Args:

            step: The step we want to goto
            exact: If True, must go to the exact step number, otherwise go to the nearest step stored in a checkpoint

        Returns:

            - None if the current step is the nearest to the given step. Otherwise the exact step we moved to

        """
        if step is None:
            return None
        step_index = np.searchsorted(self.__sorted_steps, step, side="left")
        if step_index > 0 and self.__sorted_steps[step_index - 1] == step:
            step_index -= 1
        if not exact:
            n = len(self.__sorted_steps)
            if step_index > n:
                step = n - 1
            else:
                step = self.__sorted_steps[step_index]

        if self._step_index > -1 and step == self.__sorted_steps[self._step_index]:
            return None
        filename = self.__files.get(step)
        if not filename:
            raise ValueError(f"step {step} has no file")
        self.__object = self.__object_type.from_checkpoint(filename, return_info=False)
        self._step_index = step_index
        for callback in self.__callbacks:
            callback(self.__object, step)
        return step

    def step(self) -> int | None:
        """Go one step forward in the stored steps.

        Returns:
            The number of the current step or None if we are already on the last step.

        """
        nxt_step = self._step_index + 1
        if len(self.__sorted_steps) > nxt_step:
            return self.goto(self.__sorted_steps[nxt_step], exact=True)
        return None

    def run(self):
        """Run all steps. Notice that if `register_callback` was used to register some callback functions, they will
        be called for every stored stepped during the run."""
        while self.step() is not None:
            pass

    @property
    def loaded_object(self) -> NamedObject | None:
        """The object stored in the current checkpoint"""
        return self.__object

    def fork(
        self,
        copy_past_checkpoints: bool = False,
        every: int = 1,
        folder: str | Path | None = None,
        filename: str | None = None,
        info: dict[str, Any] | None = None,
        exist_ok: bool = True,
        single: bool = True,
    ) -> NamedObject | None:
        """
        Creates a copy of the internal object that can be run safely.

        Args:
            copy_past_checkpoints: If true, all checkpoints upto and including current_step will be copied to the given
                                   folder
            every: Number of steps per checkpoint. If < 1 no checkpoints will be saved
            folder: The directory to store checkpoints under
            filename: Name of the file to save the checkpoint under. If None, a unique name will be chosen.
                                 If `single_checkpoint` was False, then multiple files will be used prefixed with the
                                 step number
            info: Any extra information to save in the json file associated with each checkpoint
            exist_ok: Override existing files if any
            single: If True, only the most recent checkpoint will be kept

        Returns:

        """
        if self.__object is None:
            return None
        if (
            not isinstance(self.__object, CheckpointMixin)
            and folder is not None
            and every > 0
        ):
            raise ValueError(
                f"Object of type {self.__object.__class__.__name__} is not implementing the "
                f"CheckpointMixin. It cannot be forked"
            )
        if copy_past_checkpoints and folder is None:
            raise ValueError(
                "Cannot copy past checkpoints because no folder for new checkpoints is given"
            )

        if folder is None:
            folder = Path()
        else:
            folder = Path(folder).absolute()

        if copy_past_checkpoints:
            files = [v for k, v in self.__files.items() if k <= self.current_step]
            for f in files:
                shutil.copy(str(f), str(folder / Path(f).name))
                shutil.copy(str(f) + ".json", str(folder / (Path(f).name + ".json")))
        x = self.__object
        if isinstance(self.__object, CheckpointMixin):
            CheckpointMixin.checkpoint_init(
                x,  # type: ignore
                every=every,
                folder=folder,
                filename=filename,
                info=info,
                exist_ok=exist_ok,
                single=single,
            )
        return x

    @property
    def steps(self) -> list[int]:
        """A list of all stored steps"""
        return self.__sorted_steps

    def reset(self) -> None:
        """Goes before the first step"""
        self._step_index = -1
        self.__object = None

    @property
    def next_step(self) -> int | None:
        """Get the  next stored step number (None if it does not exist)"""
        nxt = self._step_index + 1
        if len(self.__sorted_steps) > nxt:
            return self.__sorted_steps[nxt]
        return None

    @property
    def previous_step(self) -> int | None:
        """Get the  previous stored step number (None if it does not exist)"""
        if self._step_index < 0:
            return -1
        nxt = self._step_index - 1
        if 0 <= nxt:
            return self.__sorted_steps[nxt]
        return None

    @property
    def last_step(self) -> int | None:
        """Get the  last stored step number (None if it does not exist)"""
        return self.__sorted_steps[-1]

    @property
    def first_step(self) -> int | None:
        """Get the  first stored step number (None if it does not exist)"""
        return self.__sorted_steps[0]
