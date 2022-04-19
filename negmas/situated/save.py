from __future__ import annotations

import json
import os
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from negmas.helpers.inout import dump
from negmas.serialization import serialize

if TYPE_CHECKING:
    from .world import World

__all__ = ["save_stats"]


def save_stats(
    world: World,
    log_dir: PathLike | str,
    params: dict[str, Any] | None = None,
    stats_file_name: str | None = None,
):
    """
    Saves the statistics of a world run.

    Args:

        world: The world
        logdir_: The directory to save the stats into.
        params: A parameter list to save with the world
        stats_file_name: File name to use for stats file(s) without extension

    Returns:

    """

    def is_json_serializable(x):
        try:
            json.dumps(x)
        except:
            return False
        return True

    logdir_ = Path(log_dir)
    os.makedirs(logdir_, exist_ok=True)
    if params is None:
        d: dict = serialize(world, add_type_field=False, deep=False)  # type: ignore
        to_del = []
        for k, v in d.items():
            if isinstance(v, list) or isinstance(v, tuple):
                d[k] = str(v)
            if not is_json_serializable(v):
                to_del.append(k)
        for k in to_del:
            del d[k]
        params = d
    if stats_file_name is None:
        stats_file_name = "stats"
    agents: dict[str, dict[str, Any]] = {
        k: dict(id=a.id, name=a.name, type=a.type_name, short_type=a.short_type_name)
        for k, a in world.agents.items()
    }
    for k, v in agents.items():
        agents[k]["neg_requests_sent"] = world.neg_requests_sent[k]
        agents[k]["neg_requests_received"] = world.neg_requests_received[k]
        agents[k]["neg_requests_rejected"] = world.neg_requests_rejected[k]
        agents[k]["negs_registered"] = world.negs_registered[k]
        agents[k]["negs_initiated"] = world.negs_initiated[k]
        agents[k]["negs_succeeded"] = world.negs_succeeded[k]
        agents[k]["negs_failed"] = world.negs_failed[k]
        agents[k]["negs_timedout"] = world.negs_timedout[k]
        agents[k]["contracts_concluded"] = world.contracts_concluded[k]
        agents[k]["contracts_signed"] = world.contracts_signed[k]
        agents[k]["contracts_dropped"] = world.contracts_dropped[k]
        agents[k]["breaches_received"] = world.breaches_received[k]
        agents[k]["breaches_committed"] = world.breaches_committed[k]
        agents[k]["contracts_erred"] = world.contracts_erred[k]
        agents[k]["contracts_nullified"] = world.contracts_nullified[k]
        agents[k]["contracts_executed"] = world.contracts_executed[k]
        agents[k]["contracts_breached"] = world.contracts_breached[k]

    dump(agents, logdir_ / "agents")
    with open(logdir_ / "params.json", "w") as f_:
        f_.write(str(serialize(params)))

    dump(world.stats, logdir_ / stats_file_name)

    if world.info is not None:
        dump(world.info, logdir_ / "info")

    if hasattr(world, "info") and world.info is not None:
        dump(world.info, logdir_ / "info")

    try:
        data = pd.DataFrame.from_dict(world.stats)
        data.to_csv(str(logdir_ / f"{stats_file_name}.csv"), index_label="index")
    except:
        pass

    if world.save_negotiations:
        if len(world.saved_negotiations) > 0:
            data = pd.DataFrame(world.saved_negotiations)
            if "ended_at" in data.columns:
                data = data.sort_values(["ended_at"])
            data.to_csv(str(logdir_ / "negotiations.csv"), index_label="index")  # type: ignore
        else:
            with open(logdir_ / "negotiations.csv", "w") as f:
                f.write("")

    if world.save_resolved_breaches or world.save_unresolved_breaches:
        if len(world.saved_breaches) > 0:
            data = pd.DataFrame(world.saved_breaches)
            data.to_csv(str(logdir_ / "breaches.csv"), index_label="index")  # type: ignore
        else:
            with open(logdir_ / "breaches.csv", "w") as f:
                f.write("")

    # if world.save_signed_contracts:
    #     if len(world.signed_contracts) > 0:
    #         data = pd.DataFrame(world.signed_contracts)
    #         data.to_csv(str(logdir_ / "signed_contracts.csv"), index_label="index")
    #     else:
    #         with open(logdir_ / "signed_contracts.csv", "w") as f:
    #             f.write("")
    #
    # if world.save_cancelled_contracts:
    #     if len(world.cancelled_contracts) > 0:
    #         data = pd.DataFrame(world.cancelled_contracts)
    #         data.to_csv(str(logdir_ / "cancelled_contracts.csv"), index_label="index")
    #     else:
    #         with open(logdir_ / "cancelled_contracts.csv", "w") as f:
    #             f.write("")

    if world.save_signed_contracts or world.save_cancelled_contracts:
        if len(world.saved_contracts) > 0:
            data = pd.DataFrame(world.saved_contracts)
            for col in ("delivery_time", "time"):
                if col in data.columns:
                    data = data.sort_values(["delivery_time"])
                    break
            data.to_csv(str(logdir_ / "contracts.csv"), index_label="index")
        else:
            with open(logdir_ / "contracts.csv", "w") as f:
                f.write("")
