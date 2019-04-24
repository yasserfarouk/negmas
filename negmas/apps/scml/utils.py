import copy
import functools
import itertools
import math
import sys
from os import PathLike
from random import randint, random, shuffle, sample, choices

import numpy as np

from negmas import Agent
from negmas.apps.scml import (
    Product,
    MiningProfile,
    ReactiveMiner,
    ScheduleDrivenConsumer,
    InputOutput,
    Process,
    ManufacturingProfile,
    Factory,
    ConsumptionProfile,
    DEFAULT_NEGOTIATOR,
)
from negmas.helpers import (
    get_class,
    instantiate,
    unique_name,
    get_full_type_name,
    snake_case,
)
from negmas.java import to_dict
from negmas.situated import Entity
from negmas.tournaments import WorldRunResults, TournamentResults, tournament
from .factory_managers import GreedyFactoryManager
from .world import SCMLWorld

if True:
    from typing import (
        Tuple,
        Union,
        Type,
        Iterable,
        Sequence,
        Optional,
        Callable,
        Any,
        Dict,
        List,
    )
    from .factory_managers import FactoryManager

__all__ = [
    "anac2019_world",
    "anac2019_tournament",
    "anac2019_collusion",
    "anac2019_std",
    "balance_calculator",
    "anac2019_sabotage",
]


def integer_cut(n: int, l: int, l_m: Union[int, List[int]]) -> List[int]:
    """
    Generates l random integers that sum to n where each of them is at least l_m
    Args:
        n: total
        l: number of levels
        l_m: minimum per level

    Returns:

    """
    if not isinstance(l_m, Iterable):
        l_m = [l_m] * l
    sizes = np.asarray(l_m)
    if n < sizes.sum():
        raise ValueError(
            f"Cannot generate {l} numbers summing to {n}  with a minimum summing to {sizes.sum()}"
        )
    while sizes.sum() < n:
        sizes[randint(0, l - 1)] += 1
    return list(sizes.tolist())


def _realin(rng: Union[Tuple[float, float], float]) -> float:
    """
    Selects a random number within a range if given or the input if it was a float

    Args:
        rng: Range or single value

    Returns:

        the real within the given range
    """
    if isinstance(rng, float):
        return rng
    if abs(rng[1] - rng[0]) < 1e-8:
        return rng[0]
    return rng[0] + random() * (rng[1] - rng[0])


def _intin(rng: Union[Tuple[int, int], int]) -> int:
    """
    Selects a random number within a range if given or the input if it was an int

    Args:
        rng: Range or single value

    Returns:

        the int within the given range
    """
    if isinstance(rng, int):
        return rng
    if rng[0] == rng[1]:
        return rng[0]
    return randint(rng[0], rng[1])


def anac2019_sabotage_config_generator(
    n_competitors: int,
    n_agents_per_competitor: int,
    agent_names_reveal_type: bool = False,
    non_competitors: Optional[Tuple[Union[str, FactoryManager]]] = None,
    non_competitor_params: Optional[Tuple[Dict[str, Any]]] = None,
    compact: bool = True,
    *,
    consumption_schedule: Tuple[int, int] = (0, 5),
    consumption_horizon: Tuple[int, int] = (10, 15),
    n_retrials: Union[int, Tuple[int, int]] = 5,
    negotiator_type: str = DEFAULT_NEGOTIATOR,
    n_steps: Union[int, Tuple[int, int]] = (50, 100),
    n_miners: Union[int, Tuple[int, int]] = 5,
    n_consumers: Union[int, Tuple[int, int]] = 5,
    profile_cost: Tuple[float, float] = (1, 4),
    profile_time: Union[int, Tuple[int, int]] = 1,
    n_intermediate: Tuple[int, int] = (1, 4),
    min_factories_per_level: int = 5,
    n_default_managers: Tuple[int, int] = (1, 4),
    n_lines: int = 10,
    **kwargs,
) -> List[Dict[str, Any]]:
    return anac2019_config_generator(
        1 + (len(non_competitors) if non_competitors is not None else 1),
        n_agents_per_competitor=n_agents_per_competitor,
        agent_names_reveal_type=agent_names_reveal_type,
        consumption_schedule=consumption_schedule,
        consumption_horizon=consumption_horizon,
        n_retrials=n_retrials,
        negotiator_type=negotiator_type,
        n_steps=n_steps,
        n_miners=n_miners,
        n_consumers=n_consumers,
        profile_cost=profile_cost,
        profile_time=profile_time,
        n_intermediate=n_intermediate,
        min_factories_per_level=min_factories_per_level,
        n_default_managers=n_default_managers,
        non_competitors=non_competitors,
        non_competitor_params=non_competitor_params,
        n_lines=n_lines,
        compact=compact,
        **kwargs,
    )


def anac2019_config_generator(
    n_competitors: int,
    n_agents_per_competitor: int,
    agent_names_reveal_type: bool = False,
    non_competitors: Optional[Tuple[Union[str, FactoryManager]]] = None,
    non_competitor_params: Optional[Tuple[Dict[str, Any]]] = None,
    compact: bool = True,
    *,
    consumption_schedule: Tuple[int, int] = (0, 5),
    consumption_horizon: Tuple[int, int] = (10, 15),
    n_retrials: Union[int, Tuple[int, int]] = 5,
    negotiator_type: str = DEFAULT_NEGOTIATOR,
    n_steps: Union[int, Tuple[int, int]] = (50, 100),
    n_miners: Union[int, Tuple[int, int]] = 5,
    n_consumers: Union[int, Tuple[int, int]] = 5,
    profile_cost: Tuple[float, float] = (1, 4),
    profile_time: Union[int, Tuple[int, int]] = 1,
    n_intermediate: Tuple[int, int] = (1, 4),
    min_factories_per_level: int = 5,
    n_default_managers: Tuple[int, int] = (1, 4),
    n_lines: int = 10,
    **kwargs,
) -> List[Dict[str, Any]]:
    if non_competitors is None:
        non_competitors = (GreedyFactoryManager,)
    if isinstance(n_intermediate, Iterable):
        n_intermediate = list(n_intermediate)
    else:
        n_intermediate = [n_intermediate, n_intermediate]

    n_steps = _intin(n_steps)

    miner_type = ReactiveMiner
    consumer_type = ScheduleDrivenConsumer

    consumer_kwargs = {
        "negotiator_type": negotiator_type,
        "consumption_horizon": _intin(consumption_horizon),
    }
    miner_kwargs = {"negotiator_type": negotiator_type, "n_retrials": n_retrials}
    if negotiator_type is not None:
        for args in (consumer_kwargs, miner_kwargs):
            if "negotiator_type" not in args.keys():
                args["negotiator_type"] = negotiator_type

    n_intermediate_levels = randint(*n_intermediate)

    products = [
        Product(id=0, name="p0", catalog_price=3.0, production_level=0, expires_in=0)
    ]
    processes = []
    miners = [
        instantiate(
            miner_type,
            profiles={products[-1].id: MiningProfile()},
            name=f"m_{i}",
            **miner_kwargs,
        )
        for i in range(n_miners)
    ]
    factories = []

    def _s(x):
        return x if x is not None else 0

    if isinstance(profile_cost, tuple):
        historical_cost = (profile_cost[0] + profile_cost[1]) / 2.0
    else:
        historical_cost = profile_cost
    historical_cost = (historical_cost * 0.85, historical_cost * 1.15)

    for level in range(n_intermediate_levels + 1):
        p = Process(
            name=f"p{level + 1}",
            inputs=[InputOutput(product=level, quantity=1, step=0.0)],
            production_level=level + 1,
            outputs=[InputOutput(product=level + 1, quantity=1, step=1.0)],
            historical_cost=_realin(historical_cost),
            id=level,
        )
        new_product = Product(
            name=f"p{level + 1}",
            catalog_price=products[-1].catalog_price + p.historical_cost
            # keep this to the world to calculate _s(products[-1].catalog_price) + level + 1
            ,
            production_level=level + 1,
            id=level + 1,
            expires_in=0,
        )
        processes.append(p)
        products.append(new_product)

    n_defaults = []
    for level in range(n_intermediate_levels + 1):
        n_defaults.append(_intin(n_default_managers))
    n_agents = n_agents_per_competitor * n_competitors
    n_a_list = integer_cut(n_agents, n_intermediate_levels + 1, 0)
    for i, n_a in enumerate(n_a_list):
        if n_a + n_defaults[i] < min_factories_per_level:
            n_defaults[i] += min_factories_per_level - (n_a + n_defaults[i])
    n_f_list = [a + b for a, b in zip(n_defaults, n_a_list)]
    n_factories = sum(n_f_list)

    if non_competitor_params is None:
        non_competitor_params = [{}] * len(non_competitors)

    non_competitors = [get_full_type_name(_) for _ in non_competitors]

    for c_, p_ in zip(non_competitors, non_competitor_params):
        if c_.startswith("negmas.apps.scml.") and c_.endswith("GreedyFactoryManager"):
            p_.update({"negotiator_type": negotiator_type, "n_retrials": n_retrials})

    max_def_agents = len(non_competitors) - 1
    manager_types = [None] * n_factories
    manager_params = [None] * n_factories
    first_in_level = 0
    for level in range(n_intermediate_levels + 1):
        n_d = n_defaults[level]
        n_f = n_f_list[level]
        assert (
            n_d <= n_f
        ), f"Got {n_f} total factories at level {level} out of which {n_d} are default!!"
        for j in range(n_f):
            profiles = []
            factory_time = _intin(profile_time)
            factory_cost = _realin(profile_cost)
            for k in range(n_lines):
                profiles.append(
                    ManufacturingProfile(
                        n_steps=factory_time,
                        cost=factory_cost,
                        initial_pause_cost=0,
                        running_pause_cost=0,
                        resumption_cost=0,
                        cancellation_cost=0,
                        line=k,
                        process=processes[level],
                    )
                )
            factory = Factory(
                id=f"f{level + 1}_{j}",
                max_storage=sys.maxsize,
                profiles=profiles,
                initial_storage={},
                initial_wallet=1000.0,
            )
            factories.append(factory)
            if j >= n_f - n_d:  # default managers are last managers in the list
                def_indx = randint(0, max_def_agents)
                manager_types[first_in_level + j] = non_competitors[def_indx]
                params_ = non_competitor_params[def_indx]
                if agent_names_reveal_type:
                    params_["name"] = f"_df_{level + 1}_{j}"
                else:
                    params_["name"] = None
                manager_params[first_in_level + j] = params_
        first_in_level += n_f

    def create_schedule():
        if isinstance(consumption_schedule, tuple) and len(consumption_schedule) == 2:
            return list(
                np.random.randint(
                    consumption_schedule[0], consumption_schedule[1], n_steps
                ).tolist()
            )
        return consumption_schedule

    consumers = [
        instantiate(
            consumer_type,
            profiles={products[-1].id: ConsumptionProfile(schedule=create_schedule())},
            name=f"c_{i}",
            **consumer_kwargs,
        )
        for i in range(n_consumers)
    ]

    config = {
        "world_params": dict(
            name=unique_name("", add_time=True, rand_digits=4),
            time_limit=7200,
            neg_time_limit=120,
            neg_n_steps=20,
            neg_step_time_limit=10,
            negotiation_speed=21,
            default_signing_delay=1,
            transportation_delay=0,
            no_bank=True,
            breach_penalty_society=0.02,
            no_insurance=False,
            premium=0.03,
            premium_time_increment=0.1,
            premium_breach_increment=0.001,
            max_allowed_breach_level=None,
            breach_penalty_society_min=0.0,
            breach_penalty_victim=0.0,
            breach_move_max_product=True,
            initial_wallet_balances=1000.0,
            transfer_delay=0,
            start_negotiations_immediately=False,
            catalog_profit=0.15,
            financial_reports_period=10,
            default_price_for_products_without_one=1,
            compensation_fraction=0.5,
            n_steps=n_steps,
            compact=compact,
            **kwargs,
        ),
        "products": [to_dict(p, add_type_field=False, camel=False) for p in products],
        "processes": [to_dict(p, add_type_field=False, camel=False) for p in processes],
        "factories": [
            {
                "profile": {
                    "n_steps": f.profiles[0].n_steps,
                    "cost": f.profiles[0].cost,
                    "line": f.profiles[0].line,
                    "process.id": f.profiles[0].process.id,
                },
                "max_storage": sys.maxsize,
                "initial_wallet": 1000.0,
                "id": f.id,
                "n_lines": n_lines,
            }
            for f in factories
        ],
        "miners": [
            dict(
                id=m.id,
                name=m.name,
                type=get_full_type_name(miner_type),
                args=miner_kwargs,
                profiles={
                    k: to_dict(v, add_type_field=False, camel=False)
                    for k, v in m.profiles.items()
                },
            )
            for m in miners
        ],
        "consumers": [
            dict(
                id=c.id,
                name=c.name,
                type=get_full_type_name(consumer_type),
                args=consumer_kwargs,
                profiles={
                    k: to_dict(v, add_type_field=False, camel=False)
                    for k, v in c.profiles.items()
                },
            )
            for c in consumers
        ],
        "manager_types": [
            get_full_type_name(_) if isinstance(_, FactoryManager) else _
            for _ in manager_types
        ],
        "manager_params": manager_params,
        "n_factories_per_level": n_f_list,
        "agent_names_reveal_type": agent_names_reveal_type,
        "compact": compact,
        "scoring_context": {},
        "non_competitors": non_competitors,
        "non_competitor_params": non_competitor_params,
    }
    config.update(kwargs)
    return [config]


def anac2019_sabotage_assigner(
    config: List[Dict[str, Any]],
    max_n_worlds: int,
    n_agents_per_competitor: int = 1,
    competitors: Sequence[Type[Agent]] = (),
    params: Sequence[Dict[str, Any]] = (),
) -> List[List[Dict[str, Any]]]:
    config = config[0]
    competitors = list(
        get_full_type_name(_) if not isinstance(_, str) and _ is not None else _
        for _ in competitors
    )
    n_competitors = len(competitors)
    params = (
        list(params) if params is not None else [dict() for _ in range(n_competitors)]
    )
    n_agents = n_agents_per_competitor * 2
    agent_names_reveal_type = config.pop("agent_names_reveal_type", False)

    try:
        n_permutations = math.factorial(n_agents)
    except ArithmeticError:
        n_permutations = None

    manager_types = config["manager_types"]

    assignable_factories = [i for i, mtype in enumerate(manager_types) if mtype is None]

    configs = []

    def shorten(long_name: str, d: Dict[str, Any]) -> str:
        if long_name.endswith("JavaFactoryManager"):
            long_name = d.get("java_class_name", long_name)
        name = (
            long_name.split(".")[-1]
            .lower()
            .replace("factory_manager", "")
            .replace("manager", "")
        )
        name = (
            name.replace("factory", "")
            .replace("agent", "")
            .replace("miner", "m")
            .replace("consumer", "")
        )
        if long_name.startswith("jnegmas"):
            name = f"j:{name}"
        return name

    non_competitors = config.get(
        "non_competitors", ("negmas.apps.scml.factory_managers.GreedyFactoryManager",)
    )
    max_def = len(non_competitors) - 1
    non_competitor_params = config.get("non_competitor_params", None)
    if non_competitor_params is None:
        non_competitor_params = [{}] * (max_def + 1)

    def _type_name(c_: str, p_) -> str:
        return instantiate(c_, **p_).type_name

    def _copy_config(perm_, conf, indx, comp, c_p):
        perm_ = list(perm_)
        perm1 = copy.deepcopy(perm_)
        ctype = _type_name(comp, c_p)
        for i, (c_, p_) in enumerate(perm_):
            if c_ != "competitor":
                perm_[i] = (c_, p_)
            else:
                perm_[i] = (comp, c_p)
        new_config = copy.deepcopy(conf)
        new_config["world_params"]["name"] += f"{indx:05d}_with_{shorten(comp, c_p)}"
        new_config["scoring_context"].update(
            {"competitor": ctype, "competitor_params": c_p}
        )
        for i, (a, p_) in enumerate(perm_):
            new_config["manager_types"][assignable_factories[i]] = a
            new_config["manager_params"][assignable_factories[i]] = p_

        for i, (c_, p_) in enumerate(perm1):
            if c_ != "competitor":
                perm1[i] = (c_, p_)
            else:
                def_indx = randint(0, max_def)
                perm1[i] = (non_competitors[def_indx], non_competitor_params[def_indx])
        no_sabotage_config = copy.deepcopy(conf)
        no_sabotage_config["world_params"][
            "name"
        ] += f"{indx:05d}_no_{shorten(comp, c_p)}"
        no_sabotage_config["scoring_context"].update(
            {"competitor": ctype, "competitor_params": c_p}
        )
        for i, (a, p_) in enumerate(perm1):
            no_sabotage_config["manager_types"][assignable_factories[i]] = a
            no_sabotage_config["manager_params"][assignable_factories[i]] = p_

        return [new_config, no_sabotage_config]

    max_n_worlds = (
        int(max(1, max_n_worlds // n_competitors)) if max_n_worlds is not None else None
    )

    if n_permutations is not None and (
        max_n_worlds is None or n_permutations <= max_n_worlds
    ):
        k = 0
        others = (
            list(choices(list(zip(non_competitors, non_competitor_params))))
            * n_agents_per_competitor
        )
        agents = ["competitor"] * n_agents_per_competitor + [_[0] for _ in others]
        agent_params = ["competitor"] * n_agents_per_competitor + [_[1] for _ in others]
        for permutation in itertools.permutations(zip(agents, agent_params)):
            assert len(permutation) == len(assignable_factories)
            for competitor, c_params in zip(competitors, params):
                perm = copy.deepcopy(permutation)
                configs.append(_copy_config(perm, config, k, competitor, c_params))
                k += 1
    elif max_n_worlds is None:
        raise ValueError(f"Did not give max_n_worlds and cannot find n_permutations.")
    else:
        others = (
            list(choices(list(zip(non_competitors, non_competitor_params))))
            * n_agents_per_competitor
        )
        agents = ["competitor"] * n_agents_per_competitor + [_[0] for _ in others]
        agent_params = ["competitor"] * n_agents_per_competitor + [_[1] for _ in others]
        permutation = list(zip(agents, agent_params))
        assert len(permutation) == len(assignable_factories)
        for k in range(max_n_worlds):
            for competitor, c_params in zip(competitors, params):
                perm = copy.deepcopy(permutation)
                shuffle(perm)
                configs.append(_copy_config(perm, config, k, competitor, c_params))

    if agent_names_reveal_type:
        for config_set in configs:
            for config in config_set:
                nxt = 0
                for i, (t, p, f) in enumerate(
                    zip(
                        config["manager_types"],
                        config["manager_params"],
                        config["factories"],
                    )
                ):
                    if p.get("name", "").startswith("_df_"):
                        continue
                    p = p.copy()
                    name_ = (
                        t.short_type_name
                        if isinstance(t, Entity)
                        else get_full_type_name(t)
                        if not isinstance(t, str)
                        else shorten(t, config["manager_params"][i])
                    )
                    p["name"] = f'{name_}@{f["id"][1:]}'
                    config["manager_params"][i] = p
                    nxt = nxt + 1

    return configs


def anac2019_assigner(
    config: List[Dict[str, Any]],
    max_n_worlds: int,
    n_agents_per_competitor: int = 1,
    competitors: Sequence[Type[Agent]] = (),
    params: Sequence[Dict[str, Any]] = (),
) -> List[List[Dict[str, Any]]]:
    config = config[0]
    competitors = list(
        get_full_type_name(_) if not isinstance(_, str) and _ is not None else _
        for _ in competitors
    )
    n_competitors = len(competitors)
    params = (
        list(params) if params is not None else [dict() for _ in range(n_competitors)]
    )
    n_agents = n_agents_per_competitor * n_competitors
    agent_names_reveal_type = config.pop("agent_names_reveal_type", False)

    try:
        n_permutations = math.factorial(n_agents)
    except ArithmeticError:
        n_permutations = None

    manager_types = config["manager_types"]

    assignable_factories = [i for i, mtype in enumerate(manager_types) if mtype is None]

    agents = list(itertools.chain(*([competitors] * n_agents_per_competitor)))
    agent_params = list(itertools.chain(*([params] * n_agents_per_competitor)))
    configs = []

    def _copy_config(perm_, c, indx):
        new_config = copy.deepcopy(c)
        new_config["world_params"]["name"] += f"{indx:05d}"
        for i, (a, p_) in enumerate(perm_):
            new_config["manager_types"][assignable_factories[i]] = a
            new_config["manager_params"][assignable_factories[i]] = p_
        return [new_config]

    if n_permutations is not None and (
        max_n_worlds is None or n_permutations <= max_n_worlds
    ):
        k = 0
        for permutation in itertools.permutations(zip(agents, agent_params)):
            assert len(permutation) == len(assignable_factories)
            configs.append(_copy_config(permutation, config, k))
            k += 1
    elif max_n_worlds is None:
        raise ValueError(f"Did not give max_n_worlds and cannot find n_permutations.")
    else:
        permutation = list(zip(agents, agent_params))
        assert len(permutation) == len(assignable_factories)
        for k in range(max_n_worlds):
            perm = copy.deepcopy(permutation)
            shuffle(perm)
            configs.append(_copy_config(perm, config, k))

    def shorten(long_name: str, d: Dict[str, Any]) -> str:
        if long_name.endswith("JavaFactoryManager"):
            long_name = d.get("java_class_name", long_name)
        name = (
            long_name.split(".")[-1]
            .lower()
            .replace("factory_manager", "")
            .replace("manager", "")
        )
        name = (
            name.replace("factory", "")
            .replace("agent", "")
            .replace("miner", "m")
            .replace("consumer", "")
        )
        if long_name.startswith("jnegmas"):
            name = f"j:{name}"
        return name

    if agent_names_reveal_type:
        for config_set in configs:
            for config in config_set:
                nxt = 0
                for i, (t, p, f) in enumerate(
                    zip(
                        config["manager_types"],
                        config["manager_params"],
                        config["factories"],
                    )
                ):
                    if p.get("name", "").startswith("_df_"):
                        continue
                    p = p.copy()
                    name_ = (
                        t.short_type_name
                        if isinstance(t, Entity)
                        else get_full_type_name(t)
                        if not isinstance(t, str)
                        else shorten(t, config["manager_params"][i])
                    )
                    p["name"] = f'{name_}@{f["id"][1:]}'
                    config["manager_params"][i] = p
                    nxt = nxt + 1
    return configs


def anac2019_world_generator(**kwargs):
    products = [Product(**p) for p in kwargs["products"]]
    processes = [Process(**p) for p in kwargs["processes"]]
    for process in processes:
        process.inputs = [InputOutput(**io) for io in process.inputs]
        process.outputs = [InputOutput(**io) for io in process.outputs]
    factories = []
    for f in kwargs["factories"]:
        p = f["profile"]
        factories.append(
            Factory(
                initial_storage={},
                initial_wallet=f["initial_wallet"],
                max_storage=f["max_storage"],
                id=f'{f["id"]}',
                profiles=[
                    ManufacturingProfile(
                        n_steps=p["n_steps"],
                        cost=p["cost"],
                        line=_,
                        process=processes[p["process.id"]],
                        cancellation_cost=0.0,
                        initial_pause_cost=0.0,
                        resumption_cost=0,
                        running_pause_cost=0.0,
                    )
                    for _ in range(f["n_lines"])
                ],
            )
        )
    miners = []
    for m in kwargs["miners"]:
        miner = instantiate(
            m["type"],
            **m["args"],
            name=m["name"],
            profiles={k: MiningProfile(**v) for k, v in m["profiles"].items()},
        )
        miner.id = m["id"]
        miners.append(miner)

    consumers = []
    for c in kwargs["consumers"]:
        consumer = instantiate(
            c["type"],
            **c["args"],
            name=c["name"],
            profiles={k: ConsumptionProfile(**v) for k, v in c["profiles"].items()},
        )
        consumer.id = c["id"]
        consumers.append(consumer)

    kwargs.pop("n_factories_per_level", None)
    manager_types = kwargs.pop("manager_types", [])
    manager_params = kwargs.pop("manager_params", [])
    managers = [
        instantiate(mt, **mp)
        for mt, mp in zip(manager_types, itertools.cycle(manager_params))
    ]
    world = SCMLWorld(
        products=products,
        processes=processes,
        factories=factories,
        consumers=consumers,
        miners=miners,
        factory_managers=managers,
        **kwargs["world_params"],
    )
    return world


def anac2019_world(
    competitors: Sequence[Union[str, Type[FactoryManager]]] = (),
    params: Sequence[Dict[str, Any]] = (),
    randomize: bool = True,
    log_file_name: str = None,
    name: str = None,
    agent_names_reveal_type: bool = False,
    n_intermediate: Tuple[int, int] = (1, 4),
    n_miners=5,
    n_factories_per_level=11,
    n_agents_per_competitor=1,
    n_consumers=5,
    n_lines_per_factory=10,
    guaranteed_contracts=False,
    use_consumer=True,
    max_insurance_premium=float("inf"),
    n_retrials=5,
    negotiator_type: str = DEFAULT_NEGOTIATOR,
    transportation_delay=0,
    default_signing_delay=0,
    max_storage=sys.maxsize,
    consumption_horizon=15,
    consumption=(3, 5),
    negotiation_speed=21,
    neg_time_limit=60 * 4,
    neg_n_steps=20,
    n_steps=100,
    time_limit=60 * 90,
    n_default_per_level: int = 5,
    compact: bool = False,
) -> SCMLWorld:
    """
    Creates a world compatible with the ANAC 2019 competition. Note that

    Args:
        n_agents_per_competitor: Number of instantiations of each competing type.
        name: World name to use
        agent_names_reveal_type: If true, a snake_case version of the agent_type will prefix agent names
        randomize: If true, managers are assigned to factories randomly otherwise in the order
        they are giving (cycling).
        n_intermediate:
        n_default_per_level:
        competitors: A list of class names for the competitors
        params: A list of dictionaries giving parameters to pass to the competitors
        n_miners: number of miners of the single raw material
        n_factories_per_level: number of factories at every production level
        n_consumers: number of consumers of the final product
        n_steps: number of simulation steps
        n_lines_per_factory: number of lines in each factory
        negotiation_speed: The number of negotiation steps per simulation step. None means infinite
        default_signing_delay: The number of simulation between contract conclusion and signature
        neg_n_steps: The maximum number of steps of a single negotiation (that is double the number of rounds)
        neg_time_limit: The total time-limit of a single negotiation
        time_limit: The total time-limit of the simulation
        transportation_delay: The transportation delay
        n_retrials: The number of retrials the `Miner` and `GreedyFactoryManager` will try if negotiations fail
        max_insurance_premium: The maximum insurance premium accepted by `GreedyFactoryManager` (-1 to disable)
        use_consumer: If true, the `GreedyFactoryManager` will use an internal consumer for buying its needs
        guaranteed_contracts: If true, the `GreedyFactoryManager` will only sign contracts that it can guaratnee not to
        break.
        consumption_horizon: The number of steps for which `Consumer` publishes `CFP` s
        consumption: The consumption schedule will be sampled from a uniform distribution with these limits inclusive
        log_file_name: File name to store the logs
        negotiator_type: The negotiation factory used to create all negotiators
        max_storage: maximum storage capacity for all factory managers If None then it is unlimited
        compact: If True, then compact logs will be created to reduce memory footprint

    Returns:
        SCMLWorld ready to run

    Remarks:

        - Every production level n has one process only that takes n steps to complete


    """
    competitors = list(competitors)
    params = (
        list(params)
        if params is not None
        else [dict() for _ in range(len(competitors))]
    )
    if n_factories_per_level == n_default_per_level and len(competitors) > 0:
        raise ValueError(
            f"All factories in all levels are occupied by the default factory manager. Either decrease"
            f" n_default_per_level ({n_default_per_level}) or increase n_factories_per_level "
            f" ({n_factories_per_level})"
        )
    if isinstance(n_intermediate, Iterable):
        n_intermediate = list(n_intermediate)
    else:
        n_intermediate = [n_intermediate, n_intermediate]
    n_competitors = len(competitors)
    n_intermediate_levels_min = (
        int(math.ceil(n_competitors / (n_factories_per_level - n_default_per_level)))
        - 1
    )
    if n_intermediate_levels_min > n_intermediate[1]:
        raise ValueError(
            f"Need {n_intermediate_levels_min} intermediate levels to run {n_competitors} competitors"
        )
    n_intermediate[0] = max(n_intermediate_levels_min, n_intermediate[0])
    competitors = [get_class(c) if isinstance(c, str) else c for c in competitors]
    if len(competitors) < 1:
        competitors.append(GreedyFactoryManager)
        params.append(dict())
    world = SCMLWorld.chain_world(
        log_file_name=log_file_name,
        n_steps=n_steps,
        agent_names_reveal_type=agent_names_reveal_type,
        negotiation_speed=negotiation_speed,
        n_intermediate_levels=randint(*n_intermediate),
        n_miners=n_miners,
        n_consumers=n_consumers,
        n_factories_per_level=n_factories_per_level,
        consumption=consumption,
        consumer_kwargs={
            "negotiator_type": negotiator_type,
            "consumption_horizon": consumption_horizon,
        },
        miner_kwargs={"negotiator_type": negotiator_type, "n_retrials": n_retrials},
        default_manager_params={
            "negotiator_type": negotiator_type,
            "n_retrials": n_retrials,
            "sign_only_guaranteed_contracts": guaranteed_contracts,
            "use_consumer": use_consumer,
            "max_insurance_premium": max_insurance_premium,
        },
        transportation_delay=transportation_delay,
        time_limit=time_limit,
        neg_time_limit=neg_time_limit,
        neg_n_steps=neg_n_steps,
        default_signing_delay=default_signing_delay,
        n_lines_per_factory=n_lines_per_factory,
        max_storage=max_storage,
        manager_types=competitors,
        manager_params=params,
        n_default_per_level=n_default_per_level,
        randomize=randomize,
        name=name,
        compact=compact,
    )

    return world


def balance_calculator(
    worlds: List[SCMLWorld], scoring_context: Dict[str, Any], dry_run: bool
) -> WorldRunResults:
    """A scoring function that scores factory managers' performance by the final balance only ignoring whatever still
    in their inventory.

    Args:

        worlds: The world which is assumed to be run up to the point at which the scores are to be calculated.
        scoring_context:  A dict of context parameters passed by the world generator or assigner.
        dry_run: A boolean specifying whether this is a dry_run. For dry runs, only names and types are expected in
                 the returned `WorldRunResults`

    Returns:
        WorldRunResults giving the names, scores, and types of factory managers.

    """
    assert len(worlds) == 1
    world = worlds[0]
    result = WorldRunResults(
        world_names=[world.name], log_file_names=[world.log_file_name]
    )
    initial_balances = []
    for manager in world.factory_managers:
        if "_df_" in manager.id:
            continue
        initial_balances.append(world.a2f[manager.id].initial_balance)
    normalize = all(_ != 0 for _ in initial_balances)
    for manager in world.factory_managers:
        if "_df_" in manager.id:
            continue
        factory = world.a2f[manager.id]
        result.names.append(manager.name)
        result.types.append(manager.type_name)
        if dry_run:
            result.scores.append(None)
        if normalize:
            result.scores.append(
                (factory.balance - factory.initial_balance) / factory.initial_balance
            )
        else:
            result.scores.append(factory.balance - factory.initial_balance)
    return result


def sabotage_effectiveness(
    worlds: List[SCMLWorld], scoring_context: Dict[str, Any], dry_run: bool
) -> WorldRunResults:
    """A scoring function that scores factory managers' performance by the final balance only ignoring whatever still
    in their inventory.

    Args:

        worlds: The world which is assumed to be run up to the point at which the scores are to be calculated.
        scoring_context:  A dict of context parameters passed by the world generator or assigner.
        dry_run: A boolean specifying whether this is a dry_run. For dry runs, only names and types are expected in
                 the returned `WorldRunResults`

    Returns:
        WorldRunResults giving the names, scores, and types of factory managers.

    """
    assert len(worlds) == 2
    type_scored = scoring_context.get("competitor", None)
    if type_scored is None:
        raise ValueError("Cannot determine which is the sabotaging agent")
    if dry_run:
        results = WorldRunResults(world_names=[""], log_file_names=[""])
        results.names = [""]
        results.types = [type_scored]
        results.scores = [None]
        return results
    results = [balance_calculator([_], {}, dry_run=False) for _ in worlds]
    normal_scores, sabotaged_scores = [], []
    sabotaged_indices, normal_indices = [], []
    for i in range(len(worlds)):
        if type_scored in results[i].types:
            sabotaged_indices.append(int(i))
        else:
            normal_indices.append(int(i))
    if len(sabotaged_indices) < 1:
        raise ValueError(
            f"The sabotaging agent type {type_scored} did not participate in any worlds"
        )
    if len(normal_indices) < 1:
        raise ValueError(
            f"The sabotaging agent type {type_scored} participated in ALL worlds"
        )

    for indx in sabotaged_indices:
        sabotaged_scores.extend(
            score
            for score, type_ in zip(results[indx].scores, results[indx].types)
            if type_ != type_scored
        )
    for indx in normal_indices:
        normal_scores.extend(
            score
            for score, type_ in zip(results[indx].scores, results[indx].types)
            if type_ != type_scored
        )
    normal_score = sum(normal_scores) / len(normal_scores)
    sabotaged_score = sum(sabotaged_scores) / len(sabotaged_scores)
    result = WorldRunResults(
        world_names=[_.name for _ in worlds],
        log_file_names=[_.log_file_name for _ in worlds],
    )
    result.names = [""]
    result.scores = [(sabotaged_score - normal_score) / (normal_score + 1.0)]
    result.types = [type_scored]
    return result


def anac2019_tournament(
    competitors: Sequence[Union[str, Type[FactoryManager]]],
    agent_names_reveal_type=False,
    n_configs: int = 5,
    max_worlds_per_config: int = 1000,
    n_runs_per_world: int = 5,
    n_agents_per_competitor: int = 5,
    tournament_path: str = "./logs/tournaments",
    total_timeout: Optional[int] = 7200,
    parallelism="parallel",
    scheduler_ip: Optional[str] = None,
    scheduler_port: Optional[str] = None,
    tournament_progress_callback: Callable[[Optional[WorldRunResults]], None] = None,
    world_progress_callback: Callable[[Optional[SCMLWorld]], None] = None,
    name: str = None,
    verbose: bool = False,
    configs_only=False,
    compact=False,
    **kwargs,
) -> Union[TournamentResults, PathLike]:
    """
    The function used to run ANAC 2019 SCML tournament (collusion track).

    Args:

        name: Tournament name
        competitors: A list of class names for the competitors
        agent_names_reveal_type: If true then the type of an agent should be readable in its name (most likely at its
                                 beginning).
        n_configs: The number of different world configs (up to competitor assignment) to be generated.
        max_worlds_per_config: The maximum number of worlds to run per config. If None, then all possible assignments
                             of competitors within each config will be tried (all permutations).
        n_runs_per_world: Number of runs per world. All of these world runs will have identical competitor assignment
                          and identical world configuration.
        n_agents_per_competitor: Number of agents per competitor
        total_timeout: Total timeout for the complete process
        tournament_path: Path at which to store all results. A scores.csv file will keep the scores and logs folder will
                         keep detailed logs
        parallelism: Type of parallelism. Can be 'serial' for serial, 'parallel' for parallel and 'distributed' for distributed
        scheduler_port: Port of the dask scheduler if parallelism is dask, dist, or distributed
        scheduler_ip: IP Address of the dask scheduler if parallelism is dask, dist, or distributed
        world_progress_callback: A function to be called after everystep of every world run (only allowed for serial
                                 evaluation and should be used with cautious).
        tournament_progress_callback: A function to be called with `WorldRunResults` after each world finished
                                      processing
        verbose: Verbosity
        configs_only: If true, a config file for each
        compact: If true, effort will be made to reduce memory footprint including disableing most logs
        kwargs: Arguments to pass to the `world_generator` function

    Returns:

        `TournamentResults` The results of the tournament or a `PathLike` giving the location where configs were saved

    Remarks:

        Default parameters will be used in the league with the exception of `parallelism` which may use distributed
        processing

    """
    return anac2019_collusion(
        competitors=competitors,
        agent_names_reveal_type=agent_names_reveal_type,
        n_configs=n_configs,
        max_worlds_per_config=max_worlds_per_config,
        n_runs_per_world=n_runs_per_world,
        n_agents_per_competitor=n_agents_per_competitor,
        tournament_path=tournament_path,
        total_timeout=total_timeout,
        parallelism=parallelism,
        scheduler_ip=scheduler_ip,
        scheduler_port=scheduler_port,
        tournament_progress_callback=tournament_progress_callback,
        world_progress_callback=world_progress_callback,
        name=name,
        verbose=verbose,
        compact=compact,
        configs_only=configs_only,
        **kwargs,
    )


def anac2019_std(
    competitors: Sequence[Union[str, Type[FactoryManager]]],
    agent_names_reveal_type=False,
    n_configs: int = 5,
    max_worlds_per_config: Optional[int] = 1000,
    n_runs_per_world: int = 5,
    min_factories_per_level: int = 5,
    tournament_path: str = "./logs/tournaments",
    total_timeout: Optional[int] = 7200,
    parallelism="parallel",
    scheduler_ip: Optional[str] = None,
    scheduler_port: Optional[str] = None,
    tournament_progress_callback: Callable[[Optional[WorldRunResults]], None] = None,
    world_progress_callback: Callable[[Optional[SCMLWorld]], None] = None,
    non_competitors: Optional[Sequence[Union[str, Type[FactoryManager]]]] = None,
    non_competitor_params: Optional[Sequence[Union[str, Type[FactoryManager]]]] = None,
    name: str = None,
    verbose: bool = False,
    configs_only=False,
    compact=False,
    **kwargs,
) -> Union[TournamentResults, PathLike]:
    """
    The function used to run ANAC 2019 SCML tournament (standard track).

    Args:

        name: Tournament name
        competitors: A list of class names for the competitors
        agent_names_reveal_type: If true then the type of an agent should be readable in its name (most likely at its
                                 beginning).
        n_configs: The number of different world configs (up to competitor assignment) to be generated.
        max_worlds_per_config: The maximum number of worlds to run per config. If None, then all possible assignments
                             of competitors within each config will be tried (all permutations).
        n_runs_per_world: Number of runs per world. All of these world runs will have identical competitor assignment
                          and identical world configuration.
        min_factories_per_level: Minimum number of factories for each production level
        total_timeout: Total timeout for the complete process
        tournament_path: Path at which to store all results. A scores.csv file will keep the scores and logs folder will
                         keep detailed logs
        parallelism: Type of parallelism. Can be 'serial' for serial, 'parallel' for parallel and 'distributed' for
                     distributed
        scheduler_port: Port of the dask scheduler if parallelism is dask, dist, or distributed
        scheduler_ip: IP Address of the dask scheduler if parallelism is dask, dist, or distributed
        world_progress_callback: A function to be called after everystep of every world run (only allowed for serial
                                 evaluation and should be used with cautious).
        tournament_progress_callback: A function to be called with `WorldRunResults` after each world finished
                                      processing
        non_competitors: A list of agent types that will not be competing in the sabotage competition but will exist
                         in the world
        non_competitor_params: parameters of non competitor agents
        verbose: Verbosity
        configs_only: If true, a config file for each
        compact: If true, compact logs will be created and effort will be made to reduce the memory footprint
        kwargs: Arguments to pass to the `world_generator` function

    Returns:

        `TournamentResults` The results of the tournament or a `PathLike` giving the location where configs were saved

    Remarks:

        Default parameters will be used in the league with the exception of `parallelism` which may use distributed
        processing

    """
    return tournament(
        competitors=competitors,
        non_competitors=non_competitors,
        non_competitor_params=non_competitor_params,
        agent_names_reveal_type=agent_names_reveal_type,
        n_configs=n_configs,
        n_runs_per_world=n_runs_per_world,
        max_worlds_per_config=max_worlds_per_config,
        tournament_path=tournament_path,
        total_timeout=total_timeout,
        parallelism=parallelism,
        scheduler_ip=scheduler_ip,
        scheduler_port=scheduler_port,
        tournament_progress_callback=tournament_progress_callback,
        world_progress_callback=world_progress_callback,
        name=name,
        verbose=verbose,
        configs_only=configs_only,
        n_agents_per_competitor=1,
        world_generator=anac2019_world_generator,
        config_generator=anac2019_config_generator,
        config_assigner=anac2019_assigner,
        score_calculator=balance_calculator,
        min_factories_per_level=min_factories_per_level,
        compact=compact,
        **kwargs,
    )


def anac2019_collusion(
    competitors: Sequence[Union[str, Type[FactoryManager]]],
    agent_names_reveal_type=False,
    n_configs: int = 5,
    max_worlds_per_config: Optional[int] = 1000,
    n_runs_per_world: int = 5,
    n_agents_per_competitor: int = 5,
    min_factories_per_level: int = 5,
    tournament_path: str = "./logs/tournaments",
    total_timeout: Optional[int] = 7200,
    parallelism="parallel",
    scheduler_ip: Optional[str] = None,
    scheduler_port: Optional[str] = None,
    tournament_progress_callback: Callable[[Optional[WorldRunResults]], None] = None,
    world_progress_callback: Callable[[Optional[SCMLWorld]], None] = None,
    non_competitors: Optional[Sequence[Union[str, Type[FactoryManager]]]] = None,
    non_competitor_params: Optional[Sequence[Union[str, Type[FactoryManager]]]] = None,
    name: str = None,
    verbose: bool = False,
    configs_only=False,
    compact=False,
    **kwargs,
) -> Union[TournamentResults, PathLike]:
    """
    The function used to run ANAC 2019 SCML tournament (collusion track).

    Args:

        name: Tournament name
        competitors: A list of class names for the competitors
        agent_names_reveal_type: If true then the type of an agent should be readable in its name (most likely at its
                                 beginning).
        n_configs: The number of different world configs (up to competitor assignment) to be generated.
        max_worlds_per_config: The maximum number of worlds to run per config. If None, then all possible assignments
                             of competitors within each config will be tried (all permutations).
        n_runs_per_world: Number of runs per world. All of these world runs will have identical competitor assignment
                          and identical world configuration.
        n_agents_per_competitor: Number of agents per competitor
        min_factories_per_level: Minimum number of factories for each production level
        total_timeout: Total timeout for the complete process
        tournament_path: Path at which to store all results. A scores.csv file will keep the scores and logs folder will
                         keep detailed logs
        parallelism: Type of parallelism. Can be 'serial' for serial, 'parallel' for parallel and 'distributed' for
                     distributed
        scheduler_port: Port of the dask scheduler if parallelism is dask, dist, or distributed
        scheduler_ip: IP Address of the dask scheduler if parallelism is dask, dist, or distributed
        world_progress_callback: A function to be called after everystep of every world run (only allowed for serial
                                 evaluation and should be used with cautious).
        tournament_progress_callback: A function to be called with `WorldRunResults` after each world finished
                                      processing
        non_competitors: A list of agent types that will not be competing in the sabotage competition but will exist
                         in the world
        non_competitor_params: parameters of non competitor agents
        verbose: Verbosity
        configs_only: If true, a config file for each
        compact: If true, compact logs will be created and effort will be made to reduce the memory footprint
        kwargs: Arguments to pass to the `world_generator` function

    Returns:

        `TournamentResults` The results of the tournament or a `PathLike` giving the location where configs were saved

    Remarks:

        Default parameters will be used in the league with the exception of `parallelism` which may use distributed
        processing

    """
    return tournament(
        competitors=competitors,
        non_competitors=non_competitors,
        non_competitor_params=non_competitor_params,
        agent_names_reveal_type=agent_names_reveal_type,
        n_configs=n_configs,
        n_runs_per_world=n_runs_per_world,
        max_worlds_per_config=max_worlds_per_config,
        tournament_path=tournament_path,
        total_timeout=total_timeout,
        n_agents_per_competitor=n_agents_per_competitor,
        parallelism=parallelism,
        scheduler_ip=scheduler_ip,
        scheduler_port=scheduler_port,
        tournament_progress_callback=tournament_progress_callback,
        world_progress_callback=world_progress_callback,
        name=name,
        verbose=verbose,
        configs_only=configs_only,
        world_generator=anac2019_world_generator,
        config_generator=anac2019_config_generator,
        config_assigner=anac2019_assigner,
        score_calculator=balance_calculator,
        min_factories_per_level=min_factories_per_level,
        compact=compact,
        **kwargs,
    )


def anac2019_sabotage(
    competitors: Sequence[Union[str, Type[FactoryManager]]],
    agent_names_reveal_type=False,
    n_configs: int = 5,
    max_worlds_per_config: Optional[int] = 1000,
    n_runs_per_world: int = 5,
    n_agents_per_competitor: int = 5,
    min_factories_per_level: int = 5,
    tournament_path: str = "./logs/tournaments",
    total_timeout: Optional[int] = 7200,
    parallelism="parallel",
    scheduler_ip: Optional[str] = None,
    scheduler_port: Optional[str] = None,
    tournament_progress_callback: Callable[[Optional[WorldRunResults]], None] = None,
    world_progress_callback: Callable[[Optional[SCMLWorld]], None] = None,
    non_competitors: Optional[Sequence[Union[str, Type[FactoryManager]]]] = None,
    non_competitor_params: Optional[Sequence[Union[str, Type[FactoryManager]]]] = None,
    name: str = None,
    verbose: bool = False,
    configs_only=False,
    compact=False,
    **kwargs,
) -> Union[TournamentResults, PathLike]:
    """
    The function used to run ANAC 2019 SCML tournament (collusion track).

    Args:

        name: Tournament name
        competitors: A list of class names for the competitors
        agent_names_reveal_type: If true then the type of an agent should be readable in its name (most likely at its
                                 beginning).
        n_configs: The number of different world configs (up to competitor assignment) to be generated.
        max_worlds_per_config: The maximum number of worlds to run per config. If None, then all possible assignments
                             of competitors within each config will be tried (all permutations).
        n_runs_per_world: Number of runs per world. All of these world runs will have identical competitor assignment
                          and identical world configuration.
        n_agents_per_competitor: Number of agents per competitor
        min_factories_per_level: Minimum number of factories for each production level
        total_timeout: Total timeout for the complete process
        tournament_path: Path at which to store all results. A scores.csv file will keep the scores and logs folder will
                         keep detailed logs
        parallelism: Type of parallelism. Can be 'serial' for serial, 'parallel' for parallel and 'distributed' for
                     distributed
        scheduler_port: Port of the dask scheduler if parallelism is dask, dist, or distributed
        scheduler_ip: IP Address of the dask scheduler if parallelism is dask, dist, or distributed
        world_progress_callback: A function to be called after every step of every world run (only allowed for serial
                                 evaluation and should be used with cautious).
        tournament_progress_callback: A function to be called with `WorldRunResults` after each world finished
                                      processing
        non_competitors: A list of agent types that will not be competing in the sabotage competition but will exist
                         in the world
        non_competitor_params: parameters of non competitor agents
        verbose: Verbosity
        configs_only: If true, a config file for each
        compact: If true, compact logs will be created and effort will be made to reduce the memory footprint
        kwargs: Arguments to pass to the `world_generator` function

    Returns:

        `TournamentResults` The results of the tournament or a `PathLike` giving the location where configs were saved

    Remarks:

        Default parameters will be used in the league with the exception of `parallelism` which may use distributed
        processing

    """
    return tournament(
        competitors=competitors,
        agent_names_reveal_type=agent_names_reveal_type,
        non_competitors=non_competitors,
        non_competitor_params=non_competitor_params,
        n_configs=n_configs,
        n_runs_per_world=n_runs_per_world,
        max_worlds_per_config=max_worlds_per_config,
        tournament_path=tournament_path,
        total_timeout=total_timeout,
        n_agents_per_competitor=n_agents_per_competitor,
        parallelism=parallelism,
        scheduler_ip=scheduler_ip,
        scheduler_port=scheduler_port,
        tournament_progress_callback=tournament_progress_callback,
        world_progress_callback=world_progress_callback,
        name=name,
        verbose=verbose,
        configs_only=configs_only,
        world_generator=anac2019_world_generator,
        config_generator=anac2019_sabotage_config_generator,
        config_assigner=anac2019_sabotage_assigner,
        score_calculator=sabotage_effectiveness,
        compact=compact,
        min_factories_per_level=min_factories_per_level,
        **kwargs,
    )
