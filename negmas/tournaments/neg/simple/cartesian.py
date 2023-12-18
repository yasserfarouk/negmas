"""
Negotiation tournaments module.
"""


from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from math import exp, log
from os import cpu_count
from pathlib import Path
from random import randint, random, shuffle
from time import perf_counter
from typing import Any, Sequence

import matplotlib.pyplot as plt
import pandas as pd
from attr import asdict, define
from rich import print
from rich.progress import track

from negmas.helpers import unique_name
from negmas.helpers.inout import dump
from negmas.helpers.strings import shortest_unique_names
from negmas.helpers.types import get_class, get_full_type_name
from negmas.inout import Scenario
from negmas.mechanisms import Mechanism
from negmas.negotiators import Negotiator
from negmas.preferences.ops import (
    ScenarioStats,
    calc_outcome_distances,
    calc_outcome_optimality,
    calc_scenario_stats,
    estimate_max_dist,
)
from negmas.sao.mechanism import SAOMechanism
from negmas.serialization import serialize, to_flat_dict

__all__ = ["run_negotiation", "cartesian_tournament", "SimpleTournamentResults"]


@define
class SimpleTournamentResults:
    scores: pd.DataFrame
    """All scores per negotiator"""
    details: pd.DataFrame
    """All negotiation results"""
    scores_summary: pd.DataFrame
    """All score statistics summarized per type"""
    final_scores: pd.DataFrame
    """A list of negotiators and their final scores sorted from highest (winner) to lowest score"""
    path: Path | None = None
    """Location at which the logs are stored"""


LOG_UNIFORM_LIMIT = 10


def oneinint(x: int | tuple[int, int] | None, log_uniform=None) -> int | None:
    """Returns x or a random sample within its values.

    Args:
        x: The value or 2-valued tuple to sample from
        log_uniform: If true samples using a log-uniform distribution instead of a uniform distribution.
                     If `None`, uses a log-uniform distribution if min > 0 and max/min >= 10

    """
    if isinstance(x, tuple):
        if log_uniform is None:
            log_uniform = x[0] > 0 and x[1] / x[0] >= LOG_UNIFORM_LIMIT
        if x[0] == x[-1]:
            return x[0]
        if log_uniform:
            l = [log(_) for _ in x]
            return min(x[1], max(x[0], int(exp(random() * (l[1] - l[0]) + l[0]))))
        return randint(*x)
    return x


def oneinfloat(x: float | tuple[float, float] | None) -> float | None:
    """Returns x or a random sample within its values"""
    if isinstance(x, tuple):
        if x[0] == x[-1]:
            return x[0]
        return x[0] + random() * (x[1] - x[0])
    return x


def run_negotiation(
    s: Scenario,
    partners: tuple[type[Negotiator]],
    partner_names: tuple[str] | None = None,
    partner_params: tuple[dict[str, Any]] | None = None,
    rep: int = 0,
    path: Path | None = None,
    mechanism_type: type[Mechanism] = SAOMechanism,
    mechanism_params: dict[str, Any] | None = None,
    full_names: bool = True,
    verbosity: int = 0,
    plot=False,
    plot_params: dict[str, Any] | None = None,
    run_id: int | str | None = None,
    stats: ScenarioStats | None = None,
    annotation: dict[str, Any] | None = None,
    private_infos: tuple[dict[str, Any]] | None = None,
    id_reveals_type: bool = False,
    name_reveals_type: bool = True,
    mask_scenario_name: bool = True,
) -> dict[str, Any]:
    """
    Run a single negotiation with fully specified parameters

    Args:
        s: The `Scenario` representing the negotiation (outcome space and preferences).
        partners: The partners running the negotiation in order of addition to the mechanism.
        real_scenario_name: The real name of the scenario (used when saving logs).
        partner_names: Names of partners. Either `None` for defaults or a tuple of the same length as `partners`
        partner_params: Parameters used to create the partners. Either `None` for defaults or a tuple of the same length as `partners`
        rep: The repetition number for this run of the negotiation
        path: A folder to save the logs into. If not given, no logs will be saved.
        mechanism_type: the type of the `Mechanism` to use for this negotiation
        mechanism_params: The parameters used to create the `Mechanism` or `None` for defaults
        full_names: Use full names for partner names (only used if `partner_names` is None)
        verbosity: Verbosity level as an integer
        plot: If true, save a plot of the negotiation (only if `path` is given)
        plot_params: Parameters to pass to the plotting function
        run_id: A unique ID for this run. If not given one is generated based on date and time
        stats: statistics of the scenario. If not given or `path` is `None`, statistics are not saved
        annotation: Common information saved in the mechanism's annotation (accessible by negotiators using `self.nmi.annotation`). `None` for nothing
        private_infos: Private information saved in the negotiator's `private_info` attribute (accessible by negotiators as `self.private_info`). `None` for nothing
        id_reveals_type: Each negotiator ID will reveal its type.
        name_reveals_type: Each negotiator name will reveal its type.


    Returns:
        A dictionary of negotiation results that contains the final state of the negotiation alongside other information
    """
    if path:
        path = Path(path)
        for name in ("negotiations", "plots", "results"):
            (path / name).mkdir(exist_ok=True, parents=True)
    assert s.outcome_space is not None
    real_scenario_name = s.outcome_space.name
    if not run_id:
        run_id = unique_name("run", add_time=False, sep=".")
    run_id = str(run_id)
    effective_scenario_name = real_scenario_name
    if mask_scenario_name:
        effective_scenario_name = run_id
        new_os = type(s.outcome_space)(
            issues=s.outcome_space.issues, name=effective_scenario_name
        )
        for u in s.ufuns:
            s.outcome_space = new_os
        s = Scenario(outcome_space=new_os, ufuns=s.ufuns)
    if mechanism_params is None:
        mechanism_params = dict()
    if annotation is None:
        annotation = dict(rep=rep)
    else:
        annotation["rep"] = rep
    if partner_params is None:
        partner_params = tuple(dict() for _ in partners)  # type: ignore
    if private_infos is None:
        private_infos = tuple(dict() for _ in partners)  # type: ignore
    assert mechanism_params is not None
    assert all(_ is not None for _ in partner_params)  # type: ignore

    def _name(a: Negotiator) -> str:
        name = a.short_type_name if not full_names else a.type_name
        if name is None:
            name = get_full_type_name(type(a))
        return name

    max_utils = [_.max() for _ in s.ufuns]

    mechanism_params["name"] = effective_scenario_name
    mechanism_params["verbosity"] = verbosity - 1
    mechanism_params["annotation"] = annotation

    m = mechanism_type(outcome_space=s.outcome_space, **mechanism_params)
    complete_names, negotiators = [], []
    for type_, p, pinfo in zip(partners, partner_params, private_infos):  # type: ignore
        negotiator = type_(**p, private_info=pinfo)
        name = _name(negotiator)
        complete_names.append(name)
        negotiators.append(negotiator)
        if p:
            name += str(hash(str(p)))
    if not partner_names:
        partner_names = tuple(complete_names)  # type: ignore
    for l, (negotiator, name, u) in enumerate(zip(negotiators, partner_names, s.ufuns)):  # type: ignore
        if id_reveals_type:
            negotiator.id = f"{name}@{l}"
        else:
            negotiator.id = unique_name("n", add_time=False, sep="")
        if name_reveals_type:
            negotiator.name = f"{name}@{l}"
        else:
            negotiator.name = unique_name("n", add_time=False, sep="")
        complete_names.append(name)
        m.add(negotiator, ufun=u)

    strt = perf_counter()
    state = m.run()
    execution_time = perf_counter() - strt
    param_dump = tuple(str(to_flat_dict(_)) if _ else None for _ in partner_params)  # type: ignore
    if all(_ is None for _ in param_dump):
        param_dump = None
    agreement_utils = tuple(u(state.agreement) for u in s.ufuns)
    reservations = tuple(u.reserved_value for u in s.ufuns)
    if verbosity > 0:
        print(
            f" {partner_names} on {real_scenario_name}: {state.agreement} in {state.relative_time:4.2%} of the time with advantages: {tuple(a - b for a, b in zip(agreement_utils, reservations))}, "
        )
    run_record = asdict(state)
    run_record["utilities"] = agreement_utils
    run_record["max_utils"] = max_utils
    run_record["reserved_values"] = reservations
    run_record["partners"] = partner_names
    run_record["params"] = param_dump
    run_record["scenario"] = m.name
    run_record["run_id"] = run_id
    run_record["execution_time"] = execution_time
    run_record["negotiator_names"] = m.negotiator_names
    run_record["negotiator_ids"] = m.negotiator_ids
    run_record["negotiator_types"] = [_.type_name for _ in m.negotiators]
    run_record["negotiator_times"] = [m.negotiator_times[_] for _ in m.negotiator_ids]
    run_record["n_steps"] = m.nmi.n_steps
    run_record["time_limit"] = m.nmi.time_limit
    run_record["pend"] = m.nmi.pend
    run_record["pend_per_second"] = m.nmi.pend_per_second
    run_record["step_time_limit"] = m.nmi.step_time_limit
    run_record["negotiator_time_limit"] = m.nmi.negotiator_time_limit
    run_record["annotation"] = m.nmi.annotation
    run_record["scenario"] = real_scenario_name
    run_record["mechanism_name"] = m.name
    run_record["mechanism_type"] = m.type_name
    run_record["effective_scenario_name"] = s.outcome_space.name

    if m.nmi.annotation:
        run_record.update(m.nmi.annotation)
    if stats is not None:
        dists = calc_outcome_distances(agreement_utils, stats)
        run_record.update(
            to_flat_dict(
                calc_outcome_optimality(dists, stats, estimate_max_dist(s.ufuns))
            )
        )

    file_name = f"{real_scenario_name}_{'_'.join(partner_names)}_{rep}_{run_id}"  # type: ignore
    if path:

        def save_as_df(data: list[tuple], names, file_name):
            pd.DataFrame(data=data, columns=names).to_csv(file_name, index=False)

        full_name = path / "negotiations" / f"{file_name}.csv"
        assert not full_name.exists(), f"{full_name} already found"
        if issubclass(mechanism_type, SAOMechanism):
            save_as_df(m.extended_trace, ("step", "negotiator", "offer"), full_name)  # type: ignore
            for i, negotiator in enumerate(m.negotiators):
                neg_name = (
                    path
                    / "negotiator_behavior"
                    / file_name
                    / f"{negotiator.name}_at{i}.csv"
                )
                assert not neg_name.exists(), f"{neg_name} already found"
                neg_name.parent.mkdir(parents=True, exist_ok=True)
                save_as_df(m.negotiator_full_trace(negotiator.id), ("time", "relative_time", "step", "offer", "response"), neg_name)  # type: ignore
        else:
            pd.DataFrame.from_records(serialize(m.history)).to_csv(
                full_name, index=False
            )
        full_name = path / "results" / f"{file_name}.json"
        assert not full_name.exists(), f"{full_name} already found"
        dump(run_record, full_name)
        if plot:
            if plot_params is None:
                plot_params = dict()
            plot_params["save_fig"] = (True,)
            full_name = path / "plots" / f"{file_name}.png"
            m.plot(path=path, fig_name=full_name, **plot_params)
            try:
                plt.close(plt.gcf())
            except:
                pass
    return run_record


def cartesian_tournament(
    competitors: list[type[Negotiator] | str] | tuple[type[Negotiator] | str, ...],
    scenarios: list[Scenario] | tuple[Scenario, ...],
    competitor_params: Sequence[dict | None] | None = None,
    rotate_ufuns: bool = True,
    n_repetitions: int = 1,
    path: Path | None = None,
    njobs: int = 0,
    mechanism_type: type[Mechanism] = SAOMechanism,
    mechanism_params: dict[str, Any] | None = None,
    n_steps: int | tuple[int, int] | None = 100,
    time_limit: float | tuple[float, float] | None = None,
    pend: float | tuple[float, float] = 0.0,
    pend_per_second: float | tuple[float, float] = 0.0,
    step_time_limit: float | tuple[float, float] | None = None,
    negotiator_time_limit: float | tuple[float, float] | None = None,
    # full_names: bool = True,
    plot_fraction: float = 0.0,
    plot_params: dict[str, Any] | None = None,
    verbosity: int = 1,
    self_play: bool = True,
    randomize_runs: bool = True,
    save_every: int = 0,
    save_stats: bool = True,
    final_score: tuple[str, str] = ("advantage", "mean"),
    id_reveals_type: bool = False,
    name_reveals_type: bool = True,
    shorten_names: bool = True,
    raise_exceptions: bool = True,
    mask_scenario_names: bool = True,
) -> SimpleTournamentResults:
    """A simplified version of Cartesian tournaments not using the internal machinay of NegMAS  tournaments

    Args:
        competitors: A tuple of the competing negotiator types.
        scenarios: A tuple of base scenarios to use for the tournament.
        competitor_params: Either None for no-parameters or a tuple of dictionaries with parameters to initialize the competitors (in order).
        rotate_ufuns: If `True`, the ufuns will be rotated over negotiator positions (for bilateral negotiation this leads to two scenarios for each input scenario with reversed ufun order).
        n_repetitions: Number of times to repeat each scenario/partner combination
        path: Path on disk to save the results and details of this tournament. Pass None to disable logging
        n_jobs: Number of parallel jobs to run. -1 means running serially (useful for debugging) and 0 means using all cores.
        mechanism_type: The mechanism (protocol) used for all negotiations.
        n_steps: Number of steps/rounds allowed for the each negotiation (None for no-limit and a 2-valued tuple for sampling from a range)
        time_limit: Number of seconds allowed for the each negotiation (None for no-limit and a 2-valued tuple for sampling from a range)
        pend: Probability of ending the negotiation every step/round (None for no-limit and a 2-valued tuple for sampling from a range)
        pend_per_second: Probability of ending the negotiation every second (None for no-limit and a 2-valued tuple for sampling from a range)
        step_time_limit: Time limit for every negotiation step (None for no-limit and a 2-valued tuple for sampling from a range)
        negotiator_time_limit: Time limit for all actions of every negotiator (None for no-limit and a 2-valued tuple for sampling from a range)
        mechanism_params: Parameters of the mechanism (protocol). Usually you need to pass one or more of the following:
                          time_limit (in seconds), n_steps (in rounds), p_ending (probability of ending the negotiation every step).
        plot_fraction: fraction of negotiations for which plots are to be saved (only if `path` is not `None`)
        plot_params: Parameters to pass to the plotting function
        verbosity: Verbosity level (minimum is 0)
        self_play: Allow negotiations in which all partners are of the same type
        randomize_runs: If `True` negotiations will be run in random order, otherwise each scenario/partner combination will be finished before starting on the next
        save_every: Number of negotiations after which we dump details and scores
        save_stats: Whether to calculate and save extra statistics like pareto_optimality, nash_optimality, kalai_optimality, etc
        final_score: A tuple of two strings giving the metric used for ordering the negotiators for the final score:
                     First string can be one of the following (advantage, utility,
                     partner_welfare, welfare) or any statistic from the set calculated if `save_stats` is `True`.
                     The second string can be mean, median, min, max, or std. The default is ('advantage', 'mean')
        id_reveals_type: Each negotiator ID will reveal its type.
        name_reveals_type: Each negotiator name will reveal its type.
        shorten_names: If True, shorter versions of names will be used for results
        raise_exceptions: When given, negotiators and mechanisms are allowed to raise exceptions stopping the tournament
        mask_scenario_names: If given, scenario names will be masked so that the negotiators do not know the original scenario name

    Returns:
        A pandas DataFrame with all negotiation results.
    """
    if mechanism_params is None:
        mechanism_params = dict()
    mechanism_params["ignore_negotiator_exceptions"] = not raise_exceptions

    competitors = [get_class(_) for _ in competitors]
    if competitor_params is None:
        competitor_params = [dict() for _ in competitors]

    def make_scores(record: dict[str, Any]) -> list[dict[str, float]]:
        utils, partners = record["utilities"], record["partners"]
        reserved_values = record["reserved_values"]
        max_utils, times = record["max_utils"], record.get(
            "negotiator_times", [None] * len(utils)
        )
        scores = []
        for i, (u, r, a, m, t) in enumerate(
            zip(utils, reserved_values, partners, max_utils, times)
        ):
            n_p = len(partners)
            bilateral = n_p == 2
            basic = dict(
                strategy=a,
                utility=u,
                reserved_value=r,
                advantage=(u - r) / (m - r),
                partner_welfare=sum(_ for j, _ in enumerate(utils) if j != i)
                / (n_p - 1),
                welfare=sum(_ for _ in utils) / n_p,
                scenario=record["scenario"],
                partners=(partners[_] for _ in range(len(partners)) if _ != i)
                if not bilateral
                else partners[1 - i],
                time=t,
            )
            for c in (
                "nash_optimality",
                "kalai_optimality",
                "max_welfare_optimality",
                "pareto_optimality",
            ):
                if c in record:
                    basic[c] = record[c]
            scores.append(basic)
        return scores

    runs = []
    scenarios_path = path if path is None else Path(path) / "scenarios"
    if scenarios_path is not None:
        scenarios_path.mkdir(exist_ok=True, parents=True)
    stats = None

    def shorten(name):
        #        for s in ("Negotiator", "Agent"):
        #            x = name.replace(s, "")
        #            if not x:
        #                return name
        #            name = x
        return name

    if shorten_names:
        competitor_names = [
            shorten(_)
            for _ in shortest_unique_names([get_full_type_name(_) for _ in competitors])
        ]
    else:
        competitor_names = [get_full_type_name(_) for _ in competitors]
    competitor_info = list(
        zip(competitors, competitor_params, competitor_names, strict=True)
    )
    for s in scenarios:
        n = len(s.ufuns)
        partners_list = list(product(*tuple([competitor_info] * n)))
        if not self_play:
            partners_list = [_ for _ in partners_list if len(set(serialize(_))) > 1]

        ufun_sets = [list(s.ufuns)]
        for i, u in enumerate(s.ufuns):
            u.name = f"{i}_{u.name}"
        if rotate_ufuns:
            for _ in range(len(ufun_sets)):
                ufuns = ufun_sets[-1]
                ufun_sets.append([ufuns[-1]] + ufuns[:-1])
        original_name = s.outcome_space.name
        # original_ufun_names = [_.name for _ in s.ufuns]
        for i, ufuns in enumerate(ufun_sets):
            if len(ufun_sets) > 1:
                for j, u in enumerate(ufuns):
                    n = "_".join(u.name.split("_")[1:])
                    u.name = f"{j}_{n}"
                scenario = Scenario(
                    type(s.outcome_space)(
                        issues=s.outcome_space.issues,
                        name=f"{original_name}_{i}" if i else original_name,
                    ),
                    tuple(ufuns),
                )
            else:
                scenario = s
            this_path = None
            if scenarios_path:
                this_path = scenarios_path / str(scenario.outcome_space.name)
                scenario.to_yaml(this_path)
            if save_stats:
                stats = calc_scenario_stats(scenario.ufuns)
                if this_path:
                    dump(serialize(stats), this_path / "stats.json")

            mechanism_params.update(
                dict(
                    n_steps=oneinint(n_steps),
                    time_limit=oneinfloat(time_limit),
                    pend=oneinfloat(pend),
                    pend_per_second=oneinfloat(pend_per_second),
                    negotiator_time_limit=oneinfloat(negotiator_time_limit),
                    step_time_limit=oneinfloat(step_time_limit),
                )
            )
            for partners in partners_list:
                runs += [
                    dict(
                        s=scenario,
                        partners=[_[0] for _ in partners],
                        partner_names=[_[2] for _ in partners],
                        partner_params=[_[1] for _ in partners],
                        rep=i,
                        annotation=dict(rep=i, n_repetitions=n_repetitions),
                        path=path if path else None,
                        mechanism_type=mechanism_type,
                        mechanism_params=mechanism_params,
                        full_names=True,
                        verbosity=verbosity - 1,
                        plot=random() < plot_fraction,
                        stats=stats,
                        id_reveals_type=id_reveals_type,
                        name_reveals_type=name_reveals_type,
                        plot_params=plot_params,
                        mask_scenario_name=mask_scenario_names,
                    )
                    for i in range(n_repetitions)
                ]
    if randomize_runs:
        shuffle(runs)
    if verbosity > 0:
        print(
            f"Will run {len(runs)} negotiations on {len(scenarios)} scenarios between {len(competitors)} competitors",
            flush=True,
        )
    results, scores = [], []
    results_path = path if not path else path / "details.csv"
    scores_path = path if not path else path / "all_scores.csv"

    def process_record(record, results=results, scores=scores):
        results.append(record)
        scores += make_scores(record)
        if results_path and save_every and i % save_every == 0:
            pd.DataFrame.from_records(results).to_csv(results_path, index_label="index")
            pd.DataFrame.from_records(scores).to_csv(scores_path, index_label="index")
        return results, scores

    def get_run_id(info):
        return hash(str(serialize(info)))

    if njobs < 0:
        for i, info in enumerate(
            track(runs, total=len(runs), description="Negotiations")
        ):
            process_record(run_negotiation(**info, run_id=get_run_id(info)))

    else:
        futures = []
        n_cores = cpu_count()
        if n_cores is None:
            n_cores = 4
        cpus = min(n_cores, njobs) if njobs else cpu_count()
        with ProcessPoolExecutor(max_workers=cpus) as pool:
            for info in runs:
                futures.append(
                    pool.submit(run_negotiation, **info, run_id=get_run_id(info))
                )
            for i, f in enumerate(
                track(
                    as_completed(futures),
                    total=len(futures),
                    description="Negotiations",
                )
            ):
                process_record(f.result())

    scores_df = pd.DataFrame.from_records(scores)
    if scores_path is not None:
        scores_df.to_csv(scores_path, index_label="index")
    type_scores = (
        scores_df[[_ for _ in scores_df.columns if _ not in ("scenario", "partners")]]
        .groupby("strategy")
        .describe()
        # .agg(["min", "mean", "std", "median", "max"])
        .sort_values(final_score, ascending=False)
    )
    final = pd.DataFrame()
    if type_scores is not None:
        if scores_path:
            type_scores.to_csv(scores_path.parent / "type_scores.csv")
        final = type_scores[final_score]
        final.name = "score"
        final = final.reset_index()
        if scores_path:
            final.to_csv(scores_path.parent / "scores.csv", index_label="index")
        if verbosity > 0:
            print(final)
    details_df = pd.DataFrame.from_records(results)
    if results_path:
        details_df.to_csv(results_path, index_label="index")
    return SimpleTournamentResults(
        scores=scores_df,
        details=details_df,
        scores_summary=type_scores,
        final_scores=final,
        path=path,
    )


if __name__ == "__main__":
    from random import randint, random

    from negmas.helpers.misc import intin
    from negmas.outcomes import make_issue, make_os
    from negmas.preferences.crisp.linear import LinearAdditiveUtilityFunction as U
    from negmas.preferences.generators import generate_utility_values
    from negmas.preferences.value_fun import TableFun
    from negmas.sao.negotiators import (
        AspirationNegotiator,
        MiCRONegotiator,
        NaiveTitForTatNegotiator,
    )

    n_scenarios, n_outcomes = 5, (10, 100)
    ufun_sets = []

    for i in range(n_scenarios):
        r = random()
        n = intin(n_outcomes, log_uniform=True)
        name = "S"
        if r < 0.3:
            n_pareto = n
            name = "DivideThePieGen"
        else:
            n_pareto = randint(min(5, n // 2), n // 2)
        if r < 0.05:
            vals = generate_utility_values(
                n_pareto,
                n,
                n_ufuns=2,
                pareto_first=False,
                pareto_generator="zero_sum",
            )
            name = "DivideThePie"
        else:
            vals = generate_utility_values(
                n_pareto,
                n,
                n_ufuns=2,
                pareto_first=False,
                pareto_generator="curve" if random() < 0.5 else "piecewise_linear",
            )

        issues = (make_issue([f"{i}_{n-1 - i}" for i in range(n)], "portions"),)
        ufuns = tuple(
            U(
                values=(
                    TableFun(
                        {_: float(vals[i][k]) for i, _ in enumerate(issues[0].all)}
                    ),
                ),
                name=f"{uname}{i}",
                reserved_value=0.0,
                outcome_space=make_os(issues, name=f"{name}{i}"),
            )
            for k, uname in enumerate(("First", "Second"))
        )
        ufun_sets.append(ufuns)

    scenarios = [
        Scenario(
            outcome_space=ufuns[0].outcome_space,  # type: ignore We are sure this is not None
            ufuns=ufuns,
        )
        for ufuns in ufun_sets
    ]

    cartesian_tournament(
        competitors=(AspirationNegotiator, NaiveTitForTatNegotiator, MiCRONegotiator),
        scenarios=scenarios,
        n_repetitions=1,
    )
