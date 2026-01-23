"""
Defines import/export functionality
"""

from __future__ import annotations
from os import PathLike, listdir
from pathlib import Path
from random import shuffle
from typing import Any, Iterable, Sequence
import xml.etree.ElementTree as ET

from negmas.plots.util import DEFAULT_IMAGE_FORMAT
from attrs import define, field, evolve

from negmas.helpers.inout import dump, load
from negmas.helpers.strings import unique_name
from negmas.helpers.types import get_full_type_name
from negmas.outcomes.outcome_space import make_os
from negmas.preferences.crisp.linear import LinearAdditiveUtilityFunction
from negmas.preferences.ops import ScenarioStats, calc_scenario_stats
from negmas.sao.mechanism import SAOMechanism
from negmas.serialization import PYTHON_CLASS_IDENTIFIER, deserialize, serialize

from .mechanisms import Mechanism
from .negotiators import Negotiator
from .outcomes import (
    CartesianOutcomeSpace,
    Issue,
    OutcomeSpace,
    issues_from_genius,
    issues_from_geniusweb,
)
from .preferences import (
    BaseUtilityFunction,
    DiscountedUtilityFunction,
    UtilityFunction,
    conflict_level,
    make_discounted_ufun,
    nash_points,
    opposition_level,
    pareto_frontier,
    winwin_level,
)
from .preferences.value_fun import TableFun

__all__ = [
    "Scenario",
    "scenario_size",
    "load_genius_domain",
    "load_genius_domain_from_folder",
    "find_genius_domain_and_utility_files",
    "load_geniusweb_domain",
    "load_geniusweb_domain_from_folder",
    "find_geniusweb_domain_and_utility_files",
    "get_domain_issues",
]

STATS_MAX_CARDINALITY = 10_000_000_000
GENIUSWEB_UFUN_TYPES = ("LinearAdditiveUtilitySpace",)
INFO_FILE_NAME = "_info"
STATS_FILE_NAME = "_stats.yaml"


def scenario_size(self: Scenario):
    """Computes the scenario complexity as outcome space size times time steps.

    Args:
        self: The scenario to measure

    Returns:
        The estimated scenario size/complexity
    """
    size = self.outcome_space.cardinality
    if math.isinf(size):
        size = self.outcome_space.cardinality_if_discretized(10)
    for key in ("n_steps", "time_limit", "hiddent_time_limit"):
        n = self.mechanism_params.get(key, float("inf"))
        if n is not None and not math.isinf(n):
            size = size * n
    return size


@define
class Scenario:
    """
    A class representing a negotiation domain
    """

    outcome_space: CartesianOutcomeSpace
    ufuns: tuple[UtilityFunction, ...]
    mechanism_type: type[Mechanism] | None = SAOMechanism
    mechanism_params: dict = field(factory=dict)
    info: dict[str, Any] = field(factory=dict)
    stats: ScenarioStats | None = None
    name: str | None = field(default=None)
    source: Path | tuple[Path, ...] | None = field(default=None)

    def __attrs_post_init__(self):
        """Infer name from outcome_space if not explicitly provided."""
        if self.name is None:
            object.__setattr__(
                self,
                "name",
                self.outcome_space.name if self.outcome_space.name else None,
            )

    def __lt__(self, other: Scenario):
        """Compares scenarios by their size/complexity for sorting."""
        return scenario_size(self) < scenario_size(other)

    @property
    def issues(self) -> tuple[Issue, ...]:
        """The negotiation issues defining the outcome space."""
        return self.outcome_space.issues

    @property
    def is_linear(self) -> bool:
        """Checks if all utility functions in the scenario are linear.

        Returns:
            True if all utility functions are linear (including LinearAdditiveUtilityFunction,
            AffineUtilityFunction, LinearUtilityFunction, LinearUtilityAggregationFunction,
            or ConstUtilityFunction). Handles DiscountedUtilityFunction by checking the
            base utility function.

        Examples:
            >>> from negmas import make_issue, Scenario
            >>> from negmas.preferences import LinearUtilityFunction
            >>> issues = [make_issue([0, 1, 2], "x")]
            >>> u1 = LinearUtilityFunction(weights=[1.0], issues=issues)
            >>> u2 = LinearUtilityFunction(weights=[0.5], issues=issues)
            >>> scenario = Scenario(outcome_space=make_os(issues), ufuns=(u1, u2))
            >>> scenario.is_linear
            True
        """
        from negmas.preferences.crisp.const import ConstUtilityFunction
        from negmas.preferences.crisp.linear import (
            AffineUtilityFunction,
            LinearAdditiveUtilityFunction,
            LinearUtilityAggregationFunction,
            LinearUtilityFunction,
        )
        from negmas.preferences.discounted import DiscountedUtilityFunction

        def is_linear_ufun(ufun: BaseUtilityFunction) -> bool:
            """Recursively check if a ufun (possibly discounted) is linear."""
            if isinstance(ufun, DiscountedUtilityFunction):
                return is_linear_ufun(ufun.ufun)
            return isinstance(
                ufun,
                (
                    LinearAdditiveUtilityFunction,
                    AffineUtilityFunction,
                    LinearUtilityAggregationFunction,
                    LinearUtilityFunction,
                    ConstUtilityFunction,
                ),
            )

        return all(is_linear_ufun(u) for u in self.ufuns)

    def plot(
        self,
        ufun_indices: tuple[int, int] | None = None,
        backend: str = "matplotlib",
        **kwargs,
    ):
        """Visualizes the scenario's utility space using a 2D plot.

        Args:
            ufun_indices: Tuple of (i, j) specifying which pair of utility functions to plot.
                If None, plots the first two ufuns (indices 0 and 1).
                For scenarios with exactly 2 ufuns, this parameter is ignored.
            backend: Plotting backend to use. Either "matplotlib" or "plotly". Default is "matplotlib".
            **kwargs: Additional arguments passed to plot_2dutils.

        Returns:
            A matplotlib Figure object if backend="matplotlib", or a plotly Figure object if backend="plotly".

        Raises:
            ValueError: If the scenario has fewer than 2 utility functions, or if backend is invalid.
            IndexError: If ufun_indices contains invalid indices.
        """
        from negmas.plots.util import plot_2dutils

        if backend not in ("matplotlib", "plotly"):
            raise ValueError(
                f"Invalid backend '{backend}'. Must be 'matplotlib' or 'plotly'."
            )

        if len(self.ufuns) < 2:
            raise ValueError(
                f"Cannot plot scenario with {len(self.ufuns)} utility function(s). Need at least 2."
            )

        # Determine which pair to plot
        if ufun_indices is None:
            i, j = 0, 1
        else:
            i, j = ufun_indices
            if i < 0 or i >= len(self.ufuns) or j < 0 or j >= len(self.ufuns):
                raise IndexError(
                    f"Invalid ufun_indices ({i}, {j}). Must be in range [0, {len(self.ufuns) - 1}]."
                )
            if i == j:
                raise ValueError(
                    f"Cannot plot the same utility function twice: ufun_indices=({i}, {j})."
                )

        plotting_ufuns = (self.ufuns[i], self.ufuns[j])
        plotting_names = (self.ufuns[i].name, self.ufuns[j].name)

        return plot_2dutils(
            [],
            plotting_ufuns,
            plotting_names,
            offering_negotiators=plotting_names,
            issues=self.outcome_space.issues,
            backend=backend,
            **kwargs,
        )

    def save_plots(
        self,
        folder: Path | str,
        ext: str = DEFAULT_IMAGE_FORMAT,
        backend: str = "matplotlib",
        **plot_kwargs,
    ) -> list[Path]:
        """Saves utility space plots for all pairs of utility functions.

        Args:
            folder: Destination folder where plots will be saved.
            ext: Image file extension (e.g., 'png', 'jpg', 'svg', 'pdf', 'webp').
                Defaults to DEFAULT_IMAGE_FORMAT from negmas.plots.util (currently 'webp').
            backend: Plotting backend to use. Either "matplotlib" or "plotly". Default is "matplotlib".
            **plot_kwargs: Additional arguments passed to plot() method.

        Returns:
            List of Path objects for all saved plot files.

        Raises:
            ValueError: If the scenario has fewer than 2 utility functions, or if backend is invalid.

        Remarks:
            - For scenarios with exactly 2 ufuns: saves a single plot as "_plot.{ext}"
            - For scenarios with >2 ufuns: creates a "_plots/" subfolder and saves plots with names:
              "{u0_name}-{u1_name}.{ext}", "{u1_name}-{u2_name}.{ext}", ..., "{u{n-1}_name}-{u0_name}.{ext}"
              This creates a circular sequence of plots for all consecutive pairs plus the wraparound.
            - For matplotlib backend, uses Figure.savefig() to save plots.
            - For plotly backend, uses Figure.write_image() to save plots.
        """

        if backend not in ("matplotlib", "plotly"):
            raise ValueError(
                f"Invalid backend '{backend}'. Must be 'matplotlib' or 'plotly'."
            )

        if len(self.ufuns) < 2:
            raise ValueError(
                f"Cannot save plots for scenario with {len(self.ufuns)} utility function(s). Need at least 2."
            )

        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        # Ensure extension starts with a dot
        if not ext.startswith("."):
            ext = f".{ext}"

        saved_files = []

        if len(self.ufuns) == 2:
            # Single plot: save as _plot.{ext}
            plot_path = folder / f"_plot{ext}"
            fig = self.plot(ufun_indices=(0, 1), backend=backend, **plot_kwargs)
            if backend == "matplotlib":
                fig.savefig(str(plot_path), bbox_inches="tight")
            else:  # plotly
                fig.write_image(str(plot_path))
            saved_files.append(plot_path)
        else:
            # Multiple plots: create _plots/ subfolder
            plots_folder = folder / "_plots"
            plots_folder.mkdir(parents=True, exist_ok=True)

            # Generate circular pairs: (0,1), (1,2), (2,3), ..., (n-1,0)
            n = len(self.ufuns)
            for i in range(n):
                j = (i + 1) % n
                u_i_name = self.ufuns[i].name or f"u{i}"
                u_j_name = self.ufuns[j].name or f"u{j}"

                # Clean names to be filesystem-safe
                u_i_name = u_i_name.replace("/", "-").replace("\\", "-")
                u_j_name = u_j_name.replace("/", "-").replace("\\", "-")

                plot_path = plots_folder / f"{u_i_name}-{u_j_name}{ext}"
                fig = self.plot(ufun_indices=(i, j), backend=backend, **plot_kwargs)
                if backend == "matplotlib":
                    fig.savefig(str(plot_path), bbox_inches="tight")
                else:  # plotly
                    fig.write_image(str(plot_path))
                saved_files.append(plot_path)

        return saved_files

    def to_genius_files(self, domain_path: Path, ufun_paths: list[Path]):
        """
        Save domain and ufun files to the `path` as XML.
        """
        from negmas.preferences.discounted import DiscountedUtilityFunction

        domain_path = Path(domain_path)
        ufun_paths = [Path(_) for _ in ufun_paths]
        if len(self.ufuns) != len(ufun_paths):
            raise ValueError(
                f"I have {len(self.ufuns)} ufuns but {len(ufun_paths)} paths were passed!!"
            )
        domain_path.parent.mkdir(parents=True, exist_ok=True)
        self.outcome_space.to_genius(domain_path)
        for ufun, path in zip(self.ufuns, ufun_paths):
            # Extract discount factor if ufun is discounted
            # Genius requires <discount_factor> tag even when factor=1.0 (no discount)
            discount = 1.0  # Default: no discount
            if isinstance(ufun, DiscountedUtilityFunction):
                discount = ufun.discount_factor

            # Always pass discount_factor to ensure <discount_factor> tag is included
            ufun.to_genius(path, issues=self.issues, discount_factor=discount)
        return self

    def to_genius_folder(self, path: Path):
        """
        Save domain and ufun files to the `path` as XML.
        """
        from negmas.preferences.discounted import DiscountedUtilityFunction

        path.mkdir(parents=True, exist_ok=True)
        domain_name = (
            self.outcome_space.name.replace("\\", "/").split("/")[-1]
            if self.outcome_space.name
            else "domain"
        )
        ufun_names = [_.name.replace("\\", "/").split("/")[-1] for _ in self.ufuns]
        self.outcome_space.to_genius(path / domain_name)
        for ufun, name in zip(self.ufuns, ufun_names):
            # Extract discount factor if ufun is discounted
            # Genius requires <discount_factor> tag even when factor=1.0 (no discount)
            discount = 1.0  # Default: no discount
            if isinstance(ufun, DiscountedUtilityFunction):
                discount = ufun.discount_factor

            # Always pass discount_factor to ensure <discount_factor> tag is included
            ufun.to_genius(path / name, issues=self.issues, discount_factor=discount)
        return self

    @property
    def n_negotiators(self) -> int:
        """The number of negotiators (utility functions) in this scenario."""
        return len(self.ufuns)

    @property
    def n_issues(self) -> int:
        """The number of negotiation issues in the outcome space."""
        return len(self.outcome_space.issues)

    @property
    def issue_names(self) -> list[str]:
        """The names of all negotiation issues in the outcome space."""
        return self.outcome_space.issue_names

    def to_numeric(self) -> Scenario:
        """
        Forces all issues in the domain to become numeric

        Remarks:
            - maps the agenda and ufuns to work correctly together
        """
        raise NotImplementedError()

    def _randomize(self) -> Scenario:
        """
        Randomizes the outcomes in a single issue scneario
        """
        shuffle(self.outcome_space.issues[0].values)
        return self

    def to_single_issue(
        self,
        numeric=False,
        stringify=True,
        randomize=False,
        recalculate_stats: bool = True,
    ) -> Scenario:
        """Converts the scenario to use a single issue containing all possible outcomes.

        Args:
            numeric: If True, the output issue will be a `ContiguousIssue`, otherwise a `DiscreteCategoricalIssue`.
            stringify: If True and `numeric` is False, the output issue will have string values.
            randomize: If True, randomize outcome order when creating the single issue.
            recalculate_stats: If True and stats exist, recalculate them after conversion.
                If False and stats exist, invalidate stats by setting them to None.

        Remarks:
            - Maps the agenda and ufuns to work correctly together.
            - Only works if the outcome space is finite.
        """
        if (
            hasattr(self.outcome_space, "issues")
            and len(self.outcome_space.issues) == 1
        ):
            return self if not randomize else self._randomize()
        outcomes = list(self.outcome_space.enumerate_or_sample())
        sos = self.outcome_space.to_single_issue(numeric, stringify)
        ufuns = []
        souts = list(sos.issues[0].all)
        for u in self.ufuns:
            if isinstance(u, DiscountedUtilityFunction):
                usave = u
                v = u.ufun
                while isinstance(v, DiscountedUtilityFunction):
                    u, v = v, v.ufun
                u.ufun = LinearAdditiveUtilityFunction(
                    values=(
                        TableFun(dict(zip(souts, [v(_) for _ in outcomes]))),
                    ),  #  (The error comes from having LRU cach for table ufun's minmax which should be OK)
                    bias=0.0,
                    reserved_value=v.reserved_value,
                    name=v.name,
                    outcome_space=sos,
                )
                ufuns.append(usave)
                continue
            ufuns.append(
                LinearAdditiveUtilityFunction(
                    values=(
                        TableFun(dict(zip(souts, [u(_) for _ in outcomes]))),
                    ),  #  (The error comes from having LRU cach for table ufun's minmax which should be OK)
                    bias=0.0,
                    reserved_value=u.reserved_value,
                    name=u.name,
                    outcome_space=sos,
                )
            )
        self.ufuns = tuple(ufuns)
        self.outcome_space = sos
        if self.stats:
            if recalculate_stats:
                self.calc_stats()
            else:
                self.stats = None
        return self if not randomize else self._randomize()

    def make_session(
        self,
        negotiators: Callable[[], Negotiator]
        | type[Negotiator]
        | list[Negotiator]
        | tuple[Negotiator, ...]
        | None = None,
        n_steps: int | float | None = None,
        time_limit: float | None = None,
        roles: list[str] | None = None,
        raise_on_failure_to_enter: bool = True,
        share_ufuns: bool = False,
        share_reserved_values: bool = False,
        **kwargs,
    ):
        """
        Generates a ready to run mechanism session for this domain.
        """
        if not self.mechanism_type:
            raise ValueError(
                "Cannot create the domain because it has no `mechanism_type`"
            )

        args = self.mechanism_params
        args.update(kwargs)
        if n_steps:
            args["n_steps"] = n_steps
        if time_limit:
            args["time_limit"] = time_limit
        m = self.mechanism_type(outcome_space=self.outcome_space, **args)
        if not negotiators:
            return m
        negs: list[Negotiator]
        if share_ufuns:
            assert len(self.ufuns) == 2, (
                "Sharing ufuns in multilateral negotiations is not yet supported"
            )
            opp_ufuns = reversed(deepcopy(self.ufuns))
            if not share_reserved_values:
                for u in opp_ufuns:
                    u.reserved_value = float("nan")
        else:
            opp_ufuns = [None] * len(self.ufuns)
        if not isinstance(negotiators, Iterable):
            negs = [
                negotiators(
                    name=ufun.name  # type: ignore We trust that the class given is a negotiator and has a name
                    if ufun.name
                    else unique_name("n").replace(".xml", "").replace(".yml", ""),
                    private_info=dict(opponent_ufun=opp_ufun) if opp_ufun else None,  # type: ignore We trust that the class given is a negotiator and has private_info
                )
                for ufun, opp_ufun in zip(self.ufuns, opp_ufuns)
            ]
        else:
            negs = list(negotiators)
        if share_ufuns:
            for neg, ou in zip(negs, opp_ufuns):
                if share_ufuns and neg.opponent_ufun is None and ou is not None:
                    neg._private_info["opponent_ufun"] = ou
        if len(self.ufuns) != len(negs) or len(negs) < 1:
            raise ValueError(
                f"Invalid ufuns ({self.ufuns}) or negotiators ({negotiators})"
            )

        if not roles:
            roles = ["negotiator"] * len(negs)
        for n, r, u in zip(negs, roles, self.ufuns):
            added = m.add(n, preferences=u, role=r)
            if not added and raise_on_failure_to_enter:
                raise ValueError(
                    f"{n.name} (of type {get_full_type_name(n.__class__)}) failed to enter the negotiation {m.name}"
                )

        return m

    def scale_min(
        self,
        to: float = 0.0,
        outcome_space: OutcomeSpace | None = None,
        recalculate_stats: bool = True,
    ) -> Scenario:
        """Scales all utility functions so their minimum value equals the given target.

        This method scales each utility function independently by finding its minimum
        over the specified outcome space and multiplying by an appropriate scale factor.

        Args:
            to: Target minimum value for all utility functions.
            outcome_space: The outcome space to use when computing min/max values.
                If None, uses each ufun's own outcome space (which should match the
                scenario's outcome space).
            recalculate_stats: If True and stats exist, recalculate them after scaling.
                If False and stats exist, invalidate stats by setting them to None.

        Returns:
            Self for method chaining.

        Note:
            Each utility function is scaled independently. This is different from
            normalize() which can perform common-scale normalization across all ufuns.
        """
        os = outcome_space or self.outcome_space
        self.ufuns = tuple(_.scale_min_for(to, outcome_space=os) for _ in self.ufuns)
        if self.stats:
            if recalculate_stats:
                self.calc_stats()
            else:
                self.stats = None
        return self

    def scale_max(
        self,
        to: float = 1.0,
        outcome_space: OutcomeSpace | None = None,
        recalculate_stats: bool = True,
    ) -> Scenario:
        """Scales all utility functions so their maximum value equals the given target.

        This method scales each utility function independently by finding its maximum
        over the specified outcome space and multiplying by an appropriate scale factor.

        Args:
            to: Target maximum value for all utility functions.
            outcome_space: The outcome space to use when computing min/max values.
                If None, uses each ufun's own outcome space (which should match the
                scenario's outcome space).
            recalculate_stats: If True and stats exist, recalculate them after scaling.
                If False and stats exist, invalidate stats by setting them to None.

        Returns:
            Self for method chaining.

        Note:
            Each utility function is scaled independently. This is different from
            normalize() which can perform common-scale normalization across all ufuns.
        """
        os = outcome_space or self.outcome_space
        self.ufuns = tuple(_.scale_max_for(to, outcome_space=os) for _ in self.ufuns)
        if self.stats:
            if recalculate_stats:
                self.calc_stats()
            else:
                self.stats = None
        return self

    def normalize(
        self,
        to: tuple[float, float] = (0.0, 1.0),
        outcome_space: OutcomeSpace | None = None,
        guarantee_max: bool = True,
        guarantee_min: bool | None = None,
        independent: bool | None = None,
        common_range: bool | None = None,
        recalculate_stats: bool = True,
    ) -> Scenario:
        """Normalizes all utility functions to the given range.

        Args:
            to: Target range (min, max) to normalize all utility functions to.
            outcome_space: The outcome space to use for normalization. If None, uses the scenario's outcome space.
            guarantee_max: If True, guarantees that the maximum value is exactly to[1].
            guarantee_min: If True, guarantees that the minimum value is exactly to[0].
                When None (default), automatically set to True for common_range=True, False for common_range=False.
            common_range: If True (default), normalizes all utility functions to a common scale.
                If False, normalizes each utility function independently to span the full range.
            independent: Deprecated. Use common_range instead. If True, normalizes each utility function independently.
                If False, uses common scale normalization. This parameter will be removed in a future version.
            recalculate_stats: If True and stats exist, recalculate them after normalizing.
                If False and stats exist, invalidate stats by setting them to None.

        Remarks:
            - If either value of `to` is `None`, then all ufuns will just be scaled to match the constraint of the other value.
            - When common_range=True (default), all utility functions are normalized to a common scale,
              ensuring that utility values are comparable across agents.
            - When common_range=False, each utility function is normalized independently to span the full range.
        """
        # Handle parameter conflicts and deprecation
        if independent is not None and common_range is not None:
            raise ValueError(
                "Cannot specify both 'independent' and 'common_range'. "
                "Use 'common_range' only (independent is deprecated)."
            )

        # Convert parameters: common_range takes precedence
        if common_range is not None:
            normalize_independently = not common_range
        elif independent is not None:
            from negmas.warnings import deprecated

            deprecated(
                "The 'independent' parameter is deprecated. Use 'common_range' instead. "
                "independent=True is equivalent to common_range=False, and "
                "independent=False is equivalent to common_range=True."
            )
            normalize_independently = independent
        else:
            # Default: common_range=True (i.e., independent=False)
            normalize_independently = False

        # Set guarantee_min based on normalization mode if not specified
        if guarantee_min is None:
            # For common range, we want to guarantee min for proper scaling
            # For independent, we don't need it (defaults work fine)
            guarantee_min = not normalize_independently

        if normalize_independently:
            # Use scenario's outcome space if not provided
            os_for_norm = (
                outcome_space if outcome_space is not None else self.outcome_space
            )
            normalized_ufuns = []
            for ufun in self.ufuns:
                # Check if the ufun's normalize_for() supports guarantee_max/guarantee_min
                # Only LinearAdditiveUtilityFunction has these parameters
                from inspect import signature

                sig = signature(ufun.normalize_for)
                if "guarantee_max" in sig.parameters:
                    normalized = ufun.normalize_for(
                        to,
                        outcome_space=os_for_norm,
                        guarantee_max=guarantee_max,
                        guarantee_min=guarantee_min,
                    )
                else:
                    # Fall back to basic normalize_for for other ufun types
                    normalized = ufun.normalize_for(to, outcome_space=os_for_norm)
                normalized_ufuns.append(normalized)
            self.ufuns = tuple(normalized_ufuns)  # type: ignore The type is correct
        else:
            # Use scenario's outcome space if not provided
            os_for_norm = (
                outcome_space if outcome_space is not None else self.outcome_space
            )
            self.ufuns = tuple(
                BaseUtilityFunction.normalize_all_for(  # type: ignore
                    ufuns=self.ufuns,
                    to=to,
                    outcome_space=os_for_norm,
                    guarantee_max=guarantee_max,
                    guarantee_min=guarantee_min,
                )
            )
        if self.stats:
            if recalculate_stats:
                self.calc_stats()
            else:
                self.stats = None
        return self

    def is_normalized(
        self,
        to: tuple[float | None, float | None] = (None, 1.0),
        positive: bool = True,
        independent: bool | None = None,
        common_range: bool | None = None,
        eps: float = 1e-6,
    ) -> bool:
        """Checks that all ufuns are normalized in the given range.

        Args:
            to: Target range (min, max) to check. None means no constraint on that bound.
            positive: If True, checks that all minimums are non-negative.
            common_range: If True (default), checks common-scale normalization (all ufuns within range,
                at least one reaches each bound). If False, checks that each ufun individually spans the full range.
            independent: Deprecated. Use common_range instead. If True, checks that each ufun individually spans the full range.
                If False, checks common-scale normalization. This parameter will be removed in a future version.
            eps: Tolerance for floating point comparisons.

        Returns:
            True if the scenario is normalized according to the specified criteria.
        """
        # Handle parameter conflicts and deprecation
        if independent is not None and common_range is not None:
            raise ValueError(
                "Cannot specify both 'independent' and 'common_range'. "
                "Use 'common_range' only (independent is deprecated)."
            )

        # Convert parameters: common_range takes precedence
        if common_range is not None:
            check_independently = not common_range
        elif independent is not None:
            from negmas.warnings import deprecated

            deprecated(
                "The 'independent' parameter is deprecated. Use 'common_range' instead. "
                "independent=True is equivalent to common_range=False, and "
                "independent=False is equivalent to common_range=True."
            )
            check_independently = independent
        else:
            # Default: common_range=True (i.e., independent=False)
            check_independently = False

        mnmx = [_.minmax() for _ in self.ufuns]

        if check_independently:
            # Independent mode: Each ufun must individually span [to[0], to[1]]
            return all(
                (to[0] is None or abs(a - to[0]) < eps)
                and (to[1] is None or abs(b - to[1]) < eps)
                and (not positive or a >= -eps)
                for a, b in mnmx
            )
        else:
            # Common-scale mode: All ufuns within range, at least one reaches each bound
            mins = [a for a, _ in mnmx]
            maxs = [b for _, b in mnmx]

            # Check all ufuns are within the range
            if to[0] is not None:
                if not all(a >= to[0] - eps for a in mins):
                    return False
                # At least one ufun should reach the minimum
                if not any(abs(a - to[0]) < eps for a in mins):
                    return False

            if to[1] is not None:
                if not all(b <= to[1] + eps for b in maxs):
                    return False
                # At least one ufun should reach the maximum
                if not any(abs(b - to[1]) < eps for b in maxs):
                    return False

            # Check positivity constraint
            if positive:
                if not all(a >= -eps for a in mins):
                    return False

            return True

    def discretize(self, levels: int = 10, recalculate_stats: bool = True):
        """Discretizes all continuous issues in the outcome space.

        Args:
            levels: Number of discrete levels to create for each continuous issue.
            recalculate_stats: If True and stats exist, recalculate them after discretizing.
                If False and stats exist, invalidate stats by setting them to None.
        """
        self.outcome_space = self.outcome_space.to_discrete(levels)
        for f in self.ufuns:
            f.outcome_space = self.outcome_space
        if self.stats:
            if recalculate_stats:
                self.calc_stats()
            else:
                self.stats = None
        return self

    def remove_discounting(self, recalculate_stats: bool = True):
        """Removes time-based discounting from all utility functions.

        Args:
            recalculate_stats: If True and stats exist, recalculate them after removing discounting.
                If False and stats exist, invalidate stats by setting them to None.
        """
        self.ufuns = tuple(  # type: ignore It is the UtilityFunction, BaseUtilityFunction issue. They are equivalent here.
            [
                u.extract_base_ufun(deep=True)
                if isinstance(u, DiscountedUtilityFunction)
                else u
                for u in self.ufuns
            ]
        )
        if self.stats:
            if recalculate_stats:
                self.calc_stats()
            else:
                self.stats = None
        return self

    def remove_reserved_values(
        self, r: float = float("-inf"), recalculate_stats: bool = True
    ):
        """Replaces reserved values in all utility functions with the given value.

        Args:
            r: The value to set as the new reserved value for all utility functions.
            recalculate_stats: If True and stats exist, recalculate them after removing reserved values.
                If False and stats exist, invalidate stats by setting them to None.
        """
        for u in self.ufuns:
            u.reserved_value = r
        if self.stats:
            if recalculate_stats:
                self.calc_stats()
            else:
                self.stats = None
        return self

    def calc_stats(self) -> ScenarioStats:
        """Calculates scenario statistics and stores them in the stats attribute."""
        self.stats = calc_scenario_stats(self.ufuns)
        return self.stats

    def calc_extra_stats(
        self, max_cardinality: int = STATS_MAX_CARDINALITY
    ) -> dict[str, Any]:
        """
        Calculates and returns several stats corresponding to the domain

        Args:
            max_cardinality (int): The maximum number of outcomes considered when calculating the stats.

        Returns:
            A dictionary with the compiled stats
        """
        outcome_space, ufuns = self.outcome_space, self.ufuns
        outcomes = tuple(
            outcome_space.enumerate_or_sample(max_cardinality=max_cardinality)
        )
        minmax = [u.minmax() for u in ufuns]
        if self.stats is None:
            frontier_utils, frontier_indices = pareto_frontier(
                ufuns, outcomes=outcomes, sort_by_welfare=True
            )
            frontier_outcomes = tuple(
                outcomes[_] for _ in frontier_indices if _ is not None
            )
            pts = nash_points(ufuns, frontier_utils, outcome_space=outcome_space)
            nash_utils, nash_indx = pts[0] if pts else (None, None)
            nash_outcome = frontier_outcomes[nash_indx] if nash_indx else None
            opposition = opposition_level(
                ufuns,
                max_utils=tuple(_[1] for _ in minmax),  #
                outcomes=outcomes,
                max_tests=max_cardinality,
            )
        else:
            frontier_utils = self.stats.pareto_utils
            frontier_outcomes = self.stats.pareto_outcomes
            nash_utils = self.stats.nash_outcomes
            nash_outcome = self.stats.nash_outcomes
            opposition = self.stats.opposition
        nu, no, ol, cl, wl, fu, fo = (
            dict(),
            dict(),
            dict(),
            dict(),
            dict(),
            dict(),
            dict(),
        )
        for i, u1 in enumerate(ufuns):
            if not u1:
                continue
            for j, u2 in enumerate(ufuns[i + 1 :]):
                if not u2:
                    continue
                us = (u1, u2)
                fu_, findx = pareto_frontier(
                    us,  # type: ignore
                    outcomes=outcomes,
                    sort_by_welfare=True,
                )
                foutcomes_ = [outcomes[i] for i in findx]
                fu[(u1.name, u2.name)], fo[(u1.name, u2.name)] = (
                    fu_,
                    [outcomes[_] for _ in findx],
                )
                pts = nash_points((u1, u2), fu_, outcomes=outcomes)  # type: ignore
                nu[(u1.name, u2.name)], nindx = pts[0] if pts else (None, None)
                no[(u1.name, u2.name)] = foutcomes_[nindx] if nindx else None
                ol[(u1.name, u2.name)] = opposition_level(
                    (u1, u2),  # type: ignore
                    outcomes=outcomes,
                    max_utils=(minmax[i][1], minmax[j][1]),
                    max_tests=max_cardinality,
                )
                cl[(u1.name, u2.name)] = conflict_level(
                    u1, u2, outcomes=outcomes, max_tests=max_cardinality
                )
                wl[(u1.name, u2.name)] = winwin_level(
                    u1, u2, outcomes=outcomes, max_tests=max_cardinality
                )

        return dict(
            frontier_utils=frontier_utils,
            frontier_outcomes=frontier_outcomes,
            nash_utils=nash_utils,
            nash_outcome=nash_outcome,
            opposition_level=opposition,
            bilateral_nash_utils=nu,
            bilateral_nash_outcome=no,
            bilateral_conflict_level=cl,
            bilateral_opposition_level=ol,
            bilateral_winwin_levl=wl,
            bilateral_frontier_utils=fu,
            bilateral_frontier_outcomes=fo,
        )

    def serialize(self) -> dict[str, Any]:
        """
        Converts the current scenario into a serializable dict.

        Remarks:
            Rturns a dictionary with the following keys:
                - domain: The agenda/outcome-space
                - ufuns: A list of utility functions
        """

        def get_name(x, default):
            """Extracts a clean name from a path-like string."""
            if not x:
                return str(default)
            # Handle both forward and back slashes for cross-platform compatibility
            return x.replace("\\", "/").split("/")[-1].replace(".xml", "")

        def adjust(
            d,
            default_name,
            remove_dunder=False,
            adjust_name=True,
            ignored=("id", "n_values", "outcome_space"),
            rename={PYTHON_CLASS_IDENTIFIER: "type"},
        ):
            """Transforms a serialized dict for cleaner output format."""
            if isinstance(d, list) or isinstance(d, tuple):
                return [
                    adjust(_, default_name, remove_dunder, adjust_name, ignored)
                    for _ in d
                ]
            if not isinstance(d, dict):
                return d
            if adjust_name and "name" in d:
                d["name"] = get_name(d["name"], default_name)
            if d.get(PYTHON_CLASS_IDENTIFIER, "").startswith("negmas."):
                d[PYTHON_CLASS_IDENTIFIER] = d[PYTHON_CLASS_IDENTIFIER].split(".")[-1]
            for old, new in rename.items():
                if old in d.keys():
                    d[new] = d[old]
                    del d[old]
            for i in ignored:
                if i in d.keys():
                    del d[i]
            for k, v in d.items():
                d[k] = adjust(
                    v,
                    default_name,
                    remove_dunder=remove_dunder,
                    adjust_name=False,
                    ignored=ignored,
                )
            if not remove_dunder:
                return d
            d = {k: v for k, v in d.items() if not k.startswith("__")}
            return d

        domain = adjust(
            serialize(
                self.outcome_space, shorten_type_field=False, add_type_field=True
            ),
            "domain",
        )
        ufuns = [
            adjust(serialize(u, shorten_type_field=False, add_type_field=True), i)
            for i, u in enumerate(self.ufuns)
        ]
        return dict(domain=domain, ufuns=ufuns)

    def to_yaml(self, folder: Path | str) -> None:
        """
        Saves the scenario as yaml
        Args:
            folder: The destination path
        """
        self.dumpas(folder, "yml")

    def to_json(self, folder: Path | str) -> None:
        """
        Saves the scenario as json
        Args:
            folder: The destination path
        """
        self.dumpas(folder, "json")

    def save_info(self, folder: Path | str) -> None:
        f"""
        Save info to the given path under {INFO_FILE_NAME}

        Args:
            folder: Destination folder path.
        """
        if not self.info:
            return
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        dump(self.info, folder / f"{INFO_FILE_NAME}.{type}")

    def save_stats(
        self,
        folder: Path | str,
        compact: bool = False,
        include_pareto_frontier: bool = True,
    ) -> None:
        f"""
        Save stats to the given path under {STATS_FILE_NAME}

        Args:
            folder: Destination folder path.
            compact: If True, use compact JSON formatting.
            include_pareto_frontier: If True, include pareto_utils and pareto_outcomes
                in stats.json. If False, exclude them to save disk space. Default is True.
        """
        if not self.stats:
            return
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        dump(
            self.stats.to_dict(include_pareto_frontier=include_pareto_frontier),
            folder / STATS_FILE_NAME,
            compact=compact,
        )

    def dumpas(
        self,
        folder: Path | str,
        type="yml",
        compact: bool = False,
        save_stats=True,
        save_info=True,
        save_plot=False,
        include_pareto_frontier: bool = True,
        plot_extension: str | None = None,
        plot_kwargs: dict | None = None,
    ) -> None:
        """
        Dumps the scenario in the given file format.

        Args:
            folder: Destination folder path.
            type: File format ("yml", "json", or "xml").
            compact: If True, use compact JSON formatting.
            save_stats: If True, save scenario statistics to stats.json.
            save_info: If True, save scenario info to info file.
            save_plot: If True, save utility space plots. Default is False.
            include_pareto_frontier: If True, include pareto_utils and pareto_outcomes
                in stats.json. If False, exclude them to save disk space. Default is True.
            plot_extension: File extension for plots (e.g., 'png', 'jpg', 'svg', 'pdf', 'webp').
                If None, uses DEFAULT_IMAGE_FORMAT from negmas.plots.util (currently 'webp').
            plot_kwargs: Additional keyword arguments to pass to the plot() method. Default is None.
        """
        if type.startswith("."):
            type = type[1:]
        folder = Path(folder)
        if type == "xml":
            self.to_genius_folder(folder)
            if save_plot and len(self.ufuns) >= 2:
                kwargs = plot_kwargs or {}
                if plot_extension is not None:
                    kwargs["ext"] = plot_extension
                self.save_plots(folder, **kwargs)
            return
        folder.mkdir(parents=True, exist_ok=True)
        serialized = self.serialize()
        dump(serialized["domain"], folder / f"{serialized['domain']['name']}.{type}")
        for u in serialized["ufuns"]:
            dump(u, folder / f"{u['name']}.{type}", sort_keys=True, compact=compact)
        if self.info and save_info:
            dump(self.info, folder / f"{INFO_FILE_NAME}.{type}")
        if self.stats and save_stats:
            dump(
                self.stats.to_dict(include_pareto_frontier=include_pareto_frontier),
                folder / STATS_FILE_NAME,
            )
        if save_plot and len(self.ufuns) >= 2:
            kwargs = plot_kwargs or {}
            if plot_extension is not None:
                kwargs["ext"] = plot_extension
            self.save_plots(folder, **kwargs)

    def update(
        self,
        compact: bool = False,
        save_stats=True,
        save_info=True,
        save_plot=False,
        include_pareto_frontier: bool = True,
        plot_extension: str | None = None,
        plot_kwargs: dict | None = None,
    ) -> bool:
        """
        Updates the scenario at its source location.

        Args:
            compact: If True, use compact JSON formatting.
            save_stats: If True, save scenario statistics to stats.json.
            save_info: If True, save scenario info to info file.
            save_plot: If True, save utility space plots. Default is False.
            include_pareto_frontier: If True, include pareto_utils and pareto_outcomes
                in stats.json. If False, exclude them to save disk space. Default is True.
            plot_extension: File extension for plots (e.g., 'png', 'jpg', 'svg', 'pdf', 'webp').
                If None, uses DEFAULT_IMAGE_FORMAT from negmas.plots.util (currently 'webp').
            plot_kwargs: Additional keyword arguments to pass to the plot() method. Default is None.

        Returns:
            True if successfully saved, False if no source is available.

        Remarks:
            - Only works if the scenario has a source (was loaded from somewhere).
            - If source is a folder, saves to that folder.
            - If source is a single file, saves to the parent directory of that file.
            - If source is a tuple of files, saves to the parent directory of the first file.
            - Auto-detects format based on source file extension (yml, json, xml).
        """
        if self.source is None:
            return False

        # Determine folder and format based on source
        if isinstance(self.source, tuple):
            # Multiple files: use parent of first file (domain file)
            folder = self.source[0].parent
            file_ext = self.source[0].suffix.lstrip(".")
        else:
            # Single path (either folder or file)
            if self.source.is_dir():
                folder = self.source
                # Try to detect format from existing files
                file_ext = "yml"  # default
                for ext in ("yml", "yaml", "json", "xml"):
                    if any(folder.glob(f"*.{ext}")):
                        file_ext = ext
                        break
            else:
                # Single file: use parent as folder
                folder = self.source.parent
                file_ext = self.source.suffix.lstrip(".")

        # Default to yml if no extension detected
        if not file_ext or file_ext not in ("yml", "yaml", "json", "xml"):
            file_ext = "yml"

        # Use dumpas to save
        self.dumpas(
            folder=folder,
            type=file_ext,
            compact=compact,
            save_stats=save_stats,
            save_info=save_info,
            save_plot=save_plot,
            include_pareto_frontier=include_pareto_frontier,
            plot_extension=plot_extension,
            plot_kwargs=plot_kwargs,
        )
        return True

    def load_info_file(self, file: Path):
        """Loads scenario info from a specific file path."""
        if not file.is_file():
            return self
        self.info = load(file)
        return self

    def load_info(self, folder: PathLike | str):
        """Loads scenario info from a folder, searching for supported formats."""
        for ext in ("yml", "yaml", "json"):
            path = Path(folder) / f"{INFO_FILE_NAME}.{ext}"
            if not path.is_file():
                continue
            self.info = load(path)
            break
        return self

    def load_stats_file(self, file: Path, calc_pareto_if_missing: bool = False):
        """Load stats file.

        Args:
            file: File path to load stats from.
            calc_pareto_if_missing: If True and the loaded stats have no pareto frontier
                data, calculate it using this scenario's utility functions.

        Returns:
            Self for method chaining.

        Notes:
            Handles stats files that may have been saved without pareto frontier data
            (when include_pareto_frontier=False was used during saving).

            When calc_pareto_if_missing=True, the pareto frontier will be computed
            on-the-fly, which may be slow for large outcome spaces.
        """
        if not file.is_file():
            return self
        self.stats = ScenarioStats.from_dict(
            load(file),
            ufuns=self.ufuns if calc_pareto_if_missing else None,
            calc_pareto_if_missing=calc_pareto_if_missing,
        )
        return self

    def load_stats(self, folder: PathLike | str, calc_pareto_if_missing: bool = False):
        """Loads scenario statistics from a folder, searching for supported formats.

        Args:
            folder: Folder containing the stats file.
            calc_pareto_if_missing: If True and the loaded stats have no pareto frontier
                data, calculate it using this scenario's utility functions.

        Returns:
            Self for method chaining.

        Notes:
            Looks for stats in the following order (for backward compatibility):
            1. _stats.yaml (new format)
            2. stats.json (legacy format from cartesian_tournament < 0.14.0)

            Handles stats files that may have been saved without pareto frontier data
            (when include_pareto_frontier=False was used during saving).

            When calc_pareto_if_missing=True, the pareto frontier will be computed
            on-the-fly, which may be slow for large outcome spaces.
        """
        folder = Path(folder)
        # Try new format first (_stats.yaml), then legacy format (stats.json)
        for stats_file in (STATS_FILE_NAME, "stats.json"):
            path = folder / stats_file
            if path.is_file():
                self.stats = ScenarioStats.from_dict(
                    load(path),
                    ufuns=self.ufuns if calc_pareto_if_missing else None,
                    calc_pareto_if_missing=calc_pareto_if_missing,
                )
                break
        return self

    @staticmethod
    def from_genius_folder(
        path: PathLike | str,
        ignore_discount=False,
        ignore_reserved=False,
        safe_parsing=True,
    ) -> Scenario | None:
        """Loads a scenario from a folder containing Genius-format XML files.

        Args:
            path: Directory containing the domain and utility function XML files.
            ignore_discount: If True, ignore time-based discounting in utility functions.
            ignore_reserved: If True, set reserved values to -inf.
            safe_parsing: If True, apply more stringent validation during parsing.

        Returns:
            The loaded Scenario, or None if loading fails.
        """
        s = load_genius_domain_from_folder(
            folder_name=str(path),
            ignore_discount=ignore_discount,
            ignore_reserved=ignore_reserved,
            safe_parsing=safe_parsing,
        )
        if s is None:
            return s
        return s.load_info(path)

    @classmethod
    def load(
        cls,
        folder: Path | str,
        safe_parsing=False,
        ignore_discount=False,
        load_stats=True,
        load_info=True,
        **kwargs,
    ) -> Scenario | None:
        """
        Loads the scenario from a folder with supported formats: XML, YML
        """
        for finder, loader in (
            (find_domain_and_utility_files_yaml, cls.from_yaml_folder),
            (find_domain_and_utility_files_xml, cls.from_genius_folder),
            (find_domain_and_utility_files_geniusweb, cls.from_geniusweb_folder),
        ):
            domain, _ = finder(folder)
            if domain is None:
                continue
            s = loader(
                folder,
                safe_parsing=safe_parsing,
                ignore_discount=ignore_discount,
                **kwargs,
            )
            if s is not None and load_info:
                s.load_info(folder)
            if s is not None and load_stats:
                s.load_stats(folder)
            return s

    @classmethod
    def is_loadable(cls, path: PathLike | str):
        """Checks whether a directory contains a valid loadable scenario."""
        if not Path(path).is_dir():
            return False
        for finder in (
            find_domain_and_utility_files_yaml,
            find_domain_and_utility_files_xml,
            find_domain_and_utility_files_geniusweb,
        ):
            d, _ = finder(Path(path))
            if d is not None:
                return True
        return False

    @staticmethod
    def from_genius_files(
        domain: PathLike,
        ufuns: Iterable[PathLike],
        info: PathLike | None = None,
        ignore_discount=False,
        ignore_reserved=False,
        safe_parsing=True,
        name: str | None = None,
    ) -> Scenario | None:
        """Loads a scenario from specific Genius-format XML file paths.

        Args:
            domain: Path to the domain XML file.
            ufuns: Paths to the utility function XML files.
            info: Optional path to the scenario info file.
            ignore_discount: If True, ignore time-based discounting.
            ignore_reserved: If True, set reserved values to -inf.
            safe_parsing: If True, apply more stringent validation during parsing.
            name: Optional name for the scenario. If None, uses domain file stem.

        Returns:
            The loaded Scenario, or None if loading fails.
        """
        s = load_genius_domain(
            domain,
            [_ for _ in ufuns],
            safe_parsing=safe_parsing,
            ignore_discount=ignore_discount,
            ignore_reserved=ignore_reserved,
            normalize_utilities=False,
            normalize_max_only=False,
            name=name,  # Pass through the name parameter
        )
        if not s:
            return s
        if info is not None:
            return s.load_info_file(Path(info))
        return s

    @staticmethod
    def from_geniusweb_folder(
        path: PathLike | str,
        ignore_discount=False,
        ignore_reserved=False,
        use_reserved_outcome=False,
        safe_parsing=True,
    ) -> Scenario | None:
        """Loads a scenario from a folder containing GeniusWeb-format JSON files.

        Args:
            path: Directory containing the domain and utility function JSON files.
            ignore_discount: If True, ignore time-based discounting.
            ignore_reserved: If True, set reserved values to -inf.
            use_reserved_outcome: If True, use reserved outcome instead of reserved value.
            safe_parsing: If True, apply more stringent validation during parsing.

        Returns:
            The loaded Scenario, or None if loading fails.
        """
        s = load_geniusweb_domain_from_folder(
            folder_name=str(path),
            ignore_discount=ignore_discount,
            use_reserved_outcome=use_reserved_outcome,
            ignore_reserved=ignore_reserved,
            safe_parsing=safe_parsing,
        )
        if not s:
            return s
        return s.load_info(path)

    @staticmethod
    def from_geniusweb_files(
        domain: PathLike,
        ufuns: Iterable[PathLike],
        info: PathLike | None = None,
        ignore_discount=False,
        ignore_reserved=False,
        use_reserved_outcome=False,
        safe_parsing=True,
        name: str | None = None,
    ) -> Scenario | None:
        """Loads a scenario from specific GeniusWeb-format JSON file paths.

        Args:
            domain: Path to the domain JSON file.
            ufuns: Paths to the utility function JSON files.
            info: Optional path to the scenario info file.
            ignore_discount: If True, ignore time-based discounting.
            ignore_reserved: If True, set reserved values to -inf.
            use_reserved_outcome: If True, use reserved outcome instead of reserved value.
            safe_parsing: If True, apply more stringent validation during parsing.
            name: Optional name for the scenario. If None, uses domain file stem.

        Returns:
            The loaded Scenario, or None if loading fails.
        """
        s = load_geniusweb_domain(
            domain,
            [_ for _ in ufuns],
            safe_parsing=safe_parsing,
            ignore_discount=ignore_discount,
            ignore_reserved=ignore_reserved,
            use_reserved_outcome=use_reserved_outcome,
            normalize_utilities=False,
            normalize_max_only=False,
            name=name,  # Pass through the name parameter
        )
        if not s:
            return s
        if info is not None:
            return s.load_info_file(Path(info))
        return s

    @classmethod
    def from_yaml_folder(
        cls,
        path: PathLike | str,
        ignore_discount=False,
        ignore_reserved=False,
        safe_parsing=True,
    ) -> Scenario | None:
        """Loads a scenario from a folder containing YAML files.

        Args:
            path: Directory containing the domain and utility function YAML files.
            ignore_discount: If True, ignore time-based discounting.
            ignore_reserved: If True, set reserved values to -inf.
            safe_parsing: Unused; YAML parsing is always safe.

        Returns:
            The loaded Scenario, or None if loading fails.
        """
        domain, ufuns = find_domain_and_utility_files_yaml(path)
        if not domain:
            return None
        s = cls.from_yaml_files(
            domain=domain,
            ufuns=ufuns,
            ignore_discount=ignore_discount,
            ignore_reserved=ignore_reserved,
            safe_parsing=safe_parsing,
            name=Path(path).name,  # Use folder name as scenario name
        )
        if not s:
            return s
        # Override source to be the folder path instead of individual files
        s = evolve(s, source=Path(path))
        return s.load_info(path)

    @classmethod
    def from_yaml_files(
        cls,
        domain: PathLike,
        ufuns: Iterable[PathLike],
        info: PathLike | None = None,
        ignore_discount=False,
        ignore_reserved=False,
        safe_parsing=True,
        python_class_identifier="type",
        name: str | None = None,
    ) -> Scenario | None:
        """Loads a scenario from specific YAML file paths.

        Args:
            domain: Path to the domain YAML file.
            ufuns: Paths to the utility function YAML files.
            info: Optional path to the scenario info file.
            ignore_discount: If True, ignore time-based discounting.
            ignore_reserved: If True, set reserved values to -inf.
            safe_parsing: Unused; YAML parsing is always safe.
            python_class_identifier: Key used to identify the Python class type in YAML.
            name: Optional name for the scenario. If None, uses name from YAML or domain file stem.

        Returns:
            The loaded Scenario, or None if loading fails.
        """
        _ = safe_parsing  # yaml parsing is always safe

        def adjust_type(d: dict, base: str = "negmas", domain=None) -> dict:
            """Ensures type fields have full module paths and sets outcome space."""
            if "." not in d["type"]:
                d["type"] = f"{base}.{d['type']}"
            if domain is not None:
                d["outcome_space"] = domain
            return d

        domain_dict = adjust_type(load(domain))
        domain_dict["path"] = Path(domain)
        # If outcome space has no name in YAML, use domain file stem
        if "name" not in domain_dict or not domain_dict["name"]:
            domain_dict["name"] = Path(domain).stem

        os = deserialize(
            domain_dict,
            base_module="negmas",
            python_class_identifier=python_class_identifier,
        )
        utils = [
            deserialize(
                adjust_type(load(fname), domain=os) | {"path": path},
                python_class_identifier=python_class_identifier,
                base_module="negmas",
            )
            for fname, path in zip(ufuns, ufuns)
        ]

        # d = load(domain)
        # type_ = d.pop("type", "")
        # assert (
        #     "OutcomeSpace" in type_
        # ), f"Unknown type or no type for domain file: {domain=}\n{d=}"
        # type_ = f"negmas.outcomes.{type_}"
        # d["issues"]
        # os = instantiate(type_, **d)
        # utils = []
        # for fname in ufuns:
        #     d = load(fname)
        #     type_ = d.pop("type", "")
        #     assert (
        #         "Fun" in type_
        #     ), f"Unknown type or no type for ufun file: {domain=}\n{d=}"
        #     type_ = f"negmas.preferences.{type_}"
        #     utils.append(instantiate(type_, **d))
        assert isinstance(os, CartesianOutcomeSpace)

        # Determine scenario name: use provided name, or fallback to domain file stem
        scenario_name = name if name is not None else Path(domain).stem

        # Build source: tuple of (domain_path, ufun_path1, ufun_path2, ...)
        source_paths = (Path(domain),) + tuple(Path(u) for u in ufuns)

        s = Scenario(
            outcome_space=os,
            ufuns=tuple(utils),
            name=scenario_name,
            source=source_paths,
        )  # type: ignore
        if s and ignore_discount:
            s = s.remove_discounting()
        if s and ignore_reserved:
            s = s.remove_reserved_values()
        if info is not None:
            return s.load_info_file(Path(info))
        return s


def get_domain_issues(
    domain_file_name: PathLike | str,
    n_discretization: int | None = None,
    safe_parsing=False,
) -> Sequence[Issue] | None:
    """
    Returns the issues of a given XML domain (Genius Format)

    Args:
        domain_file_name: Name of the file
        n_discretization: Number of discrete levels per continuous variable.
        max_cardinality: Maximum number of outcomes in the outcome space after discretization. Used only if `n_discretization` is given.
        safe_parsing: Apply more checks while parsing

    Returns:
        List of issues

    """
    issues = None
    if domain_file_name is not None:
        issues, _ = issues_from_genius(
            domain_file_name,
            safe_parsing=safe_parsing,
            n_discretization=n_discretization,
        )
    return issues


def load_genius_domain(
    domain_file_name: PathLike,
    utility_file_names: Iterable[PathLike] | None = None,
    ignore_discount=False,
    ignore_reserved=False,
    safe_parsing=True,
    name: str | None = None,
    **kwargs,
) -> Scenario:
    """
    Loads a genius domain, creates appropriate negotiators if necessary

    Args:
        domain_file_name: XML file containing Genius-formatted domain spec
        utility_file_names: XML files containing Genius-fromatted ufun spec
        ignore_reserved: Sets the reserved_value of all ufuns to -inf
        ignore_discount: Ignores discounting
        safe_parsing: Applies more stringent checks during parsing
        name: Optional name for the scenario. If None, uses domain file stem.

    Returns:
        A `Domain` ready to run

    """

    issues = None
    if domain_file_name is not None:
        issues, _ = issues_from_genius(domain_file_name, safe_parsing=safe_parsing)

    agent_info = []
    if utility_file_names is None:
        utility_file_names = []
    for ufname in utility_file_names:
        try:
            utility, discount_factor = UtilityFunction.from_genius(
                file_name=ufname,
                issues=issues,
                safe_parsing=safe_parsing,
                ignore_discount=ignore_discount,
                ignore_reserved=ignore_reserved,
                name=Path(ufname).stem,
            )
        except Exception as e:
            raise OSError(
                f"Ufun named {Path(ufname).name} cannot be read: {e.__class__.__name__}({e})"
            )

        agent_info.append(
            {
                "ufun": utility,
                "ufun_name": Path(ufname).stem,
                "ufun_file_name": ufname,
                "reserved_value_func": utility.reserved_value
                if utility is not None
                else float("-inf"),
                "discount_factor": discount_factor,
            }
        )
    if domain_file_name is not None:
        kwargs["dynamic_entry"] = False
        kwargs["max_n_agents"] = None
        if not ignore_discount:
            for info in agent_info:
                info["ufun"] = (
                    info["ufun"]
                    if info["discount_factor"] is None or info["discount_factor"] == 1.0
                    else make_discounted_ufun(
                        ufun=info["ufun"],
                        discount_per_round=info["discount_factor"],
                        power_per_round=1.0,
                    )
                )
    if issues is None:
        raise ValueError(f"Could not load domain {domain_file_name}")

    # Use provided name, or fallback to domain file stem
    scenario_name = name if name is not None else Path(domain_file_name).stem

    # Build source: tuple of (domain_path, ufun_path1, ufun_path2, ...)
    source_paths = (Path(domain_file_name),) + tuple(
        Path(_["ufun_file_name"]) for _ in agent_info
    )

    return Scenario(
        outcome_space=make_os(
            issues, name=Path(domain_file_name).stem, path=Path(domain_file_name)
        ),
        ufuns=tuple(_["ufun"] for _ in agent_info),
        name=scenario_name,
        source=source_paths,
    )


def load_genius_domain_from_folder(
    folder_name: str | PathLike,
    ignore_reserved=False,
    ignore_discount=False,
    safe_parsing=False,
    **kwargs,
) -> Scenario:
    """
    Loads a genius domain from a folder. See ``load_genius_domain`` for more details.

    Args:
        folder_name: A folder containing one XML domain file and one or more ufun files in Genius format
        ignore_reserved: Sets the reserved_value of all ufuns to -inf
        ignore_discount: Ignores discounting
        safe_parsing: Applies more stringent checks during parsing
        kwargs: Extra arguments to pass verbatim to SAOMechanism constructor

    Returns:
        A domain ready for `make_session`

    Examples:

        >>> import pkg_resources
        >>> from negmas import load_genius_domain_from_folder

        Try loading and running a domain with predetermined agents:
        >>> domain = load_genius_domain_from_folder(
        ...     pkg_resources.resource_filename(
        ...         "negmas", resource_name="tests/data/Laptop"
        ...     )
        ... )


        Try loading a domain and check the resulting ufuns
        >>> domain = load_genius_domain_from_folder(
        ...     pkg_resources.resource_filename(
        ...         "negmas", resource_name="tests/data/Laptop"
        ...     )
        ... )

        >>> domain.n_issues, domain.n_negotiators
        (3, 2)

        >>> [type(_) for _ in domain.ufuns]
        [<class 'negmas.preferences.crisp.linear.LinearAdditiveUtilityFunction'>, <class 'negmas.preferences.crisp.linear.LinearAdditiveUtilityFunction'>]

        Try loading a domain forcing a single issue space
        >>> domain = load_genius_domain_from_folder(
        ...     pkg_resources.resource_filename(
        ...         "negmas", resource_name="tests/data/Laptop"
        ...     )
        ... ).to_single_issue()
        >>> domain.n_issues, domain.n_negotiators
        (1, 2)
        >>> [type(_) for _ in domain.ufuns]
        [<class 'negmas.preferences.crisp.linear.LinearAdditiveUtilityFunction'>, <class 'negmas.preferences.crisp.linear.LinearAdditiveUtilityFunction'>]


        Try loading a domain with nonlinear ufuns:
        >>> folder_name = pkg_resources.resource_filename(
        ...     "negmas", resource_name="tests/data/10issues"
        ... )
        >>> domain = load_genius_domain_from_folder(folder_name)
        >>> print(domain.n_issues)
        10
        >>> print(domain.n_negotiators)
        2
        >>> print([type(u) for u in domain.ufuns])
        [<class 'negmas.preferences.crisp.nonlinear.HyperRectangleUtilityFunction'>, <class 'negmas.preferences.crisp.nonlinear.HyperRectangleUtilityFunction'>]
        >>> u = domain.ufuns[0]
        >>> print(u.outcome_ranges[0])
        {1: (7.0, 9.0), 3: (2.0, 7.0), 5: (0.0, 8.0), 8: (0.0, 7.0)}

        >>> print(u.mappings[0])
        97.0
        >>> print(u([0.0] * domain.n_issues))
        0
        >>> print(u([0.5] * domain.n_issues))
        186.0
    """
    folder_name = str(folder_name)
    folder_path = Path(folder_name)
    files = sorted(listdir(folder_name))
    domain_file_name = None
    utility_file_names = []
    for f in files:
        if not f.endswith(".xml") or f.endswith("pareto.xml"):
            continue
        full_name = folder_name + "/" + f
        root = ET.parse(full_name).getroot()

        if root.tag == "negotiation_template":
            domain_file_name = Path(full_name)
        elif root.tag == "utility_space":
            utility_file_names.append(full_name)
    if domain_file_name is None:
        raise ValueError("Cannot find a domain file")
    s = load_genius_domain(
        domain_file_name=domain_file_name,
        utility_file_names=utility_file_names,
        safe_parsing=safe_parsing,
        ignore_reserved=ignore_reserved,
        ignore_discount=ignore_discount,
        name=folder_path.name,  # Use folder name as scenario name
        **kwargs,
    )
    # Override source to be the folder path instead of individual files
    return evolve(s, source=folder_path)


def find_domain_and_utility_files_yaml(
    folder_name,
) -> tuple[PathLike | None, list[PathLike]]:
    """Finds the domain and utility_function files in a folder"""
    files = sorted(listdir(folder_name))
    domain_file_name = None
    utility_file_names = []
    folder_name = str(folder_name)
    for f in files:
        if not f.endswith(".yml") and not f.endswith(".yaml"):
            continue
        full_name = folder_name + "/" + f
        data = load(full_name)
        if data and "OutcomeSpace" in data.get("type", ""):
            domain_file_name = full_name
        elif data and ("fun" in data.get("type", "").lower()):
            utility_file_names.append(full_name)
    return domain_file_name, utility_file_names


def find_domain_and_utility_files_geniusweb(
    folder_name,
) -> tuple[PathLike | None, list[PathLike]]:
    """Finds the domain and utility_function files in a folder"""
    files = sorted(listdir(folder_name))
    domain_file_name = None
    utility_file_names = []
    folder_name = str(folder_name)
    for f in files:
        if not f.endswith(".json") or f.endswith("specials.json"):
            continue
        full_name = folder_name + "/" + f
        data = load(full_name)

        if data and "issuesValues" in data.keys():
            domain_file_name = full_name
        elif data and "LinearAdditiveUtilitySpace" in data.keys():
            utility_file_names.append(full_name)
    return domain_file_name, utility_file_names


def find_domain_and_utility_files_xml(
    folder_name,
) -> tuple[PathLike | None, list[PathLike]]:
    """Finds the domain and utility_function files in a folder"""
    files = sorted(listdir(folder_name))
    domain_file_name = None
    utility_file_names = []
    folder_name = str(folder_name)
    for f in files:
        if not f.endswith(".xml") or f.endswith("pareto.xml"):
            continue
        full_name = folder_name + "/" + f
        root = ET.parse(full_name).getroot()

        if root.tag == "negotiation_template":
            domain_file_name = full_name
        elif root.tag == "utility_space":
            utility_file_names.append(full_name)
    return domain_file_name, utility_file_names  #


def load_geniusweb_domain_from_folder(
    folder_name: str | PathLike,
    ignore_reserved=False,
    ignore_discount=False,
    use_reserved_outcome=False,
    safe_parsing=False,
    **kwargs,
) -> Scenario:
    """
    Loads a genius-web domain from a folder. See ``load_geniusweb_domain`` for more details.

    Args:
        folder_name: A folder containing one XML domain file and one or more ufun files in Genius format
        ignore_reserved: Sets the reserved_value of all ufuns to -inf
        ignore_discount: Ignores discounting
        safe_parsing: Applies more stringent checks during parsing
        kwargs: Extra arguments to pass verbatim to SAOMechanism constructor

    Returns:
        A domain ready for `make_session`

    """
    folder_name = str(folder_name)
    folder_path = Path(folder_name)
    domain_file_name, utility_file_names = find_geniusweb_domain_and_utility_files(
        folder_name
    )

    if domain_file_name is None:
        raise ValueError("Cannot find a domain file")
    s = load_geniusweb_domain(
        domain_file_name=domain_file_name,
        utility_file_names=utility_file_names,
        safe_parsing=safe_parsing,
        ignore_reserved=ignore_reserved,
        ignore_discount=ignore_discount,
        use_reserved_outcome=use_reserved_outcome,
        name=folder_path.name,  # Use folder name as scenario name
        **kwargs,
    )
    # Override source to be the folder path instead of individual files
    return evolve(s, source=folder_path)


def find_geniusweb_domain_and_utility_files(
    folder_name,
) -> tuple[PathLike | None, list[PathLike]]:
    """Finds the domain and utility_function files in a GeniusWeb formatted json folder"""
    files = sorted(listdir(folder_name))
    domain_file_name = None
    utility_file_names = []
    folder_name = str(folder_name)
    for f in files:
        if not f.endswith(".json"):
            continue
        full_name = folder_name + "/" + f
        d = load(full_name)
        if any(_ in d.keys() for _ in GENIUSWEB_UFUN_TYPES):
            utility_file_names.append(full_name)
        elif "issuesValues" in d.keys():
            domain_file_name = full_name
    return domain_file_name, utility_file_names


def find_genius_domain_and_utility_files(
    folder_name,
) -> tuple[PathLike | None, list[PathLike]]:
    """Finds the domain and utility_function files in a folder"""
    files = sorted(listdir(folder_name))
    domain_file_name = None
    utility_file_names = []
    folder_name = str(folder_name)
    for f in files:
        if not f.endswith(".xml") or f.endswith("pareto.xml"):
            continue
        full_name = folder_name + "/" + f
        root = ET.parse(full_name).getroot()

        if root.tag == "negotiation_template":
            domain_file_name = full_name
        elif root.tag == "utility_space":
            utility_file_names.append(full_name)
    return domain_file_name, utility_file_names  #


def load_geniusweb_domain(
    domain_file_name: PathLike,
    utility_file_names: Iterable[PathLike] | None = None,
    ignore_discount=False,
    ignore_reserved=False,
    use_reserved_outcome=False,
    safe_parsing=True,
    name: str | None = None,
    **kwargs,
) -> Scenario:
    """
    Loads a geniusweb domain, creates appropriate negotiators if necessary

    Args:
        domain_file_name: JSON file containing GeniusWeb-formatted domain spec
        utility_file_names: JSON files containing GeniusWeb-fromatted ufun spec
        ignore_reserved: Sets the reserved_value of all ufuns to -inf
        ignore_discount: Ignores discounting
        use_reserved_outcome: If True, use reserved outcome instead of reserved value
        safe_parsing: Applies more stringent checks during parsing
        name: Optional name for the scenario. If None, uses name from JSON or domain file stem.
        kwargs: Extra arguments to pass verbatim to SAOMechanism constructor

    Returns:
        A `Domain` ready to run

    """

    issues = None
    # Read the domain file to get the embedded name (if any) for the outcome space
    domain_data = load(domain_file_name)
    os_name_from_file = domain_data.get("name", None)

    if domain_file_name is not None:
        issues, _ = issues_from_geniusweb(domain_file_name, safe_parsing=safe_parsing)

    agent_info = []
    if utility_file_names is None:
        utility_file_names = []
    for ufname in utility_file_names:
        utility, discount_factor = UtilityFunction.from_geniusweb(
            file_name=ufname,
            issues=issues,
            safe_parsing=safe_parsing,
            ignore_discount=ignore_discount,
            ignore_reserved=ignore_reserved,
            use_reserved_outcome=use_reserved_outcome,
            name=Path(ufname).stem,
        )
        agent_info.append(
            {
                "ufun": utility,
                "ufun_name": Path(ufname).stem,
                "ufun_file_name": ufname,
                "reserved_value_func": utility.reserved_value
                if utility is not None
                else float("-inf"),
                "discount_factor": discount_factor,
            }
        )
    if domain_file_name is not None:
        kwargs["dynamic_entry"] = False
        kwargs["max_n_agents"] = None
        if not ignore_discount:
            for info in agent_info:
                info["ufun"] = (
                    info["ufun"]
                    if info["discount_factor"] is None or info["discount_factor"] == 1.0
                    else make_discounted_ufun(
                        ufun=info["ufun"],
                        discount_per_round=info["discount_factor"],
                        power_per_round=1.0,
                    )
                )
    if issues is None:
        raise ValueError(f"Could not load domain {domain_file_name}")

    # Determine outcome space name: use name from file if present, otherwise use file stem
    os_name = os_name_from_file if os_name_from_file else Path(domain_file_name).stem

    # Determine scenario name: use provided name, or fallback to domain file stem
    scenario_name = name if name is not None else Path(domain_file_name).stem

    # Build source: tuple of (domain_path, ufun_path1, ufun_path2, ...)
    source_paths = (Path(domain_file_name),) + tuple(
        Path(_["ufun_file_name"]) for _ in agent_info
    )

    return Scenario(
        outcome_space=make_os(issues, name=os_name, path=Path(domain_file_name)),
        ufuns=[_["ufun"] for _ in agent_info],  # type: ignore We trust that the ufun will be loaded
        name=scenario_name,
        source=source_paths,
    )
