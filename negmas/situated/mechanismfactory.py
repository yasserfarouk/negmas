from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

from negmas.events import Event
from negmas.helpers import exception2str, instantiate
from negmas.helpers.inout import add_records
from negmas.mechanisms import Mechanism
from negmas.negotiators import Negotiator
from negmas.outcomes import Issue, outcome2dict

from .common import PROTOCOL_CLASS_NAME_FIELD, NegotiationInfo

if TYPE_CHECKING:
    from .agent import Agent
    from .world import World

__all__ = ["MechanismFactory"]


class MechanismFactory:
    """A mechanism creation class. It can invite agents to join a mechanism and then run it."""

    def __init__(
        self,
        world: World,
        mechanism_name: str,
        mechanism_params: dict[str, Any],
        issues: list[Issue],
        req_id: str,
        caller: Agent,
        partners: list[Agent],
        roles: list[str] | None = None,
        annotation: dict[str, Any] = None,
        neg_n_steps: int = None,
        neg_time_limit: int = None,
        neg_step_time_limit=None,
        allow_self_negotiation=False,
        log_ufuns_file=None,
        group: str = None,
    ):
        self.mechanism_name, self.mechanism_params = mechanism_name, mechanism_params
        self.caller = caller
        self.group = group
        self.partners = partners
        self.roles = roles
        self.annotation = annotation
        self.neg_n_steps = neg_n_steps
        self.neg_time_limit = neg_time_limit
        self.neg_step_time_limit = neg_step_time_limit
        self.world = world
        self.req_id = req_id
        self.issues = issues
        self.mechanism = None
        self.allow_self_negotiation = allow_self_negotiation
        self.log_ufuns_file = log_ufuns_file

    def _create_negotiation_session(
        self,
        mechanism: Mechanism,
        responses: Iterable[tuple[Negotiator, str]],
        partners: list[Agent],
    ) -> Mechanism:
        for partner in partners:
            mechanism.register_listener(event_type="negotiation_end", listener=partner)

        ufun = []
        if self.log_ufuns_file is not None:
            for outcome in mechanism.discrete_outcomes():
                record = {
                    "mechanism_id": mechanism.id,
                    "outcome": outcome2dict(outcome, mechanism.issues),
                }
                ufun.append(record)
        for i, (partner_, (_negotiator, _role)) in enumerate(zip(partners, responses)):
            if self.log_ufuns_file is not None:
                for record in ufun:
                    record[f"agent{i}"] = partner_.name
                    # record[f"agent_type{i}"] = partner_.type_name
                    # record[f"negotiator{i}"] = _negotiator.name
                    record[f"reserved{i}"] = _negotiator.reserved_value
                    try:
                        record[f"u{i}"] = _negotiator.ufun(record["outcome"])
                    except:
                        record[f"u{i}"] = None
            _negotiator.owner = partner_
            mechanism.add(negotiator=_negotiator, role=_role)

        if self.log_ufuns_file is not None:
            for record in ufun:
                outcome = record.pop("outcome", {})
                record.update(outcome)
            add_records(self.log_ufuns_file, ufun)

        return mechanism

    def _start_negotiation(
        self,
        mechanism_name,
        mechanism_params: dict[str, Any],
        roles,
        caller,
        partners,
        annotation,
        issues,
        req_id,
    ) -> NegotiationInfo | None:
        """Tries to prepare the negotiation to start by asking everyone to join"""
        mechanisms = self.world.mechanisms
        if (
            (not self.allow_self_negotiation)
            and (len({_.id if _ is not None else "" for _ in partners}) < 2)
            and len(partners) > 1
        ):
            return None
        if issues is None:
            self.world.call(
                caller, caller.on_neg_request_rejected_, req_id=req_id, by=None
            )
            return None
        if (
            mechanisms is not None
            and mechanism_name is not None
            and mechanism_name not in mechanisms.keys()
        ):
            self.world.call(
                caller, caller.on_neg_request_rejected_, req_id=req_id, by=None
            )
            return None
        if mechanisms is not None and mechanism_name is not None:
            mechanism_name = mechanisms[mechanism_name].pop(
                PROTOCOL_CLASS_NAME_FIELD, mechanism_name
            )
        if mechanism_params is None:
            mechanism_params = {}
        if mechanisms and mechanisms.get(mechanism_name, None) is not None:
            mechanism_params.update(mechanisms[mechanism_name])
        # mechanism_params = {k: v for k, v in mechanism_params.items() if k != PROTOCOL_CLASS_NAME_FIELD}
        mechanism_params["n_steps"] = self.neg_n_steps
        mechanism_params["time_limit"] = self.neg_time_limit
        mechanism_params["step_time_limit"] = self.neg_step_time_limit
        mechanism_params["issues"] = issues
        mechanism_params["annotation"] = annotation
        mechanism_params["name"] = "-".join(_.id for _ in partners)
        if mechanism_name is None:
            if mechanisms is not None and len(mechanisms) == 1:
                mechanism_name = list(mechanisms.keys())[0]
            else:
                mechanism_name = "negmas.sao.SAOMechanism"
            if mechanisms and mechanisms.get(mechanism_name, None) is not None:
                mechanism_params.update(mechanisms[mechanism_name])
        try:
            mechanism = instantiate(class_name=mechanism_name, **mechanism_params)
        except Exception as e:
            s_ = exception2str()
            self.world.mechanism_exceptions[self.world.current_step].append(s_)
            self.world.agent_exceptions[caller.id].append((self.world.current_step, s_))
            mechanism = None
            self.world.logerror(
                f"Failed to create {mechanism_name} with params {mechanism_params}",
                Event("mechanism-creation-exception", dict(exception=e)),
            )
        self.mechanism = mechanism
        if mechanism is None:
            return None

        if self.mechanism:
            self.mechanism.register_listener("negotiator_exception", self.world)
        if roles is None:
            roles = [None] * len(partners)

        partner_names = [p.id for p in partners]
        responses = [
            self.world.call(
                partner,
                partner.respond_to_negotiation_request_,
                initiator=caller.id,
                partners=partner_names,
                issues=issues,
                annotation=annotation,
                role=role,
                mechanism=mechanism.nmi,
                req_id=req_id if partner == caller else None,
            )
            for role, partner in zip(roles, partners)
        ]
        if not all(responses):
            rejectors = [p for p, response in zip(partners, responses) if not response]
            rej = [_.id for _ in rejectors]
            for r in rej:
                self.world.neg_requests_rejected[r] += 1
            self.world.call(
                caller, caller.on_neg_request_rejected_, req_id=req_id, by=rej
            )
            for partner, response in zip(partners, responses):
                if partner.id != caller.id and response:
                    self.world.call(
                        partner, partner.on_neg_request_rejected_, req_id=None, by=rej
                    )
            self.world.loginfo(
                f"{caller.name} request was rejected by {[_.name for _ in rejectors]}",
                Event(
                    "negotiation-request-rejected",
                    dict(
                        req_id=req_id,
                        caller=caller,
                        partners=partners,
                        rejectors=rejectors,
                        annotation=annotation,
                    ),
                ),
            )
            return NegotiationInfo(
                mechanism=None,
                partners=partners,
                annotation=annotation,
                issues=issues,
                rejectors=rejectors,
                requested_at=self.world.current_step,
                caller=caller,
                group=self.group,
            )
        mechanism = self._create_negotiation_session(
            mechanism=mechanism, responses=zip(responses, roles), partners=partners
        )
        neg_info = NegotiationInfo(
            mechanism=mechanism,
            partners=partners,
            annotation=annotation,
            issues=issues,
            requested_at=self.world.current_step,
            caller=caller,
            group=self.group,
        )
        self.world.call(
            caller,
            caller.on_neg_request_accepted_,
            req_id=req_id,
            mechanism=mechanism.nmi,
        )
        for partner, response in zip(partners, responses):
            if partner.id != caller.id:
                self.world.call(
                    partner,
                    partner.on_neg_request_accepted_,
                    req_id=None,
                    mechanism=mechanism,
                )
        self.world.loginfo(
            f"{caller.name} request was accepted",
            Event(
                "negotiation-request-accepted",
                dict(
                    req_id=req_id,
                    caller=caller,
                    partners=partners,
                    mechanism=mechanism,
                    annotation=annotation,
                ),
            ),
        )
        return neg_info

    def init(self) -> NegotiationInfo | None:
        return self._start_negotiation(
            mechanism_name=self.mechanism_name,
            mechanism_params=self.mechanism_params,
            roles=self.roles,
            caller=self.caller,
            partners=self.partners,
            annotation=self.annotation,
            issues=self.issues,
            req_id=self.req_id,
        )
