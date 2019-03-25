"""
Implements functionality needed to connect to JNegMAS allowing Java factory managers to participate in SCML worlds.
"""
from typing import Union, Type, List, Optional, Dict, Any, Callable

from py4j.java_collections import ListConverter

from negmas import Contract, Breach, Negotiator, MechanismState, MechanismInfo, RenegotiationRequest
from negmas.apps.scml import GreedyFactoryManager, FactoryManager, ProductionFailure, Loan, SCMLAWI, CFP
from negmas.apps.scml.simulators import FactorySimulator, FastFactorySimulator
from negmas.helpers import snake_case, instantiate
from negmas.java import JavaCallerMixin, to_java, to_java, JNegmasGateway, from_java, to_dict_for_java

__all__ = [
    'JavaFactoryManager', 'JavaDoNothingFactoryManager', 'JavaGreedyFactoryManager', 'JavaMiddleMan', 'JavaSCMLAWI'
]


class JavaSCMLAWI:
    """An SCMLAWI As seen by Java.

    This is an object that is not visible to python code. It is not directly called from python ever. It is only called
    from a corresponding Java object to represent an internal python object. Because of he way py4j works, we cannot
    just use dunders to implement this kind of object in general. We will have to implement each such class
    independently.

    This kind of classes will always have an internal Java class implementing a Java interface in Jnegmas that starts
    with Py.

    """

    def __init__(self, awi: SCMLAWI):
        self.awi = awi

    def getProducts(self):
        products = to_java(self.awi.products)
        return ListConverter().convert(products, JNegmasGateway.gateway._gateway_client)

    def getProcesses(self):
        processes = to_java(self.awi.processes)
        return ListConverter().convert(processes, JNegmasGateway.gateway._gateway_client)

    def registerCFP(self, cfp: Dict[str, Any]) -> None:
        """Registers a CFP"""
        self.awi.register_cfp(from_java(cfp))

    def registerInterest(self, products: List[int]) -> None:
        """registers interest in receiving callbacks about CFPs related to these products"""
        self.awi.register_interest(products)

    def unregisterInterest(self, products: List[int]) -> None:
        """registers interest in receiving callbacks about CFPs related to these products"""
        self.awi.unregister_interest(products)

    def removeCFP(self, cfp: Dict[str, Any]) -> bool:
        """Removes a CFP"""
        return self.awi.remove_cfp(CFP.from_java(cfp))

    def evaluateInsurance(self, contract: Dict[str, Any], t: int = None) -> Optional[float]:
        """Can be called to evaluate the premium for insuring the given contract against breaches committed by others

        Args:

            contract: hypothetical contract
            t: time at which the policy is to be bought. If None, it means current step
        """
        result = self.awi.evaluate_insurance(from_java(contract), t)
        if result < 0:
            return None
        return result

    def buyInsurance(self, contract: Dict[str, Any]) -> bool:
        """Buys insurance for the contract by the premium calculated by the insurance company.

        Remarks:
            The agent can call `evaluate_insurance` to find the premium that will be used.
        """
        return self.awi.buy_insurance(from_java(contract))

    class Java:
        implements = ['jnegmas.apps.scml.awi.SCMLAWI']


class JavaFactoryManager(FactoryManager, JavaCallerMixin):
    """Allows factory managers implemented in Java (using jnegmas) to participate in SCML worlds.

    Objects of this class is used to represent a java object to the python environment. This means that they *MUST* have
    the same interface as a python class (first class in the inheritance list). The `JavaCallerMixin` is used to enable
    it to connect to the java object it is representing.

    """

    @property
    def type_name(self):
        """Overrides type name to give the internal java type name"""
        return 'j' + snake_case(self.java_class_name.replace('jnegmas.apps.scml.factory_managers.', '').replace(
            'FactoryManager', ''))

    @property
    def awi(self):
        return self._awi

    @awi.setter
    def awi(self, value):
        self._awi = value
        if self.python_shadow is not None:
            self.python_shadow._awi = value
        self.java_awi = JavaSCMLAWI(value)
        self.java_object.setAWI(self.java_awi)

    def init(self):
        super().init()
        if self.python_shadow is not None:
            self.python_shadow.simulator = self.simulator
        self.java_object.setSimulator(self.simulator)
        self.java_object.init()

    def step(self):
        super().step()
        return self.java_object.init()

    def on_new_cfp(self, cfp: 'CFP'):
        return self.java_object.onNewCFP(to_java(cfp))

    def on_remove_cfp(self, cfp: 'CFP'):
        return self.java_object.onRemoveCFP(to_java(cfp))

    def on_contract_nullified(self, contract: Contract, bankrupt_partner: str, compensation: float) -> None:
        self.java_object.onContractNullified(contract, bankrupt_partner, compensation)

    def on_agent_bankrupt(self, agent_id: str) -> None:
        self.java_object.onAgentBankrupt(agent_id)

    def confirm_partial_execution(self, contract: Contract, breaches: List[Breach]) -> bool:
        return self.java_object.confirmParialExecution(to_java(contract), to_java(breaches))

    def on_production_failure(self, failures: List[ProductionFailure]) -> None:
        return self.java_object.onProductionFailure(failures=failures)

    def confirm_loan(self, loan: Loan) -> bool:
        return self.java_object.confirmLoan(loan)

    def confirm_contract_execution(self, contract: Contract) -> bool:
        return self.java_object.confirmContractExecution(contract=contract)

    # def _get_negotiator(self, negotiator_class_name) -> Optional[Negotiator]:
    #     if negotiator_class_name in ('', 'none', 'null'):
    #         return None
    #     if negotiator_class_name.startswith('agents'):
    #         return GeniusNegotiator(java_class_name=negotiator_class_name)
    #     if negotiator_class_name.startswith('jnegmas'):
    #         return JavaSAONegotiator(java_class_name=negotiator_class_name)
    #     return instantiate(negotiator_class_name)

    def on_negotiation_request(self, cfp: "CFP", partner: str) -> Optional[Negotiator]:
        return from_java(self.java_object.onNegotiationRequest(cfp, partner))

    def on_negotiation_failure(self, partners: List[str], annotation: Dict[str, Any], mechanism: MechanismInfo
                               , state: MechanismState) -> None:
        return self.java_object.onNegotiationFailure(partners, annotation, mechanism, state)

    def on_negotiation_success(self, contract: Contract, mechanism: MechanismInfo) -> None:
        return self.java_object.onNegotiationSuccess(contract, mechanism)

    def on_contract_signed(self, contract: Contract) -> None:
        return self.java_object.onContractSigned(contract)

    def on_contract_cancelled(self, contract: Contract, rejectors: List[str]) -> None:
        return self.java_object.onContractCancelled(contract, rejectors)

    def sign_contract(self, contract: Contract) -> Optional[str]:
        return self.java_object.signContract(contract)

    def set_renegotiation_agenda(self, contract: Contract, breaches: List[Breach]) -> Optional[RenegotiationRequest]:
        return from_java(self.java_object.setRenegotiationAgenda(contract, breaches))

    def respond_to_renegotiation_request(self, contract: Contract, breaches: List[Breach],
                                         agenda: RenegotiationRequest) -> Optional[Negotiator]:
        return from_java(self.java_object.respondToRenegotiationRequest(contract, breaches, agenda))

    @classmethod
    def do_nothing_manager(cls):
        return JavaFactoryManager(java_class_name='jnegmas.apps.scml.factory_managers.DoNothingFactoryManager')

    @classmethod
    def greedy_manager(cls):
        return JavaFactoryManager(java_class_name='jnegmas.apps.scml.factory_managers.GreedyFactoryManager')

    def __init__(self, java_class_name: str = 'jnegmas.apps.scml.factory_managers.DoNothingFactoryManager'
                 , python_object_factory: Optional[Callable[[], FactoryManager]] = None
                 , auto_load_java: bool = False
                 , name=None, simulator_type: Union[str, Type[FactorySimulator]] = FastFactorySimulator):
        super().__init__(name=name, simulator_type=simulator_type)
        self.java_awi = None
        if python_object_factory is None:
            python_shadow_object = None
        else:
            python_shadow_object = python_object_factory()
        self.python_shadow = python_shadow_object
        self.init_java_bridge(java_class_name=java_class_name, auto_load_java=auto_load_java
                              , python_shadow_object=python_shadow_object)
        map = to_dict_for_java(self)
        map['simulatorType'] = self.simulator_type.__class__.__name__
        self.java_object.construct(map)


class JavaDoNothingFactoryManager(JavaFactoryManager):
    def __init__(self, auto_load_java: bool = False
                 , name=None, simulator_type: Union[str, Type[FactorySimulator]] = FastFactorySimulator):
        super().__init__(name=name, simulator_type=simulator_type, auto_load_java=auto_load_java
                         , java_class_name='jnegmas.apps.scml.factory_managers.DoNothingFactoryManager')


class JavaGreedyFactoryManager(JavaFactoryManager):
    def __init__(self, auto_load_java: bool = False
                 , name=None, simulator_type: Union[str, Type[FactorySimulator]] = FastFactorySimulator):
        super().__init__(name=name, simulator_type=simulator_type, auto_load_java=auto_load_java
                         , java_class_name='jnegmas.apps.scml.factory_managers.GreedyFactoryManager'
                         , python_object_factory=lambda: GreedyFactoryManager(name=name
                                                                              , simulator_type=self.simulator_type))


class JavaMiddleMan(JavaFactoryManager):
    def __init__(self, auto_load_java: bool = False
                 , name=None, simulator_type: Union[str, Type[FactorySimulator]] = FastFactorySimulator):
        super().__init__(name=name, simulator_type=simulator_type, auto_load_java=auto_load_java
                         , java_class_name='jnegmas.apps.scml.factory_managers.MiddleMan')


