"""
Implements functionality needed to connect to JNegMAS allowing Java factory managers to participate in SCML worlds.
"""
from typing import Union, Type, List, Optional, Dict, Any, Callable

from py4j.java_collections import ListConverter

from negmas import Contract, Breach, Negotiator, MechanismState, AgentMechanismInterface, RenegotiationRequest
from negmas.apps.scml import GreedyFactoryManager, FactoryManager, ProductionFailure, Loan, SCMLAWI, CFP
from negmas.apps.scml.simulators import FactorySimulator, FastFactorySimulator
from negmas.helpers import snake_case, instantiate
from negmas.java import JavaCallerMixin, to_java, to_java, JNegmasGateway, from_java, to_dict_for_java

__all__ = [
    'JavaFactoryManager', 'JavaDoNothingFactoryManager', 'JavaGreedyFactoryManager', 'JavaDummyMiddleMan', 'JavaSCMLAWI'
]


class JavaFactorySimulator:
    """An FactorySimulator As seen by JNegMAS.

        This is an object that is not visible to python code. It is not directly called from python ever. It is only called
        from a corresponding Java object to represent an internal python object. Because of he way py4j works, we cannot
        just use dunders to implement this kind of object in general. We will have to implement each such class
        independently.

        This kind of classes will always have an internal Java class implementing a Java interface in Jnegmas that starts
        with Py.

    """
    def __init__(self, simulator: FactorySimulator):
        self.simulator = simulator

    def maxStorage(self) -> int:
        return to_java(self.simulator.max_storage)

    def nSteps(self) -> int:
        return to_java(self.simulator.n_steps)

    def initialWallet(self) -> float:
        return to_java(self.simulator.initial_wallet)

    def initialStorage(self) -> List[int]:
        return to_java(self.simulator.initial_storage.tolist())

    def nLines(self) -> int:
        return to_java(self.simulator.n_lines)

    def finalBalance(self) -> float:
        return to_java(self.simulator.final_balance)

    def walletTo(self, t: int) -> List[float]:
        return to_java(self.simulator.wallet_to(t).tolist())

    def walletAt(self, t: int) -> float:
        return to_java(self.simulator.wallet_at(t))

    def storageTo(self, t: int) -> List[int]:
        return to_java(self.simulator.storage_to(t).tolinst())

    def storageAt(self, t: int) -> int:
        return to_java(self.simulator.storage_at(t))
    
    def line_schedules_to(self, t: int) -> List[List[int]]:
        return to_java(self.simulator.line_schedules_to(t).tolist())
    
    def line_schedules_at(self, t: int) -> List[int]:
        return to_java(self.simulator.line_schedules_at(t).tolist())

    def totalStorageTo(self, t: int) -> List[int]:
        return to_java(self.simulator.total_storage_to(t).tolist())
    
    def totalStorageAt(self, t: int) -> int:
        return to_java(self.simulator.total_storage_at(t))

    def reservedStorageTo(self, t: int) -> List[int]:
        return to_java(self.simulator.reserved_storage_to(t).tolist())

    def reservedStorageAt(self, t: int) -> int:
        return to_java(self.simulator.reserved_storage_at(t))

    def availableStorageTo(self, t: int) -> List[int]:
        return to_java(self.simulator.available_storage_to(t).tolist())

    def availableStorageAt(self, t: int) -> int:
        return to_java(self.simulator.available_storage_at(t))

    def loansTo(self, t: int) -> List[float]:
        return to_java(self.simulator.loans_to(t).tolist())

    def loansAt(self, t: int) -> float:
        return to_java(self.simulator.loans_at(t))

    def balanceTo(self, t: int) -> List[float]:
        return to_java(self.simulator.balance_to(t).tolist())

    def balanceAt(self, t: int) -> float:
        return to_java(self.simulator.balance_at(t))

    def fixedBefore(self) -> int:
        return to_java(self.simulator.fixed_before)

    def setState(self, t: int, storage: List[int], wallet: float, loans: float, lineSchedules: List[int]) -> None:
        self.simulator.set_state(t, storage=storage, wallet=wallet, loans=loans
                                        , line_schedules=lineSchedules)

    def addLoan(self, total:float, t: int) -> bool:
        return self.simulator.add_loan(total=total , t=t)

    def receive(self, payment, t):
        return self.simulator.receive(payment, t)

    def pay(self, payment, t, ignoreMoneyShortage):
        return self.simulator.pay(payment, t, ignoreMoneyShortage)

    def transportTo(self, product, quantity, t, ignoreInventoryShortage, ignoreSpaceShortage):
        return self.simulator.transport_to(product, quantity, t, ignoreInventoryShortage, ignoreSpaceShortage)

    def buy(self, product, quantity, price, t, ignoreMoneyShortage, ignoreSpaceShortage):
        return self.simulator.buy(product, quantity, price, t, ignoreMoneyShortage, ignoreSpaceShortage)

    def sell(self, product, quantity, price, t, ignoreMoneyShortage, ignoreInventoryShortage):
        return self.simulator.sell(product, quantity, price, t, ignoreMoneyShortage, ignoreInventoryShortage)

    def schedule(self, job, ignoreInventoryShortage, ignoreMoneyShortage, ignoreSpaceShortage, override):
        return self.simulator.schedule(job, ignoreInventoryShortage, ignoreMoneyShortage, ignoreSpaceShortage, override)

    def reserve(self, product, quantity, t):
        return self.reserve(product, quantity, t)

    def fixBefore(self, t):
        return self.simulator.fix_before(t)

    def bookmark(self):
        return self.simulator.bookmark()

    def rollback(self, bookmarkId):
        return self.simulator.rollback(bookmarkId)

    def deleteBookmark(self, bookmarkId):
        return self.simulator.delete_bookmark(bookmarkId)

    class Java:
        implements = ['jengmas.apps.scml.simulators.FactorySimulator']

class JavaSCMLAWI:
    """An SCMLAWI As seen by JNegMAS.

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
        return to_java(self.awi.products)

    def getProcesses(self):
        return to_java(self.awi.processes)

    def getState(self):
        return to_java(self.awi.state)

    def relativeTime(self):
        return self.awi.relative_time

    def getCurrentStep(self):
        return self.awi.current_step

    def getNSteps(self):
        return self.awi.n_steps

    def getDefaultSigningDelay(self):
        return self.awi.default_signing_delay

    def requestNegotiation(self, cfp: CFP, req_id: str, roles: Optional[List[str]] = None
                           , mechanism_name: Optional[str] = None
                           , mechanism_params: Optional[Dict[str, Any]]=None):
        return self.awi.request_negotiation(cfp, req_id, roles, mechanism_name, mechanism_params)

    def registerCFP(self, cfp: Dict[str, Any]) -> None:
        """Registers a CFP"""
        self.awi.register_cfp(from_java(cfp))

    def removeCFP(self, cfp: Dict[str, Any]) -> bool:
        """Removes a CFP"""
        return self.awi.remove_cfp(CFP.from_java(cfp))

    def registerInterest(self, products: List[int]) -> None:
        """registers interest in receiving callbacks about CFPs related to these products"""
        self.awi.register_interest(products)

    def unregisterInterest(self, products: List[int]) -> None:
        """registers interest in receiving callbacks about CFPs related to these products"""
        self.awi.unregister_interest(products)

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

    def loginfo(self, msg: str):
        return self.awi.loginfo(msg)

    def logwarning(self, msg: str):
        return self.awi.logwarning(msg)

    def logdebug(self, msg: str):
        return self.awi.logdebug(msg)

    def logerror(self, msg: str):
        return self.awi.logerror(msg)

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
        if self.python_shadow is not None:
            self.python_shadow.simulator = self.simulator
        self.java_object.setSimulator(self.simulator)
        self.java_object.init()

    def step(self):
        return self.java_object.init()

    def on_neg_request_rejected(self, req_id: str, by: Optional[List[str]]):
        return self.java_object.onNegRequestRejected(req_id, by)

    def on_neg_request_accepted(self, req_id: str, mechanism: AgentMechanismInterface):
        return self.java_object.onNegRequestAccepted(req_id, mechanism)

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

    def respond_to_negotiation_request(self, cfp: "CFP", partner: str) -> Optional[Negotiator]:
        return from_java(self.java_object.respondToNegotiationRequest(cfp, partner))

    def on_negotiation_failure(self, partners: List[str], annotation: Dict[str, Any], mechanism: AgentMechanismInterface
                               , state: MechanismState) -> None:
        return self.java_object.onNegotiationFailure(partners, annotation, mechanism, state)

    def on_negotiation_success(self, contract: Contract, mechanism: AgentMechanismInterface) -> None:
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


class JavaDummyMiddleMan(JavaFactoryManager):
    def __init__(self, auto_load_java: bool = False
                 , name=None, simulator_type: Union[str, Type[FactorySimulator]] = FastFactorySimulator):
        super().__init__(name=name, simulator_type=simulator_type, auto_load_java=auto_load_java
                         , java_class_name='jnegmas.apps.scml.factory_managers.DummyMiddleMan')


