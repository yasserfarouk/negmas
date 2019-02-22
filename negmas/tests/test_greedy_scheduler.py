from negmas.apps.scml import Product, Process, InputOutput, ManufacturingProfileCompiled, Line, FactorySchedule, GreedyScheduler, Contract, SCMLAgreement, CFP


class TestGreedyScheduler:

    def test_something(self):
        products = [Product(name=f'pr{i}', catalog_price=i + 1, id=i, production_level=i, expires_in=0) for i in range(2)]
        print(products)

        processes = [Process(id=0,
                             name='p0',
                             # inputs={InputOutput(0, quantity=2, step=0.0), InputOutput(1, quantity=3, step=0.0)},
                             inputs={InputOutput(0, quantity=1, step=0.0)},
                             outputs={InputOutput(1, quantity=1, step=0.0)},
                             historical_cost=1, production_level=1),
                     Process(id=1,
                             name='p1',
                             # inputs={InputOutput(0, quantity=2, step=0.0), InputOutput(1, quantity=3, step=0.0)},
                             inputs={InputOutput(1, quantity=1, step=0.0)},
                             outputs={InputOutput(0, quantity=1, step=0.0)},
                             historical_cost=1, production_level=1)]
        print(processes)

        manufacturing_profiles = {0: ManufacturingProfileCompiled(n_steps=1, cost=1, cancellation_cost=5, initial_pause_cost=0, running_pause_cost=0, resumption_cost=0),
                                  1: ManufacturingProfileCompiled(n_steps=1, cost=1, cancellation_cost=5, initial_pause_cost=0, running_pause_cost=0, resumption_cost=0)}
        print(manufacturing_profiles)

        lines = [Line(profiles=manufacturing_profiles, processes=processes) for _ in range(1)]
        print(lines)

        factory = FactorySchedule(lines=lines, products=products, processes=processes)
        print("factory = ", factory)

        greedy_scheduler = GreedyScheduler(factory=factory, n_steps=99999, products=products, processes=processes, manager_id='test', awi=None, strategy='shortest')
        print(greedy_scheduler)
        print(greedy_scheduler.products)

        r = factory.init_schedule(n_steps=1000, initial_storage={0: 8, 1: 0}, initial_balance=0)
        print("init = ", r)

        a_contract = Contract(agreement=SCMLAgreement(time=20, unit_price=1, quantity=10),
                              annotation={'cfp': CFP(is_buy=True, product=0, time=-10000, unit_price=1, quantity=1, publisher='test'),
                                          'buyer': 'anyone', 'seller': 'test'})

        print("a_contract = ", a_contract)

        # something = greedy_scheduler.schedule([a_contract])
        # print(something)
        something = greedy_scheduler.schedule([a_contract])
        print("\t ###### ", type(something))
        print(something)

    assert True
