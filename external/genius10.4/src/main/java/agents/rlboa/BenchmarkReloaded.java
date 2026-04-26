package agents.rlboa;

import genius.core.boaframework.BOAagentBilateral;
import negotiator.boaframework.acceptanceconditions.other.AC_Next;
import negotiator.boaframework.offeringstrategy.anac2012.TheNegotiatorReloaded_Offering;
import negotiator.boaframework.omstrategy.NullStrategy;
import negotiator.boaframework.opponentmodel.AgentXFrequencyModel;

import java.util.HashMap;

@SuppressWarnings("deprecation")
public class BenchmarkReloaded extends BOAagentBilateral {

    /**
     * Benchmark agent to test the RL-agents against.
     * Same opponentModel and Acceptancecondition but Offeringstrategy of the Negotiator Reloaded.
     */
    private static final long serialVersionUID = 1L;

    @Override
    public void agentSetup() {

        // AverageTitForTat2 makes decisions based on its own preferences
        opponentModel = new AgentXFrequencyModel();
        opponentModel.init(negotiationSession, new HashMap<String, Double>());

        omStrategy = new NullStrategy(negotiationSession, 0.35);
        try {
            offeringStrategy = new TheNegotiatorReloaded_Offering(
                    negotiationSession, opponentModel, omStrategy);
        } catch (Exception e) {
            e.printStackTrace();
        }

        acceptConditions = new AC_Next(negotiationSession, offeringStrategy, 1, 0);
        setDecoupledComponents(acceptConditions, offeringStrategy, opponentModel, omStrategy);
    }

    @Override
    public String getName() {
        return "Project AI benchmark with offering strategy of Negotiator Reloaded";
    }

}
