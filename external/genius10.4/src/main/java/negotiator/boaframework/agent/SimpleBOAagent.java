package negotiator.boaframework.agent;

import java.util.HashMap;

import genius.core.boaframework.AcceptanceStrategy;
import genius.core.boaframework.BOAagentBilateral;
import genius.core.boaframework.OMStrategy;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import negotiator.boaframework.acceptanceconditions.other.AC_Next;
import negotiator.boaframework.offeringstrategy.other.TimeDependent_Offering;
import negotiator.boaframework.omstrategy.NullStrategy;
import negotiator.boaframework.opponentmodel.ScalableBayesianModel;

/**
 * Simple adapter which can be used to convert a BOA Agent to a normal Agent.
 * This is interesting for example for the ANAC competition in which BOA Agents
 * are not (yet) accepted.
 * 
 * @author Alex Dirkzwager
 */
public class SimpleBOAagent extends BOAagentBilateral {

	@Override
	public void agentSetup() {
		OpponentModel om = new ScalableBayesianModel();
		om.init(negotiationSession, new HashMap<String, Double>());
		OMStrategy oms = new NullStrategy(negotiationSession);
		OfferingStrategy offering = new TimeDependent_Offering(negotiationSession, om, oms, 0.2, 0, 1, 0); // Boulware
																											// agent
																											// strategy
		AcceptanceStrategy ac = new AC_Next(negotiationSession, offering, 1, 0);
		setDecoupledComponents(ac, offering, om, oms);
	}

	@Override
	public String getName() {
		return "SimpleBOAagent";
	}
}