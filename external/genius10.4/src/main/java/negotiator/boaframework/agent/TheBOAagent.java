package negotiator.boaframework.agent;

import genius.core.boaframework.BOAagentBilateral;
import genius.core.boaframework.BOAagentInfo;
import genius.core.boaframework.repository.BOAagentRepository;

/**
 * This class is used to convert a BOA agent created using the GUI to a real
 * agent. The parseStrategyParameters loads the information object, the
 * agentSetup uses the information object to load the agent using reflection.
 * 
 * For more information, see: Baarslag T., Hindriks K.V., Hendrikx M.,
 * Dirkzwager A., Jonker C.M. Decoupling Negotiating Agents to Explore the Space
 * of Negotiation Strategies. Proceedings of The Fifth International Workshop on
 * Agent-based Complex Automated Negotiations (ACAN 2012), 2012.
 * http://mmi.tudelft.nl/sites/default/files/boa.pdf
 * 
 * @author Tim Baarslag, Alex Dirkzwager, Mark Hendrikx
 */
public class TheBOAagent extends BOAagentBilateral {

	/** Name of the agent */
	private String name = "";
	/** Information object that stores the decoupled agent description */
	private BOAagentInfo dagent;

	/**
	 * Loads and initializes the decoupled components of the agent.
	 */
	@Override
	public void agentSetup() {

		// load the class names of each object
		String os = dagent.getOfferingStrategy().getClassname();
		String as = dagent.getAcceptanceStrategy().getClassname();
		String om = dagent.getOpponentModel().getClassname();
		String oms = dagent.getOMStrategy().getClassname();

		// createFrom the actual objects using reflexion

		offeringStrategy = BOAagentRepository.getInstance().getOfferingStrategy(os);
		acceptConditions = BOAagentRepository.getInstance().getAcceptanceStrategy(as);
		opponentModel = BOAagentRepository.getInstance().getOpponentModel(om);
		omStrategy = BOAagentRepository.getInstance().getOMStrategy(oms);

		// init the components.
		try {
			opponentModel.init(negotiationSession, dagent.getOpponentModel().getParameters());
			opponentModel.setOpponentUtilitySpace(fNegotiation);
			omStrategy.init(negotiationSession, opponentModel, dagent.getOMStrategy().getParameters());
			offeringStrategy.init(negotiationSession, opponentModel, omStrategy,
					dagent.getOfferingStrategy().getParameters());
			acceptConditions.init(negotiationSession, offeringStrategy, opponentModel,
					dagent.getAcceptanceStrategy().getParameters());
			acceptConditions.setOpponentUtilitySpace(fNegotiation);
		} catch (Exception e) {
			e.printStackTrace();
		}
		// remove the reference to the information object such that the garbage
		// collector can remove it.
		dagent = null;
	}

	/**
	 * Returns the name of the agent.
	 */
	@Override
	public String getName() {
		return name;
	}

	/**
	 * Removes the references to all components such that the garbage collector
	 * can remove them.
	 */
	@Override
	public void cleanUp() {
		offeringStrategy = null;
		acceptConditions = null;
		opponentModel = null;
		negotiationSession = null;
		utilitySpace = null;
		dagent = null;
	}

	/**
	 * Loads the BOA agent information object created by using the GUI. The
	 * {@link #agentSetup()} method uses this information to load the necessary
	 * components by using reflection.
	 */
	@Override
	public void parseStrategyParameters(String variables) throws Exception {
		Serializer<BOAagentInfo> serializer = new Serializer<BOAagentInfo>("");
		dagent = serializer.readStringToObject(variables);
		name = dagent.getName();
	}
}