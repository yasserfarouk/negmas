package genius.core;

import java.util.List;

import java.util.Date;
import java.util.HashMap;
import java.util.Map;

import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.EndNegotiation;
import genius.core.actions.Offer;
import genius.core.actions.OfferForVoting;
import genius.core.parties.NegotiationInfo;
import genius.core.parties.NegotiationParty;
import genius.core.protocol.MultilateralProtocol;
import genius.core.protocol.StackedAlternatingOffersProtocol;
import genius.core.tournament.VariablesAndValues.AgentParamValue;
import genius.core.tournament.VariablesAndValues.AgentParameterVariable;
import genius.core.utility.AbstractUtilitySpace;

/**
 * Adapts {@link Agent} to the {@link NegotiationParty} so that bilateral agents
 * can be run as a multiparty system. Notice that these agents can
 * handle only 1 opponent, and thus may behave weird if presented with more than
 * one opponent.
 * 
 * What is unusual (in the Java sense) is that Agent extends this, not the other
 * way round. This way, all old agents also become a NegotiationParty.
 * 
 */
@SuppressWarnings("serial")
public abstract class AgentAdapter implements NegotiationParty {

	/**
	 * 
	 * @return the actual agent that is being adapted.
	 */
	abstract protected Agent getAgent();

	private Action lastAction = null;
	private AbstractUtilitySpace utilSpace;

	@SuppressWarnings("deprecation")
	@Override
	public final void init(NegotiationInfo info) {
		this.utilSpace = info.getUtilitySpace();
		getAgent().internalInit(0, 1, new Date(),
				info.getDeadline().getTimeOrDefaultTimeout(),
				info.getTimeline(), utilSpace,
				new HashMap<AgentParameterVariable, AgentParamValue>(),
				info.getAgentID());
		getAgent().setName(info.getAgentID().toString());
		getAgent().init();
	}

	@SuppressWarnings("deprecation")
	@Override
	public final Action chooseAction(
			List<Class<? extends Action>> possibleActions) {
		lastAction = getAgent().chooseAction();
		return lastAction;
	}

	@SuppressWarnings("deprecation")
	@Override
	public final void receiveMessage(AgentID sender, Action action) {
		if (action instanceof Offer || action instanceof Accept
				|| action instanceof OfferForVoting
				|| action instanceof EndNegotiation) {
			getAgent().ReceiveMessage(action);
		}
	}

	/**
	 * This is a convenience wrapper so that we don't have to fix all old agent
	 * descriptions (these used to be in the xml file)
	 */
	@Override
	public String getDescription() {
		return "Agent " + getAgent().getClass().getSimpleName();
	}

	@Override
	public final Class<? extends MultilateralProtocol> getProtocol() {
		return StackedAlternatingOffersProtocol.class;
	}

	@SuppressWarnings("deprecation")
	@Override
	public final Map<String, String> negotiationEnded(Bid acceptedBid) {
		double util = 0;
		if (acceptedBid != null) {
			try {
				util = getAgent().getUtility(acceptedBid);
			} catch (Exception e) {
			}
		}
		getAgent().endSession(
				new NegotiationResult(util, lastAction, acceptedBid));
		return null;
	}

}
