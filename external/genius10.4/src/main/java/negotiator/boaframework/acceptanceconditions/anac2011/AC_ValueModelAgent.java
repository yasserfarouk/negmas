package negotiator.boaframework.acceptanceconditions.anac2011;

import java.util.Map;

import genius.core.boaframework.AcceptanceStrategy;
import genius.core.boaframework.Actions;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import negotiator.boaframework.sharedagentstate.anac2011.ValueModelAgentSAS;

/**
 * This is the decoupled Acceptance Conditions for ValueModelAgent (ANAC2011).
 * The code was taken from the ANAC2011 ValueModelAgent and adapted to work
 * within the BOA framework.
 * 
 * Decoupling Negotiating Agents to Explore the Space of Negotiation Strategies
 * T. Baarslag, K. Hindriks, M. Hendrikx, A. Dirkzwager, C.M. Jonker
 * 
 * @author Mark Hendrikx
 */
public class AC_ValueModelAgent extends AcceptanceStrategy {

	/**
	 * Empty constructor for the BOA framework.
	 */
	public AC_ValueModelAgent() {
	}

	public AC_ValueModelAgent(NegotiationSession negoSession, OfferingStrategy strat) throws Exception {
		initializeAgent(negoSession, strat);
	}

	@Override
	public void init(NegotiationSession negoSession, OfferingStrategy strat, OpponentModel opponentModel,
			Map<String, Double> parameters) throws Exception {
		initializeAgent(negoSession, strat);
	}

	public void initializeAgent(NegotiationSession negotiationSession, OfferingStrategy os) throws Exception {
		this.negotiationSession = negotiationSession;
		this.offeringStrategy = os;

		if (os.getHelper() instanceof ValueModelAgentSAS) {
			helper = os.getHelper();
		} else {
			helper = new ValueModelAgentSAS();
		}
	}

	@Override
	public Actions determineAcceptability() {
		boolean skip = ((ValueModelAgentSAS) helper).shouldSkipAcceptDueToCrash();
		if (negotiationSession.getOpponentBidHistory().size() > 0) {

			if (!skip && negotiationSession.getTime() > 0.98 && negotiationSession.getTime() <= 0.99) {
				if (((ValueModelAgentSAS) helper)
						.getOpponentUtil() >= ((ValueModelAgentSAS) helper).getLowestApprovedInitial() - 0.01) {
					return Actions.Accept;
				}
			}
			if (!skip && negotiationSession.getTime() > 0.995
					&& ((ValueModelAgentSAS) helper).getOpponentMaxBidUtil() > 0.55) {
				if (((ValueModelAgentSAS) helper)
						.getOpponentUtil() >= ((ValueModelAgentSAS) helper).getOpponentMaxBidUtil() * 0.99) {
					return Actions.Accept;
				}
			}

			// if our opponent settled enough for us we accept, and there is
			// a discount factor we accept
			// if(opponent.expectedDiscountRatioToConvergence()*opponentUtil
			// > lowestApproved){
			if (!skip
					&& ((ValueModelAgentSAS) helper).getOpponentUtil() > ((ValueModelAgentSAS) helper)
							.getLowestApproved()
					&& (negotiationSession.getDiscountFactor() > 0.02
							|| ((ValueModelAgentSAS) helper).getOpponentUtil() > 0.975)) {
				return Actions.Accept;
			}
			if (!skip && negotiationSession.getTime() > 0.9) {
				if (((ValueModelAgentSAS) helper)
						.getOpponentUtil() >= ((ValueModelAgentSAS) helper).getPlannedThreshold() - 0.01) {
					return Actions.Accept;
				}
			}
		}
		return Actions.Reject;
	}

	@Override
	public String getName() {
		return "2011 - ValueModelAgent";
	}
}