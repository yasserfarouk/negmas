package negotiator.boaframework.acceptanceconditions.anac2011;

import java.util.Map;

import genius.core.boaframework.AcceptanceStrategy;
import genius.core.boaframework.Actions;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import negotiator.boaframework.sharedagentstate.anac2011.TheNegotiatorSAS;

/**
 * This is the decoupled Acceptance Conditions for TheNegotiator (ANAC2011). The
 * code was taken from the ANAC2011 TheNegotiator and adapted to work within the
 * BOA framework.
 * 
 * Decoupling Negotiating Agents to Explore the Space of Negotiation Strategies
 * T. Baarslag, K. Hindriks, M. Hendrikx, A. Dirkzwager, C.M. Jonker
 * 
 * @author Alex Dirkzwager, Mark Hendrikx
 */
public class AC_TheNegotiator extends AcceptanceStrategy {

	/**
	 * Empty constructor for the BOA framework.
	 */
	public AC_TheNegotiator() {
	}

	public AC_TheNegotiator(NegotiationSession negoSession, OfferingStrategy strat) throws Exception {
		init(negoSession, strat, null, null);
	}

	@Override
	public void init(NegotiationSession negoSession, OfferingStrategy strat, OpponentModel opponentModel,
			Map<String, Double> parameters) throws Exception {
		this.negotiationSession = negoSession;
		offeringStrategy = strat;

		// checking if offeringStrategy helper is a TheNegotiatorHelper
		if (offeringStrategy.getHelper() == null || (!offeringStrategy.getHelper().getName().equals("TheNegotiator"))) {
			helper = new TheNegotiatorSAS(negotiationSession);
		} else {
			helper = (TheNegotiatorSAS) offeringStrategy.getHelper();
		}
	}

	@Override
	public Actions determineAcceptability() {
		Actions decision = Actions.Reject;
		double utility = 0;

		try {
			// effectively this will ensure that the utility is 0 if our agent
			// is first
			if (negotiationSession.getOpponentBidHistory().getHistory().size() > 0) {
				// get the last opponent bid
				utility = negotiationSession.getOpponentBidHistory().getLastBidDetails().getMyUndiscountedUtil();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}

		int phase;
		int movesLeft;
		double threshold;

		phase = ((TheNegotiatorSAS) helper).calculateCurrentPhase(negotiationSession.getTime());
		movesLeft = ((TheNegotiatorSAS) helper).calculateMovesLeft();
		threshold = ((TheNegotiatorSAS) helper).calculateThreshold(negotiationSession.getTime());

		if (phase < 3) {
			if (utility >= threshold) {
				decision = Actions.Accept;
			}
		} else { // phase 3
			if (movesLeft >= 15) {
				if (utility >= threshold) {
					decision = Actions.Accept;
				}
			} else {
				if (movesLeft < 15) {
					decision = Actions.Accept;
				}
			}
		}
		return decision;
	}

	@Override
	public String getName() {
		return "2011 - TheNegotiator";
	}
}
