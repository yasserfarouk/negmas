package negotiator.boaframework.acceptanceconditions.other;

import genius.core.boaframework.AcceptanceStrategy;
import genius.core.boaframework.Actions;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OfferingStrategy;

/**
 * This is the decoupled Acceptance Condition from ABMP Agent. The code was
 * taken from the ABMP Agent and adapted to work within the BOA framework.
 * 
 * http://www.verwaart.nl/culture/posterBNAIC2009ABMP.pdf
 * http://www.iids.org/publications/IJCAI01.ABMP.pdf
 * 
 * Decoupling Negotiating Agents to Explore the Space of Negotiation Strategies
 * T. Baarslag, K. Hindriks, M. Hendrikx, A. Dirkzwager, C.M. Jonker
 *
 * @author Alex Dirkzwager
 */
public class AC_ABMP extends AcceptanceStrategy {

	private static final double UTIlITYGAPSIZE = 0.05;

	/**
	 * Empty constructor for the BOA framework.
	 */
	public AC_ABMP() {
	}

	public AC_ABMP(NegotiationSession negoSession, OfferingStrategy strat) throws Exception {
		init(negoSession, strat, null, null);
	}

	@Override
	public Actions determineAcceptability() {

		Actions decision = Actions.Reject;

		if (negotiationSession.getOwnBidHistory().getLastBidDetails() != null && negotiationSession
				.getOpponentBidHistory().getLastBidDetails().getMyUndiscountedUtil() >= negotiationSession
						.getOwnBidHistory().getLastBidDetails().getMyUndiscountedUtil() - UTIlITYGAPSIZE) {
			decision = Actions.Accept;
		}
		return decision;
	}

	@Override
	public String getName() {
		return "Other - ABMP";
	}
}