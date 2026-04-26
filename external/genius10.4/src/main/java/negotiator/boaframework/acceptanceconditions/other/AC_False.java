package negotiator.boaframework.acceptanceconditions.other;

import genius.core.boaframework.AcceptanceStrategy;
import genius.core.boaframework.Actions;

/**
 * This Acceptance Condition never accepts an opponent offer.
 * 
 * Decoupling Negotiating Agents to Explore the Space of Negotiation Strategies
 * T. Baarslag, K. Hindriks, M. Hendrikx, A. Dirkzwager, C.M. Jonker
 * 
 * @author Alex Dirkzwager, Mark Hendrikx
 * @version 18/12/11
 */
public class AC_False extends AcceptanceStrategy {

	/**
	 * Empty constructor for the BOA framework.
	 */
	public AC_False() {
	}

	/**
	 * @return reject any bid.
	 */
	@Override
	public Actions determineAcceptability() {
		return Actions.Reject;
	}

	@Override
	public String getName() {
		return "Other - False";
	}
}