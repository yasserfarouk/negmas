package negotiator.boaframework.acceptanceconditions.other;

import genius.core.boaframework.AcceptanceStrategy;
import genius.core.boaframework.Actions;

/**
 * This Acceptance Condition will accept any opponent offer. Very handy for
 * debugging.
 * 
 * Decoupling Negotiating Agents to Explore the Space of Negotiation Strategies
 * T. Baarslag, K. Hindriks, M. Hendrikx, A. Dirkzwager, C.M. Jonker
 * 
 * @author Alex Dirkzwager, Mark Hendrikx
 */
public class AC_True extends AcceptanceStrategy {

	/**
	 * Empty constructor for the BOA framework.
	 */
	public AC_True() {
	}

	/**
	 * @return accept any bid.
	 */
	@Override
	public Actions determineAcceptability() {
		return Actions.Accept;
	}

	@Override
	public String getName() {
		return "Other - True";
	}
}