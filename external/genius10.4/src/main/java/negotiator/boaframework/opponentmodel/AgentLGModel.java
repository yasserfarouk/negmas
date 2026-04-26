package negotiator.boaframework.opponentmodel;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import genius.core.Bid;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OpponentModel;
import genius.core.issue.Issue;
import genius.core.issue.Value;
import genius.core.utility.AdditiveUtilitySpace;
import negotiator.boaframework.opponentmodel.agentlg.BidStatistic;
import negotiator.boaframework.opponentmodel.tools.UtilitySpaceAdapter;

/**
 * Adaptation of the opponent model used by AgentLG in the ANAC2012 to be
 * compatible with the BOA framework.
 * 
 * Note that originally, the model sums up all value scores which entails that
 * the total preference of a bid can be as high as offered bids * issues. The
 * value score was equal to the amount of times the value was offered divided by
 * total amount of bids offered by the opponent.
 * 
 * In my implementation I normalize each value score by the highest value,
 * similar to the implementation of preference profiles in Genius. Finally I sum
 * all results and divide by the amount of issues. This is identical to assuming
 * that the issue weights are uniform.
 * 
 * Tim Baarslag, Koen Hindriks, Mark Hendrikx, Alex Dirkzwager and Catholijn M.
 * Jonker. Decoupling Negotiating Agents to Explore the Space of Negotiation
 * Strategies
 *
 * @author Mark Hendrikx
 */
public class AgentLGModel extends OpponentModel {

	/**
	 * Creates objects to keep track of how many times each value has been
	 * offered for an issue
	 */
	private HashMap<Issue, BidStatistic> statistic = new HashMap<Issue, BidStatistic>();
	/** Cache the issues of the domain for performanc reasons */
	private List<Issue> issues;

	/**
	 * Initialize the opponent model by creating an object to keep track of the
	 * values for each issue.
	 */
	@Override
	public void init(NegotiationSession negotiationSession, Map<String, Double> parameters) {
		this.negotiationSession = negotiationSession;
		issues = negotiationSession.getUtilitySpace().getDomain().getIssues();
		for (Issue issue : issues) {
			statistic.put(issue, new BidStatistic(issue));
		}
	}

	/**
	 * Update the opponent model by updating the value score for each issue.
	 * 
	 * @param opponentBid
	 * @param time
	 *            of offering
	 */
	@Override
	public void updateModel(Bid opponentBid, double time) {
		try {
			// updates statistics
			for (Issue issue : statistic.keySet()) {
				Value v = opponentBid.getValue(issue.getNumber());
				statistic.get(issue).add(v);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * @return utility of the opponent's bid
	 */
	public double getBidEvaluation(Bid bid) {
		double ret = 0;
		for (Issue issue : issues) {
			try {
				ret += statistic.get(issue).getValueUtility(bid.getValue(issue.getNumber()));
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		return (ret / issues.size());
	}

	/**
	 * @return the uniform issue weight
	 */
	public double getWeight(Issue issue) {
		return (1.0 / issues.size());
	}

	/**
	 * @return utilityspace created by using the opponent model adapter.
	 */
	@Override
	public AdditiveUtilitySpace getOpponentUtilitySpace() {
		return new UtilitySpaceAdapter(this, negotiationSession.getUtilitySpace().getDomain());
	}

	public void cleanUp() {
		super.cleanUp();
	}

	@Override
	public String getName() {
		return "AgentLG Model";
	}
}