package negotiator.boaframework.opponentmodel;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import genius.core.Bid;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OpponentModel;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.Value;
import genius.core.utility.AdditiveUtilitySpace;
import negotiator.boaframework.opponentmodel.tools.UtilitySpaceAdapter;

/**
 * Optimized version of the ANAC2012 CUKHAgent opponent model. Adapted by Mark
 * Hendrikx to be compatible with the BOA framework.
 *
 * This model keeps track of how many times each value has been offered. In the
 * author's implementation the sum of the value scores is used to quantify the
 * quality of the bid. In my BOA-compatible implementation I divide the sum of
 * the value scores by the maximum possible score to normalize the score to a
 * utility.
 * 
 * Note that after 100 bids the model is no longer updated.
 * 
 * Tim Baarslag, Koen Hindriks, Mark Hendrikx, Alex Dirkzwager and Catholijn M.
 * Jonker. Decoupling Negotiating Agents to Explore the Space of Negotiation
 * Strategies
 *
 * @author Mark Hendrikx
 */
public class CUHKFrequencyModelV2 extends OpponentModel {

	/** History of unique bids, contains at most 100 bids */
	private ArrayList<Bid> bidHistory;
	/**
	 * List of Hashmaps storing how much each value has been offered for each
	 * issue
	 */
	private ArrayList<HashMap<Value, Integer>> opponentBidsStatisticsDiscrete;
	/**
	 * Optimization which stores the score of the most prefered value for each
	 * issue
	 */
	private HashMap<Integer, Integer> maxPreferencePerIssue;
	/**
	 * Maximum amount unique bids which may be stored by the agent. After this
	 * limit the OM is not updated
	 */
	private int maximumBidsStored = 100;
	/** Cache the issues of the domain to improve performance */
	private List<Issue> issues;
	/** Highest possible sum of values */
	private int maxPossibleTotal = 0;
	/**
	 * After 100 bids the utilityspace does not chance, and can therefore be
	 * cached
	 */
	private AdditiveUtilitySpace cache = null;
	/** Boolean which indicates if the utilityspaced is arleady cached */
	private boolean cached = false;

	/**
	 * initialization of the model in which the issues are cached and the score
	 * keeper for each issue is created.
	 */
	@Override
	public void init(NegotiationSession negotiationSession, Map<String, Double> parameters) {
		this.bidHistory = new ArrayList<Bid>();
		opponentBidsStatisticsDiscrete = new ArrayList<HashMap<Value, Integer>>();
		maxPreferencePerIssue = new HashMap<Integer, Integer>();
		this.negotiationSession = negotiationSession;
		try {
			issues = negotiationSession.getUtilitySpace().getDomain().getIssues();

			for (int i = 0; i < issues.size(); i++) {
				IssueDiscrete lIssueDiscrete = (IssueDiscrete) issues.get(i);
				HashMap<Value, Integer> discreteIssueValuesMap = new HashMap<Value, Integer>();
				for (int j = 0; j < lIssueDiscrete.getNumberOfValues(); j++) {
					Value v = lIssueDiscrete.getValue(j);
					discreteIssueValuesMap.put(v, 0);
				}
				maxPreferencePerIssue.put(i, 0);
				opponentBidsStatisticsDiscrete.add(discreteIssueValuesMap);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * This function updates the opponent's Model by calling the
	 * updateStatistics method
	 */
	public void updateModel(Bid bid, double time) {
		if (this.bidHistory.size() > this.maximumBidsStored) {
			return;
		}
		if (bidHistory.indexOf(bid) == -1) {
			this.bidHistory.add(bid);
		}
		if (this.bidHistory.size() <= this.maximumBidsStored) {
			this.updateStatistics(bid);
		}
	}

	/**
	 * This function updates the statistics of the bids that were received from
	 * the opponent.
	 */
	private void updateStatistics(Bid bidToUpdate) {
		try {
			// counters for each type of the issues
			int discreteIndex = 0;
			for (Issue lIssue : issues) {
				int issueNum = lIssue.getNumber();
				Value v = bidToUpdate.getValue(issueNum);
				if (opponentBidsStatisticsDiscrete == null) {
					System.out.println("opponentBidsStatisticsDiscrete is NULL");
				} else if (opponentBidsStatisticsDiscrete.get(discreteIndex) != null) {
					int counterPerValue = opponentBidsStatisticsDiscrete.get(discreteIndex).get(v);
					counterPerValue++;
					if (counterPerValue > maxPreferencePerIssue.get(discreteIndex)) {
						maxPreferencePerIssue.put(discreteIndex, counterPerValue);
						maxPossibleTotal++; // must be an increase by 1 one the
											// total
					}
					opponentBidsStatisticsDiscrete.get(discreteIndex).put(v, counterPerValue);
				}
				discreteIndex++;
			}
		} catch (Exception e) {
			System.out.println("Exception in updateStatistics: " + e.getMessage());
		}
	}

	public double getBidEvaluation(Bid bid) {
		int discreteIndex = 0;
		int totalBidValue = 0;
		try {
			for (int j = 0; j < issues.size(); j++) {
				Value v = bid.getValue(issues.get(j).getNumber());
				if (opponentBidsStatisticsDiscrete == null) {
					System.err.println("opponentBidsStatisticsDiscrete is NULL");
				} else if (opponentBidsStatisticsDiscrete.get(discreteIndex) != null) {
					int counterPerValue = opponentBidsStatisticsDiscrete.get(discreteIndex).get(v);
					totalBidValue += counterPerValue;
				}
				discreteIndex++;
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		if (totalBidValue == 0) {
			return 0.0;
		}
		return (double) totalBidValue / (double) maxPossibleTotal;
	}

	/**
	 * Returns the estimated utilityspace. After 100 unique bids the opponent
	 * model is not longer updated, and therefore a cached version can be used.
	 */
	public AdditiveUtilitySpace getOpponentUtilitySpace() {
		if (!cached && this.bidHistory.size() >= this.maximumBidsStored) {
			cached = true;
			cache = new UtilitySpaceAdapter(this, negotiationSession.getUtilitySpace().getDomain());
		} else if (this.bidHistory.size() < this.maximumBidsStored) {
			return new UtilitySpaceAdapter(this, negotiationSession.getUtilitySpace().getDomain());
		}
		return cache;
	}

	/**
	 * This model does not rely on issue weights. Therefore, the issue weight is
	 * uniform.
	 */
	public double getWeight(Issue issue) {
		return (1.0 / (double) issues.size());
	}

	public String getName() {
		return "CUHK Frequency Model V2";
	}
}