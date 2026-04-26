package negotiator.boaframework.opponentmodel;

import java.util.ArrayList;
import java.util.Map;

import genius.core.Bid;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OpponentModel;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.utility.AdditiveUtilitySpace;
import negotiator.boaframework.opponentmodel.agentx.DiscreteIssueProcessor;
import negotiator.boaframework.opponentmodel.agentx.DiscreteValueProcessor;
import negotiator.boaframework.opponentmodel.tools.UtilitySpaceAdapter;

/**
 * Class for building an opponent model in discrete space. Contains value- and
 * issue-processors to deal with opponent bids.
 * 
 * Adapted by Mark Hendrikx to be compatible with the BOA framework.
 *
 * Tim Baarslag, Koen Hindriks, Mark Hendrikx, Alex Dirkzwager and Catholijn M.
 * Jonker. Decoupling Negotiating Agents to Explore the Space of Negotiation
 * Strategies
 *
 * @author E. Jacobs, Mark Hendrikx
 */
public class AgentXFrequencyModel extends OpponentModel {

	private DiscreteIssueProcessor issueProcessor;
	private ArrayList<IssueDiscrete> discreteIssueList = new ArrayList<IssueDiscrete>();
	private ArrayList<DiscreteValueProcessor> valueProcessList = new ArrayList<DiscreteValueProcessor>();
	private int startingBidIssue = 0;

	/**
	 * Creates an opponent model for the given utility space
	 * 
	 * @param u
	 *            The utility space
	 */
	@Override
	public void init(NegotiationSession negotiationSession, Map<String, Double> parameters) {
		this.negotiationSession = negotiationSession;
		issueProcessor = new DiscreteIssueProcessor(negotiationSession.getUtilitySpace().getDomain());
		discreteIssueList = issueProcessor.getIssueList();
		createValueProcessors();
		while (!testIndexOfFirstIssue(negotiationSession.getUtilitySpace().getDomain().getRandomBid(null),
				startingBidIssue)) {
			startingBidIssue++;
		}
	}

	/**
	 * Just an auxiliar funtion to calculate the index where issues start on a
	 * bid because we found out that it depends on the domain.
	 * 
	 * @return true when the received index is the proper index
	 */
	private boolean testIndexOfFirstIssue(Bid bid, int i) {
		try {
			@SuppressWarnings("unused")
			ValueDiscrete valueOfIssue = (ValueDiscrete) bid.getValue(i);
		} catch (Exception e) {
			return false;
		}
		return true;
	}

	/**
	 * Creates discreteValueProcessors for all issues
	 */
	private void createValueProcessors() {

		for (IssueDiscrete i : discreteIssueList) {
			valueProcessList.add(new DiscreteValueProcessor(i.getValues()));
		}
	}

	/**
	 * Returns a DiscreteValueProcessor, allowing the use of several methods
	 * 
	 * @param i
	 *            The issue for which a ValueProcessor is needed
	 * @return a DiscreteValueProcessor for the issue
	 */
	public DiscreteValueProcessor getValueProcessor(IssueDiscrete i) {

		int index = discreteIssueList.indexOf(i);

		if (index != -1)
			return valueProcessList.get(index);

		return null;

	}

	/**
	 * Gives the discreteIssueProcessor for this opponent model
	 * 
	 * @return The discreteIssueProcessor
	 */
	public DiscreteIssueProcessor getIssueProcessor() {
		return issueProcessor;
	}

	/**
	 * Processes a bid, possibly changing value ranks for the internal opponent
	 * model. Currently, values on which more bids are made are ranked higher.
	 * 
	 * @param b
	 *            The bid done by the opponent
	 * @param time
	 *            Time at which the bid was done
	 */
	public void updateModel(Bid b, double time) {

		issueProcessor.adaptWeightsByBid(b, time);
		for (IssueDiscrete i : discreteIssueList) {
			Value v = null;

			try {
				v = b.getValue(i.getNumber());
			} catch (Exception e) {
			}

			if (v != null) {
				ValueDiscrete vDisc = (ValueDiscrete) v;
				getValueProcessor(i).addBidForValue(vDisc);
			}
		}
	}

	/**
	 * Gives the normalized value rank of some value within a certain Issue
	 * 
	 * @param issueIndex
	 *            The index of the issue. Same index as in the ArrayList
	 *            provided by utilitySpace.getDomain().getIssues()
	 * @param valueOfIssue
	 *            The value from that issue for which the normalized rank is
	 *            required
	 * @return The normalized valueRank for the value of the issue given
	 */
	public double getEvaluationOfValue(int issueIndex, ValueDiscrete valueOfIssue) {
		return valueProcessList.get(issueIndex).getNormalizedValueRank(valueOfIssue);
	}

	/**
	 * Calculates the utility to our opponent of the bid received as a parameter
	 * using the current knowledge given by our opponent model
	 * 
	 * @param bid
	 * @return utility value for our opponent
	 */
	public double getBidEvaluation(Bid bid) {

		double utility = 0;

		// Taking into account that Opponent Issue list is in the same our of
		// the domain
		int nrIssues = negotiationSession.getUtilitySpace().getDomain().getIssues().size();

		for (int i = 0; i < nrIssues; i++) {
			try {
				// It was needed to use an auxiliar variable startingBidIssue,
				// because we found out it varies from domain to domain
				ValueDiscrete valueOfIssue = (ValueDiscrete) bid.getValue(i + startingBidIssue);
				double w = getIssueProcessor().getWeightByIssue(
						(IssueDiscrete) negotiationSession.getUtilitySpace().getDomain().getIssues().get(i));
				double eval = getEvaluationOfValue(i, valueOfIssue);

				utility += w * eval;

			} catch (Exception e) {
				// e.printStackTrace();
			}
		}

		return utility;
	}

	/**
	 * @return the weight of the issue.
	 */
	public double getWeight(Issue issue) {
		return getIssueProcessor().getWeightByIssue((IssueDiscrete) issue);
	}

	/**
	 * @return utilityspace created by using the opponent model adapter.
	 */
	public AdditiveUtilitySpace getOpponentUtilitySpace() {
		return new UtilitySpaceAdapter(this, negotiationSession.getUtilitySpace().getDomain());
	}

	public String getName() {
		return "AgentX Frequency Model";
	}
}