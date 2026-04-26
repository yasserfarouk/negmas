package negotiator.boaframework.opponentmodel;

import java.util.List;
import java.util.Map;

import genius.core.Bid;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OpponentModel;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.IssueReal;
import genius.core.issue.Value;
import genius.core.utility.AdditiveUtilitySpace;
import negotiator.boaframework.opponentmodel.nash.AIssueEvaluation;
import negotiator.boaframework.opponentmodel.nash.IssueEvaluationDiscrete;
import negotiator.boaframework.opponentmodel.nash.IssueEvaluationInteger;
import negotiator.boaframework.opponentmodel.nash.IssueEvaluationList;
import negotiator.boaframework.opponentmodel.nash.IssueEvaluationReal;
import negotiator.boaframework.opponentmodel.nash.Range;
import negotiator.boaframework.opponentmodel.tools.UtilitySpaceAdapter;

/**
 * This class holds the model of a negotiator, which will be constructed by it's
 * bids. The opponentmodel will be based on frequency/distribution analysis.
 * 
 * - The importance of a discrete issue and it's values will be calculated by
 * the difference in the number of times a certain value is chosen. - The
 * importance of a numerical issue will be calculated by the first offered
 * values and the range in which our own issue has a utility > 0. We will then
 * interpolate between the max utility value and the min utility value.
 *
 * Adapted by Mark Hendrikx to be compatible with the BOA framework.
 *
 * Tim Baarslag, Koen Hindriks, Mark Hendrikx, Alex Dirkzwager and Catholijn M.
 * Jonker. Decoupling Negotiating Agents to Explore the Space of Negotiation
 * Strategies
 * 
 * @author Roland van der Linden, Mark Hendrikx
 */
public class NashFrequencyModel extends OpponentModel {
	// **************************************
	// Fields
	// **************************************

	// The list of issueEvaluations which hold the analysis.
	private IssueEvaluationList issueEvaluationList;

	@Override
	public void init(NegotiationSession domainKnow, Map<String, Double> parameters) {
		negotiationSession = domainKnow;
		initModel();
	}

	/**
	 * This initializes the negotiatormodel, creating an issueEvaluation for
	 * each issue in the utilitySpace.
	 */
	private void initModel() {
		List<Issue> issues = negotiationSession.getUtilitySpace().getDomain().getIssues();

		this.issueEvaluationList = new IssueEvaluationList(issues.size());

		// Create an empty issueEvaluation object for each issue in the domain.
		// This will later contain all information we can gather on the
		// negotiator.
		for (int index = 0; index < issues.size(); index++) {
			Issue issue = issues.get(index);
			AIssueEvaluation issueEvaluation = null;

			if (issue instanceof IssueDiscrete)
				issueEvaluation = new IssueEvaluationDiscrete((IssueDiscrete) issue);
			else if (issue instanceof IssueInteger) {
				// We use the range in which our utility is non-zero to estimate
				// the distribution of the opponent.
				IssueInteger issueI = (IssueInteger) issue;
				Range ourNonZeroUtilityRange = new Range(issueI.getLowerBound(), issueI.getUpperBound());
				issueEvaluation = new IssueEvaluationInteger((IssueInteger) issue, ourNonZeroUtilityRange);
			} else if (issue instanceof IssueReal) {
				// We use the range in which our utility is non-zero to estimate
				// the distribution of the opponent.
				IssueReal issueR = (IssueReal) issue;
				Range ourNonZeroUtilityRange = new Range(issueR.getLowerBound(), issueR.getUpperBound());
				issueEvaluation = new IssueEvaluationReal((IssueReal) issue, ourNonZeroUtilityRange);
			} else
				throw new UnsupportedOperationException("There is no implementation for that issueType.");

			// Add the new issueEvaluation to the list.
			this.issueEvaluationList.addIssueEvaluation(issueEvaluation);
		}
	}

	// **************************************
	// Update model
	// **************************************

	/**
	 * This will receiveMessage the negotiatormodel based on a new bid that has
	 * just been offered by the negotiator.
	 * 
	 * @param bid
	 *            The bid that has just been offered.
	 */
	public void updateModel(Bid bid, double time) {
		if (bid != null) {
			// We receiveMessage each issueEvaluation with the value that has
			// been offered in the bid.
			List<Issue> issues = negotiationSession.getUtilitySpace().getDomain().getIssues();
			for (Issue issue : issues) {
				try {
					int issueID = issue.getNumber();
					Value offeredValue = bid.getValue(issueID);
					this.issueEvaluationList.updateIssueEvaluation(issueID, offeredValue);
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
			// After all issueEvaluations have been updated, we can calculate
			// the new
			// estimated weights for the issues themselves.
			this.issueEvaluationList.updateIssueWeightMap();
		}
	}

	// ****************************************
	// Utility
	// ****************************************

	/**
	 * This method estimates the utility of the negotiator given that it has
	 * just offered the given bid. The utility estimation is done by multiplying
	 * the normalized issue weights with the normalized offered-value weight of
	 * that issue.
	 * 
	 * Note that the utility of a bid will always be in between 0 - 1. Note that
	 * this method does not take discount into account.
	 * 
	 * @param bid
	 *            The bid that has just been offered by the negotiator that we
	 *            are evaluating.
	 * @return The estimated utility of the bid that has been offered by the
	 *         negotiator we are evaluating.
	 */
	public double getBidEvaluation(Bid bid) {
		double result = 0;
		if (issueEvaluationList.isReady()) {
			List<Issue> issues = negotiationSession.getUtilitySpace().getDomain().getIssues();

			double totalEstimatedUtility = 0;
			for (Issue issue : issues) {
				try {
					int issueID = issue.getNumber();

					// Get the estimated normalized weight of the issue, which
					// indicates how important the issue is.
					double issueWeight = this.issueEvaluationList.getNormalizedIssueWeight(issueID);

					// Get the estimated normalized weight of the value of the
					// issue, which corresponds to the value that has
					// been offered in the bid.
					Value offeredValue = bid.getValue(issueID);
					AIssueEvaluation issueEvaluation = this.issueEvaluationList.getIssueEvaluation(issueID);
					double offeredValueWeight = issueEvaluation.getNormalizedValueWeight(offeredValue);

					// Since all issueWeights combined should add up to 1, and
					// the maximum value of the valueWeight is 1,
					// the estimated utility should be exactly 1 if all offered
					// valueWeights are 1. So to calculate the partial estimated
					// utility for this specific issue-value combination, we
					// need to multiply the weights.
					totalEstimatedUtility += (issueWeight * offeredValueWeight);
				} catch (Exception e) {
					e.printStackTrace();
				}
			}

			// Make sure to remove roundoff errors from the utility.
			result = Math.min(1, Math.max(0, totalEstimatedUtility));
		}
		return result;
	}

	// ****************************************
	// toString
	// ****************************************

	/**
	 * This returns a string representation of the negotiatormodel.
	 */
	public String toString() {
		return this.issueEvaluationList.toString();
	}

	public double getWeight(Issue issue) {
		// Get the estimated normalized weight of the issue, which indicates how
		// important the issue is.
		return this.issueEvaluationList.getNormalizedIssueWeight(issue.getNumber());
	}

	@Override
	public String getName() {
		return "NASH Frequency Model";
	}

	public AdditiveUtilitySpace getOpponentUtilitySpace() {
		if (issueEvaluationList.isReady()) {
			return new UtilitySpaceAdapter(this, negotiationSession.getUtilitySpace().getDomain());
		} else {
			System.out.println("Returned own utilityspace to avoid an error (normal on first turn).");
			return (AdditiveUtilitySpace) negotiationSession.getUtilitySpace();
		}
	}
}