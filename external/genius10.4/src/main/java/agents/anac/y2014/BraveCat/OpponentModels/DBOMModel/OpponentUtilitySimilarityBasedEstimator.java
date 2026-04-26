package agents.anac.y2014.BraveCat.OpponentModels.DBOMModel;

import java.util.List;

import agents.anac.y2014.BraveCat.necessaryClasses.NegotiationSession;
import genius.core.Bid;
import genius.core.bidding.BidDetails;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.IssueReal;

public class OpponentUtilitySimilarityBasedEstimator {

	int maximumNumOfLastReceivedBidsUsed = 0;
	NegotiationSession negotiationSession;

	public OpponentUtilitySimilarityBasedEstimator(NegotiationSession nego,
			int max) {
		negotiationSession = nego;
		maximumNumOfLastReceivedBidsUsed = max;
	}

	public double GetBidUtility(BidDetails bid1) throws Exception {
		int repeatingTimes = Math.min(maximumNumOfLastReceivedBidsUsed,
				negotiationSession.getOpponentBidHistory().getHistory().size());
		int i = 1;
		double sum = 0;
		double avgUtility = 0;
		double[] CalculatedSimilarities = new double[repeatingTimes + 1];

		for (i = 1; i <= repeatingTimes; i++) {
			Bid tempBid = negotiationSession
					.getOpponentBidHistory()
					.getHistory()
					.get(negotiationSession.getOpponentBidHistory()
							.getHistory().size()
							- i).getBid();
			double tempUtility = 1 - (0.3 * negotiationSession
					.getOpponentBidHistory()
					.getHistory()
					.get(negotiationSession.getOpponentBidHistory()
							.getHistory().size()
							- i).getTime());
			double tempTime = negotiationSession
					.getOpponentBidHistory()
					.getHistory()
					.get(negotiationSession.getOpponentBidHistory()
							.getHistory().size()
							- i).getTime();
			BidDetails bid2 = new BidDetails(tempBid, tempUtility, tempTime);

			CalculatedSimilarities[i] = GetSimilarity(bid1, bid2);
			sum += CalculatedSimilarities[i];
		}

		for (i = 1; i <= repeatingTimes; i++)
			CalculatedSimilarities[i] = (double) CalculatedSimilarities[i]
					/ sum;

		for (i = 1; i <= repeatingTimes; i++) {
			Bid tempBid = negotiationSession
					.getOpponentBidHistory()
					.getHistory()
					.get(negotiationSession.getOpponentBidHistory()
							.getHistory().size()
							- i).getBid();
			double tempUtility = 1 - (0.3 * negotiationSession
					.getOpponentBidHistory()
					.getHistory()
					.get(negotiationSession.getOpponentBidHistory()
							.getHistory().size()
							- i).getTime());
			double tempTime = negotiationSession
					.getOpponentBidHistory()
					.getHistory()
					.get(negotiationSession.getOpponentBidHistory()
							.getHistory().size()
							- i).getTime();
			BidDetails bid2 = new BidDetails(tempBid, tempUtility, tempTime);
			avgUtility += CalculatedSimilarities[i]
					* bid2.getMyUndiscountedUtil();
		}
		return avgUtility;
	}

	private double GetSimilarity(BidDetails bid1, BidDetails bid2)
			throws Exception {
		double TemporalSimilarity = Math.abs(bid1.getTime() - bid2.getTime());
		double NaturalSimilarity = GetBidDistance(bid1.getBid(), bid2.getBid());
		return TemporalSimilarity * NaturalSimilarity;
	}

	private double GetBidDistance(Bid bid1, Bid bid2) throws Exception {
		double avgDistance = 0;
		List<Issue> issues = negotiationSession.getDomain().getIssues();
		for (int i = 1; i <= issues.size(); i++) {
			Issue lIssue = issues.get(i - 1);
			switch (lIssue.getType()) {
			case DISCRETE:
				IssueDiscrete lIssueDiscrete = (IssueDiscrete) lIssue;
				double t1 = (double) (lIssueDiscrete.getValueIndex(bid1
						.getValue(i).toString()) - lIssueDiscrete
						.getValueIndex(bid2.getValue(i).toString()))
						/ lIssueDiscrete.getNumberOfValues();
				avgDistance += Math.pow(t1, 2);
				break;
			case REAL:
				IssueReal lIssueReal = (IssueReal) lIssue;
				double t2 = (double) (Double.parseDouble(bid1.getValue(i)
						.toString()) - Double.parseDouble(bid2.getValue(i)
						.toString()))
						/ (lIssueReal.getUpperBound() - lIssueReal
								.getLowerBound());
				avgDistance += Math.pow(t2, 2);
				break;
			case INTEGER:
				IssueInteger lIssueInteger = (IssueInteger) lIssue;
				double t3 = (double) (Integer.parseInt(bid1.getValue(i)
						.toString()) - Integer.parseInt(bid2.getValue(i)
						.toString()))
						/ (lIssueInteger.getUpperBound() - lIssueInteger
								.getLowerBound());
				avgDistance += Math.pow(t3, 2);
				break;
			default:
				throw new Exception("issue type " + lIssue.getType()
						+ " not supported!");
			}
		}
		return Math.sqrt(avgDistance);
	}
}
