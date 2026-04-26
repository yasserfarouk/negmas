package agents.anac.y2014.BraveCat.necessaryClasses;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import genius.core.Bid;
import genius.core.bidding.BidDetails;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.IssueReal;
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;

public class BidGenerator {
	Random random200;
	Random random300;
	NegotiationSession negotiationSession;

	public BidGenerator(NegotiationSession negoSession) {
		negotiationSession = negoSession;
		random200 = new Random();
		random300 = new Random();
	}

	private BidDetails searchBid() throws Exception {
		HashMap values = new HashMap();
		List issues = negotiationSession.getDomain().getIssues();
		Bid bid = null;
		for (Iterator it = issues.iterator(); it.hasNext();) {
			Issue lIssue = (Issue) it.next();
			switch (lIssue.getType()) {
			case DISCRETE:
				IssueDiscrete lIssueDiscrete = (IssueDiscrete) lIssue;
				int optionIndex = this.random300.nextInt(lIssueDiscrete
						.getNumberOfValues());
				values.put(Integer.valueOf(lIssue.getNumber()),
						lIssueDiscrete.getValue(optionIndex));
				break;
			case REAL:
				IssueReal lIssueReal = (IssueReal) lIssue;
				int optionInd = this.random300.nextInt(lIssueReal
						.getNumberOfDiscretizationSteps() - 1);
				values.put(
						Integer.valueOf(lIssueReal.getNumber()),
						new ValueReal(lIssueReal.getLowerBound()
								+ (lIssueReal.getUpperBound() - lIssueReal
										.getLowerBound()) * optionInd
								/ lIssueReal.getNumberOfDiscretizationSteps()));
				break;
			case INTEGER:
				IssueInteger lIssueInteger = (IssueInteger) lIssue;
				int optionIndex2 = lIssueInteger.getLowerBound()
						+ this.random300.nextInt(lIssueInteger.getUpperBound()
								- lIssueInteger.getLowerBound());
				values.put(Integer.valueOf(lIssueInteger.getNumber()),
						new ValueInteger(optionIndex2));
				break;
			default:
				throw new Exception("issue type " + lIssue.getType()
						+ " not supported!");
			}
		}
		bid = new Bid(negotiationSession.getDomain(), values);
		BidDetails bidDetails = new BidDetails(bid, this.negotiationSession
				.getUtilitySpace().getUtility(bid),
				this.negotiationSession.getTime());
		return bidDetails;
	}

	public BidDetails selectBid(double bidTarget) throws Exception {
		BidDetails nextBid = null;
		double searchUtil = 0.0D;
		try {
			int loop = 0;
			while (searchUtil < bidTarget) {
				if (loop > 50) {
					bidTarget -= 0.01D;
					loop = 0;
				}
				nextBid = searchBid();
				searchUtil = negotiationSession.getUtilitySpace().getUtility(
						nextBid.getBid());
				loop++;
			}
		} catch (Exception localException) {
			System.out.println("Exception occured when selecting a bid!");
		}
		if (nextBid == null)
			nextBid = searchBid();
		return nextBid;
	}

	public List<BidDetails> NBidsNearUtility(double bidTarget, int n)
			throws Exception {
		ArrayList<BidDetails> tempList = new ArrayList();
		for (int i = 0; i < n; i++)
			tempList.add(selectBid(bidTarget));
		return tempList;
	}
}
