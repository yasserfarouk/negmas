package agents.anac.y2015.JonnyBlack;

import java.util.Vector;

import genius.core.Bid;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.ValueDiscrete;
import genius.core.utility.AdditiveUtilitySpace;

public class Functions {
	public static int[] calcOrderOfIssues(AdditiveUtilitySpace us) {
		int noOfIssues = us.getNrOfEvaluators();
		int[] issueOrder = new int[noOfIssues];
		for (int i = 0; i < issueOrder.length; i++) {
			issueOrder[i] = i;
		}
		for (int i = 0; i < issueOrder.length; i++) {
			for (int j = i + 1; j < issueOrder.length; j++) {
				if (us.getWeight(issueOrder[i]) < us.getWeight(issueOrder[j])) {
					int temp = issueOrder[i];
					issueOrder[i] = issueOrder[j];
					issueOrder[j] = temp;
				}
			}
		}
		return issueOrder;
	}

	public static int[][] calcOrderOfIssueVals(AdditiveUtilitySpace us) {
		int noOfIssues = us.getNrOfEvaluators();
		int[][] orderOfVals = new int[noOfIssues][];
		for (int i = 0; i < noOfIssues; i++) {
			int noVals = ((IssueDiscrete) (us.getIssue(i))).getNumberOfValues();
			orderOfVals[i] = new int[noVals];
		}
		for (int i = 0; i < orderOfVals.length; i++) {
			for (int j = 0; j < orderOfVals[i].length; j++) {
				orderOfVals[i][j] = j + 1;
			}
		}
		// TODO Not Ordering
		for (int i = 0; i < orderOfVals.length; i++) {
			for (int j = 0; j < orderOfVals[i].length; j++) {
				double value1 = getValueOfIssueVal(us, i, j);
				for (int k = j; k < orderOfVals[i].length; k++) {
					double value2 = getValueOfIssueVal(us, i, k);
					if (value1 < value2) {
						int t = orderOfVals[i][j];
						orderOfVals[i][j] = orderOfVals[i][k];
						orderOfVals[i][k] = t;
					}
				}
			}
		}
		return orderOfVals;
	}

	public static double getValueOfIssueVal(AdditiveUtilitySpace us, int issue, int val) {
		Bid temp = getCopyOfBestBid(us);
		ValueDiscrete vd = getVal(us, issue, val);
		temp = temp.putValue(issue + 1, vd);
		double value = evaluateOneIssue(us, issue, temp);
		return value;
	}

	public static ValueDiscrete getVal(AdditiveUtilitySpace us, int issue, int valID) {
		IssueDiscrete is = (IssueDiscrete) us.getIssue(issue);
		return is.getValue(valID);
	}

	public static double evaluateOneIssue(AdditiveUtilitySpace us, int issue, Bid b) {
		try {
			return us.getEvaluator(issue + 1).getEvaluation(us, b, issue + 1);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return -1;
	}

	public static Bid getCopyOfBestBid(AdditiveUtilitySpace us) {
		try {
			return new Bid(us.getMaxUtilityBid());
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}

	public static double getBidValue(AdditiveUtilitySpace us, Bid b) {
		try {
			return us.getUtility(b);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return -1;
	}

	public static double calcStopVal(Vector<Party> parties, int n,
			AdditiveUtilitySpace us) {
		for (Party p : parties) {
			p.orderBids(us);
		}
		Vector<BidHolder> topNbids = new Vector<BidHolder>();
		for (int i = 0; i < n && i < parties.get(0).orderedBids.size(); i++) {
			topNbids.add(parties.get(0).orderedBids.get(i));
		}
		for (int i = 0; i < topNbids.size(); i++) {
			for (Party p : parties) {
				if (p.orderedBids.indexOf(topNbids.get(i)) > n
						|| p.orderedBids.indexOf(topNbids.get(i)) == -1) {
					topNbids.remove(i);
					i--;
					break;
				}
			}
		}
		System.out.println("Common Bids : " + topNbids.size());
		double max = 0;
		for (BidHolder b : topNbids) {
			double v = Functions.getBidValue(us, b.b);
			if (v > max)
				max = v;
		}
		return Math.max(max, 0.6);
	}
}
