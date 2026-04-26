package agents.anac.y2015.TUDMixedStrategyAgent;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Set;

import genius.core.Bid;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.EvaluatorDiscrete;

public class BidGenerator {

	// Just static methods
	private BidGenerator() {
		super();
	}

	// Unused
	// retreives the value with the highest evaluation for this Issue
	public static Value getMaxValue(EvaluatorDiscrete Eval) {
		double currentValue = 0;
		ValueDiscrete maxvalue = null;
		double maxevaluated = 0;

		Set<ValueDiscrete> valueSet = Eval.getValues();
		for (ValueDiscrete value : valueSet) {
			try {
				currentValue = Eval.getEvaluation(value);
				if (maxevaluated < currentValue) {
					maxevaluated = currentValue;
					maxvalue = value;
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		return maxvalue;
	}

	// Generates a List with the utility of the list of bids it receives
	public static double[] utilitylist(ArrayList<Bid> bids,
			TUDMixedStrategyAgent info) {

		double[] utilityarray = new double[bids.size()];

		for (int i = 0; i < bids.size(); i++) {
			utilityarray[i] = info.getUtility(bids.get(i));
		}

		return utilityarray;
	}

	// Generates a List of all possible bids in this UtilitySpace
	public static ArrayList<Bid> BidList(AdditiveUtilitySpace utilitySpace) {
		List<Issue> issueList = utilitySpace.getDomain().getIssues();
		ArrayList<ArrayList<ValueDiscrete>> listValueList = new ArrayList<ArrayList<ValueDiscrete>>();
		ArrayList<Bid> bidList = new ArrayList<Bid>();
		int bidListsize = 1;
		int nIssues = issueList.size();
		int[] nValues = new int[nIssues];
		for (int i = 0; i < issueList.size(); i++) {
			listValueList
					.add((ArrayList<ValueDiscrete>) ((IssueDiscrete) issueList
							.get(i)).getValues());
			nValues[i] = listValueList.get(i).size();
			bidListsize = bidListsize * nValues[i];
		}
		ValueDiscrete[][] valueMatrix = new ValueDiscrete[bidListsize][nIssues];
		for (int i = 0; i < nIssues; i++) {
			int before = 1, actual = nValues[i], after = 1;
			for (int k = 0; k < i; k++) {
				before = before * nValues[k];
			}
			for (int k = nIssues - 1; k > i; k--) {
				after = after * nValues[k];
			}
			for (int j = 0; j < before; j++) {
				for (int k = 0; k < actual; k++) {
					for (int l = 0; l < after; l++) {
						valueMatrix[l + actual * after * j + k * after][i] = listValueList
								.get(i).get(k);
					}
				}
			}

		}
		for (int i = 0; i < bidListsize; i++) {
			HashMap<Integer, Value> bidMap = new HashMap<Integer, Value>();
			for (int j = 0; j < nIssues; j++) {
				bidMap.put(issueList.get(j).getNumber(), valueMatrix[i][j]);
			}
			Bid currentBid;
			try {
				currentBid = new Bid(utilitySpace.getDomain(), bidMap);
				bidList.add(currentBid);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		return bidList;
	}

	/*
	 * Returns ArrayList of Bids within the range given, if there isn't any
	 * returns closest Bid with higher utility, If still there isn't any will
	 * return Bid with closest lower utility. If there isn't any will return an
	 * empty ArrayList
	 */
	public static ArrayList<Bid> getBidsInRange(ArrayList<Bid> bids,
			double low_bound, double up_bound, TUDMixedStrategyAgent info) {
		ArrayList<Bid> bidsInRange = new ArrayList<Bid>();
		// No bids
		if (bids.isEmpty())
			return bidsInRange;
		for (int i = 0; i < bids.size(); i++) {
			if (low_bound < info.getUtility(bids.get(i))
					&& up_bound > info.getUtility(bids.get(i))) {
				bidsInRange.add(bids.get(i));
			}
		}
		if (bidsInRange.isEmpty()) {
			// There are none within range, search for larger than range
			for (int i = 0; i < bids.size(); i++) {
				if (low_bound < info.getUtility(bids.get(i))) {
					bidsInRange.add(bids.get(i));
				}
			}
		}
		if (bidsInRange.isEmpty()) {
			// There are none inside or larger than range, search for closest
			// smaller
			double distance = Double.MAX_VALUE;
			Bid closestBid = null;
			for (int i = 0; i < bids.size(); i++) {
				if (distance > low_bound - info.getUtility(bids.get(i))) {
					closestBid = bids.get(i);
					distance = low_bound - info.getUtility(bids.get(i));
				}
			}
			bidsInRange.add(closestBid);
		}
		return bidsInRange;

	}
}
