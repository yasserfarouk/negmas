package agents.anac.y2011.ValueModelAgent;

import java.util.HashMap;
import java.util.Map;

import genius.core.Bid;
import genius.core.issue.ISSUETYPE;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.Value;
import genius.core.utility.AdditiveUtilitySpace;

public class ValueSeperatedBids {

	private class ValueBidData {
		BidList approvedSorted = new BidList();
		int lastTimeBidden = -1;
		boolean changed = false;

		void addWrapper(BidWrapper bid) {
			approvedSorted.addIfNew(bid);
			changed = true;
		}
	}

	private class IssueSeperatedBids {
		public Map<String, ValueBidData> values = new HashMap<String, ValueBidData>();
	}

	AdditiveUtilitySpace utilitySpace;
	IssueSeperatedBids[] issues;
	ValueModeler model;

	public void addApproved(BidWrapper bid) {
		for (int i = 0; i < issues.length; i++) {
			if (issues[i] != null) {

				try {
					Value value = bid.bid.getValue(utilitySpace.getIssue(i)
							.getNumber());
					ValueBidData data = issues[i].values.get(value.toString());
					data.addWrapper(bid);
				} catch (Exception e) {
				}
			}
		}

	}

	public void bidden(Bid bid, int roundID) {

		for (int i = 0; i < issues.length; i++) {
			if (issues[i] != null) {

				try {
					Value value = bid.getValue(utilitySpace.getIssue(i)
							.getNumber());
					ValueBidData data = issues[i].values.get(value.toString());
					data.lastTimeBidden = roundID;
				} catch (Exception e) {
				}
			}
		}
	}

	public void init(AdditiveUtilitySpace space, ValueModeler model) {
		this.model = model;
		utilitySpace = space;
		int issueCount = utilitySpace.getDomain().getIssues().size();
		issues = new IssueSeperatedBids[issueCount];
		for (int i = 0; i < issueCount; i++) {
			Issue issue = utilitySpace.getDomain().getIssues().get(i);
			if (issue.getType() == ISSUETYPE.DISCRETE) {
				issues[i] = new IssueSeperatedBids();
				IssueDiscrete issueD = (IssueDiscrete) issue;
				int s = issueD.getNumberOfValues();
				for (int j = 0; j < s; j++) {
					String key = issueD.getValue(j).toString();
					issues[i].values.put(key, new ValueBidData());
				}

			} else {
				issues[i] = null;
			}
		}
	}

	public BidWrapper explore(int round) {

		ValueBidData minData = null;
		int minRound = round;
		int maxBidsAvaliable = 1;
		int issueCount = utilitySpace.getDomain().getIssues().size();
		for (int i = 0; i < issueCount; i++) {
			Issue issue = utilitySpace.getDomain().getIssues().get(i);
			if (issue.getType() == ISSUETYPE.DISCRETE) {
				IssueDiscrete issueD = (IssueDiscrete) issue;
				int s = issueD.getNumberOfValues();
				for (int j = 0; j < s; j++) {
					String key = issueD.getValue(j).toString();
					ValueBidData data = issues[i].values.get(key);
					int s2 = data.approvedSorted.bids.size();
					if (s2 > 0
							&& (data.lastTimeBidden < minRound || (data.lastTimeBidden == minRound && maxBidsAvaliable < s2))) {
						minRound = data.lastTimeBidden;
						maxBidsAvaliable = s2;
						minData = data;
					}
				}

			}
		}
		if (minData != null) {
			if (minData.changed == true) {
				minData.approvedSorted.sortByOpponentUtil(model);
				minData.changed = false;
			}
			if (minRound == round)
				return null;
			for (int i = 0; i < minData.approvedSorted.bids.size(); i++) {
				BidWrapper tempBid = minData.approvedSorted.bids.get(i);
				if (!tempBid.sentByUs) {
					return tempBid;
				}
			}
			minData.lastTimeBidden = round;
			// if all bids in this value were taken, trying to explore a
			// different
			// value
			return explore(round);
		}
		return null;
	}

	public void clear() {
		int issueCount = utilitySpace.getDomain().getIssues().size();
		for (int i = 0; i < issueCount; i++) {
			Issue issue = utilitySpace.getDomain().getIssues().get(i);
			if (issue.getType() == ISSUETYPE.DISCRETE) {
				IssueDiscrete issueD = (IssueDiscrete) issue;
				int s = issueD.getNumberOfValues();
				for (int j = 0; j < s; j++) {
					String key = issueD.getValue(j).toString();
					ValueBidData data = issues[i].values.get(key);
					data.approvedSorted.bids.clear();
				}

			}
		}
	}
}