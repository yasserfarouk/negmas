package agents.anac.y2016.caduceus.agents.Caduceus.sanity;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import genius.core.Bid;
import genius.core.issue.Objective;
import genius.core.issue.ValueDiscrete;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.Evaluator;
import genius.core.utility.EvaluatorDiscrete;

/**
 * AI2015Group3Assignment
 * <p/>
 * Created by Taha Doğan Güneş on 28/11/15. Copyright (c) 2015. All rights
 * reserved.
 */
public class SaneUtilitySpace {
	public class IssueSpace {
		HashMap<String, SaneValue> values = new HashMap<String, SaneValue>();
		SaneIssue saneIssue;

		public IssueSpace(SaneIssue saneIssue, ArrayList<SaneValue> values) {
			this.saneIssue = saneIssue;
			for (SaneValue value : values) {
				this.values.put(value.name, value);
			}
		}

		public SaneValue findValue(String name) {
			return values.get(name);
		}

		@Override
		public String toString() {
			String str = "";

			for (Map.Entry<String, SaneValue> entry : values.entrySet()) {
				str += entry.getKey() + ": " + entry.getValue();
			}

			return str;
		}
	}

	public HashMap<String, IssueSpace> saneSpaceMap = new HashMap<String, IssueSpace>(); // Issue
																							// index
																							// (beware
																							// Genius
																							// issue
																							// indexes
																							// may
																							// start
																							// from
																							// 1)
	public HashMap<String, SaneIssue> saneIssueMap = new HashMap<String, SaneIssue>();

	public SaneUtilitySpace() {

	}

	@Override
	public String toString() {
		String str = "";
		for (Map.Entry<String, IssueSpace> entry : saneSpaceMap.entrySet()) {
			IssueSpace space = entry.getValue();
			String issueName = entry.getKey();
			str += issueName + ": \n" + saneIssueMap.get(issueName).weight
					+ " \n";
			str += space.toString();
		}

		return str + "\n";
	}

	public void initZero(AdditiveUtilitySpace space) {
		for (Map.Entry<Objective, Evaluator> entry : space.getEvaluators()) {

			String issueName = ((Objective) entry.getKey()).getName();
			EvaluatorDiscrete evaluator = ((EvaluatorDiscrete) entry.getValue());

			SaneIssue issue = new SaneIssue(0, issueName);
			ArrayList<SaneValue> saneValues = new ArrayList<SaneValue>();

			for (ValueDiscrete valueDiscrete : evaluator.getValues()) {
				String valueName = valueDiscrete.getValue();
				SaneValue saneValue = new SaneValue(valueName, 0);
				saneValues.add(saneValue);
			}

			IssueSpace issueSpace = new IssueSpace(issue, saneValues);
			saneSpaceMap.put(issueName, issueSpace);
			saneIssueMap.put(issueName, issue);
		}
	}

	public void initWithCopy(AdditiveUtilitySpace space) {
		for (Map.Entry<Objective, Evaluator> entry : space.getEvaluators()) {

			double issueWeight = ((Evaluator) entry.getValue()).getWeight();
			String issueName = ((Objective) entry.getKey()).getName();
			EvaluatorDiscrete evaluator = ((EvaluatorDiscrete) entry.getValue());

			SaneIssue issue = new SaneIssue(issueWeight, issueName);
			ArrayList<SaneValue> saneValues = new ArrayList<SaneValue>();

			for (ValueDiscrete valueDiscrete : evaluator.getValues()) {
				String valueName = valueDiscrete.getValue();
				double utility = 0;
				try {
					utility = evaluator.getEvaluation(valueDiscrete);
				} catch (Exception e) {
					e.printStackTrace();
				}
				SaneValue saneValue = new SaneValue(valueName, utility);
				saneValues.add(saneValue);
			}

			IssueSpace issueSpace = new IssueSpace(issue, saneValues);
			saneSpaceMap.put(issueName, issueSpace);
			saneIssueMap.put(issueName, issue);
		}
	}

	public double getBidUtility(SaneBid bid) {
		double utility = 0;

		Iterator<Pair<SaneIssue, SaneValue>> iterator = bid.getIterator();

		while (iterator.hasNext()) {
			Pair<SaneIssue, SaneValue> pair = iterator.next();
			SaneIssue issue = pair.first;
			SaneValue value = pair.second;

			utility += saneIssueMap.get(issue.name).weight
					* saneSpaceMap.get(issue.name).findValue(value.name).utility;
		}

		return utility;
	}

	// public double getBidUtility(Bid uglyBid) {
	// SaneBid bid = new SaneBid(uglyBid, this);
	//
	// return this.getBidUtility(bid);
	// }

	public double getDiscountedUtility(Bid uglyBid, double discountFactor,
			double time) {
		SaneBid bid = new SaneBid(uglyBid, this);

		return this.getBidUtility(bid) * Math.pow(discountFactor, time);
	}

	public void normalize() {
		double issueSum = 0;
		for (Map.Entry<String, IssueSpace> entry : saneSpaceMap.entrySet()) {
			String issueName = entry.getKey();
			SaneIssue issue = saneIssueMap.get(issueName);

			double valueSum = 0;

			IssueSpace issueSpace = entry.getValue();

			for (Map.Entry<String, SaneValue> valueEntry : issueSpace.values
					.entrySet()) {
				valueSum += valueEntry.getValue().utility;
			}

			for (Map.Entry<String, SaneValue> valueEntry : issueSpace.values
					.entrySet()) {
				SaneValue saneValue = valueEntry.getValue();
				saneValue.utility /= valueSum;
			}

			issueSum += issue.weight;
		}

		for (Map.Entry<String, IssueSpace> entry : saneSpaceMap.entrySet()) {
			String issueName = entry.getKey();
			SaneIssue issue = saneIssueMap.get(issueName);
			issue.weight /= issueSum;
		}
	}

}
