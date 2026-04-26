package agents.anac.y2015.TUDMixedStrategyAgent;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.BidHistory;
import genius.core.issue.Issue;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.utility.EvaluatorDiscrete;

public class AgentUtils {

	AgentID agent;
	BidHistory bidHistory;
	ArrayList<EvaluatorDiscrete> issueList;
	int nIssues;
	int[] issueIds;

	public AgentUtils(AgentID agent, BidHistory bidHistory, int nIssues) {
		super();
		this.agent = agent;
		this.bidHistory = bidHistory;
		this.nIssues = nIssues;
		issueIds = new int[nIssues];
		issueList = new ArrayList<EvaluatorDiscrete>();
		for (int i = 0; i < nIssues; i++) {
			issueList.add(new EvaluatorDiscrete());
		}
	}

	// Calculates the utility of this bid for the Agent
	public double getAgentUtil(Bid bid) {
		double utility = 0;
		// This HashMap connects the issues IDs to its values
		HashMap<Integer, Value> bidValuesHashMap = bid.getValues();
		for (int i = 0; i < nIssues; i++) {
			double weight = issueList.get(i).getWeight();
			ValueDiscrete VD = (ValueDiscrete) bidValuesHashMap
					.get(issueIds[i]);
			if (issueList.get(i).getValues().contains(VD))
				utility = utility + issueList.get(i).getDoubleValue(VD)
						* weight;
		}
		return utility;
	}

	// based on the most recent bids received recalculates the issue and value
	// weights
	public void recalculateUtilFunction() {
		List<Issue> bidIssueList = bidHistory.getHistory().get(0).getBid()
				.getIssues();
		for (int i = 0; i < bidIssueList.size(); i++) {
			issueIds[i] = bidIssueList.get(i).getNumber();
		}

		double[] bidWeights = calculateBidWeights();
		int[] nChanges = calculateNChanges();

		for (int i = 0; i < nIssues; i++) {
			issueList.get(i).clear();
			// First the weight of each issue
			issueList.get(i).setWeight(calculateIssueWeight(i, nChanges));
			ArrayList<ValueDiscrete> valueList = new ArrayList<ValueDiscrete>();
			double[] valueEvaluation = new double[100]; // random large number,
														// just so we don't have
														// to use a List
			for (int j = 0; j < bidHistory.getHistory().size(); j++) {
				ValueDiscrete currentValue = (ValueDiscrete) bidHistory
						.getHistory().get(j).getBid().getValues()
						.get(bidIssueList.get(i).getNumber());
				if (!valueList.contains(currentValue)) {
					valueList.add(currentValue);
				}
				valueEvaluation[(valueList.indexOf(currentValue))] = valueEvaluation[valueList
						.indexOf(currentValue)] + bidWeights[j];
			}
			int k = 0;
			do {
				try {
					issueList.get(i).setEvaluationDouble(valueList.get(k),
							valueEvaluation[k]);
				} catch (Exception e) {
					e.printStackTrace();
				}
				k++;
			} while (valueEvaluation[k] != 0);
		}
	}

	// Returns the number of changes in values for each issue through the entire
	// bid history
	private int[] calculateNChanges() {
		int[] nChanges = new int[nIssues];
		Arrays.fill(nChanges, 1);
		Value oldValue = null;
		for (int i = 0; i < nIssues; i++) {
			for (int j = 0; j < bidHistory.getHistory().size(); j++) {
				try {
					if (oldValue != null
							&& !oldValue.equals(bidHistory.getHistory().get(j)
									.getBid().getValue(issueIds[i]))) {
						nChanges[i]++;
					}
					oldValue = bidHistory.getHistory().get(j).getBid()
							.getValue(issueIds[i]);
				} catch (Exception e) {
					e.printStackTrace();
				}

			}
			oldValue = null;
		}
		return nChanges;
	}

	// Calculates Issue i weight, based on the number of changes
	private double calculateIssueWeight(int i, int[] nChanges) {
		double weight = 0;
		for (int j = 0; j < nIssues; j++) {
			weight = weight + 1 / nChanges[j];
		}
		return 1 / nChanges[i] * 1 / weight;
	}

	// Gives a weight to each bid we received, the first bids are worth much
	// more than the last ones
	private double[] calculateBidWeights() {
		int N = bidHistory.getHistory().size();
		double factor = 0;
		for (int i = 1; i <= N; i++) {
			factor = factor + 1 / i;
		}
		double[] bidWeights = new double[N];
		for (int i = 1; i <= N; i++) {
			bidWeights[i - 1] = 1 / i * 1 / factor;
		}
		return bidWeights;
	}

}
