package genius.core.uncertainty;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import genius.core.Bid;
import genius.core.Domain;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.ValueDiscrete;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.EvaluatorDiscrete;
import java.util.Random;


/**
 * This class contains helper methods used in the estimateUtilitySpace function kS_Movement (which can be found in EstimateUtilityLibrary).
 *
 */

public class Helper_KS_Movement {
	
	private Domain domain;
	private BidRanking bidRank;
	protected Random rand = new Random();
	
	public Helper_KS_Movement(Domain domain, BidRanking bidRank){
		this.domain = domain;
		this.bidRank = bidRank;
	}
	
	public double getScore(AdditiveUtilitySpace additiveUtilitySpace, boolean isPrint) {

		Map<Bid, Integer> realRanks = new HashMap<Bid, Integer>();
		List<Double> estimatedUtils = new ArrayList<Double>();
		for (Bid bid : bidRank.getBidOrder()) {
			realRanks.put(bid, realRanks.size());
			estimatedUtils.add(additiveUtilitySpace.getUtility(bid));
		}
		Collections.sort(estimatedUtils);

		Map<Bid, Integer> estimatedRanks = new HashMap<Bid, Integer>();
		for (Bid bid : bidRank.getBidOrder()) {
			estimatedRanks.put(bid, estimatedUtils.indexOf(additiveUtilitySpace.getUtility(bid)));
		}

		double errors = 0;
		for (Bid bid : bidRank.getBidOrder()) {
			errors += Math.pow(realRanks.get(bid) - estimatedRanks.get(bid), 2);
		}

		double spearman = 1.0D - 6.0D * errors / (Math.pow(realRanks.size(), 3) - realRanks.size());
		double lowDiff = Math.abs(bidRank.getLowUtility().doubleValue() - additiveUtilitySpace.getUtility(bidRank.getMinimalBid()));
		double highDiff = Math.abs(bidRank.getHighUtility().doubleValue() - additiveUtilitySpace.getUtility(bidRank.getMaximalBid()));
		if(isPrint) {
			System.out.println("spearman = " + spearman + ", lowDiff = " + lowDiff + ", highDiff = " + highDiff);
		}

		return spearman * 10.0D + (1.0D - lowDiff) + (1.0D - highDiff);
	}


	public AdditiveUtilitySpace generateRandomUtilitySpace() {
		AdditiveUtilitySpaceFactory additiveUtilitySpaceFactory = new AdditiveUtilitySpaceFactory(domain);
		for (IssueDiscrete issue : additiveUtilitySpaceFactory.getIssues()) {
			additiveUtilitySpaceFactory.setWeight(issue, this.rand.nextDouble());
			for (ValueDiscrete value : issue.getValues()) {
				additiveUtilitySpaceFactory.setUtility(issue, value, this.rand.nextDouble());
			}
		}

		normalize(additiveUtilitySpaceFactory);
		return additiveUtilitySpaceFactory.getUtilitySpace();
	}

	public AdditiveUtilitySpace getNeighbor(AdditiveUtilitySpace current, Movement movement) {
		double speed = 1.0D;
		AdditiveUtilitySpaceFactory neighbor = getAdditiveUtilitySpaceFactoryFrom(current);

		IssueDiscrete issue = neighbor.getIssues().get(movement.getIssueID());
		if (movement.getIsWeight()) {
			if (rand.nextBoolean()) {
				neighbor.setWeight(issue, current.getWeight(issue) + speed * rand.nextDouble() / neighbor.getIssues().size());
			} else {
				neighbor.setWeight(issue, Math.abs(current.getWeight(issue) - speed * rand.nextDouble() / neighbor.getIssues().size()));
			}

		} else {
			double averageEval = 0;
			for (ValueDiscrete v : issue.getValues()) {
				averageEval += ((EvaluatorDiscrete)current.getEvaluator(issue)).getDoubleValue(v);
			}
			averageEval /= issue.getValues().size();

			ValueDiscrete value = issue.getValue(movement.getValueID());
			double eval = ((EvaluatorDiscrete)current.getEvaluator(issue)).getDoubleValue(value);
			if (rand.nextBoolean()) {
				neighbor.setUtility(issue, value, eval + speed * averageEval * rand.nextDouble());
			} else {
				neighbor.setUtility(issue, value, Math.abs(eval - speed * averageEval * rand.nextDouble()));
			}
		}

		normalize(neighbor);
		return neighbor.getUtilitySpace();
	}

	public AdditiveUtilitySpaceFactory getAdditiveUtilitySpaceFactoryFrom(AdditiveUtilitySpace additiveUtilitySpace){
		AdditiveUtilitySpaceFactory additiveUtilitySpaceFactory = new AdditiveUtilitySpaceFactory(domain);
		for (IssueDiscrete issue : additiveUtilitySpaceFactory.getIssues()) {
			additiveUtilitySpaceFactory.setWeight(issue, additiveUtilitySpace.getWeight(issue));
			for (ValueDiscrete value : issue.getValues()) {
				additiveUtilitySpaceFactory.setUtility(issue, value, ((EvaluatorDiscrete) additiveUtilitySpace.getEvaluator(issue)).getDoubleValue(value));
			}
		}
		return additiveUtilitySpaceFactory;
	}

	private void normalize(AdditiveUtilitySpaceFactory additiveUtilitySpaceFactory){
		additiveUtilitySpaceFactory.normalizeWeights();

		for (IssueDiscrete issue : additiveUtilitySpaceFactory.getIssues()) {
			double max = issue.getValues().stream().map(value -> additiveUtilitySpaceFactory.getUtility(issue, value)).max(Double::compareTo).get();
			for (ValueDiscrete value : issue.getValues()) {
				double eval = additiveUtilitySpaceFactory.getUtility(issue, value);
				additiveUtilitySpaceFactory.setUtility(issue, value, eval / max);
			}
		}
	}

	public class Movement {
		private int issueID;
		private int valueID;
		private boolean isWeight;

		public Movement(int issueID, int valueID) {
			this.issueID = issueID;
			this.valueID = valueID;
			this.isWeight = false;
		}

		public Movement(int issueID) {
			this.issueID = issueID;
			this.valueID = 0;
			this.isWeight = true;
		}

		public Movement(Domain domain, double wightRate) {
			List<Issue> issues = domain.getIssues();
			this.issueID = rand.nextInt(issues.size());

			if (rand.nextDouble() > wightRate) {
				this.isWeight = true;
				this.valueID = 0;
			} else {
				this.isWeight = false;
				this.valueID = rand.nextInt(((IssueDiscrete) issues.get(this.issueID)).getNumberOfValues());
			}
		}

		private int getIssueID() {
			return this.issueID;
		}

		private int getValueID() {
			return this.valueID;
		}

		private boolean getIsWeight() {
			return this.isWeight;
		}
	}
	
	
}
