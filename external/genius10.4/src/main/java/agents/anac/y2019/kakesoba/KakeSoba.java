package agents.anac.y2019.kakesoba;

import genius.core.Bid;
import genius.core.BidIterator;
import genius.core.Domain;
import genius.core.actions.*;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.ValueDiscrete;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.uncertainty.AdditiveUtilitySpaceFactory;
import genius.core.uncertainty.BidRanking;
import genius.core.utility.AbstractUtilitySpace;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.EvaluatorDiscrete;
import genius.core.timeline.Timeline.Type;

import java.util.*;

/**
 * KakeSoba
 */
public class KakeSoba extends AbstractNegotiationParty {
	private int nrChosenActions = 0; // number of times chosenAction was called.
	private boolean isFirstParty;
	private boolean isDebug = false;
	private boolean isInvalid = false;
	private Map<String, Map<String, Double>> counts = new HashMap<String, Map<String, Double>>();

	@Override
	public void init(NegotiationInfo info) {
		super.init(info);

		if (!(utilitySpace instanceof AdditiveUtilitySpace)) {
			System.out.println("This agent displays more interesting behavior with a additive utility function; now it simply generates random bids.");
			isInvalid = true;
			return;
		}

		AdditiveUtilitySpace additiveUtilitySpace = (AdditiveUtilitySpace) utilitySpace;
		for (Issue issue : additiveUtilitySpace.getDomain().getIssues()) {
			if (!(issue instanceof IssueDiscrete)) {
				System.out.println("This agent displays more interesting behavior with a discrete issue domain; now it simply generates random bids.");
				isInvalid = true;
				return;
			}
			Map<String, Double> h = new HashMap<String, Double>();
			for (ValueDiscrete value : ((IssueDiscrete)issue).getValues()) {
				h.put(value.getValue(), 0.0D);
			}
			counts.put(issue.getName(), h);
		}
	}

	public Action chooseAction(List<Class<? extends Action>> validActions) {
		double t = this.getTimeLine().getTime();
		nrChosenActions++;

		if (this.isInvalid) {
			return new Offer(getPartyId(), generateRandomBid());
		}

		if (!(getLastReceivedAction() instanceof Offer) && !(getLastReceivedAction() instanceof Accept)){
			isFirstParty = true;
		}

		if (getLastReceivedAction() instanceof Offer) {
			Bid receivedBid = ((Offer) getLastReceivedAction()).getBid();

			if (isAcceptableBid(receivedBid, t)) {
				return new Accept(getPartyId(), receivedBid);
			}

			if (timeline.getType() == Type.Rounds) {
				if (!isFirstParty && timeline.getCurrentTime() >= timeline.getTotalTime() - 1) {
					if (this.utilitySpace.getUtilityWithDiscount(receivedBid, t) > this.utilitySpace.getReservationValueWithDiscount(t)) {
						return new Accept(getPartyId(), receivedBid);
					}
				}
			}
		}

		if (isDebug) {
			System.out.println(getPartyId() + ": " + getError());
			System.out.println(getPartyId() + ": " + counts);
			System.out.println(getPartyId() + ": " + utilitySpace);
		}

		if(nrChosenActions == 1) {
			try {
				Bid bid = getUtilitySpace().getMaxUtilityBid();
				countBid(bid);
				return new Offer(getPartyId(), bid);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}

		Bid bid = generateBid(t);
		countBid(bid);
		return new Offer(getPartyId(), bid);
	}

	private void countBid(Bid bid, double weight) {
		for (Issue issue : bid.getIssues()) {
			Map<String, Double> h = counts.get(issue.getName());
			String value = ((ValueDiscrete) bid.getValue(issue)).getValue();
			h.put(value, h.get(value) + weight);
		}
	}

	private void countBid(Bid bid) { countBid(bid, 1.0D); }

	private double getError() {
		double error = 0.0D;

		AdditiveUtilitySpace additiveUtilitySpace = (AdditiveUtilitySpace) utilitySpace;
		for (Issue issue : additiveUtilitySpace.getDomain().getIssues()) {
			Map<String, Double> h = counts.get(issue.getName());
			EvaluatorDiscrete evaluator = (EvaluatorDiscrete) additiveUtilitySpace.getEvaluator(issue);
			double max = h.values().stream().mapToDouble(Double::doubleValue).max().getAsDouble();
			for (ValueDiscrete value : ((IssueDiscrete) issue).getValues()) {
				error += Math.abs((h.get(value.getValue()) / max) - evaluator.getDoubleValue(value));
			}
		}
		return error;
	}

	private double getErrorWithNewBid(Bid bid) {
		double error;

		countBid(bid, 1.0D);
		error = getError();
		countBid(bid, -1.0D);

		return error;
	}

	private Bid generateBid(double t) {
		BidIterator bidIterator = new BidIterator(this.getDomain());

		Bid bestBid;
		try {
			bestBid = utilitySpace.getMaxUtilityBid();
		} catch (Exception e) {
			bestBid = bidIterator.next();
		}
		double minError = getErrorWithNewBid(bestBid);
		while (bidIterator.hasNext()) {
			Bid bid = bidIterator.next();
			if (!isProposableBid(bid, t)) {
				continue;
			}
			double error = getErrorWithNewBid(bid);
			if (error < minError) {
				bestBid = bid;
				minError = error;
			}
		}
		return bestBid;
	}

	private boolean isProposableBid(Bid bid, double t) {
		double utility ;
		try {
			utility = utilitySpace.getUtilityWithDiscount(bid, t);
		} catch (Exception e) {
			utility = -1.0D;
		}

		return getLowerBound(t) <= utility && utility <= getUpperBound(t) &&  utility >= utilitySpace.getReservationValueWithDiscount(t);
	}

	private boolean isAcceptableBid(Bid bid, double t) {
		double utility ;
		try {
			utility = utilitySpace.getUtilityWithDiscount(bid, t);
		} catch (Exception e) {
			utility = -1.0D;
		}

		return getLowerBound(t) <= utility &&  utility >= utilitySpace.getReservationValueWithDiscount(t);
	}

	private double getUpperBound(double t) {
		return 1.0D;
	}

	private double getLowerBound(double t) {
		return 0.85D;
	}

	public AbstractUtilitySpace estimateUtilitySpace() {
		List<Movement> TabuList = new ArrayList<Movement>();
		AdditiveUtilitySpace additiveUtilitySpace = generateRandomUtilitySpace();
		AdditiveUtilitySpace hallOfFame = additiveUtilitySpace;
		double hallOfFameScore = getScore(hallOfFame, false);

		if (this.isInvalid) {
			return defaultUtilitySpaceEstimator(getDomain(), userModel);
		}

		int domainSize = 0;
		for (Issue issue : this.getDomain().getIssues()) {
			domainSize += ((IssueDiscrete) issue).getValues().size() + 1;
		}

		int numOfMovement = 5000;
		final double wightRate = this.getDomain().getIssues().size() * 1.0D / domainSize;

		for (int i = 0; i < numOfMovement; i ++) {
			Map<Movement, AdditiveUtilitySpace> moveToNeighbors = new HashMap<Movement, AdditiveUtilitySpace>();

			for (int j = 0; j < domainSize; j ++) {
				Movement movement = new Movement(this.getDomain(), wightRate);
				while (TabuList.contains(movement)) {
					movement = new Movement(this.getDomain(), wightRate);
				}
				moveToNeighbors.put(movement, getNeighbor(additiveUtilitySpace, movement));
			}

			Iterator<Map.Entry<Movement, AdditiveUtilitySpace>> iterator = moveToNeighbors.entrySet().iterator();
			Map.Entry<Movement, AdditiveUtilitySpace> bestEntry = iterator.next();
			double bestScore = -100.0D;
			while (iterator.hasNext()) {
				Map.Entry<Movement, AdditiveUtilitySpace> entry = iterator.next();
				double score = getScore(entry.getValue(), false);
				if (score > bestScore) {
					bestEntry = entry;
					bestScore = score;
				}
			}

			additiveUtilitySpace = bestEntry.getValue();
			if (bestScore > hallOfFameScore) {
				hallOfFame = additiveUtilitySpace;
				hallOfFameScore = bestScore;
			}

			TabuList.add(bestEntry.getKey());
			if (TabuList.size() > Math.sqrt(domainSize) / 2) {
				TabuList.remove(0);
			}

			if (isDebug) {
				getScore(additiveUtilitySpace, true);
			}
		}

		if (isDebug) {
			getScore(additiveUtilitySpace, true);
		}

		return hallOfFame;
	}

	private double getScore(AdditiveUtilitySpace additiveUtilitySpace, boolean isPrint) {
		BidRanking bidRank = this.userModel.getBidRanking();

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


	private AdditiveUtilitySpace generateRandomUtilitySpace() {
		AdditiveUtilitySpaceFactory additiveUtilitySpaceFactory = new AdditiveUtilitySpaceFactory(this.getDomain());

		for (IssueDiscrete issue : additiveUtilitySpaceFactory.getIssues()) {
			additiveUtilitySpaceFactory.setWeight(issue, this.rand.nextDouble());

			for (ValueDiscrete value : issue.getValues()) {
				additiveUtilitySpaceFactory.setUtility(issue, value, this.rand.nextDouble());
			}
		}

		normalize(additiveUtilitySpaceFactory);
		return additiveUtilitySpaceFactory.getUtilitySpace();
	}

	private AdditiveUtilitySpace getNeighbor(AdditiveUtilitySpace current, Movement movement) {
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

	private AdditiveUtilitySpaceFactory getAdditiveUtilitySpaceFactoryFrom(AdditiveUtilitySpace additiveUtilitySpace){
		AdditiveUtilitySpaceFactory additiveUtilitySpaceFactory = new AdditiveUtilitySpaceFactory(getDomain());
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

	public String getDescription() {
		return "KakeSoba";
	}


	private class Movement {
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
