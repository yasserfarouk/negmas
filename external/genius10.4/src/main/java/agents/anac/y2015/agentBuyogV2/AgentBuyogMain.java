package agents.anac.y2015.agentBuyogV2;

import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;
import java.util.Random;

import agents.anac.y2015.agentBuyogV2.flanagan.analysis.Regression;
import genius.core.AgentID;
import genius.core.Bid;
import genius.core.BidHistory;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.EndNegotiation;
import genius.core.actions.Offer;
import genius.core.analysis.BidSpace;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.SortedOutcomeSpace;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.Objective;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.issue.ValueInteger;
import genius.core.misc.Range;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.Evaluator;
import genius.core.utility.EvaluatorDiscrete;
import genius.core.utility.EvaluatorInteger;

public class AgentBuyogMain extends AbstractNegotiationParty {

	// PARAMETER LIST. THIS IS WHAT YOU'RE SUPPOSED TO PLAY AROUND WITH.

	private static final double alphaDefault = 0, betaDefault = 0,
			issueWeightsConstant = 0.3, issueValuesConstant = 100,
			minimumHistorySize = 50, learningTimeController = 1.3; // Discount
																	// is
																	// added
																	// to
																	// the
																	// learningTimeController.
																	// So
																	// it's
																	// actually
																	// learningTimeController
																	// +
																	// discount
	private static final int maxWeightForBidPoint = 300;
	private static final double leniencyAdjuster = 1,
			domainWeightController = 1.75, timeConcessionController = 1.8;
	private static final double lastSecondConcessionFactor = 0.5;
	private static final double kalaiPointCorrection = 0.1;

	// PARAMETER LIST ENDS HERE. DO NOT TOUCH CODE BELOW THIS POINT.

	private OpponentInfo infoA, infoB;
	private BidHistory myBidHistory, AandBscommonBids, totalHistory;
	private boolean initialized = false;
	private SortedOutcomeSpace sortedUtilitySpace;
	private int numberOfRounds = 0;

	@Override
	public void init(NegotiationInfo info) {
		super.init(info);
		this.totalHistory = new BidHistory();
		this.myBidHistory = new BidHistory();
		this.AandBscommonBids = new BidHistory();
		this.sortedUtilitySpace = new SortedOutcomeSpace(utilitySpace);

	}

	@Override
	public Action chooseAction(List<Class<? extends Action>> possibleActions) {

		if (timeline.getTime() >= 1) {
			return new EndNegotiation(getPartyId());
		}

		numberOfRounds++;
		double timePerRound = timeline.getTime() / numberOfRounds;
		double remainingRounds = (1 - timeline.getTime()) / timePerRound;

		BidDetails bestBid = null;
		double minimumPoint = utilitySpace.getDiscountFactor() * 0.7
				+ utilitySpace.getReservationValueUndiscounted() * 0.3;

		BidDetails bestAgreeableBidSoFar = null;
		double bestAgreeableBidsUtility = 0;
		double mostRecentBidsUtility = 0;

		if (AandBscommonBids != null && AandBscommonBids.size() > 0) {
			bestAgreeableBidSoFar = AandBscommonBids.getBestBidDetails();
			bestAgreeableBidsUtility = AandBscommonBids.getBestBidDetails()
					.getMyUndiscountedUtil();
		}

		if (totalHistory != null && totalHistory.size() > 0) {
			mostRecentBidsUtility = totalHistory.getLastBidDetails()
					.getMyUndiscountedUtil();
		}

		OpponentInfo difficultAgent = null;

		if (infoA != null && infoB != null && infoA.getAgentDifficulty() != null
				&& infoB.getAgentDifficulty() != null) {

			if (infoA.getAgentDifficulty() <= infoB.getAgentDifficulty()) {
				difficultAgent = infoA;
			} else {
				difficultAgent = infoB;
			}

			minimumPoint = utilitySpace.getDiscountFactor()
					* difficultAgent.getAgentDifficulty();
		}

		double acceptanceThreshold = minimumPoint + (1 - minimumPoint)
				* (1 - Math.pow(timeline.getTime(), timeConcessionController));

		if (remainingRounds <= 3) {
			acceptanceThreshold = acceptanceThreshold
					* lastSecondConcessionFactor;
		}

		if (acceptanceThreshold < utilitySpace
				.getReservationValueUndiscounted()) {
			acceptanceThreshold = utilitySpace
					.getReservationValueUndiscounted();

			if (utilitySpace.getDiscountFactor() < 1 && remainingRounds > 3) {
				return new EndNegotiation(getPartyId());
			}
		}

		if (possibleActions.contains(Accept.class)) {
			if (mostRecentBidsUtility >= acceptanceThreshold
					&& mostRecentBidsUtility >= bestAgreeableBidsUtility
					&& remainingRounds <= 3) {
				return new Accept(getPartyId(),
						totalHistory.getLastBidDetails().getBid());
			}
		}

		if (bestAgreeableBidsUtility > acceptanceThreshold) {
			bestBid = bestAgreeableBidSoFar;
		} else {
			Range range = new Range(acceptanceThreshold, 1);
			List<BidDetails> bidsInWindow = sortedUtilitySpace
					.getBidsinRange(range);
			bestBid = getBestBidFromList(bidsInWindow);
		}

		if (possibleActions.contains(Accept.class)) {
			if (mostRecentBidsUtility >= acceptanceThreshold
					&& mostRecentBidsUtility >= bestAgreeableBidsUtility
					&& mostRecentBidsUtility >= bestBid
							.getMyUndiscountedUtil()) {
				return new Accept(getPartyId(),
						totalHistory.getLastBidDetails().getBid());
			}
		}

		totalHistory.add(bestBid);
		return new Offer(getPartyId(), bestBid.getBid());

	}

	private BidDetails getBestBidFromList(List<BidDetails> bidsInWindow) {
		double bestBidOpponentsUtil = 0;
		BidDetails bestBid = null;

		if (infoA == null || infoB == null || infoA.getAgentDifficulty() == null
				|| infoB.getAgentDifficulty() == null) {
			Random random = new Random();
			return bidsInWindow.get(random.nextInt(bidsInWindow.size()));
		}

		try {
			bestBid = findNearestBidToKalai(bidsInWindow, 1D, 1D);
		} catch (Exception e) {
			e.printStackTrace();
		}

		return bestBid;
	}

	private BidDetails findNearestBidToKalai(List<BidDetails> bidsInWindow,
			Double infoAKalai, Double infoBKalai) {
		Double shortestDistance;
		BidDetails nearestBid;

		nearestBid = bidsInWindow.get(0);
		shortestDistance = getDistance(nearestBid, infoAKalai, infoBKalai);

		for (BidDetails bid : bidsInWindow) {
			Double bidDistance = getDistance(bid, infoAKalai, infoBKalai);
			if (bidDistance < shortestDistance) {
				shortestDistance = bidDistance;
				nearestBid = bid;
			}
		}
		return nearestBid;
	}

	private Double getDistance(BidDetails bid, Double infoAKalai,
			Double infoBKalai) {
		try {
			return Math.sqrt((1 - infoA.getAgentDifficulty())
					* Math.pow(infoA.getOpponentUtilitySpace()
							.getUtility(bid.getBid()) - infoAKalai, 2)
					+ (1 - infoB.getAgentDifficulty())
							* Math.pow(infoB.getOpponentUtilitySpace()
									.getUtility(bid.getBid()) - infoBKalai, 2));
		} catch (Exception e) {
			e.printStackTrace();
		}
		return 0D;
	}

	@Override
	public void receiveMessage(AgentID sender, Action action) {

		super.receiveMessage(sender, action);

		if (sender == null
				|| !((action instanceof Offer) || (action instanceof Accept))) {
			return;
		}

		Bid bid = null;

		if (!initialized) {
			initializeOpponentInfo(sender);
		}

		if (action instanceof Offer) {
			bid = ((Offer) action).getBid();
			try {
				totalHistory.add(new BidDetails(bid,
						utilitySpace.getUtility(bid), timeline.getTime()));
			} catch (Exception e) {
				e.printStackTrace();
			}
		} else if (action instanceof Accept) {
			bid = totalHistory.getLastBid();
		}

		OpponentInfo senderInfo = getOpponentInfoObjectOfSender(sender);
		OpponentInfo otherInfo = getOpponentInfoObjectOfOther(sender);
		updateOpponentBidHistory(senderInfo, bid);
		updateCommonBids(otherInfo, bid);
		updateOpponentModel(senderInfo);
	}

	private void updateCommonBids(OpponentInfo otherInfo, Bid bid) {
		if (otherInfo == null) {
			return;
		}
		if (otherInfo.containsBid(bid)) {
			try {
				this.AandBscommonBids
						.add(new BidDetails(bid, utilitySpace.getUtility(bid)));
			} catch (Exception e) {
				e.printStackTrace();
			}
		}

	}

	private OpponentInfo getOpponentInfoObjectOfOther(Object sender) {

		if (infoA != null && infoA.getAgentID().equals(sender.toString())) {
			return infoB;
		} else if (infoB != null
				&& infoB.getAgentID().equals(sender.toString())) {
			return infoA;
		}

		return null;

	}

	private void updateOpponentModel(OpponentInfo senderInfo) {

		if (senderInfo == null) {
			return;
		}

		// CODE TO LEARN OPPONENT CONCESSION
		if (senderInfo.getAgentBidHistory().size() >= minimumHistorySize) {
			LearningFunction function = new LearningFunction(
					senderInfo.getAgentBidHistory().getHistory().get(0)
							.getMyUndiscountedUtil());
			double xData[] = new double[senderInfo.getBestBids().size()];
			double yData[] = new double[senderInfo.getBestBids().size()];
			double yWeights[] = new double[senderInfo.getBestBids().size()];
			double step[] = { 0.005, 0.005 };
			double initialEstimates[] = { alphaDefault, betaDefault };
			for (int i = 0; i < senderInfo.getBestBids().size(); i++) {
				xData[i] = senderInfo.getBestBids().getHistory().get(i)
						.getTime();
				yData[i] = senderInfo.getBestBids().getHistory().get(i)
						.getMyUndiscountedUtil();
				yWeights[i] = senderInfo.getBidPointWeights().get(i);
			}
			Regression regression = new Regression(xData, yData, yWeights);

			regression.simplex(function, initialEstimates, step);

			double[] bestEstimates = regression.getBestEstimates();
			double alpha = bestEstimates[0];
			double beta = bestEstimates[1];
			double slopeStandardScale = Math.pow(Math.E, alpha) * beta
					* Math.pow(timeline.getTime(), beta - 1);
			double slopeFromZeroToOne = Math.atan(slopeStandardScale)
					/ (Math.PI / 2);
			double adjustedLeniency = slopeFromZeroToOne
					+ slopeFromZeroToOne / leniencyAdjuster;
			if (adjustedLeniency > 1) {
				adjustedLeniency = 1;
			}
			senderInfo.setLeniency(adjustedLeniency);
		} else {
			senderInfo.setLeniency(-1D);
		}

		// CODE TO LEARN OPPONENT PREFERENCES
		AdditiveUtilitySpace opponentUtilitySpace = senderInfo
				.getOpponentUtilitySpace();

		if (senderInfo.getAgentBidHistory().size() < 2) {
			return;
		}

		int numberOfUnchanged = 0;
		int numberOfIssues = opponentUtilitySpace.getDomain().getIssues()
				.size();
		BidHistory opponentsBidHistory = senderInfo.getAgentBidHistory();
		BidDetails opponentsLatestBid = opponentsBidHistory.getLastBidDetails();
		BidDetails opponentsSecondLastBid = opponentsBidHistory.getHistory()
				.get(opponentsBidHistory.size() - 2);
		HashMap<Integer, Boolean> changed = determineDifference(senderInfo,
				opponentsSecondLastBid, opponentsLatestBid);

		for (Boolean hasChanged : changed.values()) {
			if (!hasChanged) {
				numberOfUnchanged++;
			}
		}

		double goldenValue = issueWeightsConstant
				* (1 - (Math.pow(timeline.getTime(),
						learningTimeController
								+ utilitySpace.getDiscountFactor())))
				/ numberOfIssues;
		double totalSum = 1D + goldenValue * numberOfUnchanged;
		double maximumWeight = 1D - (numberOfIssues) * goldenValue / totalSum;

		for (Integer issueNumber : changed.keySet()) {
			if (!changed.get(issueNumber) && opponentUtilitySpace
					.getWeight(issueNumber) < maximumWeight) {
				opponentUtilitySpace.setWeight(
						opponentUtilitySpace.getDomain().getObjectivesRoot()
								.getObjective(issueNumber),
						(opponentUtilitySpace.getWeight(issueNumber)
								+ goldenValue) / totalSum);
			} else {
				opponentUtilitySpace.setWeight(
						opponentUtilitySpace.getDomain().getObjectivesRoot()
								.getObjective(issueNumber),
						opponentUtilitySpace.getWeight(issueNumber) / totalSum);
			}
		}

		try {
			for (Entry<Objective, Evaluator> issueEvaluatorEntry : opponentUtilitySpace
					.getEvaluators()) {
				if (issueEvaluatorEntry.getKey() instanceof IssueDiscrete) {
					((EvaluatorDiscrete) issueEvaluatorEntry.getValue())
							.setEvaluation(
									opponentsLatestBid.getBid().getValue(
											((IssueDiscrete) issueEvaluatorEntry
													.getKey()).getNumber()),
									(int) (issueValuesConstant
											* (1 - Math.pow(timeline.getTime(),
													learningTimeController
															+ utilitySpace
																	.getDiscountFactor()))
											+ ((EvaluatorDiscrete) issueEvaluatorEntry
													.getValue())
															.getEvaluationNotNormalized(
																	(ValueDiscrete) opponentsLatestBid
																			.getBid()
																			.getValue(
																					((IssueDiscrete) issueEvaluatorEntry
																							.getKey())
																									.getNumber()))));
				} else if (issueEvaluatorEntry
						.getKey() instanceof IssueInteger) {
					int issueNumber = ((IssueInteger) issueEvaluatorEntry
							.getKey()).getNumber();
					Value opponentsLatestValueForIssue = opponentsLatestBid
							.getBid().getValue(issueNumber);
					int opponentsLatestValueForIssueAsInteger = ((ValueInteger) opponentsLatestValueForIssue)
							.getValue();

					int upperBound = ((IssueInteger) issueEvaluatorEntry
							.getKey()).getUpperBound();
					int lowerBound = ((IssueInteger) issueEvaluatorEntry
							.getKey()).getLowerBound();
					double midPoint = Math.ceil(
							lowerBound + (upperBound - lowerBound) / 2) + 1;

					if (midPoint > opponentsLatestValueForIssueAsInteger) {
						double distanceFromMidPoint = midPoint
								- opponentsLatestValueForIssueAsInteger;
						double normalizedDistanceFromMidPoint = distanceFromMidPoint
								/ (midPoint - lowerBound);

						double total = 1;
						double newLowEndEvaluation = ((EvaluatorInteger) issueEvaluatorEntry
								.getValue()).getEvaluation(lowerBound)
								+ (issueValuesConstant / 10000)
										* normalizedDistanceFromMidPoint
										* (1 - Math.pow(timeline.getTime(),
												learningTimeController
														+ utilitySpace
																.getDiscountFactor()));
						double highEndEvaluation = ((EvaluatorInteger) issueEvaluatorEntry
								.getValue()).getEvaluation(upperBound);

						if (newLowEndEvaluation > 1) {
							total = newLowEndEvaluation + highEndEvaluation;
						}

						((EvaluatorInteger) issueEvaluatorEntry.getValue())
								.setLinearFunction(newLowEndEvaluation / total,
										highEndEvaluation / total);
					} else {
						double distanceFromMidPoint = opponentsLatestValueForIssueAsInteger
								- midPoint + 1; // because
												// midPoint
												// is
												// included
												// and
												// I
												// don't
												// want
												// a
												// 0
												// value.
						double normalizedDistanceFromMidPoint = distanceFromMidPoint
								/ (upperBound - midPoint + 1);

						double total = 1;
						double newHighEndEvaluation = ((EvaluatorInteger) issueEvaluatorEntry
								.getValue()).getEvaluation(upperBound)
								+ (issueValuesConstant / 10000)
										* normalizedDistanceFromMidPoint
										* (1 - Math.pow(timeline.getTime(),
												learningTimeController
														+ utilitySpace
																.getDiscountFactor()));
						double lowEndEvaluation = ((EvaluatorInteger) issueEvaluatorEntry
								.getValue()).getEvaluation(lowerBound);

						if (newHighEndEvaluation > 1) {
							total = newHighEndEvaluation + lowEndEvaluation;
						}

						((EvaluatorInteger) issueEvaluatorEntry.getValue())
								.setLinearFunction(lowEndEvaluation / total,
										newHighEndEvaluation / total);
					}
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}

		try {

			BidSpace bidSpace = new BidSpace(this.utilitySpace,
					opponentUtilitySpace, true);
			double kalaiPoint = bidSpace.getKalaiSmorodinsky().getUtilityA();

			// UNCOMMENT THE FOLLOWING LINES IF THE ESTIMATED KALAI POINT IS A
			// BIT TOO CONSERVATIVE

			if (kalaiPoint <= 0.4) {
				kalaiPoint = kalaiPoint + kalaiPointCorrection;
			} else if (kalaiPoint <= 0.7) {
				kalaiPoint = kalaiPoint + (kalaiPointCorrection / 2);
			}

			if (kalaiPoint > senderInfo.getBestBids().getBestBidDetails()
					.getMyUndiscountedUtil()) {
				senderInfo.setDomainCompetitiveness(kalaiPoint);
			} else {
				senderInfo.setDomainCompetitiveness(senderInfo.getBestBids()
						.getBestBidDetails().getMyUndiscountedUtil());
			}

			if (senderInfo.getAgentBidHistory().size() >= minimumHistorySize) {
				// TODO Changeable formulas
				double domainWeight = (1 - Math.pow(senderInfo.getLeniency(),
						domainWeightController));
				double agentDifficulty = (1 - domainWeight)
						* senderInfo.getLeniency()
						+ domainWeight * senderInfo.getDomainCompetitiveness();
				senderInfo.setAgentDifficulty(agentDifficulty);
			} else {
				double agentDifficulty = senderInfo.getDomainCompetitiveness();
				senderInfo.setAgentDifficulty(agentDifficulty);
			}
		} catch (Exception e) {

			e.printStackTrace();
		}

	}

	private HashMap<Integer, Boolean> determineDifference(
			OpponentInfo senderInfo, BidDetails first, BidDetails second) {
		HashMap<Integer, Boolean> changed = new HashMap<Integer, Boolean>();
		try {
			for (Issue i : senderInfo.getOpponentUtilitySpace().getDomain()
					.getIssues()) {
				if (i instanceof IssueDiscrete) {
					changed.put(i.getNumber(),
							(((ValueDiscrete) first.getBid()
									.getValue(i.getNumber()))
											.equals(second.getBid()
													.getValue(i.getNumber())))
															? false : true);
				} else if (i instanceof IssueInteger) {
					changed.put(i.getNumber(),
							(((ValueInteger) first.getBid()
									.getValue(i.getNumber()))
											.equals(second.getBid()
													.getValue(i.getNumber())))
															? false : true);
				}
			}
		} catch (Exception ex) {
			ex.printStackTrace();
		}

		return changed;
	}

	private OpponentInfo getOpponentInfoObjectOfSender(Object sender) {
		if (infoA != null && infoA.getAgentID().equals(sender.toString())) {
			return infoA;
		} else if (infoB != null
				&& infoB.getAgentID().equals(sender.toString())) {
			return infoB;
		}

		return null;
	}

	private void initializeOpponentInfo(Object sender) {
		if (infoA == null) {
			infoA = new OpponentInfo(sender.toString(),
					(AdditiveUtilitySpace) utilitySpace);
		} else if (infoB == null) {
			infoB = new OpponentInfo(sender.toString(),
					(AdditiveUtilitySpace) utilitySpace);
		}

		if (infoA != null && infoB != null) {
			initialized = true;
		}
	}

	private void updateOpponentBidHistory(OpponentInfo opponent, Bid bid) {

		if (opponent == null || bid == null) {
			return;
		}

		try {
			opponent.getAgentBidHistory().add(new BidDetails(bid,
					utilitySpace.getUtility(bid), timeline.getTime()));

			for (Integer i : opponent.getBidPointWeights()) {
				if (i > 1) {
					i--;
				}
			}
			opponent.getBidPointWeights().add(maxWeightForBidPoint);

			if (opponent.getBestBid() == null
					|| utilitySpace.getUtility(bid) >= utilitySpace
							.getUtility(opponent.getBestBid())) {
				opponent.setBestBid(bid);
				opponent.getBestBids().add(new BidDetails(bid,
						utilitySpace.getUtility(bid), timeline.getTime()));
			} else {
				opponent.getBestBids()
						.add(new BidDetails(opponent.getBestBid(),
								utilitySpace.getUtility(opponent.getBestBid()),
								timeline.getTime()));

			}
		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	@Override
	public String getDescription() {
		return "ANAC2015";
	}

}
