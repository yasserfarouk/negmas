package negotiator.boaframework.opponentmodel;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import genius.core.Bid;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OpponentModel;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.Objective;
import genius.core.issue.ValueDiscrete;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.Evaluator;
import genius.core.utility.EvaluatorDiscrete;

/**
 * BOA framework implementation of an Adapted Frequency Model.
 * 
 * This model increases its weights only when the opponent does not constanstly
 * offer the same bid. The values of this model are scaled before they are used
 * to calculate the utility of a bid.
 * 
 * @author Ruben van Zessen, Mariana Branco
 */
public class InoxAgent_OM extends OpponentModel {

	/** Issue weight coefficient */
	private double weightCoef = 0.4;
	/** Issue weight decrease with repeated bids */
	private double weightDecr = 0.2;
	/** Initial value of the issue weight coefficient */
	private double initCoef = 0.4;

	/** Value weight coefficient */
	private int learnValueAddition = 1;
	/** Amount of issues in the negotiation */
	private int amountOfIssues;
	/**
	 * Smaller version of the opponent's bidding history, containing only unique
	 * bids
	 */
	private ArrayList<Bid> smallHistory;

	/** Time of the previous iteration */
	private double lastTime = 0.0;
	/** List of time differences between iterations */
	private ArrayList<Double> timeList = new ArrayList<Double>();
	/** Estimated total number of rounds in the negotiation */
	private int roundsEst = 15000;

	/**
	 * Empty constructor.
	 */
	public InoxAgent_OM() {
	}

	/**
	 * Regular constructor.
	 */
	public InoxAgent_OM(NegotiationSession negotiationSession) {
		this.negotiationSession = negotiationSession;
		smallHistory = new ArrayList<Bid>();
		initializeModel();
	}

	/**
	 * Initialization function.
	 * 
	 * Does the same as the regular constructor.
	 */
	@Override
	public void init(NegotiationSession negotiationSession, Map<String, Double> parameters) {
		this.negotiationSession = negotiationSession;
		smallHistory = new ArrayList<Bid>();
		initializeModel();
	}

	/**
	 * Method to initialize the model.
	 */
	private void initializeModel() {
		opponentUtilitySpace = (AdditiveUtilitySpace) negotiationSession.getUtilitySpace().copy();
		amountOfIssues = opponentUtilitySpace.getDomain().getIssues().size();
		double commonWeight = 1D / (double) amountOfIssues;

		// initialize the weights
		for (Entry<Objective, Evaluator> e : opponentUtilitySpace.getEvaluators()) {
			// set the issue weights
			opponentUtilitySpace.unlock(e.getKey());
			e.getValue().setWeight(commonWeight);
			try {
				// set all value weights to one (they are normalized when
				// calculating the utility)
				for (ValueDiscrete vd : ((IssueDiscrete) e.getKey()).getValues())
					((EvaluatorDiscrete) e.getValue()).setEvaluation(vd, 1);
			} catch (Exception ex) {
				ex.printStackTrace();
			}
		}
	}

	/**
	 * Determines the difference between bids. For each issue, it is determined
	 * if the value changed. If this is the case, a 1 is stored in a hashmap for
	 * that issue, else a 0.
	 * 
	 * @param a
	 *            bid of the opponent
	 * @param another
	 *            bid
	 * @return
	 */
	private HashMap<Integer, Integer> determineDifference(BidDetails first, BidDetails second) {

		HashMap<Integer, Integer> diff = new HashMap<Integer, Integer>();
		try {
			for (Issue i : opponentUtilitySpace.getDomain().getIssues()) {
				diff.put(i.getNumber(), (((ValueDiscrete) first.getBid().getValue(i.getNumber()))
						.equals((ValueDiscrete) second.getBid().getValue(i.getNumber()))) ? 0 : 1);
			}
		} catch (Exception ex) {
			ex.printStackTrace();
		}

		return diff;
	}

	/**
	 * Updates the opponent model given a bid.
	 */
	@Override
	public void updateModel(Bid opponentBid, double time) {
		// Do nothing with the model if the opponent's bidding history is too
		// small for analysis
		if (negotiationSession.getOpponentBidHistory().size() < 2) {
			smallHistory.add(opponentBid);
			return;
		}

		// Update rounds estimation
		updateRoundEst(time);

		// Determine number of unchanged issues
		int numberOfUnchanged = 0;
		BidDetails oppBid = negotiationSession.getOpponentBidHistory().getHistory()
				.get(negotiationSession.getOpponentBidHistory().size() - 1);
		BidDetails prevOppBid = negotiationSession.getOpponentBidHistory().getHistory()
				.get(negotiationSession.getOpponentBidHistory().size() - 2);
		HashMap<Integer, Integer> lastDiffSet = determineDifference(prevOppBid, oppBid);

		// count the number of changes in value
		for (Integer i : lastDiffSet.keySet()) {
			if (lastDiffSet.get(i) == 0)
				numberOfUnchanged++;
		}

		// If the bid is unchanged, decrease issueweight
		if (numberOfUnchanged == amountOfIssues) {
			weightCoef -= weightDecr;
			if (weightCoef < 0.0) {
				weightCoef = 0;
			}
			// If the bid has changed, reset issueweight
		} else {
			weightCoef = initCoef;
			if (!smallHistory.contains(opponentBid)) {
				smallHistory.add(opponentBid);
			}
		}

		// This is the value to be added to weights of unchanged issues before
		// normalization.
		// Also the value that is taken as the minimum possible weight,
		// (therefore defining the maximum possible also).
		double goldenValue = weightCoef / (double) amountOfIssues;
		// The total sum of weights before normalization.
		double totalSum = 1D + goldenValue * (double) numberOfUnchanged;
		// The maximum possible weight
		double maximumWeight = 1D - ((double) amountOfIssues) * goldenValue / totalSum;

		// re-weighing issues while making sure that the sum remains 1
		for (Integer i : lastDiffSet.keySet()) {
			if (lastDiffSet.get(i) == 0 && opponentUtilitySpace.getWeight(i) < maximumWeight)
				opponentUtilitySpace.setWeight(opponentUtilitySpace.getDomain().getObjectivesRoot().getObjective(i),
						(opponentUtilitySpace.getWeight(i) + goldenValue) / totalSum);
			else
				opponentUtilitySpace.setWeight(opponentUtilitySpace.getDomain().getObjectivesRoot().getObjective(i),
						opponentUtilitySpace.getWeight(i) / totalSum);
		}

		// Then for each issue value that has been offered last time, a constant
		// value is added to its corresponding ValueDiscrete.
		try {
			for (Entry<Objective, Evaluator> e : opponentUtilitySpace.getEvaluators()) {
				((EvaluatorDiscrete) e.getValue()).setEvaluation(
						oppBid.getBid().getValue(((IssueDiscrete) e.getKey()).getNumber()),
						(learnValueAddition + ((EvaluatorDiscrete) e.getValue()).getEvaluationNotNormalized(
								((ValueDiscrete) oppBid.getBid().getValue(((IssueDiscrete) e.getKey()).getNumber())))));
			}
		} catch (Exception ex) {
			ex.printStackTrace();
		}
	}

	@Override
	public double getBidEvaluation(Bid bid) {
		double result = 0;
		try {
			// If opponent is friendly, conceding opponent we can use the normal
			// frequency
			// model without implementing our scaling
			if (smallHistory.size() > 0.2 * negotiationSession.getOpponentBidHistory().size()) {
				result = opponentUtilitySpace.getUtility(bid);
				// Else receiveMessage with scaling
			} else {
				Objective root = opponentUtilitySpace.getDomain().getObjectivesRoot();
				Enumeration<Objective> issueEnum = root.getPreorderIssueEnumeration();
				while (issueEnum.hasMoreElements()) {
					Objective is = (Objective) issueEnum.nextElement();
					Evaluator eval = (Evaluator) opponentUtilitySpace.getEvaluator(is.getNumber());
					result += eval.getWeight() * valueEval(is, bid);
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return result;
	}

	/**
	 * Scaling function used to improve the opponent model.
	 */
	private double mapToEval(int freq) {
		return (70 - (1.0 / (((10000.0 / roundsEst) * freq / 15000) + (1.0 / 70))));
	}

	/**
	 * Method used to scale a value.
	 */
	public double valueEval(Objective obj, Bid bid) throws Exception {
		EvaluatorDiscrete lEval = (EvaluatorDiscrete) opponentUtilitySpace.getEvaluator(obj.getNumber());
		int variable = 0;
		for (ValueDiscrete vd : lEval.getValues()) {
			if (variable < lEval.getEvaluationNotNormalized(vd)) {
				variable = lEval.getEvaluationNotNormalized(vd);
			}
		}
		int idc = lEval.getEvaluationNotNormalized(bid, obj.getNumber());
		// Return normalized scaled value
		return ((mapToEval(idc)) / (mapToEval(variable)));
	}

	/**
	 * Method used to estimate the total number of rounds by evaluating the
	 * iteration time in the last 10 rounds.
	 */
	private void updateRoundEst(double t) {
		timeList.add(t - lastTime);
		lastTime = t;
		if (timeList.size() >= 10) {
			if (timeList.size() > 10) {
				timeList.remove(0);
			}

			double sum = 0;
			for (int i = 0; i < timeList.size(); i++) {
				sum += timeList.get(i);
			}
			roundsEst = (int) (timeList.size() / sum);
		}
	}
}