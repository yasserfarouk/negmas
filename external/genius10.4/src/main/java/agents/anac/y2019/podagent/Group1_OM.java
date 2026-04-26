package agents.anac.y2019.podagent;

import java.util.*;
import java.util.Map.Entry;

import genius.core.Bid;
import genius.core.BidHistory;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.BOAparameter;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OpponentModel;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.Objective;
import genius.core.issue.ValueDiscrete;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.Evaluator;
import genius.core.utility.EvaluatorDiscrete;

public class Group1_OM extends OpponentModel {
	
	boolean hardHeadedUntilNow = false;
	private int amountOfIssues;
	private double lastStepBidsAverage = 0.0;

	@Override
	public void init(NegotiationSession negotiationSession, Map<String, Double> parameters) {
		this.negotiationSession = negotiationSession;
		opponentUtilitySpace = (AdditiveUtilitySpace) negotiationSession
				.getUtilitySpace().copy();
		amountOfIssues = opponentUtilitySpace.getDomain().getIssues().size();

		initializeModel();
	}

	/**
	 * Model update function altered from the HardHeaded frequency model.
	 *
	 * Instead of incrementing issue weights after two of the same successive values in a bid,
	 * the range of frequencies per issue is calculated, normalized and used as weight instead
	 *
	 *
	 * @param opponentBid
	 * @param time
	 */
	@Override
	public void updateModel(Bid opponentBid, double time) {
		if (negotiationSession.getOpponentBidHistory().size() < 1) {
			return;
		}
		
		BidDetails oppBid = negotiationSession.getOpponentBidHistory().getHistory().get(negotiationSession.getOpponentBidHistory().size() - 1);
		
		// Like HardHeaded, add a constant value of 1 to each value from the bid
		try {
			for (Entry<Objective, Evaluator> e : opponentUtilitySpace.getEvaluators()) {
				EvaluatorDiscrete value = (EvaluatorDiscrete) e.getValue();
				IssueDiscrete issue = ((IssueDiscrete) e.getKey());

				ValueDiscrete issuevalue = (ValueDiscrete) oppBid.getBid().getValue(issue.getNumber());
				Integer eval = value.getEvaluationNotNormalized(issuevalue);
				value.setEvaluation(issuevalue, (1 + eval));
			}
		} catch (Exception ex) {
			ex.printStackTrace();
		}
		
		// Calculate the difference between the highest value and the lowest value for each issue
		double totalDist = 0;
		Map<IssueDiscrete,Integer> distances = new HashMap<IssueDiscrete,Integer>();
		for (Entry<Objective, Evaluator> e : opponentUtilitySpace.getEvaluators()) {
			try {
				EvaluatorDiscrete value = (EvaluatorDiscrete) e.getValue();
				IssueDiscrete issue = ((IssueDiscrete) e.getKey());
				Integer max = 0;
				Integer min = Integer.MAX_VALUE;
				for (ValueDiscrete vd : ((IssueDiscrete) e.getKey()).getValues()) {
					Integer eval = value.getEvaluationNotNormalized(vd);
					min = eval < min ? eval : min;
					max = eval > max ? eval : max;
				}
				Integer dist = max - min;
				totalDist += max - min;
				distances.put(issue, dist);
			} catch (Exception ex) {
				ex.printStackTrace();
			}
		}

		// Update weights
		for (Entry<Objective, Evaluator> e : opponentUtilitySpace.getEvaluators()) {
			try {
				EvaluatorDiscrete value = (EvaluatorDiscrete) e.getValue();
				IssueDiscrete issue = ((IssueDiscrete) e.getKey());
				double dist = (double) distances.get(issue);
				// normalize the distance value and update the issue weight
				double newWeight = ((double) dist) / totalDist;
				opponentUtilitySpace.setWeight(issue, newWeight);
			} catch (Exception ex) {
				ex.printStackTrace();
			}
		}
	}

	@Override
	public double getBidEvaluation(Bid bid) {
		double result = 0;
		try {
			result = opponentUtilitySpace.getUtility(bid);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return result;
	}

	@Override
	public String getName() {
		return "Group1_OM";
	}

	@Override
	public Set<BOAparameter> getParameterSpec() {
		Set<BOAparameter> set = new HashSet<BOAparameter>();
		set.add(new BOAparameter("l", 0.2,
				"The learning coefficient determines how quickly the issue weights are learned"));
		return set;
	}

	/**
	 * Init to flat weight and flat evaluation distribution
	 */
	private void initializeModel() {
		double commonWeight = 1D / amountOfIssues;

		for (Entry<Objective, Evaluator> e : opponentUtilitySpace
				.getEvaluators()) {

			opponentUtilitySpace.unlock(e.getKey());
			e.getValue().setWeight(commonWeight);
			try {
				// set all value weights to one (they are normalized when
				// calculating the utility)
				for (ValueDiscrete vd : ((IssueDiscrete) e.getKey())
						.getValues())
					((EvaluatorDiscrete) e.getValue()).setEvaluation(vd, 1);
			} catch (Exception ex) {
				ex.printStackTrace();
			}
		}
	}


	//Should only be called when concede() is called	
	public double getOpponentSentiment(double lastSteptime) {
		//Not sure why this happens sometimes :/
		if(negotiationSession == null)
			return 0.5;
		
		//For the first step, use the first bid ever
		if(lastStepBidsAverage == 0 && negotiationSession.getOpponentBidHistory().size() != 0) {
			lastStepBidsAverage = getBidEvaluation(negotiationSession.getOpponentBidHistory().getLastBidDetails().getBid());
			return 0.5;
		}
		
		double newLastBidsAverage = negotiationSession.getOpponentBidHistory()
				.filterBetweenTime(lastSteptime, this.negotiationSession.getTime()).getHistory().stream().mapToDouble(a -> getBidEvaluation(a.getBid())).average().orElse(0.0);
				
		double opponentSentiment = (newLastBidsAverage - lastStepBidsAverage);
		//Update last step average
		lastStepBidsAverage = newLastBidsAverage;	
		if(opponentSentiment < 0) {
			return opponentSentiment - 0.5;
		}else if(opponentSentiment > 0) {
			return opponentSentiment + 0.5;
		}else {
			return 0;
		}
	}
	
	/**
	 * Counts the amount of different bids offered by the opponent to determine whether he is following a hard headed strategy
	 * @return hard headed status of the opponent
	 */
	public boolean isHardHeaded() {
		BidHistory hist = negotiationSession.getOpponentBidHistory();
		int diffBids = 0;
		ArrayList seenBids = new ArrayList<Bid>();
		for(BidDetails b : hist.getHistory()) {
			Bid bid = b.getBid();
			if(!seenBids.contains(bid)) {
				seenBids.add(bid);
				diffBids++;
			}
		}
		if(diffBids > 3) {
			return false;
		}
		return true;
	}

}
