package negotiator.boaframework.opponentmodel;

import java.util.Map;

import agents.bayesianopponentmodel.BayesianOpponentModelScalable;
import agents.bayesianopponentmodel.OpponentModelUtilSpace;
import genius.core.Bid;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OpponentModel;
import genius.core.issue.Issue;
import genius.core.issue.ValueDiscrete;
import genius.core.utility.AdditiveUtilitySpace;

/**
 * Adapter for BayesianOpponentModelScalable for the BOA framework.
 *
 * Tim Baarslag, Koen Hindriks, Mark Hendrikx, Alex Dirkzwager and Catholijn M.
 * Jonker. Decoupling Negotiating Agents to Explore the Space of Negotiation
 * Strategies
 *
 * KNOWN BUGS: 1. Opponent model does not take the opponent's strategy into
 * account, in contrast to the original paper which depicts an assumption about
 * the opponent'strategy which adapts over time.
 * 
 * 2. The opponent model becomes invalid after a while as NaN occurs in some
 * hypotheses, corrupting the overall estimation.
 * 
 * @author Mark Hendrikx
 */
public class ScalableBayesianModel extends OpponentModel {

	private BayesianOpponentModelScalable model;
	private int startingBidIssue = 0;

	@Override
	public void init(NegotiationSession negoSession, Map<String, Double> parameters) {
		initializeModel(negoSession);
	}

	public void initializeModel(NegotiationSession negotiationSession) {
		this.negotiationSession = negotiationSession;
		while (!testIndexOfFirstIssue(negotiationSession.getUtilitySpace().getDomain().getRandomBid(null),
				startingBidIssue)) {
			startingBidIssue++;
		}

		model = new BayesianOpponentModelScalable((AdditiveUtilitySpace) negotiationSession.getUtilitySpace());

	}

	/**
	 * Just an auxiliary function to calculate the index where issues start on a
	 * bid because we found out that it depends on the domain.
	 * 
	 * @return true when the received index is the proper index
	 */
	private boolean testIndexOfFirstIssue(Bid bid, int i) {
		try {
			@SuppressWarnings("unused")
			ValueDiscrete valueOfIssue = (ValueDiscrete) bid.getValue(i);
		} catch (Exception e) {
			return false;
		}
		return true;
	}

	@Override
	public void updateModel(Bid opponentBid, double time) {
		try {
			// time is not used by this opponent model
			model.updateBeliefs(opponentBid);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	@Override
	public double getBidEvaluation(Bid bid) {
		try {
			return model.getNormalizedUtility(bid);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return 0;
	}

	public double getWeight(Issue issue) {
		return model.getNormalizedWeight(issue, startingBidIssue);
	}

	@Override
	public AdditiveUtilitySpace getOpponentUtilitySpace() {
		return new OpponentModelUtilSpace(model);
	}

	public void cleanUp() {
		super.cleanUp();
	}

	@Override
	public String getName() {
		return "Scalable Bayesian Model";
	}
}