package negotiator.boaframework.opponentmodel;

import java.util.Map;

import agents.bayesianopponentmodel.OpponentModelUtilSpace;
import genius.core.Bid;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OpponentModel;
import genius.core.issue.Issue;
import genius.core.issue.ValueDiscrete;
import genius.core.utility.AdditiveUtilitySpace;
import negotiator.boaframework.opponentmodel.fsegaagent.FSEGAOpponentModel;

/**
 * Adapter to opponent model of FSEGA. This opponent model gives a nullpointer
 * in UtilitySpaceHypothesis and therefore does not work.
 * 
 * Adapted by Mark Hendrikx to be compatible with the BOA framework.
 *
 * Tim Baarslag, Koen Hindriks, Mark Hendrikx, Alex Dirkzwager and Catholijn M.
 * Jonker. Decoupling Negotiating Agents to Explore the Space of Negotiation
 * Strategies
 *
 * @author Mark Hendrikx
 */
public class FSEGABayesianModel extends OpponentModel {

	FSEGAOpponentModel model;
	private int startingBidIssue = 0;

	@Override
	public void init(NegotiationSession negotiationSession, Map<String, Double> parameters) {
		this.negotiationSession = negotiationSession;
		model = new FSEGAOpponentModel((AdditiveUtilitySpace) negotiationSession.getUtilitySpace());
		while (!testIndexOfFirstIssue(negotiationSession.getUtilitySpace().getDomain().getRandomBid(null),
				startingBidIssue)) {
			startingBidIssue++;
		}
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

	@Override
	public AdditiveUtilitySpace getOpponentUtilitySpace() {
		return new OpponentModelUtilSpace(model);
	}

	public double getWeight(Issue issue) {
		return model.getNormalizedWeight(issue, startingBidIssue);
	}

	public void cleanUp() {
		super.cleanUp();
		model = null;
	}
}