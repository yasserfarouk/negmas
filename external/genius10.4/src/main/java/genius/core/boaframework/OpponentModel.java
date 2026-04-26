package genius.core.boaframework;

import java.io.Serializable;
import java.util.List;
import java.util.Map;

import genius.core.Bid;
import genius.core.issue.Issue;
import genius.core.protocol.BilateralAtomicNegotiationSession;
import genius.core.utility.AbstractUtilitySpace;
import genius.core.utility.AdditiveUtilitySpace;

/**
 * Describes an opponent model of an agent of the BOA framework. This model
 * assumes issue weights hence only supports {@link AdditiveUtilitySpace}. Also
 * notice that most implementations are assuming bi-lateral negotiation (2
 * parties).
 * 
 * paper: https://ii.tudelft.nl/sites/default/files/boa.pdf
 * 
 */
public abstract class OpponentModel extends BOA {

	/** Reference to the estimated opponent's utility state */
	protected AdditiveUtilitySpace opponentUtilitySpace;
	/** Boolean to indicate that the model has been cleared to free resources */
	private boolean cleared;

	/**
	 * Initializes the model. The init method should always be called after
	 * creating an opponent model.
	 * 
	 * @param negotiationSession
	 *            reference to the state of the negotiation
	 * @param parameters
	 * @throws Exception
	 */
	@Override
	public void init(NegotiationSession negotiationSession,
			Map<String, Double> parameters) {
		super.init(negotiationSession, parameters);
		opponentUtilitySpace = (AdditiveUtilitySpace) negotiationSession
				.getUtilitySpace().copy();
		cleared = false;
	}

	/**
	 * Called to inform about a new {@link Bid} done by the opponent.
	 * 
	 * @param opponentBid
	 *            the bid received from the opponent
	 */
	public void updateModel(Bid opponentBid) {
		updateModel(opponentBid, negotiationSession.getTime());
	}

	/**
	 * As {@link #updateModel(Bid)} but with the current time added.
	 */
	protected abstract void updateModel(Bid bid, double time);

	/**
	 * Support function. Determines the utility of a bid according to the
	 * preference profile.
	 * 
	 * @param bid
	 *            of which the utility is calculated.
	 * @return Utility of the bid
	 */
	public double getBidEvaluation(Bid bid) {
		try {
			return opponentUtilitySpace.getUtility(bid);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return -1;
	}

	/**
	 * @return the estimated utility space of the opponent
	 */
	public AbstractUtilitySpace getOpponentUtilitySpace() {
		return opponentUtilitySpace;
	}

	/**
	 * Method which may be overwritten by an opponent model to get access to the
	 * opponent's utilityspace.
	 * 
	 * @param fNegotiation
	 */
	public void setOpponentUtilitySpace(
			BilateralAtomicNegotiationSession fNegotiation) {
	}

	/**
	 * Method which may be overwritten by an opponent model to get access to the
	 * opponent's utilityspace.
	 * 
	 * @param opponentUtilitySpace
	 */
	public void setOpponentUtilitySpace(
			AdditiveUtilitySpace opponentUtilitySpace) {
	}

	/**
	 * Returns the weight of a particular issue in the domain. Only works with
	 * {@link AdditiveUtilitySpace}.
	 * 
	 * @param issue
	 *            from which the weight should be returned
	 * @return weight of the given issue
	 */
	public double getWeight(Issue issue) {
		return opponentUtilitySpace.getWeight(issue.getNumber());
	}

	/**
	 * @return set of all estimated issue weights.
	 */
	public double[] getIssueWeights() {
		List<Issue> issues = negotiationSession.getUtilitySpace().getDomain()
				.getIssues();
		double estimatedIssueWeights[] = new double[issues.size()];
		int i = 0;
		for (Issue issue : issues) {
			estimatedIssueWeights[i] = getWeight(issue);
			i++;
		}
		return estimatedIssueWeights;
	}

	/**
	 * Removes references to the objects used by the opponent model.
	 */
	public void cleanUp() {
		negotiationSession = null;
		cleared = true;
	}

	/**
	 * @return if the opponent model is in a usable state.
	 */
	public boolean isCleared() {
		return cleared;
	}

	/**
	 * @return name of the opponent model.
	 */
	@Override
	public String getName() {
		return "Default";
	}

	@Override
	public final void storeData(Serializable object) {
		negotiationSession.setData(BoaType.OPPONENTMODEL, object);
	}

	@Override
	public final Serializable loadData() {
		return negotiationSession.getData(BoaType.OPPONENTMODEL);
	}

}
