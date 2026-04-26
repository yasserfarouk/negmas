package negotiator.boaframework.offeringstrategy.anac2012.TheNegotiatorReloaded;

import genius.core.analysis.BidPoint;
import genius.core.analysis.BidSpace;
import genius.core.boaframework.NoModel;
import genius.core.boaframework.OMStrategy;
import genius.core.boaframework.OpponentModel;
import genius.core.utility.AdditiveUtilitySpace;

/**
 * Class used to estimate the Kalai point. This approach works well
 * with Bayesian models in which all bids are used in the updating process.
 * 
 * @author Mark Hendrikx
 */
public class DomainAnalyzer {
	
	// when an opponent model is given, the model is used in the Kalai estimation
	private OpponentModel opponentModel;
	// the utility space of our agent
	private AdditiveUtilitySpace ownUtilSpace;
	// default value for Kalai which is used when the opponent model is unavailable or unreliable
	private static double DEFAULT_KALAI = 0.7;
	// opponent model strategy, used to check if the model can be updated
	private OMStrategy omStrategy; 
	private double previousKalaiPoint;
	
	/**
	 * Set the domain analyzer variables and determine the domain size.
	 * 
	 * @param ownUtilSpace utility space of our agent.
	 * @param opponentModel used to estimate the opponent's preference profile.
	 * @param omStrategy used to check if the opponent model may be updated.
	 */
	public DomainAnalyzer(AdditiveUtilitySpace ownUtilSpace, OpponentModel opponentModel, OMStrategy omStrategy) {
		this.opponentModel = opponentModel;
		this.omStrategy = omStrategy;
		this.ownUtilSpace = ownUtilSpace;
	}
	
	/**
	 * Calculates the Kalai point by optionally using an opponent model.
	 * When an opponent model in unavailable, unreliable, or the domain too large,
	 * a default value is used.
	 * 
	 * @return coordinate of the Kalai point from our utilityspace
	 */
	public double calculateKalaiPoint() {
		double kalaiPoint = DEFAULT_KALAI;
		// check if the opponent model is a real model
		if (opponentModel != null && !(opponentModel instanceof NoModel)) {
			// check if the opponent model may be updated, signaling that its estimate potentially changed
			if (omStrategy.canUpdateOM()) {
				// calculate new estimate of the Kalai-point
				try {
					BidSpace space = new BidSpace(ownUtilSpace, opponentModel.getOpponentUtilitySpace(), true, true);
					BidPoint kalai = space.getKalaiSmorodinsky();
					kalaiPoint = kalai.getUtilityA();
					previousKalaiPoint = kalaiPoint;
				} catch (Exception e) {
					e.printStackTrace();
				}
			} else {
				// return previous Kalai point. In case there was no OM, or it was never updated, this
				// value is set to a safe defualt.
				kalaiPoint = previousKalaiPoint;
			}
		}
		return kalaiPoint;
	}
}