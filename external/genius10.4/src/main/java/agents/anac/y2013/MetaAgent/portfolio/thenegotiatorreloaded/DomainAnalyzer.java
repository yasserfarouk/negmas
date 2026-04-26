package agents.anac.y2013.MetaAgent.portfolio.thenegotiatorreloaded;

import genius.core.analysis.BidPoint;
import genius.core.analysis.BidSpace;
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
	 * @param ownUtilSpace utility space of our agen
	 * @param opponentModel used to estimate the opponent's preference profile
	 * @param opponentHistory
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
		if (opponentModel != null && !(opponentModel instanceof NullModel) && opponentModel.getOpponentUtilitySpace() != null) {
			if (omStrategy.canUpdateOM()) {
				try {
					BidSpace space = new BidSpace(ownUtilSpace, opponentModel.getOpponentUtilitySpace(), true);
					BidPoint kalai = space.getKalaiSmorodinsky();
					kalaiPoint = kalai.getUtilityA();
					previousKalaiPoint = kalaiPoint;
				} catch (Exception e) {
					//e.printStackTrace();
				}
			} else {
				kalaiPoint = previousKalaiPoint;
			}
		}
		return kalaiPoint;
	}
}