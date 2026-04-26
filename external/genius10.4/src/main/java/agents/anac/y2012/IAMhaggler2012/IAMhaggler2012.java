package agents.anac.y2012.IAMhaggler2012;

import agents.anac.y2012.IAMhaggler2012.agents2011.IAMhaggler2011;
import agents.anac.y2012.IAMhaggler2012.utility.SouthamptonUtilitySpace;
import genius.core.Bid;
import genius.core.SupportedNegotiationSetting;
import genius.core.utility.AdditiveUtilitySpace;

/**
 * @author Colin Williams
 * 
 *         The IAMhaggler Agent, created for ANAC 2012. Designed by C. R.
 *         Williams, V. Robu, E. H. Gerding and N. R. Jennings.
 * 
 */
public class IAMhaggler2012 extends IAMhaggler2011 {

	private SouthamptonUtilitySpace sus;

	/*
	 * (non-Javadoc)
	 * 
	 * @see agents.southampton.SouthamptonAgent#init()
	 */
	@Override
	public void init() {
		debug = false;
		super.init();
		sus = new SouthamptonUtilitySpace((AdditiveUtilitySpace) utilitySpace);
	}

	/*
	 * 
	 * (non-Javadoc)
	 * 
	 * @see agents2011.southampton.IAMhaggler2011#proposeInitialBid()
	 */
	@Override
	protected Bid proposeInitialBid() throws Exception {
		Bid b = sus.getMaxUtilityBid();
		if (utilitySpace.getUtilityWithDiscount(b, timeline) < utilitySpace
				.getReservationValueWithDiscount(timeline)) {
			return null;
		}
		return b;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see agents.southampton.SouthamptonAgent#proposeNextBid(negotiator.Bid)
	 */
	@Override
	protected Bid proposeNextBid(Bid opponentBid) throws Exception {
		Bid b = super.proposeNextBid(opponentBid);
		if (utilitySpace.getUtilityWithDiscount(b, timeline) < utilitySpace
				.getReservationValueWithDiscount(timeline)) {
			return proposeInitialBid();
		}
		return b;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see agents2011.southampton.IAMhaggler2011#getTarget(double, double)
	 */
	@Override
	protected double getTarget(double opponentUtility, double time) {
		return Math.max(utilitySpace.getReservationValueWithDiscount(time),
				super.getTarget(opponentUtility, time));
	}

	@Override
	public SupportedNegotiationSetting getSupportedNegotiationSetting() {
		return SupportedNegotiationSetting.getLinearUtilitySpaceInstance();
	}

	@Override
	public String getDescription() {
		return "ANAC2012";
	}
}
