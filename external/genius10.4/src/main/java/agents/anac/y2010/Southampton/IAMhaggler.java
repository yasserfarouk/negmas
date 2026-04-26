package agents.anac.y2010.Southampton;

import agents.anac.y2010.Southampton.similarity.LinearSimilarityAgent;
import agents.anac.y2010.Southampton.utils.concession.SpecialTimeConcessionFunction;
import agents.anac.y2010.Southampton.utils.concession.TimeConcessionFunction;
import genius.core.SupportedNegotiationSetting;

/**
 * @author Colin Williams The IAMhaggler Agent, created for ANAC 2010. Designed
 *         by C. R. Williams, V. Robu, E. Gerding and N. R. Jennings.
 * 
 */
public class IAMhaggler extends LinearSimilarityAgent {

	/**
	 * The lowest target we have tried to use (initialised to 1).
	 */
	private double lowestTarget = 1.0;

	/**
	 * The minimum discounting factor.
	 */
	private final double minDiscounting = 0.1;

	/**
	 * The minimum beta value, which stops the agent from becoming too tough.
	 */
	private final double minBeta = 0.01;

	/**
	 * The maximum beta value, which stops the agent from becoming too weak.
	 */
	private double maxBeta = 2.0;

	/**
	 * The default value for beta, used when there is not enough information to
	 * perform the linear regression.
	 */
	private double defaultBeta = 1;

	/**
	 * The minimum target value when the opponent is considered to be a
	 * hardhaed.
	 */
	private final double hardHeadTarget = 0.8;

	/*
	 * (non-Javadoc)
	 * 
	 * @see agents.southampton.similarity.SimilarityAgent#init()
	 */
	@Override
	public void init() {
		super.init();
		int discreteCombinationCount = this.bidSpace
				.getDiscreteCombinationsCount();
		if (this.bidSpace.isContinuousWeightsZero()) {
			defaultBeta = Math.min(0.5,
					Math.max(0.0625, discreteCombinationCount * 0.001));
			if (utilitySpace.isDiscounted()) {
				maxBeta = (2.0 + (4.0
						* Math.min(0.5, utilitySpace.getDiscountFactor())))
						* defaultBeta;
			} else {
				maxBeta = 2.0 + 4.0 * defaultBeta;
			}
		}
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see agents.southampton.similarity.VariableConcessionSimilarityAgent#
	 * getTargetUtility(double, double)
	 */
	@Override
	protected double getTargetUtility(double myUtility, double oppntUtility) {
		double currentTime = timeline.getTime() * timeline.getTotalTime()
				* 1000;
		double beta = bidSpace.getBeta(bestOpponentBidUtilityHistory,
				timeline.getTime(), utility0, utility1, minDiscounting, minBeta,
				maxBeta, defaultBeta, currentTime, currentTime);

		cf = new SpecialTimeConcessionFunction(beta, defaultBeta,
				TimeConcessionFunction.DEFAULT_BREAKOFF);

		double target = super.getTargetUtility(myUtility, oppntUtility);

		lowestTarget = Math.min(target, lowestTarget);

		if (opponentIsHardHead) {
			return Math.max(hardHeadTarget, lowestTarget);
		}
		return lowestTarget;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see negotiator.Agent#getName()
	 */
	@Override
	public String getName() {
		return "IAMhaggler";
	}

	/**
	 * @return
	 */
	@Override
	public String getVersion() {
		return "2.0 (Genius 3.1)";
	}

	@Override
	public SupportedNegotiationSetting getSupportedNegotiationSetting() {
		return SupportedNegotiationSetting.getLinearUtilitySpaceInstance();
	}

	@Override
	public String getDescription() {
		return "ANAC2010";
	}
}
