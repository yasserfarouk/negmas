package genius.core.uncertainty;

import genius.core.boaframework.SortedOutcomeSpace;
import genius.core.utility.AbstractUtilitySpace;
import genius.core.utility.AdditiveUtilitySpaceWithPerturbation;
import genius.core.utility.UncertainAdditiveUtilitySpace;

/**
 * 
 * Keeps track of both the real preferences of the user as well as the
 * perception of the agent. The preference profile that is given to the agents
 * can be a "distorted utility space" in the cases that uncertainty is included
 * in the form of a perturbation to the real utilities (see
 * {@link AdditiveUtilitySpaceWithPerturbation}}) or a flattened Utility Space
 * (see {@link FlattenedUtilitySpace}), OR in the form of a Pairwise Comparison
 * User Model (see {@link UserModel}). Used only in the core.
 *
 */

public class UncertainPreferenceContainer {

	private UNCERTAINTYTYPE type;
	private UncertainAdditiveUtilitySpace realUtilitySpace;
	private UserModel pairwiseCompUserModel;

	/**
	 * 
	 * @param realUtilitySpace
	 * @param type
	 */
	
	public UncertainPreferenceContainer(UncertainAdditiveUtilitySpace utilspace,
			UNCERTAINTYTYPE type) {
		this.realUtilitySpace = utilspace;
		this.type = type;
		
		createOutcomeComparisonUserModel(utilspace.getComparisons(),
					utilspace.getErrors(), utilspace.getSeed(), utilspace.isExperimental());
	}
	
	public UncertainPreferenceContainer(UncertainAdditiveUtilitySpace utilspace,
			UNCERTAINTYTYPE type, int sizeOfBidRank) {
		this.realUtilitySpace = utilspace;
		this.type = type;
		createOutcomeComparisonUserModel(sizeOfBidRank, utilspace.getErrors(), utilspace.getSeed(), utilspace.isExperimental());
	}

	public UncertainPreferenceContainer(
			UncertainPreferenceContainer container) {
		this.realUtilitySpace = container.realUtilitySpace;
		this.type = container.type;
		this.pairwiseCompUserModel = container.pairwiseCompUserModel;
	}

	public UncertainPreferenceContainer(ExperimentalUserModel userModel) {
		this.realUtilitySpace = userModel.getRealUtilitySpace();
		this.type = UNCERTAINTYTYPE.PAIRWISECOMP;
		this.pairwiseCompUserModel = userModel;
	}

	/**
	 * Generates comparisons between different outcomes. At the moment each
	 * outcome is part of only one comparison.
	 * 
	 * 
	 * @param amountOfBids,
	 *            amount of bids to be compared
	 * @para error the number of errors. UNUSED
	 * @param seed
	 *            if nonzero, this seed is used to pick the random bids
	 * 
	 */
	public void createOutcomeComparisonUserModel(int amountOfBids,
			double error, long seed, boolean experimental) {
		SortedOutcomeSpace outcomeSpace = (new SortedOutcomeSpace(this.getRealUtilitySpace()));
		BidRanking ranking = (new ComparisonGenerator(outcomeSpace))
				.generateRankingByAmount(amountOfBids, seed);
		if (experimental)
			this.pairwiseCompUserModel = new ExperimentalUserModel(ranking,	realUtilitySpace);
		else
			this.pairwiseCompUserModel = new UserModel(ranking);
	}

	public AbstractUtilitySpace getRealUtilitySpace() {
		return realUtilitySpace;
	}

	public UserModel getPairwiseCompUserModel() {
		return pairwiseCompUserModel;
	}

	public void setPairwiseCompUserModel(UserModel pairwiseCompUserModel) {
		this.pairwiseCompUserModel = pairwiseCompUserModel;
	}

	public UNCERTAINTYTYPE getType() {
		return type;
	}
}
