package genius.core.uncertainty;

import genius.core.Bid;
import genius.core.utility.UncertainAdditiveUtilitySpace;

/**
 * Extends the {@link UserModel} (which contains *perceived* preferences) with
 * access to the *real* preferences. This class can be used by an agent for
 * debugging purposes to access the real utilites.
 * 
 * @author Tim Baarslag, Dimitrios Tsimpoukis
 */

public class ExperimentalUserModel extends UserModel {
	private UncertainAdditiveUtilitySpace realUtilitySpace;

	public ExperimentalUserModel(BidRanking ranking,
			UncertainAdditiveUtilitySpace realUSpace) {
		super(ranking);
		this.realUtilitySpace = realUSpace;
	}

	public UncertainAdditiveUtilitySpace getRealUtilitySpace() {
		return realUtilitySpace;
	}

	public double getRealUtility(Bid bid) {
		return realUtilitySpace.getUtility(bid);
	}

}
