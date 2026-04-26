package agents.anac.y2014.BraveCat.OpponentModels;

import genius.core.Bid;
import genius.core.Domain;
import genius.core.utility.AdditiveUtilitySpace;

@SuppressWarnings("serial")
public class UtilitySpaceAdapter extends AdditiveUtilitySpace {
	private OpponentModel opponentModel;

	public UtilitySpaceAdapter(OpponentModel opponentModel, Domain domain) {
		super(domain);
		this.opponentModel = opponentModel;
	}

	@Override
	public double getUtility(Bid b) {
		double u = 0.0D;
		try {
			u = this.opponentModel.getBidEvaluation(b);
		} catch (Exception e) {
			System.err.println("getNormalizedUtility failed. returning 0");
			u = 0.0D;
		}
		return u;
	}

	@Override
	public double getWeight(int i) {
		System.err
				.println("The opponent model should overwrite getWeight() when using the UtilitySpaceAdapter");
		return i;
	}
}