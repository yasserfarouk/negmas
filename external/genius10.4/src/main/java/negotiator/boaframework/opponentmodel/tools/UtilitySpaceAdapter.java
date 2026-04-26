package negotiator.boaframework.opponentmodel.tools;

import genius.core.Bid;
import genius.core.Domain;
import genius.core.boaframework.OpponentModel;
import genius.core.utility.AdditiveUtilitySpace;

/**
 * Some opponent models do not use the UtilitySpace-object. Using this object, a
 * UtilitySpace-object can be created for each opponent model.
 * 
 * @author Mark Hendrikx
 */
public class UtilitySpaceAdapter extends AdditiveUtilitySpace {

	private OpponentModel opponentModel;

	public UtilitySpaceAdapter(OpponentModel opponentModel, Domain domain) {
		super(domain);
		this.opponentModel = opponentModel;
	}

	public double getUtility(Bid b) {
		double u = 0.;
		try {
			u = opponentModel.getBidEvaluation(b);
		} catch (Exception e) {
			e.printStackTrace();
			System.err.println("getNormalizedUtility failed. returning 0");
			u = 0.0;
		}
		return u;
	}

	public double getWeight(int i) {
		System.err
				.println("The opponent model should overwrite getWeight() when using the UtilitySpaceAdapter");
		return i;
	}
}