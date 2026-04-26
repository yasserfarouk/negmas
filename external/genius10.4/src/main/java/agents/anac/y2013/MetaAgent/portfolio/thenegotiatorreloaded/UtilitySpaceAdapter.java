package agents.anac.y2013.MetaAgent.portfolio.thenegotiatorreloaded;

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

	public double getUtility(Bid b) {
		double u = 0.;
		try {
			u = opponentModel.getBidEvaluation(b);
		} catch (Exception e) {
			e.printStackTrace();
			System.out.println("getNormalizedUtility failed. returning 0");
			u = 0.;
		}
		return u;
	}
}
