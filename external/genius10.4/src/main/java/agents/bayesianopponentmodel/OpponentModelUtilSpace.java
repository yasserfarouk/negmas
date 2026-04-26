package agents.bayesianopponentmodel;

import genius.core.Bid;
import genius.core.utility.AdditiveUtilitySpace;

@SuppressWarnings("serial")
public class OpponentModelUtilSpace extends AdditiveUtilitySpace {
	OpponentModel opponentmodel;

	public OpponentModelUtilSpace(OpponentModel opmod) {
		super(opmod.getDomain());
		opponentmodel = opmod;
	}

	public double getUtility(Bid b) {
		double u = 0.;
		try {
			u = opponentmodel.getNormalizedUtility(b);
		} catch (Exception e) {
			e.printStackTrace();
			System.out.println("getNormalizedUtility failed. returning 0");
			u = 0.;
		}
		return u;
	}
}
