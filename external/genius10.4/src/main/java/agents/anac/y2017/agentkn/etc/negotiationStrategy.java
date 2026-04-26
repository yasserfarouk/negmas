package agents.anac.y2017.agentkn.etc;

import java.util.ArrayList;
import java.util.Iterator;

import genius.core.Bid;
import genius.core.utility.AbstractUtilitySpace;

public class negotiationStrategy {
	private AbstractUtilitySpace utilitySpace;
	private negotiationInfo negotiationInfo;
	private double rv = 0.0; // 留保価格

	private boolean isPrinting = false; // デバッグ用

	public negotiationStrategy(AbstractUtilitySpace utilitySpace, negotiationInfo negotiationInfo, boolean isPrinting) {
		this.utilitySpace = utilitySpace;
		this.negotiationInfo = negotiationInfo;
		this.isPrinting = isPrinting;
		rv = utilitySpace.getReservationValue();
	}

	// 受容判定
	public boolean selectAccept(Bid offeredBid, double time) {
		try {
			if (utilitySpace.getUtility(offeredBid) >= getThreshold(time)) {
				return true;
			} else {
				return false;
			}
		} catch (Exception e) {
			System.out.println("受容判定に失敗しました");
			e.printStackTrace();
			return false;
		}
	}

	// 交渉終了判定
	public boolean selectEndNegotiation(double time) {
		return false;
	}

	// 閾値を返す
	public double getThreshold(double time) {
		System.out.println("time : " + time);
		double threshold = 1.0D;
		double mi = 0.0D;
		double ave = 0.0D;
		double extra = 0.0D;
		ArrayList opponents = negotiationInfo.getOpponents();

		double sd;
		for (Iterator i = opponents.iterator(); i.hasNext(); extra = sd) {
			Object sender = i.next();
			if (negotiationInfo.getPartnerBidNum(sender) % 10 == 0) {
				ave = 0.0D;
				extra = 0.0D;
			}

			double m = negotiationInfo.getAverage(sender);
			sd = negotiationInfo.getStandardDeviation(sender);
			ave = m;
		}

		double c = 1.0D - Math.pow(time, 5D);
		double emax = emax();
		threshold = 1 - (1 - emax) * Math.pow(time, 5D);

		return threshold;
	}

	private double emax() {
		double ave = 0.0D;
		double extra = 0.0D;
		ArrayList opponents = negotiationInfo.getOpponents();

		double sd = 0.0D;
		for (Iterator i = opponents.iterator(); i.hasNext(); extra = sd) {
			Object sender = i.next();
			if (negotiationInfo.getPartnerBidNum(sender) % 10 == 0) {
				ave = 0.0D;
				extra = 0.0D;
			}

			double m = negotiationInfo.getAverage(sender);
			sd = negotiationInfo.getStandardDeviation(sender);
			ave = m;
		}

		double d = Math.sqrt(3) * sd / Math.sqrt(ave * (1 - ave));

		return 0.7D * ave + (1 - ave) * d * 0.7D;
	}
}
