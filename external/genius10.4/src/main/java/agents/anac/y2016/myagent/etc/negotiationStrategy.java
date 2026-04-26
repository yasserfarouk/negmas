package agents.anac.y2016.myagent.etc;

import java.util.ArrayList;

import genius.core.Bid;
import genius.core.utility.UtilitySpace;

public class negotiationStrategy {
	private UtilitySpace utilitySpace;
	private negotiationInfo negotiationInfo;
	private double df = -1.0; // 割引率
	private double rv = 0.0; // 留保価格

	private boolean isPrinting = false; // デバッグ用

	public negotiationStrategy(UtilitySpace utilitySpace,
			negotiationInfo negotiationInfo, boolean isPrinting) {
		this.utilitySpace = utilitySpace;
		this.negotiationInfo = negotiationInfo;
		this.isPrinting = isPrinting;
		rv = utilitySpace.getReservationValue();
	}

	// 受容判定
	public boolean selectAccept(Bid offeredBid, double time) {

		// 割引率を求める（discountFactorが使えないため）
		if (df == -1.0) {
			double util = utilitySpace.getUtility(offeredBid);
			double discounted = utilitySpace.discount(util, time);
			df = Math.pow(discounted / util, 1 / time);
			// System.out.println("util:" + util + "\tdiscounted:" + discounted
			// + "\tdf:" + df);
		}

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
		if (getThreshold(time) < rv) {
			return true;
		}
		return false;
	}

	// 閾値を返す
	public double getThreshold(double time) {

		/* 交渉戦略に基づきthreshold(t)を設計する */
		/* negotiationInfoから提案履歴の統計情報を取得できるので使っても良い */
		/* （統計情報についてはあまりデバッグできていないので，バグが見つかった場合は報告をお願いします） */

		double threshold = 0.95;
		double e = 0.01;

		threshold = 1.0 - Math.pow(time, 1 / e);

		ArrayList<Object> arrogants = negotiationInfo.getArrogants();

		if (arrogants.size() == 1) {
			Object arrogant = arrogants.get(0);

			double slant = negotiationInfo.getSlant(arrogant);
			// System.out.println("---------------\nSlant: " + slant);

			double rushValue = threshold * df * slant; // 強気＊強気(0.5は勝利確率)
			double concedeValue = negotiationInfo.getAverage(arrogant)
					* Math.pow(df, time); // 弱気＊強気
			// System.out.println("rushValue:" + rushValue);
			// System.out.println("concedeValue:" + concedeValue);

			if (concedeValue > rushValue) {
				if (negotiationInfo.getOpponentBidNum(arrogant) < 70) {
					e = 0.5;
				} else if (slant < 0.15) {
					if (df == 1.0) {
						e = 0.3;
					} else {
						e = 1.65;
					}
				} else {
					e = 0.1;
				}
			}
		}

		threshold = 1.0 - Math.pow(time, 1 / e);

		// 例：
		ArrayList<Object> opponents = negotiationInfo.getOpponents();

		for (Object sender : opponents) {
			// System.out.println("sender:" + sender);
			// System.out.println("time:" + time);
			// double m = negotiationInfo.getAverage(sender);
			// double v = negotiationInfo.getVariancer(sender);
			// double sd = negotiationInfo.getStandardDeviation(sender);
			// System.out.println("ave:" + m + "\nvar:" + v + "\nsta:" + sd);
			// System.out.println("AcceptedNum:" +
			// negotiationInfo.getAcceptedNum(sender));
			// System.out.println("ArrogantsNum:" + arrogants.size());
		}

		return threshold;
	}

}
