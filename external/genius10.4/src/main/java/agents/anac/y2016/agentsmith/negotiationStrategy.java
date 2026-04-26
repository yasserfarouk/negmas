package agents.anac.y2016.agentsmith;

import java.util.ArrayList;

import genius.core.Bid;
import genius.core.utility.AbstractUtilitySpace;

public class negotiationStrategy {
	private AbstractUtilitySpace utilitySpace;
	private negotiationInfo negotiationInfo;
	public double rv = 0.0; // 留保価格
	public double df = 0.0;

	private boolean isPrinting = false; // デバッグ用

	public negotiationStrategy(AbstractUtilitySpace utilitySpace,
			negotiationInfo negotiationInfo, boolean isPrinting) {
		this.utilitySpace = utilitySpace;
		this.negotiationInfo = negotiationInfo;
		this.isPrinting = isPrinting;
		rv = utilitySpace.getReservationValue();
		df = utilitySpace.getDiscountFactor();
	}

	// u:utility
	// t:time
	// int i =0;

	// 受容判定
	public boolean selectAccept(Bid offeredBid, double time) {
		try {
			// System.out.println("time:" +time);
			//
			// System.out.println("dislity "+i +":" +discountedUtility );
			// i++;
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
	public boolean selectEndNegotiation(Bid offerBid, double time) {
		if (getThreshold(time) <= rv * Math.pow(df, time)) {
			return true;
		}
		return false;
	}

	// 閾値を返す
	public double getThreshold(double time) {
		double threshold = 1.0;
		// if (rv != 0.0 && df == 1.0){ //留保価格あり，割引効用なし
		//
		// }
		// else if (rv !=0.0 && df != 1.0){//留保価格あり，割引効用あり
		// }
		// else if (rv == 0.0 && df == 1.0){//留保価格なし，割引効用なし
		// }
		// else if (rv == 0.0 && df != 1.0){//留保価格なし，割引効用あり
		//
		// }
		/* 交渉戦略に基づきthreshold(t)を設計する */
		/* negotiationInfoから提案履歴の統計情報を取得できるので使っても良い */
		/* （統計情報についてはあまりデバッグできていないので，バグが見つかった場合は報告をお願いします） */

		// 例：
		ArrayList<Object> opponents = negotiationInfo.getOpponents();
		for (Object sender : opponents) {
			double m = negotiationInfo.getAverage(sender);
			double v = negotiationInfo.getVariancer(sender);
			double sd = negotiationInfo.getStandardDeviation(sender);
		}

		return threshold;
	}
}
