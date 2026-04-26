package agents.anac.y2016.farma.etc;

import java.util.ArrayList;
import java.util.HashMap;

import genius.core.Bid;
import genius.core.utility.AbstractUtilitySpace;

public class negotiationStrategy {
	private AbstractUtilitySpace utilitySpace;
	private negotiationInfo negotiationInfo;
	private double df = 0.0; // 割引係数
	private double rv = 0.0; // 留保価格

	private boolean isPrinting = false; // デバッグ用

	public negotiationStrategy(AbstractUtilitySpace utilitySpace,
			negotiationInfo negotiationInfo, boolean isPrinting) {
		this.utilitySpace = utilitySpace;
		this.negotiationInfo = negotiationInfo;
		this.isPrinting = isPrinting;
		df = utilitySpace.getDiscountFactor();
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
		// 時刻timeの割引効用を考慮した留保価格が閾値を上回った時EndNegotiation
		if (utilitySpace.getReservationValueWithDiscount(time) >= getThreshold(time))
			return true;

		double weightedDF = df * 2.0;
		if (time > 0.5 && rv > weightedDF) {
			ArrayList<Object> opponents = negotiationInfo.getOpponents();
			double minDIST = 1.0;
			double maxEmax = 0.0;
			for (Object sender : opponents) {
				minDIST = Math.min(minDIST,
						negotiationInfo.calSpaceDistByInitOffer(sender));
				maxEmax = Math.max(maxEmax, calEmax(sender));
			}
			if (maxEmax < minDIST)
				return true;
		}
		return false;
	}

	public boolean isWorthNegotiation() {
		boolean isWorth = true;
		// HashMap

		return isWorth;
	}

	// 閾値を返す
	public double getThreshold(double time) {
		double lowerLimitThreshold = calLowerLimitThreshold(time);
		double threshold = 1.0;

		// 割引効用が存在するかどうか
		if (utilitySpace.isDiscounted()) {
			/*
			 * 割引効用が存在する場合、早期の合意形成が要求される したがって、閾値を早めに下げておく必要性がある
			 */
			threshold = 0.95; // 序盤から閾値を少し下げておくことでOfferの柔軟性を少し上げる
			double temp = 1.0 - (1 - df) * Math.log1p(time * (Math.E - 1));
			// double temp = 1.0 - (1-df) * Math.pow(Math.E, time - df);
			threshold = Math.min(threshold, temp);

		} else {
			/*
			 * 割引効用が存在しない場合、早くに交渉を成立させる意味がなくなるため、より長く交渉を続けることが要求される
			 * つまり、bidデータを多く活用できるため統計的なアプローチでも一定の意味を持つという判断のもと以下のような閾値をとることとする
			 */
			threshold = statisticalThreshold(time, threshold);

		}
		// 閾値の下限との比較
		threshold = Math.max(threshold, lowerLimitThreshold);

		return threshold;

	}

	public double calEmax(Object sender) {
		double m = negotiationInfo.getAverage(sender);
		double sd = negotiationInfo.getStandardDeviation(sender);
		return m + (1 - m) * calWidth(m, sd);
	}

	public double statisticalThreshold(double time, double threshold) {
		double originalAlpha = 3.5 + rv; // 最初は線形的に閾値を下げていく
		HashMap<Object, Double> emaxs = new HashMap<Object, Double>();
		HashMap<Object, Double> targets = new HashMap<Object, Double>();

		ArrayList<Object> opponents = negotiationInfo.getOpponents();
		for (Object sender : opponents) {
			// alphaが大きいほど強硬、小さいほど譲歩に柔軟な姿勢を示す
			double alpha = Math
					.max(0.1,
							originalAlpha
									- calAlpha(negotiationInfo
											.calAssertiveness(sender),
											negotiationInfo
													.calCooperativeness(sender)));

			// double m = negotiationInfo.getAverage(sender);
			// double v = negotiationInfo.getVariancer(sender);
			// double sd = negotiationInfo.getStandardDeviation(sender);

			emaxs.put(sender, calEmax(sender));
			double target = 1 - (1 - emaxs.get(sender)) * Math.pow(time, alpha);
			targets.put(sender, target);

			threshold = Math.min(threshold, target);
		}
		return threshold;
	}

	/**
	 * 統計情報を元に相手の変位幅を推定
	 * 
	 * @param m
	 * @param sd
	 * @return
	 */
	public double calWidth(double m, double sd) {
		if (m > 0.0 && m < 1.0) {
			return Math.sqrt(3.0 / (m - m * m)) * sd;
		} else {
			return Math.sqrt(12) * sd;
		}
	}

	/**
	 * 割引効用がない場合の譲歩の強硬・柔軟性を調整する
	 * 
	 * @param assertiveValue
	 * @param cooperativeValue
	 * @return
	 */
	public double calAlpha(double assertiveValue, double cooperativeValue) {
		double ans = cooperativeValue - assertiveValue;
		return ans;
	}

	/**
	 * 現在の割引効用下の留保価格以上の効用値を得ることができる効用値を閾値の下限とする
	 * 
	 * @param time
	 * @return
	 */
	public double calLowerLimitThreshold(double time) {
		double lowerLimitThreshold = rv;
		if (utilitySpace.isDiscounted()) {
			lowerLimitThreshold = 0.30 + rv * 0.2;
		} else {
			lowerLimitThreshold = 0.55 + rv * 0.2;
		}

		ArrayList<Object> opponents = negotiationInfo.getOpponents();
		double minEmax = 1.0;
		for (Object sender : opponents) {
			minEmax = Math.min(minEmax, calEmax(sender));
		}

		lowerLimitThreshold = Math.max(lowerLimitThreshold, minEmax);
		if (rv == 0.0 && time > 0.99) {
			lowerLimitThreshold *= 0.85;
		}
		System.out.println("LowerThreshold4: " + lowerLimitThreshold);
		return lowerLimitThreshold;
	}

}
