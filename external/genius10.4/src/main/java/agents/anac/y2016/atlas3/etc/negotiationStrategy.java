package agents.anac.y2016.atlas3.etc;

import genius.core.Bid;
import genius.core.utility.AbstractUtilitySpace;

public class negotiationStrategy {
	private AbstractUtilitySpace utilitySpace;
	private negotiationInfo negotiationInfo;

	private double df = 1.0; // 割引係数
	private double rv = 0.0; // 留保価格

	private double A11 = 0.0; // 効用値・・・A:強硬戦略，B:強硬戦略
	private double A12 = 0.0; // 効用値・・・A:強硬戦略，B:妥協戦略
	private double A21 = 0.0; // 効用値・・・A:妥協戦略，B:強硬戦略
	private double A22 = 0.0; // 効用値・・・A:妥協戦略，B:妥協戦略

	static private double TF = 1.0; // 最終提案フェーズの時刻
	static private double PF = 0.5; // 最終提案フェーズにおいて互いのが妥協戦略を選択した場合に，自身が相手よりも先に譲歩する確率

	private boolean isPrinting = false;

	public negotiationStrategy(AbstractUtilitySpace utilitySpace,
			negotiationInfo negotiationInfo, boolean isPrinting) {
		this.utilitySpace = utilitySpace;
		this.negotiationInfo = negotiationInfo;
		this.isPrinting = isPrinting;
		df = utilitySpace.getDiscountFactor();
		rv = utilitySpace.getReservationValue();

		if (this.isPrinting) {
			System.out.println("negotiationStrategy:success");
		}
	}

	// 受容判定
	public boolean selectAccept(Bid offeredBid, double time) {
		try {
			double offeredBidUtil = utilitySpace.getUtility(offeredBid);
			if (offeredBidUtil >= getThreshold(time)) {
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
		// 閾値が留保価格を下回るとき交渉を放棄
		if (utilitySpace.discount(rv, time) >= getThreshold(time)) {
			return true;
		} else {
			return false;
		}
	}

	// 割引後の効用値から割引前の効用値を導出する
	public double pureUtility(double discounted_util, double time) {
		return discounted_util / Math.pow(df, time);
	}

	// 閾値を返す
	public double getThreshold(double time) {
		double threshold = 1.0;
		updateGameMatrix(); // ゲームの表を更新する

		double target = pureUtility(getExpectedUtilityinFOP(), time);

		// 最終提案フェーズの期待効用に基づき，譲歩を行う
		if (utilitySpace.discount(1.0, time) == 1.0) {
			threshold = target + (1.0 - target) * (1.0 - time);
		} else {
			threshold = Math.max(1.0 - time / utilitySpace.discount(1.0, time),
					target);
		}

		// デバッグ用
		if (isPrinting) {
			System.out.println("threshold = " + threshold + ", opponents:"
					+ negotiationInfo.getOpponents());
		}

		return threshold;
	}

	// 最終提案フェイズの混合戦略の期待効用
	private double getExpectedUtilityinFOP() {
		double q = getOpponentEES();
		return q * A21 + (1 - q) * A22;
	}

	// 最終提案ゲームにおける最適混合戦略の均衡点での，相手の混合戦略(p,1.0-p)=(強硬戦略を選択する確率，妥協戦略を選択する確率)を導出する
	private double getOpponentEES() {
		double q = 1.0;
		if ((A12 - A22 != 0) && (1.0 - (A11 - A21) / (A12 - A22) != 0)) {
			q = 1.0 / (1.0 - (A11 - A21) / (A12 - A22));
		}
		if (q < 0.0 || q > 1.0) {
			q = 1.0;
		}
		return q;
	}

	// ゲームの表を更新する
	private void updateGameMatrix() {
		double C; // 妥協案の推定効用値
		if (negotiationInfo.getNegotiatorNum() == 2) {
			C = negotiationInfo.getBOU();
		} else {
			C = negotiationInfo.getMPBU();
		}

		A11 = utilitySpace.discount(rv, 1.0);
		A12 = utilitySpace.discount(1.0, TF);
		if (C >= rv) {
			A21 = utilitySpace.discount(C, TF);
		} else {
			A21 = utilitySpace.discount(rv, 1.0);
		}
		A22 = PF * A21 + (1.0 - PF) * A12;
	}
}
