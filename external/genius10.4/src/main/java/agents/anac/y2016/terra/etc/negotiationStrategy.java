package agents.anac.y2016.terra.etc;

import java.util.HashMap;

import genius.core.Bid;
import genius.core.utility.AbstractUtilitySpace;

//	戦略
//	提案＋合意の双方を考える必要性
//		序盤、中盤、終盤で交渉のまとめ方を変える
//		前半(0~90)
//			基本的には自分の効用値の高いオファー行う(相手が自分の提案から良いのを見つけてくれる可能性も高まるため)
//			効用値の閾値は高め
//			留保価格を考え早めに交渉がまとまるのであればまとめてしまう
//		中盤	~終盤(90~170)
//			自分にとって欲張ってどのくらい相手に提案を飲ませられるかの最大値を提示
//		終盤(170~180)
//			まとめの時期
//			相手のこれまでの行動により自分の閾値を決める(基本は相手が妥協してくれそうなギリギリのラインを狙う)
//			相手が提案(二人の提案の最大値のうち小さい方)よりも低い値で提案+合意を行わないようにする
//			また、minの値を決めそれより低い数値では提案+合意を行わない
//
//	その他のアイデア
//				
//

public class negotiationStrategy {
	private AbstractUtilitySpace utilitySpace;
	private negotiationInfo negotiationInfo;
	private double rv;// 留保価格

	//
	public negotiationStrategy(AbstractUtilitySpace utilitySpace,
			negotiationInfo negotiationInfo) {
		this.utilitySpace = utilitySpace;
		this.negotiationInfo = negotiationInfo;
		rv = utilitySpace.getReservationValue();
	}

	// 受容判定
	public boolean selectAccept(Bid offeredBid, double time) {
		try {
			if (utilitySpace.getUtility(offeredBid) >= getThreshold(time)/*-0.1*(1-time)*/) {
				return true;
			} else {
				if (time > (173.0 / 180.0)
						|| (time > (15.0 / 180.0) && utilitySpace
								.getDiscountFactor() < 0.9)
						|| (negotiationInfo.getIssues().size() < 5 && time > (10.0 / 180.0))) {
					HashMap<Object, Double> recentOfferedUtility = negotiationInfo
							.getUtilityAverage(negotiationInfo
									.getRecentOfferedList());
					if (recentOfferedUtility.isEmpty())
						return false;
					double ave = 0;
					for (Double tave : recentOfferedUtility.values()) {
						ave += tave;
					}
					ave /= 2;
					if (time < (178.0 / 180.0)) {
						if (time < (30.0 / 180.0)) {
							if ((utilitySpace.getUtility(offeredBid) > 0.72))
								return true;
							else
								return false;
						}
						if ((utilitySpace.getUtility(offeredBid) - utilitySpace
								.getUtility(offeredBid)
								* (1.0 - utilitySpace.getDiscountFactor())
								* time)
								- (ave - ave
										* (1.0 - utilitySpace
												.getDiscountFactor()) * time) > 0.0)
							return true;
					} else
						return true;
				}
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
		HashMap<Object, Double> recentOfferedUtility = negotiationInfo
				.getUtilityAverage(negotiationInfo.getRecentOfferedList());

		// System.out.println(utilitySpace.getDiscountFactor());
		if (recentOfferedUtility.isEmpty() || time < (15.0 / 180.0) || rv < 0.1)
			return false;

		double ave = 0;
		for (Double tave : recentOfferedUtility.values()) {
			ave += tave;
		}
		ave /= 2;

		if ((ave - ave * (1.0 - utilitySpace.getDiscountFactor()) * time)
				- (rv - rv * (1.0 - utilitySpace.getDiscountFactor() * time)) <= 0)
			return true;
		return false;
	}

	enum NegotiationPhase {
		FIRST_Claim, Claim, SUMMARIZE
	};

	private NegotiationPhase selectPhase(double time, double ave) {
		if ((time < (30.0 / 180.0))
				|| (utilitySpace.getDiscountFactor() > 0.9
						&& time < (170.0 / 180.0) || (rv < 0.1)))
			return NegotiationPhase.FIRST_Claim;
		if ((ave - ave * (1.0 - utilitySpace.getDiscountFactor()) * time) > 0.3)
			return NegotiationPhase.Claim;
		return NegotiationPhase.SUMMARIZE;
	}

	public double getThreshold(double time) {
		HashMap<Object, Double> recentOfferedUtility = negotiationInfo
				.getUtilityAverage(negotiationInfo.getRecentOfferedList());
		double ave = 0;
		for (Double tave : recentOfferedUtility.values()) {
			ave += tave;
		}
		ave /= 2;
		NegotiationPhase phase = selectPhase(time, ave);

		if (phase == NegotiationPhase.FIRST_Claim) {
			double threshold = 0;

			if (utilitySpace.getDiscountFactor() > 0.9) {// 割引効用がない場合
				if (time < (60.0 / 180.0))
					threshold = 1.0 - time * 0.5;//
				else
					threshold = 0.9;
			} else {// ある場合
			// System.out.println(time*(1-utilitySpace.getDiscountFactor()));
				threshold = 1 - time * (1 - utilitySpace.getDiscountFactor())
						* 2.3;// Math.pow(time*time*time,(1-utilitySpace.getDiscountFactor()));
				// threshold=1.0-time*0.5;
				if (threshold < 0.7)
					threshold = 0.7;
			}
			return threshold;
		} else if (phase == NegotiationPhase.Claim) {
			HashMap<Object, Double> utilityMax = new HashMap<>();
			HashMap<Object, Double> agreedUtilityMax = negotiationInfo
					.getUtilityMax(negotiationInfo.getAgreedList());
			HashMap<Object, Double> offeredUtilityAverage = negotiationInfo
					.getUtilityAverage(negotiationInfo.getRecentOfferedList());

			for (Object agent : negotiationInfo.getOpponet()) {
				if (agreedUtilityMax.containsKey(agent)) {
					if (offeredUtilityAverage.get(agent) > agreedUtilityMax
							.get(agent)) {
						utilityMax.put(agent, offeredUtilityAverage.get(agent));
					} else {
						utilityMax.put(agent, agreedUtilityMax.get(agent));
					}
				} else {
					utilityMax.put(agent, offeredUtilityAverage.get(agent));
				}
			}

			double threshold = 0.5;
			double p1 = 1.3;
			for (double tvalue : utilityMax.values()) {
				if (threshold < tvalue)
					threshold = tvalue;
			}
			if (threshold * p1 > 0.9)
				return threshold = 0.9;
			return threshold = threshold * p1;
		} else {
			// minを相手が合意するできるだけmaxに設定
			// HashMap<Object, Double>
			// offeredUtilityMax=negotiationInfo.getUtilityMax(negotiationInfo.getOfferedList());
			HashMap<Object, Double> offeredUtilityMax = negotiationInfo
					.getUtilityAverage(negotiationInfo.getRecentOfferedList());
			HashMap<Object, Double> agreedUtilityMax = negotiationInfo
					.getUtilityMax(negotiationInfo.getAgreedList());

			HashMap<Object, Double> utilityMax = new HashMap<>();
			for (Object agent : negotiationInfo.getOpponet()) {
				if (agreedUtilityMax.containsKey(agent)) {
					if (offeredUtilityMax.get(agent) > agreedUtilityMax
							.get(agent)) {
						utilityMax.put(agent, offeredUtilityMax.get(agent));
					} else {
						utilityMax.put(agent, agreedUtilityMax.get(agent));
					}
				} else {
					utilityMax.put(agent, offeredUtilityMax.get(agent));
				}
			}
			double threshold = 2.0;// 2人のうち小さい方の効用値を閾値とする

			double min = 0.75 - time * (1 - utilitySpace.getDiscountFactor())
					* 0.5;
			// if(negotiationInfo.getRound()<100)min=0.55; //minよりも閾値が小さくならないように

			for (double tvalue : utilityMax.values()) {
				if (threshold > tvalue)
					threshold = tvalue;
			}
			if (threshold < min)
				threshold = min;
			return threshold;
		}

	}
}
