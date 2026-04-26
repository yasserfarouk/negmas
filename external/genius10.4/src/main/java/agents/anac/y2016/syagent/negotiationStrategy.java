package agents.anac.y2016.syagent;

import genius.core.Bid;
import genius.core.utility.AbstractUtilitySpace;

public class negotiationStrategy {
	private AbstractUtilitySpace utilitySpace;
	private negotiationInfo negotiationInfo;
	private double rv = 0.0; // 留保価格
	private double df = 1.0; // 割引係数
	private boolean rvFlag = false; // 留保価格 rv が存在する(true)か存在しないか(false)を表すフラグ
	private boolean dfFlag = false; // 割引があるか(df != 1.0 なら true)ないか(df == 1.0
									// ならfalse)を表すフラグ
	private double endNegoTime = 1.0; // endNegotiation するべき時間を取得
	private double styleChangeTime = 0.6;// 妥協をやめて、強気になり始める時間

	private boolean isPrinting = false; // 本番用

	// private boolean isPrinting = true; // デバッグ用

	public negotiationStrategy(AbstractUtilitySpace utilitySpace,
			negotiationInfo negotiationInfo, boolean isPrinting) {
		this.utilitySpace = utilitySpace;
		this.negotiationInfo = negotiationInfo;
		this.isPrinting = isPrinting;
		this.rv = utilitySpace.getReservationValue();
		this.df = utilitySpace.getDiscountFactor();
		this.rvFlag = negotiationInfo.setRvFlag();
		this.dfFlag = negotiationInfo.setDfFlag();
		this.endNegoTime = getEndNegotiationTime(); // endNegotiation するべき時間を取得
		this.styleChangeTime = getStyleChangeTime(endNegoTime);// 妥協をやめて、強気になり始める時間

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
		if (df == 0)
			return true; //
		double discountedRv = utilitySpace
				.getReservationValueWithDiscount(time);// 割引後の留保価格
		if (discountedRv >= getThreshold(time)
				|| (Math.pow(df, time) < 0.5 && rvFlag)) {
			return true;
		} else {
			return false;
		}
	} // End of selectEndNegotiation

	/*
	 * // 閾値を返す public double getThreshold(double time) { double threshold =
	 * 1.0; double yamada = getYamada();// styleChangeTime まで
	 * thresholdをいい感じにする係数 yamada if(rvFlag && dfFlag){// 留保価格 - あり 割引 - あり
	 * 
	 * }else if(rvFlag && !dfFlag){// 留保価格 - あり 割引 - なし
	 * 
	 * }else if(!rvFlag && dfFlag){// 留保価格 - なし 割引 - あり
	 * 
	 * }else if(!rvFlag && !dfFlag){// 留保価格 - なし 割引 - なし
	 * 
	 * }
	 * 
	 * 
	 * return threshold; }
	 * 
	 * private double getYamada() {// FIXME double styleChangeTime =
	 * getStyleChangeTime(); double lastUtilityTarget = (df + rv * df)/2; double
	 * yamada = 0.0; return yamada; }
	 * 
	 * // 妥協をやめて、強気になり始める時間 private double getStyleChangeTime() {// FIXME
	 * if(dfFlag) return df; else return 0.6; }
	 */

	/* 以下,ルール間違えてるやつ */
	/*
	 * // 交渉終了判定 public boolean selectEndNegotiation(double time) { if (!rvFlag)
	 * return false; // rv = 0.0 の時はendNegotiation する必要がない //
	 * 割引後の効用が留保価格を下回るとき交渉を放棄 double discountedMaxUtility = Math.pow(df, time);
	 * if (discountedUtility <= rv && discountedMaxUtility <= rv) { return true;
	 * } else { return false; }
	 * 
	 * }else{ return false; }
	 * 
	 * } // End of selectEndNegotiation
	 */
	// 閾値を返す
	public double getThreshold(double time) {
		double threshold = 1.0;
		if (df == 0.0) {
			return 0.0;
		}
		if (isPrinting) {
			System.out.println("rv              :  " + rv);
			System.out.println("df              :  " + df);
			System.out.println("rvFlag          :  " + rvFlag);
			System.out.println("dfFlag          :  " + dfFlag);
			System.out.println("endNegoTime     : " + endNegoTime);
			System.out.println("styleChangeTime : " + styleChangeTime);
		}

		if (rvFlag) {// 留保価格 - あり
			if (dfFlag) { // 留保価格 - あり 割引 - あり
				double yamada = getYamada(styleChangeTime); // styleChangeTime
															// まで
															// thresholdをいい感じにする係数
															// yamada
				double minthreshold = 1.0 + (Math.log(1 - styleChangeTime))
						* yamada; // 閾値の下限
				double shotaro = getShotaro(minthreshold);// styleChangeTime から
															// endNegoTimeまで
															// thresholdをいい感じにする係数
															// shotaro

				if (time <= styleChangeTime) {
					threshold = 1.0 + (Math.log(1.0 - time) * yamada);
				} else if (time < endNegoTime) {
					threshold = minthreshold
							- (Math.log(1.0 + styleChangeTime - time))
							* shotaro;
				}

			} else { // 留保価格 - あり 割引 - なし
				double minthreshold = 1.0 + (Math.log(1 - styleChangeTime)) / 4.7; // 閾値の下限

				if (time <= styleChangeTime) {
					threshold = 1.0 + (Math.log(1.0 - time) / 4.7);
				} else if (time < endNegoTime) {
					threshold = minthreshold
							- (Math.log(1.0 + styleChangeTime - time)) / 3.0;
				}
			}

		} else {
			if (dfFlag) { // 留保価格 - なし 割引 - あり
				double convertTime = time % df;
				double yamada = getYamada(styleChangeTime); // styleChangeTime
															// まで
															// thresholdをいい感じにする係数
															// yamada
				if (convertTime <= styleChangeTime) {
					threshold = 1.0 + (Math.log(1.0 - convertTime) * yamada);
				}
			} else { // 留保価格 - なし 割引 - なし
				threshold = 1.0 - 0.1 * (1.0 + Math.sin(Math.PI
						* (6.0 * time - 0.5)));
			}
		}

		if (time > 0.985) {
			threshold = negotiationInfo.getMyBidAverage();// 一応合意できるように最後にテキトーに下げる
		}
		if (isPrinting) {
			System.out.println("time: " + time);
			System.out.println("threshold: " + threshold);
			System.out.println(":::::::::::::::::::::::::::::::::::::::::::");

		}

		double minthreshold = 1.0 + (Math.log(0.4)) / 4.7; // 閾値の下限

		if (time <= 0.6) {
			threshold = 1.0 + (Math.log(1.0 - time) / 4.7);
		} else {
			threshold = minthreshold - (Math.log(1.6 - time)) / 3.0;
		}

		return threshold;
	}

	private double getShotaro(double minthreshold) {

		// minthreshold - (Math.log(1.0+styleChangeTime-endNegoTime)) * shotaro
		// = 1.0 となればいいので以下で shotaro を計算
		double shotaro = (minthreshold - 1.0)
				/ (Math.log(1.0 + styleChangeTime - endNegoTime));
		return shotaro;
	}

	private double getYamada(double styleChangeTime) {
		// minthreshold (= rv) = (1.0 + (Math.log(1-styleChangeTime)) * yamada)
		// * Math.pow(df,styleChangeTime) とするので以下で yamada を計算
		double yamada = (rv / Math.pow(df, styleChangeTime) - 1.0)
				/ (Math.log(1 - styleChangeTime));
		if (!rvFlag) {
			yamada = (0.7 / Math.pow(df, styleChangeTime) - 1.0)
					/ (Math.log(1 - styleChangeTime)); // テキトーに rv = 0.7 として
														// yamada を計算
		}
		return yamada;
	}

	// 妥協をやめて、強気になり始める時間を求めるメソッド
	// 割引が大きいほど早く threshold を低くする必要がある -> df によって変化させる
	// 自分のさじ加減だけど。
	private double getStyleChangeTime(double endNegoTime) {
		if (endNegoTime == 1.0) {
			if (rvFlag)
				return 0.6; // テキトー
			else
				return df;
		} else {
			return df * endNegoTime; // 　ここもテキトー
		}
	}

	/*
	 * endNegotiation するべき時間を計算するメソッド(留保価格と自分の提案の割引後の効用から) discountedUtility =
	 * Utility * pow(df,time) より、自分が maxUtility(=1.0) で offer した時
	 * discountedUtility = pow(df,time) んで、 rv > discountedUtility となる time に
	 * endNegotiation その time を計算するメソッド
	 */
	private double getEndNegotiationTime() {
		if (df != 0.0) {
			if (rv > df && rvFlag) { // 0.0 < endNegoTime < 1.0 となる場合
				return Math.log(rv) / Math.log(df);
			} else { // endNegotiation する必要がない場合
				return 1.0;
			}
		} else { // df == 0.0 の時
			return 0.0;
		}
	}

}
