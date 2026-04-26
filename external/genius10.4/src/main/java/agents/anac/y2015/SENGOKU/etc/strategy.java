package agents.anac.y2015.SENGOKU.etc;

import java.util.ArrayList;

import genius.core.Bid;
import genius.core.utility.AdditiveUtilitySpace;

public class strategy {
	private AdditiveUtilitySpace utilitySpace;
	private negotiatingInfo negotiatingInfo;
	private bidSearch bidSearch;

	private double df = 1.0; // 割引係数
	private double rv = 0.0; // 留保価格
	public double myThreshold = 1.0; // 閾値
	public int firstState = 0;
	public boolean endNegotieitionFlag = false; // エンドネゴシエーションするときtrue
	private double startTime = 0.0;
	public int myState = 0; // 協力的は１　非協力的は０
	private double eBase = 1.0; // hardheadを使うときの係数の基本
	private double e = eBase; // hardheadを使うときの係数
	private double cThreshold = 1.0; // 以前自分が協力的な時は
	private int cooperationNum = 0; // 非協力的な人の数

	// デバッグ用
	public static boolean isPrinting = false; // メッセージを表示する

	public strategy(AdditiveUtilitySpace utilitySpace, negotiatingInfo negotiatingInfo,
			bidSearch bidSeach) {
		this.utilitySpace = utilitySpace;
		this.negotiatingInfo = negotiatingInfo;
		df = utilitySpace.getDiscountFactor();
		rv = utilitySpace.getReservationValue();
	}

	public void updateThreshold(double num) {
		myThreshold = num;
	}

	// 自身のラウンド数を返す
	public double getThreshold() {
		return myThreshold;
	}

	// 受容判定
	// 閾値より上なら真を返す！
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
		return false;
	}

	// 閾値を決めるときに最初に呼ばれるメソッド
	public double getThreshold(double time) {

		if (negotiatingInfo.getMyBidHistory().size() < 50) { // 最初はデータ収集で強気でいく
			if (negotiatingInfo.getMyBidHistory().size() > 10) {
				cooperationCheck();
			}
			return strongThreshold();
		}

		switch (negotiatingInfo.getActionFlag()) {
		case 0:// オファーするとき
			return getOfferThreshold(time);
		case 1:// アクセプトする時
			return getAcceptThreshold(time);
		}
		return myThreshold;
	}

	// これが閾値！　オファーするときの閾値決め
	public double getOfferThreshold(double time) {
		double threshold = maxthreshold(0); // 閾値の最大を取得
		switch (myState) {
		case 0: // 裏切る　
			startTime = time;
			return bThreshold();

		case 1: // 協力する //おふぁーで相手に戦略を公開するから偽の裏切っているようにみせる！少数はに入るようにする　１　１
			return cooperateThreshold(time);
			// threshold = maxthreshold(1); //閾値の最小でいい
			// return hardThreshold(threshold, time);
			// return hardThresholdDf(time);
		case 2: // 協力する
			return cooperateThreshold(time);
			// threshold = maxthreshold(1); //閾値の最小でいい
			// return hardThreshold(threshold, time);
			// return hardThresholdDf(time);
		default:
			return strongThreshold();
		}
	}

	// これが閾値！ accept用
	public double getAcceptThreshold(double time) {

		double threshold = maxthreshold(0); // 閾値の最大を取得
		// System.out.println("最大閾値:"+threshold);
		int num = cooperationCheck(); // 非協力的な人数
		myState = num;

		switch (num) {
		case 0: // 裏切る　
			return bThreshold();
		case 1: // 協力する
			return cooperateThreshold(time);
			// threshold = maxthreshold(1); //閾値の最小でいい
			// return hardThreshold(threshold, time);
			// return hardThresholdDf(time);
		case 2: // 協力する
			return cooperateThreshold(time);
			// threshold = maxthreshold(1); //閾値の最小でいい
			// return hardThreshold(threshold, time);
			// return hardThresholdDf(time);
		default:
			// System.out.println("警告-----------------------------");
			return 1.0;
		}
	}

	// 非協力的な人の人数を返す
	private int cooperationCheck() {
		ArrayList<Object> opponents = negotiatingInfo.getOpponents();
		int count = 0;

		for (Object sender : opponents) {
			// アクセプトがあるかないか
			double a = negotiatingInfo.getNewAccept(sender);
			if (a > 0) {
				// System.out.println("アクセプトで協力"+sender);
				negotiatingInfo.updateopponentsCooperateHistory(sender, true);
				continue;
			}

			// 効用値が上昇傾向にある時
			if (thresholdUpRate(sender, 7) > 0.0) {
				// System.out.println("効用値の上昇で協力"+sender);
				negotiatingInfo.updateopponentsCooperateHistory(sender, true);
				continue;
			}

			// 同じ提案数 半分以上同じ提案のとき
			double s = negotiatingInfo.getNewSumNum(sender);
			// System.out.println("NewSumNum"+s);
			if (s > 2) {
				// System.out.println("同じ提案で非協力"+sender);
				negotiatingInfo.updateopponentsCooperateHistory(sender, false);
				count++;
				continue;
			}

			// 標準偏差
			double sd = negotiatingInfo.getNewStandardDeviation(sender);
			double aveSd = negotiatingInfo.getStandardDeviation(sender);
			// System.out.println("最新の標準偏差"+sd);
			if (sd < aveSd) {
				// System.out.println("偏差が平均より小さくて非協力"+sender);
				negotiatingInfo.updateopponentsCooperateHistory(sender, false);
				count++;
				continue;
			}

			// １０回の平均が期待値以下の時 その人の！
			double ave = negotiatingInfo.getNewAverage(sender);
			if (ave < negotiatingInfo.getAverage(sender)) {
				// System.out.println("平均以下で非協力"+sender +"平均：" + ave);
				negotiatingInfo.updateopponentsCooperateHistory(sender, false);
				count++;
				continue;
			}
			negotiatingInfo.updateopponentsCooperateHistory(sender, true);

		}
		// System.out.println("非協力的人数：" + count);
		return count;
	}

	// 相手の協力する確率をこれまでの情報より計算して出す。　すべての相手の合計で計算する。
	public ArrayList<Double> getCooperateRate() {
		ArrayList<Double> rateList = new ArrayList<Double>();
		ArrayList<Object> opponents = negotiatingInfo.getOpponents();
		for (Object sender : opponents) {
			double allCount = 0;
			double copperateCount = 0;
			ArrayList<Boolean> states = negotiatingInfo
					.getopponentsCooperateHistory(sender);
			for (Boolean state : states) {
				allCount++;
				if (state) {
					copperateCount++;
				}
			}
			if (copperateCount == 0) {
				rateList.add(0.0);
			} else {
				rateList.add(copperateCount / allCount);
			}
		}
		return rateList;
	}

	// 協力状態するときの閾値
	public double cooperateThreshold(double time) {

		// 相手の協力度の計算
		ArrayList<Double> rateList = getCooperateRate();
		double cooperateRate = 0.0; // 他２人の合計の協力レートを算出する
		for (int i = 0; i < rateList.size(); i++) {
			if (i == 0) {
				cooperateRate = rateList.get(i);
			} else {
				cooperateRate = cooperateRate * rateList.get(i);
			}
		}

		double minThreshold = maxthreshold(1); // 相手の平均と分散から得られる最低
		double maxThreshold = bThreshold(); // 非協力のときの裏切りの閾値
		double fThreshold = cooperateRate * maxThreshold + (1 - cooperateRate)
				* minThreshold; // 最終的な効用値を協力レートから算出
		double dffThreshold = fThreshold * df; // 最終の結果に割引効用を適用する。
		double rvUtil = rv * Math.pow(df, time); // 現在もらうことができる留保価格を算出する。

		double hardThreshold = fThreshold + (1 - fThreshold)
				* (1 - Math.pow(time, 1 / e)); // 実際の使用する閾値　ファティマの式を線形でいれる

		if (dffThreshold < rvUtil) { // エンドネゴシエーション
			if (isPrinting) {
				System.out.println("最後の閾値にリザ：" + dffThreshold + "リザベーションバリュー："
						+ rvUtil);
			}
			double harddffThreshold = dffThreshold + (1 - dffThreshold)
					* (1 - Math.pow(time, 0.5)); // 最後の予想値にファティマで落としていく
			if (isPrinting) {
				System.out.println("線形で最後のリザにおとしていくときの閾値:" + harddffThreshold);
			}
			if (harddffThreshold < rvUtil) { // エンドネゴシエーション
				endNegotieitionFlag = true;
				return 1.0;
			} else { // 割引効用に応じてもっと早めに落としていく
				return harddffThreshold;
			}
		}

		/*
		 * if (df == 0.0){ //割引効用がないとき //期待予想値にファティマの式を代入する。 hardThreshold =
		 * fThreshold + (1 -fThreshold) * (1 - Math.pow(time,1/e));
		 * //System.out.println("協力閾値"+hardThreshold); }else { //割引効用があるとき
		 * hardThreshold = fThreshold / Math.pow(df,time); double
		 * firstHardthreshold = dffThreshold + (1 - dffThreshold) * (1 -
		 * Math.pow(time,1/e)); if (hardThreshold < firstHardthreshold) {
		 * hardThreshold = firstHardthreshold; } }
		 */

		if (isPrinting) {
			System.out.println("今の協力値" + hardThreshold + "最後の予想値" + fThreshold);
		}
		cThreshold = hardThreshold;
		return hardThreshold;
	}

	// 期待できる一番高い閾値を返す
	public double maxthreshold(int flag) {
		// opponents 交渉相手の一覧を返す
		ArrayList<Object> opponents = negotiatingInfo.getOpponents();
		double threshold = 1.0;
		double mAll = 0.0;
		double sdAll = 0.0;

		for (Object sender : opponents) {
			// 平均
			double m = negotiatingInfo.getAverage(sender);
			mAll = mAll + m;
			// 分散
			// double v = negotiatingInfo.getVariancer(sender);
			// 標準偏差
			double sd = negotiatingInfo.getStandardDeviation(sender);
			sdAll = sdAll + sd;
		}

		mAll = mAll / opponents.size();
		// System.out.println("平均all"+mAll);
		sdAll = sdAll / opponents.size();
		// System.out.println("偏差all"+sdAll);

		if (flag == 0) { // このメソッドの引数によって変えれる
			threshold = mAll + sdAll;
		} else if (flag == 1) {
			threshold = mAll - sdAll;
		}
		// System.out.println("相手から今の状況でとれる最大閾値"+threshold);

		return threshold;

	}

	// ある区間で相手ごとに閾値が何パーセント上昇するか調べる numはどれふぁけのを観察するか＞
	private double thresholdUpRate(Object sender, int num) {
		ArrayList<Double> List = negotiatingInfo.getBidValue(sender);
		double rateSum = 0;
		if (List.size() < num) {
			return 0;
		} // 閾値の配列に要素がないとき
		for (int i = num; i < num; i++) {
			double now = List.get(List.size() - 1 - i);
			double old = List.get(List.size() - 2 - i);
			double rate = (now - old) / now;
			rateSum = rateSum + rate;
		}
		return rateSum;
	}

	// 閾値に対して　引数に対して収束していく関数
	private double hardThreshold(double threshold, double time) {
		double rateTime = 1 / (1 - startTime);
		double useTime = (time - startTime) * rateTime;
		double hardthreshold = threshold + (1 - threshold)
				* (1 - Math.pow(useTime, 1 / e));
		return hardthreshold;
	}

	// 非協力のときの閾値を返す
	private double hardThresholdDf(double time) {
		double hThreshold = maxthreshold(1) * Math.pow(df, time); // 割引効用を考えたあたい
		if (myThreshold * df > hThreshold) {
			firstState = 0;
			startTime = time;
			return myThreshold;
		} else {
			firstState = 1;
			return hardThreshold(hThreshold, time);
		}
	}

	// 裏切りの閾値を返す 相手提案のなかの最大の閾値に分散を適用させて使用
	private double bThreshold() {
		ArrayList<Object> opponents = negotiatingInfo.getOpponents();
		double util = 0.0; // 最大有効値
		double sdAll = 0.0; // 標準偏差

		for (Object sender : opponents) {
			double sd = negotiatingInfo.getStandardDeviation(sender);
			sdAll = sdAll + sd;

			double senderUtil = negotiatingInfo.getmaxUtil(sender);
			if (util == 0.0) {
				util = senderUtil;
			} else if (util > senderUtil) {
				util = senderUtil;
			}
		}
		sdAll = sdAll / opponents.size();

		double threshold = util + sdAll * Math.random();

		if (maxthreshold(1) > threshold) { // 裏切り閾値が最低期待値を下回っているときは平均の分散の最大と交換
			threshold = maxthreshold(0);
		}

		if (cThreshold != 1.0 && cThreshold > threshold) {
			threshold = cThreshold;
		}

		if (threshold > 1.0) { // 裏切りの閾値が１を超えてしまった時
			return 1.0 - (sdAll * Math.random());
		}

		return threshold;
	}

	// 強気の閾値は自分のいい効用情報を公開していく
	private double strongThreshold() {
		return 1.0 - (Math.random() * 0.1);

		/*
		 * ArrayList<Object> opponents = negotiatingInfo.getOpponents(); if
		 * (negotiatingInfo.getRound() < 3){ //相手の情報がないとき return 1.0 -
		 * (Math.random() * 0.1); }else{ double sdAll=0.0; //標準偏差
		 * 
		 * for(Object sender:opponents){ double sd =
		 * negotiatingInfo.getStandardDeviation(sender); sdAll = sdAll + sd; }
		 * sdAll = sdAll/opponents.size();
		 * 
		 * return 1.0 - (Math.random() * sdAll); }
		 */
	}

	// 係数調節システム
	private void eSet(double threshold, double time) {
		System.out.println("--------set------------" + time + "------"
				+ (double) 20 / 180);
		if (time < (double) 20 / 180) {
			return;
		}// 最初はデータがないからできない

		ArrayList<Object> opponents = negotiatingInfo.getOpponents();
		double rateMax = 0.0;
		for (Object sender : opponents) {
			double rate = thresholdUpRate(sender, 7);
			if (rate > rateMax) {
				rateMax = rate; // 上がる予定
			}
		}

		int flag = 0;
		double myRate = (hardThreshold(threshold, time) - hardThreshold(
				threshold, time + (double) 1 / 180))
				/ hardThreshold(threshold, time);
		if (myRate > rateMax) { // 自分の方が下げている
			flag = 0;
		} else { // 相手の方が下げている
			e = e + 0.1;
			flag = 1;
		}

		while (true) {
			// １つ後の閾値とのレート 自分レート　プラスに出るようにする
			myRate = (hardThreshold(threshold, time) - hardThreshold(threshold,
					time + 1 / 180)) / hardThreshold(threshold, time);
			if (myRate > rateMax) { // 自分の方が下げている
				if (flag == 1) {
					return;
				}
				e = e - 0.1;
			} else { // 相手の方が下げている
				if (flag == 0) {
					return;
				}
				e = e + 0.1;
			}
		}
	}

}
