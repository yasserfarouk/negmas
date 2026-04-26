package agents.anac.y2015.SENGOKU.etc;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import genius.core.Bid;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.utility.AdditiveUtilitySpace;

public class negotiatingInfo {
	private AdditiveUtilitySpace utilitySpace; // 効用空間
	private List<Issue> issues; // 論点
	private ArrayList<Object> opponents; // 自身以外の交渉参加者のsender
	private ArrayList<Bid> MyBidHistory = null; // 提案履歴 自分！
	private HashMap<Object, ArrayList<Bid>> opponentsBidHistory = null; // 提案履歴　相手！　ハッシュマップで保存！
	private HashMap<Object, ArrayList<Boolean>> opponentsCooperateHistory = null; // 相手の協力履歴
	private HashMap<Object, Double> opponentsAverage; // 平均
	private double allAverage = 0.0; // すべての平均
	private HashMap<Object, Double> opponentsVariance; // 分散
	private double allVariance = 0.0; // すべての分散
	private HashMap<Object, Double> opponentsSum; // 和
	private HashMap<Object, Double> opponentsPowSum; // 二乗和
	private HashMap<Object, Double> opponentsStandardDeviation; // 標準偏差
	private HashMap<Issue, HashMap<Value, Double>> valueRelativeUtility = null; // 自身の効用空間における各論点値の相対効用値行列（線形効用空間用）
	private HashMap<Issue, HashMap<Value, Double>> valueNumUtility = null; // 自身の効用空間における各論点値の相対効用値行列（線形効用空間用
	private int round = 0; // 自分の手番数
	private int negotiatorNum = 3; // 交渉者数

	private int acceptNum = 0; // アクセプト数
	private double acceptRate = 0.0; // アクセプト率１ラウンドあたり

	private boolean isLinerUtilitySpace = true; // 線形効用空間であるかどうか

	// オリジナルプロパティー
	private int ActionFlag = 0; // 戦術番号
	private HashMap<Object, ArrayList<Bid>> newBidHistory = null; // 提案履歴　相手！　ハッシュマップで保存！近い履歴
	private HashMap<Object, Double> newAverage; // 平均
	private HashMap<Object, Double> newVariance; // 分散
	private HashMap<Object, Double> newStandardDeviation; // 標準偏差
	private HashMap<Object, Double> newSumNum; // 重複回数
	private HashMap<Object, Double> newAccept; // アクセプトフラグ
	private ArrayList<Double> allMaxShreshold; // 最大の閾値をたえず保存
	private HashMap<Object, ArrayList<Double>> BidValue; // 提案したビットの自分の効用値
	private Object OfferPlayer = null;
	private HashMap<Object, Double> maxUtil; // 相手の提案の最大値
	private ArrayList<Bid> LastOfferBidHistory = null; // 最後の提案に使う用のビッドリスト
	private ArrayList<Double> aveTime = new ArrayList<Double>(); // 自分のターンが回ってくるまでの時間
	public Boolean lastFlag = false; // 最後の妥協にしようするフラグ

	public negotiatingInfo(AdditiveUtilitySpace utilitySpace) {
		// 初期化
		this.utilitySpace = utilitySpace;
		issues = utilitySpace.getDomain().getIssues();
		opponents = new ArrayList<Object>();
		MyBidHistory = new ArrayList<Bid>();
		opponentsBidHistory = new HashMap<Object, ArrayList<Bid>>();
		opponentsCooperateHistory = new HashMap<Object, ArrayList<Boolean>>();
		opponentsAverage = new HashMap<Object, Double>();
		opponentsVariance = new HashMap<Object, Double>();
		opponentsSum = new HashMap<Object, Double>();
		opponentsPowSum = new HashMap<Object, Double>();
		opponentsStandardDeviation = new HashMap<Object, Double>();
		valueRelativeUtility = new HashMap<Issue, HashMap<Value, Double>>();
		valueNumUtility = new HashMap<Issue, HashMap<Value, Double>>();

		// 新しいのの初期化
		newBidHistory = new HashMap<Object, ArrayList<Bid>>();
		newAverage = new HashMap<Object, Double>();
		newVariance = new HashMap<Object, Double>();
		newStandardDeviation = new HashMap<Object, Double>();
		newSumNum = new HashMap<Object, Double>();
		newAccept = new HashMap<Object, Double>();
		BidValue = new HashMap<Object, ArrayList<Double>>();
		allMaxShreshold = new ArrayList<Double>();
		maxUtil = new HashMap<Object, Double>();
		LastOfferBidHistory = new ArrayList<Bid>();

		try {
			initValueRelativeUtility();
		} catch (Exception e) {
			System.out.println("相対効用行列の初期化に失敗しました");
			e.printStackTrace();
		}
	}

	public void initOpponent(Object sender) {
		initNegotiatingInfo(sender); // 交渉情報を初期化
		opponents.add(sender); // 交渉参加者にsenderを追加
	}

	public void updateInfo(Object sender, Bid offeredBid) {
		try {
			updateNegotiatingInfo(sender, offeredBid);
		} // 交渉情報の更新
		catch (Exception e1) {
			System.out.println("交渉情報の更新に失敗しました");
			e1.printStackTrace();
		}
	}

	private void initNegotiatingInfo(Object sender) {
		opponentsBidHistory.put(sender, new ArrayList<Bid>());
		opponentsCooperateHistory.put(sender, new ArrayList<Boolean>());
		opponentsAverage.put(sender, 0.0);
		opponentsVariance.put(sender, 0.0);
		opponentsSum.put(sender, 0.0);
		opponentsPowSum.put(sender, 0.0);
		opponentsStandardDeviation.put(sender, 0.0);
		newBidHistory.put(sender, new ArrayList<Bid>());
		newAverage.put(sender, 0.0);
		newVariance.put(sender, 0.0);
		newStandardDeviation.put(sender, 0.0);
		newSumNum.put(sender, 0.0);
		newAccept.put(sender, 0.0);
		BidValue.put(sender, new ArrayList<Double>());
		maxUtil.put(sender, 0.0);
	}

	// 相対効用行列の初期化
	private void initValueRelativeUtility() throws Exception {
		ArrayList<Value> values = null;
		for (Issue issue : issues) {
			valueRelativeUtility.put(issue, new HashMap<Value, Double>()); // 論点行の初期化
			valueNumUtility.put(issue, new HashMap<Value, Double>()); // 論点行の初期化
			values = getValues(issue);
			for (Value value : values) { // 論点行の要素の初期化
				valueRelativeUtility.get(issue).put(value, 0.0);
				valueNumUtility.get(issue).put(value, 0.0);
			}
		}
	}

	// 相対効用行列の導出
	public void setValueRelativeUtility(Bid maxBid) throws Exception {
		ArrayList<Value> values = null;
		Bid currentBid = null;
		for (Issue issue : issues) {
			currentBid = new Bid(maxBid);
			values = getValues(issue);
			for (Value value : values) {
				currentBid = currentBid.putValue(issue.getNumber(), value);
				valueRelativeUtility.get(issue).put(
						value,
						utilitySpace.getUtility(currentBid)
								- utilitySpace.getUtility(maxBid));
			}
		}
	}

	// 相手の提案している値の数を数えていく 相手すべての頻度行列
	public void setValueNumUtility(Bid maxBid) throws Exception {
		ArrayList<Value> values = null;
		Bid currentBid = null;
		for (Issue issue : issues) {
			currentBid = new Bid(maxBid);
			Value onValue = currentBid.getValue(issue.getNumber());
			values = getValues(issue);
			for (Value value : values) {
				if (onValue.equals(value)) {
					Double i = valueRelativeUtility.get(issue).get(value);
					valueRelativeUtility.get(issue).put(value, i + 1.0);
				}
			}
		}
	}

	public void updateNegotiatingInfo(Object sender, Bid offeredBid)
			throws Exception {
		OfferPlayer = sender;
		opponentsBidHistory.get(sender).add(offeredBid); // 提案履歴

		// utilは相手の提案に対するそのあたい効用値である！　自分の提案は自分でわかる！
		// utilitySpace.getUtility 引数ビットにたいする公用地を返す自分のみ！
		double util = utilitySpace.getUtility(offeredBid);
		updateBidValue(sender, util);
		if (maxUtil.get(sender) < util) {
			maxUtil.put(sender, util);
		}

		if (!lastFlag) {
			if (LastOfferBidHistory.size() == 0) {
				LastOfferBidHistory.add(offeredBid);
			} else if (utilitySpace.getUtility(LastOfferBidHistory
					.get(LastOfferBidHistory.size() - 1)) < util) {
				LastOfferBidHistory.add(offeredBid);
			}
		}

		// 新しいものを更新していくクラス！
		test(sender);

		round++;
		opponentsSum.put(sender, opponentsSum.get(sender) + util); // 和
		opponentsPowSum.put(sender,
				opponentsPowSum.get(sender) + Math.pow(util, 2)); // 二乗和

		int round_num = opponentsBidHistory.get(sender).size();
		opponentsAverage.put(sender, opponentsSum.get(sender) / round_num); // 平均
		opponentsVariance.put(sender, (opponentsPowSum.get(sender) / round_num)
				- Math.pow(opponentsAverage.get(sender), 2)); // 分散

		if (opponentsVariance.get(sender) < 0) {
			opponentsVariance.put(sender, 0.0);
		}
		opponentsStandardDeviation.put(sender,
				Math.sqrt(opponentsVariance.get(sender))); // 標準偏差

	}

	// 提案履歴から最新x個を取り出して平均とかを求める！
	public void test(Object sender) throws Exception {
		// System.out.println(opponentsBidHistory.get(sender));
		ArrayList<Bid> bidlist = opponentsBidHistory.get(sender);
		int x = 6; // どれだけの最新でーたを採用するか
		double sumNum = 0; // どれだけ重複しているか
		int bitSize = bidlist.size(); // どれだけ履歴があるか
		int count = x; // 実際の入れる配列数
		if (bitSize < x) {
			count = bitSize;
		}

		double sum = 0;
		double ave = 0;
		double v = 0;
		double sd = 0;

		ArrayList<Bid> newlist = new ArrayList<Bid>();

		for (int i = 0; count > i; i++) {
			Bid inBit = bidlist.get(bitSize - i - 1); // 入れるビット
			// 何回同じ提案があるか計算
			if (newlist.contains(inBit)) {
				sumNum = sumNum + 1.0;
			}

			newlist.add(inBit);
			double util = utilitySpace.getUtility(inBit);
			sum = sum + util;
		}

		ave = sum / count; // 平均を出す
		for (int i = 0; count > i; i++) {
			newlist.add(bidlist.get(bitSize - i - 1));
			double util = utilitySpace.getUtility(bidlist.get(bitSize - i - 1));
			v = v + Math.pow((ave - util), 2);
		}

		sd = Math.sqrt(v);
		newBidHistory.put(sender, newlist);
		newAverage.put(sender, ave);
		newVariance.put(sender, v);
		newStandardDeviation.put(sender, sd);
		newSumNum.put(sender, sumNum);
		if (newAccept.get(sender) > 0.0) {
			newAccept.put(sender, newAccept.get(sender) - 1.0 / x);
		}
	}

	// 他人がアクセプトしたときのデータ更新
	public void updateAccept(Object sender) {
		// すべてのアクセプトの数
		acceptNum++;
		// １巡のアクセプト率
		double rate = 1.0 / opponents.size();
		acceptRate = acceptRate + rate;
		// ある人の最近アクセプト
		newAccept.put(sender, 1.0);
	}

	// 得られる最大をすべて保存
	public void updateMaxThreshold(double threshold) {
		allMaxShreshold.add(threshold);
	}

	// //得られる最大を返す
	public ArrayList<Double> getMaxThreshold() {
		return allMaxShreshold;
	}

	// アクセプト率をリセット
	public void resetAcceptRate() {
		acceptRate = 0.0;
	}

	// 自分が何をするか　オファーのとき
	public void myActionOffer() {
		ActionFlag = 0;
	}

	// 自分が何をするか　アクセプトの時
	public void myActionAccept() {
		ActionFlag = 1;
	}

	// 交渉者数を返す
	public void updateOpponentsNum(int num) {
		negotiatorNum = num;
	}

	// 線形効用空間でない場合
	public void utilitySpaceTypeisNonLiner() {
		isLinerUtilitySpace = false;
	}

	// 自身の提案情報の更新
	public void updateMyBidHistory(Bid offerBid) {
		MyBidHistory.add(offerBid);
	}

	// 平均
	public double getAverage(Object sender) {
		return opponentsAverage.get(sender);
	}

	// 分散
	public double getVariancer(Object sender) {
		return opponentsVariance.get(sender);
	}

	// 最新の平均
	public double getNewAverage(Object sender) {
		return newAverage.get(sender);
	}

	// 最新の分散
	public double getNewVariancer(Object sender) {
		return newVariance.get(sender);
	}

	// 平均
	public double getAllAverage(double m) {
		return allAverage;
	}

	// 分散
	public double getAllVariancer(double v) {
		return allVariance;
	}

	// 標準偏差
	public double getStandardDeviation(Object sender) {
		return opponentsStandardDeviation.get(sender);
	}

	// 新しい標準偏差
	public double getNewStandardDeviation(Object sender) {
		return newStandardDeviation.get(sender);
	}

	// 相手の協力状態を返す
	public ArrayList<Double> getBidValue(Object sender) {
		return BidValue.get(sender);
	}

	// 相手の協力状態を更新
	public void updateBidValue(Object sender, double util) {
		BidValue.get(sender).add(util);
	}

	// 相手の提案履歴の要素数を返す
	public int getPartnerBidNum(Object sender) {
		return opponentsBidHistory.get(sender).size();
	}

	// 相手の提案履歴を返す
	public ArrayList<Bid> getPartnerBid(Object sender) {
		return opponentsBidHistory.get(sender);
	}

	// 相手の協力情報を更新する　協力ならtrue
	public void updateopponentsCooperateHistory(Object sender, boolean state) {
		opponentsCooperateHistory.get(sender).add(state);
	}

	// 相手の協力情報をリストで返す
	public ArrayList<Boolean> getopponentsCooperateHistory(Object sender) {
		return opponentsCooperateHistory.get(sender);
	}

	// 最後のリストを返す
	public Bid getLastOfferBidHistry() {
		int size = LastOfferBidHistory.size();
		return LastOfferBidHistory.get(size - 1);
	}

	// 最後のビットのリストのサイズを返す
	public int getLastOfferBidNum() {
		return LastOfferBidHistory.size();
	}

	public void removeLastOfferBidHistry() {
		int size = LastOfferBidHistory.size();
		LastOfferBidHistory.remove(size - 1);
	}

	// 自分の提案履歴を返す
	public ArrayList<Bid> getMyBidHistory() {
		return MyBidHistory;
	}

	// オファーしている相手を返す
	public Object getOfferPlayer() {
		return OfferPlayer;
	}

	// 新しい提案の重複数
	public double getNewSumNum(Object sender) {
		return newSumNum.get(sender);
	}

	// 新しいアクセプト数
	public double getNewAccept(Object sender) {
		return newAccept.get(sender);
	}

	// 自身のラウンド数を返す
	public int getRound() {
		return round;
	}

	// 自身の行動フラグを返す
	public int getActionFlag() {
		return ActionFlag;
	}

	// アクセプト数を返す
	public int getAccept() {
		return acceptNum;
	}

	// アクセプト数を返す
	public double getAcceptRate() {
		return acceptRate;
	}

	// 相手の提案の有効値の最大を返す
	public double getmaxUtil(Object sender) {
		return maxUtil.get(sender);
	}

	// 交渉者数を返す
	public int getNegotiatorNum() {
		return negotiatorNum;
	}

	// 相対効用行列を返す
	public HashMap<Issue, HashMap<Value, Double>> getValueRelativeUtility() {
		return valueRelativeUtility;
	}

	// 線形効用空間であるかどうかを返す
	public boolean isLinerUtilitySpace() {
		return isLinerUtilitySpace;
	}

	// 論点一覧を返す
	public List<Issue> getIssues() {
		return issues;
	}

	// 論点における取り得る値の一覧を返す　それほど大したものではない
	public ArrayList<Value> getValues(Issue issue) {
		ArrayList<Value> values = new ArrayList<Value>();
		switch (issue.getType()) {
		case DISCRETE:
			List<ValueDiscrete> valuesDis = ((IssueDiscrete) issue).getValues();
			for (Value value : valuesDis) {
				values.add(value);
			}
			break;
		case INTEGER:
			int min_value = ((IssueInteger) issue).getUpperBound();
			int max_value = ((IssueInteger) issue).getUpperBound();
			for (int j = min_value; j <= max_value; j++) {
				Object valueObject = new Integer(j);
				values.add((Value) valueObject);
			}
			break;
		default:
			try {
				throw new Exception("issue type " + issue.getType()
						+ " not supported by Atlas3");
			} catch (Exception e) {
				System.out.println("論点の取り得る値の取得に失敗しました");
				e.printStackTrace();
			}
		}
		return values;
	}

	// すべての論点の内容を返す 勝手に出力
	public void getValueAll() {
		for (int i = 0; getIssues().size() > i; i++) {
			System.out.println(getValues(getIssues().get(i)));
		}
	}

	public void updateTime(double time) {

		if (aveTime.size() == 0) {
			aveTime.add(time);
		} else {
			double allTime = 0.0;
			for (int i = 0; i < aveTime.size(); i++) {
				allTime = allTime + aveTime.get(i);
			}
			aveTime.add(time - allTime);
		}
	}

	public double getAveTime() {
		double allTime = 0.0;
		int count = aveTime.size();
		for (int i = 0; i < count; i++) {
			allTime = allTime + aveTime.get(i);
		}
		return allTime / (double) count;
	}

	public void laststrategy(double time) {
		updateTime(time);
		double lastTime = 1 - time;
		int bidNum = LastOfferBidHistory.size() + 2;
		// System.out.println(LastOfferBidHistory.size() +
		// "-----list"+LastOfferBidHistory);
		double yosokuTime = (double) bidNum * getAveTime();
		if (lastTime < yosokuTime) {
			lastFlag = true;
		}
	}

	// 交渉相手の一覧を返す
	public ArrayList<Object> getOpponents() {
		return opponents;
	}
}
