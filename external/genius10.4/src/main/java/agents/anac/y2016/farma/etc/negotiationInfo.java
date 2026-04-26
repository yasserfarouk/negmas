package agents.anac.y2016.farma.etc;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import genius.core.Bid;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.utility.AbstractUtilitySpace;

public class negotiationInfo {
	private AbstractUtilitySpace utilitySpace; // 効用空間
	private List<Issue> issues; // 論点
	private ArrayList<Object> opponents; // 自身以外の交渉参加者のsender
	private ArrayList<Bid> MyBidHistory = null; // 提案履歴
	private HashMap<Object, ArrayList<Bid>> opponentsBidHistory = null; // 提案履歴
	private HashMap<Object, Double> opponentsAverage; // 平均
	private HashMap<Object, Double> opponentsVariance; // 分散
	private HashMap<Object, Double> opponentsSum; // 和
	private HashMap<Object, Double> opponentsPowSum; // 二乗和
	private HashMap<Object, Double> opponentsStandardDeviation; // 標準偏差
	// 自身の効用空間における各論点値の相対効用値行列（線形効用空間用）
	private HashMap<Issue, HashMap<Value, Double>> valueRelativeUtility = null;
	private int round = 0;
	private int negotiatorNum = 0; // 交渉者数
	private boolean isLinerUtilitySpace = true; // 線形効用空間であるかどうか

	private boolean isPrinting = false; // デバッグ用

	// new parameter
	private double myAverage;
	private double myVariance;
	private double mySum;
	private double myPowSum;
	private double myStandardDeviation;

	private int valueNum;
	private boolean isWorth; // これ以上交渉する余地があるか

	public HashMap<Bid, CntBySender> bidAcceptNum; // Bid別のAccept数（OfferもAcceptと加算）
	public HashMap<Value, CntBySender> offeredValueNum; // Value別のOffer数
	public CntBySender opponentsAcceptNum; // Sender別のAccept数（Offerは含まない）
	public HashMap<Bid, Integer> neighborhoodBid; // 近傍・頻度探索した結果
	public HashMap<Bid, Integer> myOfferedBids; // 自身がOfferしたビットの回数
	public ArrayList<Pair<Double, Bid>> myUtilityAllBits; // bid集合が少ない場合、

	public negotiationInfo(AbstractUtilitySpace utilitySpace, boolean isPrinting) {
		// 初期化
		this.utilitySpace = utilitySpace;
		this.isPrinting = isPrinting;

		issues = utilitySpace.getDomain().getIssues();
		opponents = new ArrayList<Object>();
		MyBidHistory = new ArrayList<>();
		opponentsBidHistory = new HashMap<Object, ArrayList<Bid>>();
		opponentsAverage = new HashMap<Object, Double>();
		opponentsVariance = new HashMap<Object, Double>();
		opponentsSum = new HashMap<Object, Double>();
		opponentsPowSum = new HashMap<Object, Double>();
		opponentsStandardDeviation = new HashMap<Object, Double>();
		valueRelativeUtility = new HashMap<Issue, HashMap<Value, Double>>();

		isWorth = true;
		bidAcceptNum = new HashMap<Bid, CntBySender>();
		offeredValueNum = new HashMap<Value, CntBySender>();
		opponentsAcceptNum = new CntBySender(0);
		neighborhoodBid = new HashMap<Bid, Integer>();
		myOfferedBids = new HashMap<Bid, Integer>();
		myUtilityAllBits = new ArrayList<Pair<Double, Bid>>();

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
		// 交渉情報の更新
		try {
			updateNegotiatingInfo(sender, offeredBid);
		} catch (Exception e1) {
			System.out.println("交渉情報の更新に失敗しました");
			e1.printStackTrace();
		}
	}

	private void initNegotiatingInfo(Object sender) {
		opponentsBidHistory.put(sender, new ArrayList<Bid>());
		opponentsAverage.put(sender, 0.0);
		opponentsVariance.put(sender, 0.0);
		opponentsSum.put(sender, 0.0);
		opponentsPowSum.put(sender, 0.0);
		opponentsStandardDeviation.put(sender, 0.0);
	}

	/**
	 * 相対効用行列の初期化
	 * 
	 * @throws Exception
	 */
	private void initValueRelativeUtility() throws Exception {
		ArrayList<Value> values = null;
		for (Issue issue : issues) {
			// 論点行の初期化
			valueRelativeUtility.put(issue, new HashMap<Value, Double>());
			values = getValues(issue);

			// 論点行の要素の初期化
			for (Value value : values) {
				valueRelativeUtility.get(issue).put(value, 0.0);
			}
		}
	}

	/**
	 * 相対効用行列の導出
	 * 
	 * @param maxBid
	 * @throws Exception
	 */
	public void setValueRelativeUtility(Bid maxBid) throws Exception {
		ArrayList<Value> values = null;
		Bid currentBid = null;
		valueNum = issues.size();
		for (Issue issue : issues) {
			currentBid = new Bid(maxBid);
			values = getValues(issue);
			valueNum += values.size();
			for (Value value : values) {
				currentBid = currentBid.putValue(issue.getNumber(), value);
				valueRelativeUtility.get(issue).put(
						value,
						utilitySpace.getUtility(currentBid)
								- utilitySpace.getUtility(maxBid));
			}
		}
	}

	public void updateNegotiatingInfo(Object sender, Bid offeredBid)
			throws Exception {
		opponentsBidHistory.get(sender).add(offeredBid); // 提案履歴

		double util = utilitySpace.getUtility(offeredBid);
		opponentsSum.put(sender, opponentsSum.get(sender) + util); // 和
		opponentsPowSum.put(sender,
				opponentsPowSum.get(sender) + Math.pow(util, 2)); // 二乗和

		int round_num = opponentsBidHistory.get(sender).size();
		opponentsAverage.put(sender, opponentsSum.get(sender) / round_num); // 平均
		// 分散
		opponentsVariance.put(sender, (opponentsPowSum.get(sender) / round_num)
				- Math.pow(opponentsAverage.get(sender), 2));

		if (opponentsVariance.get(sender) < 0) {
			opponentsVariance.put(sender, 0.0);
		}
		// 標準偏差
		opponentsStandardDeviation.put(sender,
				Math.sqrt(opponentsVariance.get(sender)));
	}

	public void updateRound() {
		round += 1;
	}

	public void updateMyNegotiatingInfo(Bid offeredBid) throws Exception {
		double util = utilitySpace.getUtility(offeredBid);
		mySum += util; // 和
		myPowSum += Math.pow(util, 2); // 二乗和

		int round_num = MyBidHistory.size();
		myAverage = mySum / round_num; // 平均

		myVariance = Math.max(0,
				(myPowSum / round_num) - Math.pow(myAverage, 2)); // 分散
		myStandardDeviation = Math.sqrt(myVariance); // 標準偏差
	}

	/**
	 * 交渉者数を返す
	 * 
	 * @param num
	 */
	public void updateOpponentsNum(int num) {
		negotiatorNum = num;
	}

	/**
	 * 線形効用空間でない場合
	 */
	public void utilitySpaceTypeisNonLiner() {
		isLinerUtilitySpace = false;
	}

	/**
	 * 自身の提案情報の更新
	 * 
	 * @param offerBid
	 */
	public void updateMyBidHistory(Bid offerBid) {
		MyBidHistory.add(offerBid);
	}

	/**
	 * 平均
	 * 
	 * @param sender
	 * @return
	 */
	public double getAverage(Object sender) {
		return opponentsAverage.get(sender);
	}

	/**
	 * 分散
	 * 
	 * @param sender
	 * @return
	 */
	public double getVariancer(Object sender) {
		return opponentsVariance.get(sender);
	}

	/**
	 * 標準偏差
	 * 
	 * @param sender
	 * @return
	 */
	public double getStandardDeviation(Object sender) {
		return opponentsStandardDeviation.get(sender);
	}

	/**
	 * 相手の提案履歴の要素数を返す
	 * 
	 * @param sender
	 * @return
	 */
	public int getPartnerBidNum(Object sender) {
		return opponentsBidHistory.get(sender).size();
	}

	/**
	 * 自身のラウンド数を返す
	 * 
	 * @return
	 */
	public int getRound() {
		return round;
	}

	/**
	 * 交渉者数を返す
	 * 
	 * @return
	 */
	public int getNegotiatorNum() {
		return negotiatorNum;
	}

	/**
	 * 相対効用行列を返す
	 * 
	 * @return
	 */
	public HashMap<Issue, HashMap<Value, Double>> getValueRelativeUtility() {
		return valueRelativeUtility;
	}

	/**
	 * 線形効用空間であるかどうかを返す
	 * 
	 * @return
	 */
	public boolean isLinerUtilitySpace() {
		return isLinerUtilitySpace;
	}

	/**
	 * 論点一覧を返す
	 * 
	 * @return
	 */
	public List<Issue> getIssues() {
		return issues;
	}

	/**
	 * 論点における取り得る値の一覧を返す
	 * 
	 * @param issue
	 * @return
	 */
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

	/**
	 * 交渉相手の一覧を返す
	 * 
	 * @return
	 */
	public ArrayList<Object> getOpponents() {
		return opponents;
	}

	// new functions

	// EndNegotiation関連
	public int getValueNum() {
		return valueNum;
	}

	public void updateMyOfferedBids(Bid bid) {
		if (myOfferedBids.containsKey(bid)) {
			myOfferedBids.put(bid, myOfferedBids.get(bid) + 1);
		} else {
			myOfferedBids.put(bid, 1);
		}
	}

	public HashMap<Bid, Integer> getMyOfferedBids() {
		return myOfferedBids;
	}

	public void updateIsWorth(boolean isworth) {
		isWorth = isworth;
	}

	public boolean getIsWorth() {
		return isWorth;
	}

	/**
	 * エージェントsenderのidx番目の提案履歴を返す
	 * 
	 * @param sender
	 * @param idx
	 * @return
	 */
	public Bid getOpponentsBidHistory(Object sender, int idx) {
		return opponentsBidHistory.get(sender).get(idx);
	}

	/**
	 * Bid1とBid2について、cos類似度を計算. TODO 各valueについて重みを設定する。（でないと、単にvalueの一致度と一緒）.
	 * 
	 * @param b1
	 * @param b2
	 * @return
	 */
	public Double cosSimilarity(Bid b1, Bid b2) {
		Double sim = 0.0;

		for (Issue issue : issues) {
			int issueIdx = issue.getNumber();
			Value v1 = b1.getValue(issueIdx);
			Value v2 = b2.getValue(issueIdx);

			if (v1.equals(v2)) {
				sim += 1.0 - valueRelativeUtility.get(issue).get(v1);
			}
		}
		sim /= issues.size();
		return sim;
	}

	// 相手の性格の把握

	/**
	 * 相手の自己主張性を求める
	 * 
	 * @param sender
	 * @return
	 */
	public double calAssertiveness(Object sender) {
		double assertiveValue = myVariance - opponentsVariance.get(sender);
		return assertiveValue;
	}

	/**
	 * 相手の協調性を求める
	 * 
	 * @param sender
	 * @return
	 */
	public double calCooperativeness(Object sender) {
		double cooperativeValue = myAverage - opponentsAverage.get(sender);
		return cooperativeValue;
	}

	// 頻度情報の更新

	/**
	 * 
	 * @param sender
	 * @param bid
	 * @param time
	 */
	public void updateOfferedValueNum(Object sender, Bid bid, double time) {
		for (Issue issue : issues) {
			Value value = bid.getValue(issue.getNumber());
			if (!offeredValueNum.containsKey(value)) {
				offeredValueNum.put(value, new CntBySender(0));
			}
			offeredValueNum.get(value).incrementCnt(sender, time);
		}
	}

	public void updateBidAcceptNum(Object sender, Bid bid, double time) {
		if (!bidAcceptNum.containsKey(bid)) {
			bidAcceptNum.put(bid, new CntBySender(0));
		}
		bidAcceptNum.get(bid).incrementCnt(sender, time);
	}

	public void updateOpponentsAcceptNum(Object sender, double time) {
		opponentsAcceptNum.incrementCnt(sender, time);
	}

	// 頻度情報を用いた値を返す

	/**
	 * エージェントSenderの全体に対するAccept率を返す[0.0, 1.0]
	 * 
	 * @param sender
	 * @return
	 */
	public double calAcceptRate(Object sender) {
		return (opponentsAcceptNum.getSimpleCnt(sender) + 1)
				/ (opponentsAcceptNum.getSimpleSum() + opponents.size());
	}

	/**
	 * 相手の効用空間との差とAccept率から、どのエージェントを優先的にするか決める
	 * 
	 * @param sender
	 * @return
	 */
	public double calImportanceRate(Object sender) {
		// TODO:どっちか選ぶ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！

		// log_4 Accept率が低くて、効用空間の差が大きい人優先
		return 1 - calAcceptRate(sender) + calSpaceDistByInitOffer(sender);
		// log_5 Accept率が低くて、効用空間の差が小さい人優先
		// return 2 - calAcceptRate(sender) - calSpaceDistByInitOffer(sender);
	}

	/**
	 * BidのAccept数を返す
	 * 
	 * @param bid
	 * @param isWeighted
	 *            Accept率を考慮するかどうか
	 * @return
	 */
	public double getAcceptNumByBid(Bid bid, boolean isWeighted) {
		double acceptNum = 0.0;

		// Accept率を考慮したBidに置けるAccept数を返す
		if (isWeighted) {
			for (Object sender : opponents) {
				if (bidAcceptNum.containsKey(bid)
						&& bidAcceptNum.get(bid).isContainOpponents(sender)) {
					acceptNum += calImportanceRate(sender)
							* bidAcceptNum.get(bid).getSimpleCnt(sender);
				}
			}
		} else {
			if (bidAcceptNum.containsKey(bid)) {
				acceptNum = bidAcceptNum.get(bid).getSimpleSum();
			}
		}
		return acceptNum;
	}

	/**
	 * BidのAccept率を返す[0.0, 1.0]
	 * 
	 * @param bid
	 * @param isWeighted
	 * @return
	 */
	public double getAcceptRateByBid(Bid bid, boolean isWeighted) {
		return getAcceptNumByBid(bid, isWeighted)
				/ opponentsAcceptNum.getSimpleSum();
	}

	/**
	 * エージェントSenderの最高頻度Bidを返す
	 * 
	 * @param sender
	 * @param baseBid
	 * @return
	 */
	public Bid getHighFrequencyBid(Object sender, Bid baseBid) {
		Bid ansBid = new Bid(baseBid);
		for (Issue issue : issues) {
			Value ansValue = getHighFrequencyValue(sender, issue);
			ansBid = ansBid.putValue(issue.getNumber(), ansValue);
		}
		return ansBid;
	}

	/**
	 * エージェントSenderの論点issueの最高頻度のValueを返す
	 * 
	 * @param sender
	 * @param issue
	 * @return
	 */
	public Value getHighFrequencyValue(Object sender, Issue issue) {
		ArrayList<Value> values = getValues(issue);

		double tempMax = 0.0;
		Value tempValue = values.get(0);
		for (Value value : values) {
			// 最大頻度のValueを求める
			if (offeredValueNum.containsKey(value)) {
				double nowCnt = offeredValueNum.get(value).getSimpleCnt(sender);
				if (tempMax < nowCnt) {
					tempMax = nowCnt;
					tempValue = value;
				}
			}
		}
		return tempValue;
	}

	// 効用空間の違い
	/**
	 * エージェントsenderと自分の効用空間との差を、自身の効用空間の値で数値化[0.0, 1.0]
	 * 
	 * @param sender
	 * @return
	 */
	public double calSpaceDistByInitOffer(Object sender) {
		if (opponentsBidHistory.get(sender).isEmpty()) {
			return 0.0;
		}
		return 1.0 - utilitySpace.getUtility(opponentsBidHistory.get(sender)
				.get(0));
	}

	// 近傍・頻度探索
	public void updateNeighborhoodBid(Bid bid) {
		if (neighborhoodBid.containsKey(bid)) {
			neighborhoodBid.put(bid, neighborhoodBid.get(bid) + 1);
		} else {
			neighborhoodBid.put(bid, 1);
		}
	}

	public boolean isNeighborhoodBidContain(Bid bid) {
		return neighborhoodBid.containsKey(bid);
	}

	public HashMap<Bid, Integer> getNeighborhoodBid() {
		return neighborhoodBid;
	}

	public int getNeighborhoodBidSize() {
		return neighborhoodBid.size();
	}

}
