package agents.anac.y2016.syagent;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.utility.AbstractUtilitySpace;
import genius.core.utility.UtilitySpace;

public class negotiationInfo {
	/* Unchangeable Information */
	private UtilitySpace utilitySpace; // 効用空間
	private boolean isLinerUtilitySpace = true; // 線形効用空間であるかどうか
	private List<Issue> issues; // 論点
	private ArrayList<Object> opponents; // 自身以外の交渉参加者のsender
	private int negotiatorNum = 0; // 交渉者数
	private double rv = 0.0; // 留保価格の初期値
	private double df = 1.0; // 割引係数

	/* Changeable Information */
	private HashMap<Issue, HashMap<Value, Integer>> allFrequency = null; // 全員分の頻度

	/* My Negotiation Information */
	private HashMap<Issue, HashMap<Value, Double>> valueRelativeUtility = null; // 自身の効用空間における各論点値の相対効用値行列（線形効用空間用）
	private ArrayList<Bid> MyBidHistory = null; // 自分の提案履歴
	private double MyBidAverage = 0.0;// 自分の平均
	private int round = 0; // 自分の手番数

	/* Opponents Negotiation Information */
	private HashMap<Object, ArrayList<Bid>> opponentsBidHistory = null; // 相手の提案履歴
	private HashMap<Object, Double> opponentsAverage; // 相手の平均
	private HashMap<Object, Double> opponentsVariance; // 相手の分散
	private HashMap<Object, Double> opponentsSum; // 相手の和
	private HashMap<Object, Double> opponentsPowSum; // 相手の二乗和
	private HashMap<Object, Double> opponentsStandardDeviation; // 相手の標準偏差
	private HashMap<Object, HashMap<Issue, HashMap<Value, Integer>>> opponentsFrequency = null; // 相手の頻度
	private HashMap<Object, ArrayList<Bid>> opponentsAcceptBid = null; // 相手の受容履歴

	private boolean isPrinting = false; // デバッグ用

	public negotiationInfo(AbstractUtilitySpace utilitySpace, boolean isPrinting) {
		/* Initialize Unchangeable Information */
		this.utilitySpace = utilitySpace;
		this.isPrinting = isPrinting;
		issues = utilitySpace.getDomain().getIssues();
		opponents = new ArrayList<Object>();
		negotiatorNum = new Integer(0);
		rv = utilitySpace.getReservationValue();
		df = utilitySpace.getDiscountFactor();

		/* Initialize Changeable Information */
		allFrequency = new HashMap<Issue, HashMap<Value, Integer>>();
		try {
			initAllValueFrequency();
		} catch (Exception e1) {
			System.out.println("全員の頻度行列の初期化に失敗しました");
			e1.printStackTrace();
		}

		/* Initialize My Negotiation Information */
		valueRelativeUtility = new HashMap<Issue, HashMap<Value, Double>>();
		try {
			initValueRelativeUtility();
		} catch (Exception e) {
			System.out.println("相対効用行列の初期化に失敗しました");
			e.printStackTrace();
		}
		MyBidHistory = new ArrayList<>();
		MyBidAverage = new Double(0.0);
		round = new Integer(0);

		/* Initialize Opponents Negotiation Information */
		opponentsBidHistory = new HashMap<Object, ArrayList<Bid>>();
		opponentsAverage = new HashMap<Object, Double>();
		opponentsVariance = new HashMap<Object, Double>();
		opponentsSum = new HashMap<Object, Double>();
		opponentsPowSum = new HashMap<Object, Double>();
		opponentsStandardDeviation = new HashMap<Object, Double>();
		opponentsFrequency = new HashMap<Object, HashMap<Issue, HashMap<Value, Integer>>>();
		opponentsAcceptBid = new HashMap<Object, ArrayList<Bid>>();
	}

	/*** 初期化に関するメソッドたち ***/

	// 全員分の頻度の初期化
	private void initAllValueFrequency() throws Exception {
		ArrayList<Value> values = null;
		for (Issue issue : issues) {
			allFrequency.put(issue, new HashMap<Value, Integer>());
			values = getValues(issue);
			for (Value value : values) {
				allFrequency.get(issue).put(value, 0);
			}
		}
	}

	// 相対効用行列の初期化
	private void initValueRelativeUtility() throws Exception {
		ArrayList<Value> values = null;
		for (Issue issue : issues) {
			valueRelativeUtility.put(issue, new HashMap<Value, Double>()); // 論点(issue)行の初期化
			values = getValues(issue);
			for (Value value : values) {
				valueRelativeUtility.get(issue).put(value, 0.0);
			} // 論点(issue)行の要素の初期化
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

	// 初出のsenderの情報を初期化
	public void initOpponent(Object sender) {
		opponents.add(sender); // 交渉参加者にsenderを追加
		initNegotiationInfo(sender); // 交渉情報を初期化
		try {
			initOpponentsValueFrequency(sender);
		} catch (Exception e) {
			System.out.println("相手の頻度の初期化に失敗しました");
			e.printStackTrace();
		}
	}

	// senderのNegotiationInfoの初期化
	private void initNegotiationInfo(Object sender) {
		opponentsBidHistory.put(sender, new ArrayList<Bid>());
		opponentsAverage.put(sender, 0.0);
		opponentsVariance.put(sender, 0.0);
		opponentsSum.put(sender, 0.0);
		opponentsPowSum.put(sender, 0.0);
		opponentsStandardDeviation.put(sender, 0.0);
		opponentsAcceptBid.put(sender, new ArrayList<Bid>());
	}

	// 相手の頻度の初期化
	private void initOpponentsValueFrequency(Object sender) throws Exception {
		opponentsFrequency.put(sender,
				new HashMap<Issue, HashMap<Value, Integer>>()); // senderの頻度の初期化
		for (Issue issue : issues) {
			opponentsFrequency.get(sender).put(issue,
					new HashMap<Value, Integer>());
			ArrayList<Value> values = getValues(issue);
			for (Value value : values) {
				opponentsFrequency.get(sender).get(issue).put(value, 0);
			}
		}
	}

	/*** 更新に関するメソッドたち (void) ***/

	// 交渉者数を更新
	public void updateOpponentsNum(int num) {
		negotiatorNum = num;
	}

	// 交渉情報の更新
	public void updateInfo(Object sender, Bid offeredBid) {
		try {
			updateNegotiationInfo(sender, offeredBid);
		} // 交渉情報の更新
		catch (Exception e1) {
			System.out.println("交渉情報の更新に失敗しました");
			e1.printStackTrace();
		}
		try {
			updateFrequency(sender, offeredBid);
		} // senderの頻度の更新
		catch (Exception e) {
			System.out.println("頻度の更新に失敗しました");
			e.printStackTrace();
		}
	}

	// 頻度を更新
	private void updateFrequency(Object sender, Bid offeredBid)
			throws Exception {
		for (Issue issue : issues) {
			Value value = offeredBid.getValue(issue.getNumber());
			opponentsFrequency
					.get(sender)
					.get(issue)
					.put(value,
							opponentsFrequency.get(sender).get(issue)
									.get(value) + 1); // リストを更新
			allFrequency.get(issue).put(value,
					allFrequency.get(issue).get(value) + 1); // リストを更新
		}
	}

	// 自身の提案情報の更新
	public void updateMyBidHistory(Bid offerBid) {
		MyBidHistory.add(offerBid);
		MyBidAverage = (MyBidAverage * (MyBidHistory.size() - 1) + utilitySpace
				.getUtility(offerBid)) / MyBidHistory.size();
	}

	// round を数える
	public void countRound() {
		round++;
	}

	// sender の NegotiationInfo を更新
	public void updateNegotiationInfo(Object sender, Bid offeredBid)
			throws Exception {

		opponentsBidHistory.get(sender).add(offeredBid); // 相手の提案履歴

		double util = utilitySpace.getUtility(offeredBid);

		opponentsSum.put(sender, opponentsSum.get(sender) + util); // 和の更新
		opponentsPowSum.put(sender,
				opponentsPowSum.get(sender) + Math.pow(util, 2)); // 二乗和の更新

		int round_num = opponentsBidHistory.get(sender).size();
		opponentsAverage.put(sender, opponentsSum.get(sender) / round_num); // 平均の更新
		opponentsVariance.put(sender, (opponentsPowSum.get(sender) / round_num)
				- Math.pow(opponentsAverage.get(sender), 2)); // 分散の更新

		if (opponentsVariance.get(sender) < 0) {
			opponentsVariance.put(sender, 0.0);
		}
		opponentsStandardDeviation.put(sender,
				Math.sqrt(opponentsVariance.get(sender))); // 標準偏差の更新
	}

	/*** 値を返すメソッドたち (return) ***/

	// sender の提案履歴を返す
	public HashMap<Object, ArrayList<Bid>> getOpponentsBidHistory(Object sender) {
		return opponentsBidHistory;
	}

	// 自分の提案の平均を返す
	public double getMyBidAverage() {
		return MyBidAverage;
	}

	// 線形効用空間でない場合
	public void utilitySpaceTypeisNonLiner() {
		isLinerUtilitySpace = false;
	}

	// sender の平均を返す
	public double getAverage(Object sender) {
		return opponentsAverage.get(sender);
	}

	// sender の分散を返す
	public double getVariancer(Object sender) {
		return opponentsVariance.get(sender);
	}

	// sender の標準偏差を返す
	public double getStandardDeviation(Object sender) {
		return opponentsStandardDeviation.get(sender);
	}

	// sender の提案履歴の要素数を返す
	public int getPartnerBidNum(Object sender) {
		return opponentsBidHistory.get(sender).size();
	}

	// 自身のラウンド数を返す
	public int getRound() {
		return round;
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

	// 論点における取り得る値の一覧を返す
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
						+ " not supported by SYAgent");
			} catch (Exception e) {
				System.out.println("論点の取り得る値の取得に失敗しました");
				e.printStackTrace();
			}
		}
		return values;
	}

	// 交渉相手の一覧を返す
	public ArrayList<Object> getOpponents() {
		return opponents;
	}

	// 論点の最頻値を返すメソッド
	public Value getFrequentValue(Issue issue) {
		Value maxValue = null; // 最頻値
		int frequency = 0;// 要素の頻度
		int maxFrequency = 0; // 最頻値の数
		ArrayList<Value> values = getValues(issue);
		Collections.shuffle(values);

		for (Value value : values) {
			frequency = allFrequency.get(issue).get(value);
			if (maxValue == null || frequency > maxFrequency) {
				maxFrequency = frequency;
				maxValue = value;
			}
		}
		return maxValue;
	}

	// df が 1.0 でないか 1.0 であるかを示す boolean を決定するメソッド
	public boolean setDfFlag() {
		if (df == 1.0) {
			return false;// df == 1.0 (割引なし)
		} else {
			return true; // df != 1.0 (割引あり)
		}

	}

	// rv が 0.0 か否かを示す boolean を決定するメソッド
	public boolean setRvFlag() {
		if (rv == 0.0) {
			return false;// rv == 0.0 (留保価格なし)
		} else {
			return true; // rv != 0.0 (留保価格あり)
		}
	}

	// sender が accept した Bid の記録を更新
	public void updateAcceptedBid(AgentID sender, Bid bidFromAction) {
		opponentsAcceptBid.get(sender).add(bidFromAction);
	}

}
