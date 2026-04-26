package agents.anac.y2015.DrageKnight.etc;

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
	private ArrayList<Bid> MyBidHistory = null; // 提案履歴
	private HashMap<Object, ArrayList<Bid>> opponentsBidHistory = null; // 提案履歴
	private HashMap<Object, Double> opponentsAverage; // 平均
	private HashMap<Object, Double> opponentsVariance; // 分散
	private HashMap<Object, Double> opponentsSum; // 和
	private HashMap<Object, Double> opponentsPowSum; // 二乗和
	private HashMap<Object, Double> opponentsStandardDeviation; // 標準偏差
	private HashMap<Issue, HashMap<Value, Double>> valueRelativeUtility = null; // 自身の効用空間における各論点値の相対効用値行列（線形効用空間用）
	private int round = 0; // 自分の手番数
	private int negotiatorNum = 0; // 交渉者数
	private boolean isLinerUtilitySpace = true; // 線形効用空間であるかどうか

	public negotiatingInfo(AdditiveUtilitySpace utilitySpace) {
		// 初期化
		this.utilitySpace = utilitySpace;
		issues = utilitySpace.getDomain().getIssues();
		opponents = new ArrayList<Object>();
		MyBidHistory = new ArrayList<Bid>();
		opponentsBidHistory = new HashMap<Object, ArrayList<Bid>>();
		opponentsAverage = new HashMap<Object, Double>();
		opponentsVariance = new HashMap<Object, Double>();
		opponentsSum = new HashMap<Object, Double>();
		opponentsPowSum = new HashMap<Object, Double>();
		opponentsStandardDeviation = new HashMap<Object, Double>();
		valueRelativeUtility = new HashMap<Issue, HashMap<Value, Double>>();

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
		opponentsAverage.put(sender, 0.0);
		opponentsVariance.put(sender, 0.0);
		opponentsSum.put(sender, 0.0);
		opponentsPowSum.put(sender, 0.0);
		opponentsStandardDeviation.put(sender, 0.0);
	}

	// 相対効用行列の初期化
	private void initValueRelativeUtility() throws Exception {
		ArrayList<Value> values = null;
		for (Issue issue : issues) {
			valueRelativeUtility.put(issue, new HashMap<Value, Double>()); // 論点行の初期化
			values = getValues(issue);
			for (Value value : values) {
				valueRelativeUtility.get(issue).put(value, 0.0);
			} // 論点行の要素の初期化
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

	public void updateNegotiatingInfo(Object sender, Bid offeredBid)
			throws Exception {
		opponentsBidHistory.get(sender).add(offeredBid); // 提案履歴

		double util = utilitySpace.getUtility(offeredBid);
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

	// 標準偏差
	public double getStandardDeviation(Object sender) {
		return opponentsStandardDeviation.get(sender);
	}

	// 相手の提案履歴の要素数を返す
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
						+ " not supported by Atlas3");
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
}
