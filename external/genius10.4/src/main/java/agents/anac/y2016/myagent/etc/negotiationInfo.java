package agents.anac.y2016.myagent.etc;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import genius.core.Bid;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.utility.UtilitySpace;

public class negotiationInfo {
	private UtilitySpace utilitySpace; // 効用空間
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

	// original
	private boolean isLast = false; // 受けると終わり
	private HashMap<Object, ArrayList<Bid>> opponentsAccept; // 受け入れた提案
	private HashMap<Object, HashMap<Issue, HashMap<Value, Integer>>> opponentsIssueWeight; // 敵の論点の重み

	private Bid insurance; // 保険ビッド

	private boolean isPrinting = false; // デバッグ用

	public negotiationInfo(UtilitySpace utilitySpace, boolean isPrinting) {
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

		opponentsAccept = new HashMap<>();
		opponentsIssueWeight = new HashMap<>();

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

	// 最後のフェーズかを確認
	public boolean isLast() {
		return isLast;
	}

	// 最後のフェーズかを更新
	public void updateLast(boolean b) {
		isLast = b;
	}

	// Accept履歴を更新
	public void updateAcceptList(Object sender, Bid acceptedBid) {
		// 初期化
		if (!opponentsAccept.containsKey(sender)) {
			ArrayList<Bid> temp = new ArrayList<>();
			opponentsAccept.put(sender, temp);
		}

		// 追加
		opponentsAccept.get(sender).add(acceptedBid);
	}

	public ArrayList<Bid> getAcceptList(Object sender) {
		if (!opponentsAccept.containsKey(sender)) {
			return new ArrayList<Bid>();
		}
		return opponentsAccept.get(sender);
	}

	// 今までにAcceptした回数を返す
	public int getAcceptedNum(Object opponent) {
		// 初期化されていない：１回もAcceptされていない
		if (!opponentsAccept.containsKey(opponent)) {
			return 0;
		}
		return opponentsAccept.get(opponent).size();
	}

	// 弱気の連中を返す
	public ArrayList<Object> getCowards() {
		ArrayList<Object> cowards = new ArrayList<>();

		for (Object opponent : opponents) {
			if (getAcceptedNum(opponent) > 3) {
				cowards.add(opponent);
			}
		}

		return cowards;
	}

	// 強気の連中を返す
	public ArrayList<Object> getArrogants() {
		ArrayList<Object> arrogants = new ArrayList<>();

		for (Object opponent : opponents) {
			if (getAcceptedNum(opponent) <= 2) {
				arrogants.add(opponent);
			}
		}

		return arrogants;
	}

	// 最低ラインを求める
	public double getLeastValue(ArrayList<Object> arrogants) {
		double least = 0.0;
		for (Object opponent : arrogants) {
			double ave = getAverage(opponent);
			double var = getVariancer(opponent);
			double temp = ave + var * 0.1;
			if (temp > least) {
				least = temp;
			}
		}

		return least;
	}

	// 相手の履歴を確認
	public ArrayList<Bid> getOpponentHistory(Object sender, int num) {

		ArrayList<Bid> recentHistory = new ArrayList<>();

		if (!this.opponentsBidHistory.containsKey(sender)) {
			return recentHistory;
		}

		int anum = num > opponentsBidHistory.get(sender).size() ? opponentsBidHistory
				.get(sender).size() : num;
		for (int i = 0; i < anum; i++) {
			recentHistory.add(opponentsBidHistory.get(sender).get(
					opponentsBidHistory.get(sender).size() - 1 - i));
		}

		return recentHistory;
	}

	// 相手の履歴の数を返す
	public int getOpponentBidNum(Object sender) {
		return opponentsBidHistory.get(sender).size();
	}

	// 相手の全体の傾きを求める
	public double getSlant(Object sender) {

		int num = 20; // 最近の数

		HashMap<Issue, HashMap<Value, Double>> allWeight = getOpponentMaxValues(sender);
		HashMap<Issue, HashMap<Value, Double>> recentWeight = getRecentMaxWeight(
				sender, num);

		double sumSlant = 0.0;

		for (Issue issue : getIssues()) {

			for (Value v : allWeight.get(issue).keySet()) {

				// maxValueが異なる場合
				if (!recentWeight.get(issue).containsKey(v)) {
					continue;
				}

				double rawSlant = allWeight.get(issue).get(v)
						- recentWeight.get(issue).get(v);

				// 傾きがマイナスの場合
				if (rawSlant < 0) {
					continue;
				} else {
					double slant = rawSlant * allWeight.get(issue).get(v);
					sumSlant += slant;
				}
			}
		}

		return sumSlant;
	}

	// 相手の最近の論点の回数を求める
	public HashMap<Issue, HashMap<Value, Integer>> getRecentCount(
			Object sender, int num) {
		HashMap<Issue, HashMap<Value, Integer>> recentCount = initIssueWeight();
		for (Bid bid : getOpponentHistory(sender, num)) {
			for (Issue issue : bid.getIssues()) {
				Value v = bid.getValue(issue.getNumber());
				int c = recentCount.get(issue).get(v);
				recentCount.get(issue).put(v, c + 1);
			}
		}

		return recentCount;
	}

	// 相手の最近の論点の重みを求める
	public HashMap<Issue, HashMap<Value, Double>> getRecentWeight(
			Object sender, int num) {

		HashMap<Issue, HashMap<Value, Double>> recentWeight = initRecentWeight();
		HashMap<Issue, HashMap<Value, Integer>> recentCount = getRecentCount(
				sender, num);

		for (Issue issue : getIssues()) {
			int sum = 0;

			for (Value value : getValues(issue)) {
				sum += recentCount.get(issue).get(value);
			}

			for (Value value : getValues(issue)) {
				recentWeight.get(issue).put(value,
						(recentCount.get(issue).get(value) + 0.0) / sum);
			}
		}

		return recentWeight;
	}

	public HashMap<Issue, HashMap<Value, Double>> getRecentMaxWeight(
			Object sender, int num) {

		HashMap<Issue, HashMap<Value, Double>> recentMaxWeight = initRecentWeight();
		HashMap<Issue, HashMap<Value, Double>> recentWeight = getRecentWeight(
				sender, num);

		for (Issue issue : getIssues()) {
			Value maxValue = null;
			for (Map.Entry<Value, Double> e : recentWeight.get(issue)
					.entrySet()) {
				if (maxValue == null) {
					maxValue = e.getKey();
				} else {
					if (e.getValue() > recentWeight.get(issue).get(maxValue)) {
						maxValue = e.getKey();
					}
				}
			}
			HashMap<Value, Double> weight = new HashMap<>();
			weight.put(maxValue, recentWeight.get(issue).get(maxValue));
			recentMaxWeight.put(issue, weight);
		}

		return recentMaxWeight;
	}

	// 初期化
	private HashMap<Issue, HashMap<Value, Double>> initRecentWeight() {
		HashMap<Issue, HashMap<Value, Double>> recentWeight = new HashMap<>();

		for (Issue issue : getIssues()) {
			HashMap<Value, Double> weights = new HashMap<>();
			for (Value value : this.getValues(issue)) {
				weights.put(value, 0.0);
			}

			recentWeight.put(issue, weights);
		}

		return recentWeight;
	}

	// 論点の重みを更新
	public void updateOpponentIssueWeight(Object sender, Bid bid) {
		// 初期化
		if (!opponentsIssueWeight.containsKey(sender)) {
			opponentsIssueWeight.put(sender, initIssueWeight());
		}

		for (Issue issue : bid.getIssues()) {
			int current = opponentsIssueWeight.get(sender).get(issue)
					.get(bid.getValue(issue.getNumber()));
			opponentsIssueWeight.get(sender).get(issue)
					.put(bid.getValue(issue.getNumber()), current + 1);
		}
	}

	// 論点の重みを獲得
	public HashMap<Issue, HashMap<Value, Integer>> getWeights(Object sender) {
		return opponentsIssueWeight.get(sender);
	}

	// 論点の重みを獲得
	public HashMap<Value, Integer> getWeight(Object sender, Issue issue) {
		return opponentsIssueWeight.get(sender).get(issue);
	}

	public int getCount(Object sender, Issue issue, Value value) {
		return getWeight(sender, issue).get(value);
	}

	// 論点Mapの初期化
	private HashMap<Issue, HashMap<Value, Integer>> initIssueWeight() {

		HashMap<Issue, HashMap<Value, Integer>> IssueWeight = new HashMap<>();

		for (Issue issue : getIssues()) {
			HashMap<Value, Integer> values = new HashMap<>();
			for (Value value : getValues(issue)) {
				values.put(value, 0);
			}
			IssueWeight.put(issue, values);
		}

		return IssueWeight;
	}

	public HashMap<Issue, HashMap<Value, Double>> getOpponentMaxValues(
			Object sender) {

		HashMap<Issue, HashMap<Value, Double>> opponentMaxValues = new HashMap<>();

		for (Issue issue : getIssues()) {
			Value maxValue = null;
			double sum = 0;
			for (Value value : getWeight(sender, issue).keySet()) {
				if (maxValue == null) {
					maxValue = value;
				} else if (getCount(sender, issue, maxValue) < getCount(sender,
						issue, value)) {
					maxValue = value;
				}

				sum += getCount(sender, issue, value);
			}

			HashMap<Value, Double> temp = new HashMap<>();
			temp.put(maxValue, getCount(sender, issue, maxValue) / sum);
			opponentMaxValues.put(issue, temp);
		}

		return opponentMaxValues;
	}

	public HashMap<Issue, Value> getCriticalIssues(Object sender) {

		HashMap<Issue, Value> criticalIssues = new HashMap<>();

		for (Map.Entry<Issue, HashMap<Value, Double>> e : getOpponentMaxValues(
				sender).entrySet()) {
			for (Value value : e.getValue().keySet()) {
				double weight = e.getValue().get(value);
				if (weight > 0.95) {
					criticalIssues.put(e.getKey(), value);
				}
			}
		}

		return criticalIssues;
	}

	/* ------------------------------- */
}
