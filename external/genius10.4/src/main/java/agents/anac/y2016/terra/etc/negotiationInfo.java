package agents.anac.y2016.terra.etc;

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

/**
 * 交渉者の情報を蓄えて整理
 */
public class negotiationInfo {

	/* 交渉に関する情報 */
	private AbstractUtilitySpace utilitySpace; // 効用空間
	private List<Issue> issues; // 論点
	private int round = 0; // 自分の手番数
	private boolean isLinerUtilitySpace = true; // 線形効用空間であるかどうか

	/* 自分の情報 */
	private ArrayList<Bid> myBidHistory;
	private HashMap<Issue, ArrayList<Value>> myOfferValue;
	private HashMap<Issue, HashMap<Value, Double>> valueRelativeUtility = null; // 自身の効用空間における各論点値の相対効用値行列（線形効用空間用)

	/* 相手の情報 */
	private ArrayList<Object> opponentList; // 交渉者リスト
	private int negotiatonNum = 0; // 交渉者数

	private HashMap<Object, ArrayList<Bid>> disagreedList; // 却下された提案リスト
	private HashMap<Object, ArrayList<Bid>> agreedList; // 賛成された提案リスト
	private HashMap<Object, ArrayList<Bid>> offeredList; // 提案された提案リスト

	private HashMap<Object, ArrayList<Bid>> recentDisagreedList; // 最近却下された提案リスト
	private HashMap<Object, ArrayList<Bid>> recentAgreedList; // 最近賛成された提案リスト
	private HashMap<Object, ArrayList<Bid>> recentOfferedList; // 最近提案された提案リスト

	private HashMap<Object, HashMap<Issue, ArrayList<Value>>> offeredValue;// 相手の提案内容の組み合わせ

	/* init */
	public negotiationInfo(AbstractUtilitySpace utilitySpace) {

		/* 交渉に関する情報 */
		this.utilitySpace = utilitySpace;
		issues = utilitySpace.getDomain().getIssues();

		/* 自分の情報 */
		myBidHistory = new ArrayList<>();
		myOfferValue = new HashMap<>();

		valueRelativeUtility = new HashMap<Issue, HashMap<Value, Double>>();//
		try {
			initValueRelativeUtility();
		} catch (Exception e) {
			System.out.println("相対効用行列の初期化に失敗しました");
			e.printStackTrace();
		}

		/* 相手の情報 */
		opponentList = new ArrayList<>();

		disagreedList = new HashMap<>();
		agreedList = new HashMap<>();
		offeredList = new HashMap<>();

		recentDisagreedList = new HashMap<>();
		recentAgreedList = new HashMap<>();
		recentOfferedList = new HashMap<>();

		offeredValue = new HashMap<>();
	}

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

	/* Set */
	public void updateOpponents(Object agent, int negotiationNum) {
		// if(agent.equals(null))return;
		if (!opponentList.contains(agent))
			opponentList.add(agent);
		this.negotiatonNum = negotiationNum;

		if (!offeredList.containsKey(agent))
			offeredList.put(agent, new ArrayList<Bid>());

		if (!disagreedList.containsKey(agent))
			disagreedList.put(agent, new ArrayList<Bid>());

		if (!agreedList.containsKey(agent))
			agreedList.put(agent, new ArrayList<Bid>());

		if (!offeredList.containsKey(agent))
			offeredList.put(agent, new ArrayList<Bid>());

		if (!offeredValue.containsKey(agent))
			offeredValue.put(agent, new HashMap<Issue, ArrayList<Value>>());
	}

	public void utilitySpaceTypeisNonLiner() {
		isLinerUtilitySpace = false;
	}

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

	/* Add */
	public void addMyBidHistory(Bid bid) {
		round++;
		myBidHistory.add(bid);

		HashMap<Issue, ArrayList<Value>> tHashMap = new HashMap<>();
		tHashMap = myOfferValue;
		for (Issue issues : getIssues()) {
			ArrayList<Value> tValues = new ArrayList<>();
			if (tHashMap.containsKey(issues)) {
				tValues = myOfferValue.get(issues);
			}

			tValues.add(bid.getValue(issues.getNumber()));

			tHashMap.put(issues, tValues);
		}
		myOfferValue = tHashMap;
	}

	public void addDisagreedList(Object agent, Bid bid) {
		ArrayList<Bid> tList = new ArrayList<>();
		tList = disagreedList.get(agent);
		tList.add(bid);
		disagreedList.put(agent, tList);

		ArrayList<Bid> tList2 = new ArrayList<>();
		for (int i = 1; i <= 12; i++) {
			if (tList.size() - i < 0)
				break;
			tList2.add(tList.get(tList.size() - i));
		}
		recentDisagreedList.put(agent, tList2);
		return;
	}

	public void addAgreedList(Object agent, Bid bid) {
		ArrayList<Bid> tList = new ArrayList<>();
		tList = agreedList.get(agent);
		tList.add(bid);
		agreedList.put(agent, tList);

		ArrayList<Bid> tList2 = new ArrayList<>();
		for (int i = 1; i <= 12; i++) {
			if (tList.size() - i < 0)
				break;
			tList2.add(tList.get(tList.size() - i));
		}
		recentAgreedList.put(agent, tList2);
		return;
	}

	public void addOfferedList(Object agent, Bid bid) {
		ArrayList<Bid> tList = new ArrayList<>();
		tList = offeredList.get(agent);
		tList.add(bid);
		offeredList.put(agent, tList);

		ArrayList<Bid> tList2 = new ArrayList<>();
		for (int i = 1; i <= 6; i++) {
			if (tList.size() - i < 0)
				break;
			tList2.add(tList.get(tList.size() - i));
		}
		recentOfferedList.put(agent, tList2);

		addOfferedValue(agent, bid);
		return;
	}

	private void addOfferedValue(Object agent, Bid bid) {
		HashMap<Issue, ArrayList<Value>> tHashMap = new HashMap<>();
		tHashMap = offeredValue.get(agent);

		for (Issue issues : getIssues()) {
			ArrayList<Value> tValues = new ArrayList<>();
			if (tHashMap.containsKey(issues)) {
				tValues = offeredValue.get(agent).get(issues);
			}

			tValues.add(bid.getValue(issues.getNumber()));

			tHashMap.put(issues, tValues);
		}
		offeredValue.put(agent, tHashMap);
	}

	/* Get */
	public ArrayList<Bid> getMyHistory() {
		return myBidHistory;
	}

	public HashMap<Issue, ArrayList<Value>> getMyOfferValue() {
		return myOfferValue;
	}

	public List<Issue> getIssues() {
		return issues;
	}

	public HashMap<Issue, HashMap<Value, Double>> getValueRelativeUtility() {
		return valueRelativeUtility;
	}

	public HashMap<Object, ArrayList<Bid>> getDisagreedList() {
		return disagreedList;
	}; // 却下された提案リスト

	public HashMap<Object, ArrayList<Bid>> getAgreedList() {
		return agreedList;
	} // 賛成された提案リスト

	public HashMap<Object, ArrayList<Bid>> getOfferedList() {
		return offeredList;
	} // 提案された提案リスト

	public ArrayList<Object> getOpponet() {
		return opponentList;
	} // 相手リスト

	public HashMap<Object, HashMap<Issue, ArrayList<Value>>> getOfferedValue() {
		return offeredValue;
	}// 提案されたValue

	public HashMap<Object, ArrayList<Bid>> getRecentDisagreedList() {
		return recentDisagreedList;
	}; // 最近の却下された提案リスト

	public HashMap<Object, ArrayList<Bid>> getRecentAgreedList() {
		return recentAgreedList;
	} // 最近の賛成された提案リスト

	public HashMap<Object, ArrayList<Bid>> getRecentOfferedList() {
		return recentOfferedList;
	} // 最近の提案された提案リスト

	// Issue毎に何回Valueが提案されたかを数える
	public HashMap<Issue, HashMap<Value, Integer>> getOfferedValueNum(
			Object agent) {
		HashMap<Issue, ArrayList<Value>> tHashMap = offeredValue.get(agent);
		return getOfferedValueNum(tHashMap);
	}

	public HashMap<Issue, HashMap<Value, Integer>> getOfferedValueNum() {
		HashMap<Issue, HashMap<Value, Integer>> answer = new HashMap<>();
		for (Object agent : getOpponet()) {
			HashMap<Issue, ArrayList<Value>> hashMap = offeredValue.get(agent);
			for (Issue issue : getIssues()) {
				if (hashMap.containsKey(issue)) {
					HashMap<Value, Integer> tHashMap = new HashMap<>();
					if (answer.containsKey(issue))
						tHashMap = answer.get(issue);// 前のエージェントの論点の値を考慮

					for (Value tValue : hashMap.get(issue)) {
						if (tHashMap.containsKey(tValue)) {// 値がすでに出ていれば数に1足す
							tHashMap.put(tValue, tHashMap.get(tValue) + 1);
						} else {// 未出であれば1を加える
							tHashMap.put(tValue, 1);
						}
					}
					answer.put(issue, tHashMap);
				}
			}
		}
		return answer;
	}

	public HashMap<Issue, HashMap<Value, Integer>> getOfferedValueNum(
			HashMap<Issue, ArrayList<Value>> hashMap) {
		HashMap<Issue, HashMap<Value, Integer>> answer = new HashMap<>();

		for (Issue issue : getIssues()) {
			if (hashMap.containsKey(issue)) {
				HashMap<Value, Integer> tHashMap = new HashMap<>();
				for (Value tValue : hashMap.get(issue)) {
					if (tHashMap.containsKey(tValue)) {// 値がすでに出ていれば数に1足す
						tHashMap.put(tValue, tHashMap.get(tValue) + 1);
					} else {// 未出であれば1を加える
						tHashMap.put(tValue, 1);
					}
				}
				answer.put(issue, tHashMap);
			}
		}
		return answer;
	}

	// 与えられたBidリストの効用値の平均値を返す
	public HashMap<Object, Double> getUtilityAverage(
			HashMap<Object, ArrayList<Bid>> bidList) {
		HashMap<Object, Double> ans = new HashMap<>();
		for (Object agent : bidList.keySet()) {
			if (bidList.containsKey(agent)) {
				ArrayList<Bid> tList = bidList.get(agent);
				double ave = 0;
				for (Bid tBid : tList) {
					ave += utilitySpace.getUtility(tBid);
				}
				ave /= tList.size();
				ans.put(agent, ave);
			}
		}
		return ans;
	};

	// 与えられたBidリストの効用値の最大値を返す
	public HashMap<Object, Double> getUtilityMax(
			HashMap<Object, ArrayList<Bid>> bidList) {
		HashMap<Object, Double> ans = new HashMap<>();
		for (Object agent : opponentList) {
			ArrayList<Bid> tList = bidList.get(agent);
			ans.put(agent, 0.0);
			for (Bid tBid : tList) {
				double tvalue = utilitySpace.getUtility(tBid);
				if (tvalue > ans.get(agent)) {
					ans.put(agent, tvalue);
				}
			}
		}
		return ans;
	};

	// ターン数を取得
	// public int getRound(){return round;}

	// Issue毎のValueを取得
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

	/* Bool */
	public boolean isLinerUtilitySpace() {
		return isLinerUtilitySpace;
	}
}
