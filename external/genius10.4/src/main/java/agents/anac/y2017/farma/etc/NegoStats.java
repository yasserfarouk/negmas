package agents.anac.y2017.farma.etc;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import genius.core.Bid;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.parties.NegotiationInfo;

/**
 * Created by tatsuya_toyama on 2017/05/07.
 */
public class NegoStats {
	private NegotiationInfo info;
	private boolean isPrinting = false; // デバッグ用
	private boolean isPrinting_Stats = false;

	// 交渉における基本情報
	private List<Issue> issues;
	private ArrayList<Object> rivals;
	private int negotiatorNum = 0; // 交渉者数
	private boolean isLinerUtilitySpace = true; // 線形効用空間であるかどうか
	private HashMap<Issue, HashMap<Value, Double>> valueRelativeUtility = null; // 自身の効用空間における各論点値の相対効用値行列（線形効用空間用）

	// current session の提案履歴
	private ArrayList<Bid> myBidHist = null;
	private HashMap<Object, ArrayList<Bid>> rivalsBidHist = null;
	private HashMap<Object, HashMap<Issue, HashMap<Value, Integer>>> agreedValueFrequency = null;
	private HashMap<Object, HashMap<Issue, HashMap<Value, Integer>>> rejectedValueFrequency = null;

	// 交渉相手の統計情報
	private HashMap<Object, Double> rivalsMean = null;
	private HashMap<Object, Double> rivalsVar = null;
	private HashMap<Object, Double> rivalsSum = null;
	private HashMap<Object, Double> rivalsPowSum = null;
	private HashMap<Object, Double> rivalsSD = null;

	private HashMap<Object, Double> rivalsMax = null; // 今sessionにおける相手の提案に対する自身の最大効用値
	private HashMap<Object, Double> rivalsMin = null; // 今sessionにおける相手の提案に対する自身の最低効用値

	public NegoStats(NegotiationInfo info, boolean isPrinting) {
		this.info = info;
		this.isPrinting = isPrinting;

		// 交渉における基本情報
		issues = info.getUtilitySpace().getDomain().getIssues();
		rivals = new ArrayList<Object>();
		valueRelativeUtility = new HashMap<Issue, HashMap<Value, Double>>();

		// current session の提案履歴
		myBidHist = new ArrayList<Bid>();
		rivalsBidHist = new HashMap<Object, ArrayList<Bid>>();
		agreedValueFrequency = new HashMap<Object, HashMap<Issue, HashMap<Value, Integer>>>(); // Value毎のAccept数
		rejectedValueFrequency = new HashMap<Object, HashMap<Issue, HashMap<Value, Integer>>>(); // Value毎のReject数

		try {
			initValueRelativeUtility();
		} catch (Exception e) {
			System.out.println("[Exception_Stats] 相対効用行列の初期化に失敗しました");
			e.printStackTrace();
		}

		// 交渉相手の統計情報
		rivalsMean = new HashMap<Object, Double>();
		rivalsVar = new HashMap<Object, Double>();
		rivalsSum = new HashMap<Object, Double>();
		rivalsPowSum = new HashMap<Object, Double>();
		rivalsSD = new HashMap<Object, Double>();

		rivalsMax = new HashMap<Object, Double>();
		rivalsMin = new HashMap<Object, Double>();

		if (this.isPrinting) {
			System.out.println("[isPrinting] NegoStats: success");
		}

	}

	public void initRivals(Object sender) {
		initNegotiatingInfo(sender); // 交渉情報を初期化
		rivals.add(sender); // 交渉参加者にsenderを追加
	}

	public void updateInfo(Object sender, Bid offeredBid) {
		try {
			updateNegoStats(sender, offeredBid); // 交渉情報の更新
		} catch (Exception e1) {
			System.out.println("[Exception_Stats] 交渉情報の更新に失敗しました");
			e1.printStackTrace();
		}
	}

	private void initNegotiatingInfo(Object sender) {
		rivalsBidHist.put(sender, new ArrayList<Bid>());
		rivalsMean.put(sender, 0.0);
		rivalsVar.put(sender, 0.0);
		rivalsSum.put(sender, 0.0);
		rivalsPowSum.put(sender, 0.0);
		rivalsSD.put(sender, 0.0);

		rivalsMax.put(sender, 0.0);
		rivalsMin.put(sender, 1.0);
	}

	/**
	 * 交渉者数を更新する
	 * 
	 * @param num
	 */
	public void updateNegotiatorsNum(int num) {
		negotiatorNum = num;
	}

	/**
	 * 線形効用空間でない場合
	 */
	public void utilSpaceTypeisNonLiner() {
		isLinerUtilitySpace = false;
	}

	/**
	 * 相対効用行列の初期化
	 * 
	 * @throws Exception
	 */
	private void initValueRelativeUtility() throws Exception {
		// ArrayList<Value> values = null;
		ArrayList<Object> rivals = getRivals();

		for (Issue issue : issues) {
			valueRelativeUtility.put(issue, new HashMap<Value, Double>()); // 論点行の初期化

			// 論点行の要素の初期化
			ArrayList<Value> values = getValues(issue);
			for (Value value : values) {
				valueRelativeUtility.get(issue).put(value, 0.0);
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
				valueRelativeUtility.get(issue).put(value,
						info.getUtilitySpace().getUtility(currentBid) - info.getUtilitySpace().getUtility(maxBid));
			}
		}
	}

	/**
	 * Agent senderが受け入れたValueの頻度を更新
	 * 
	 * @param sender
	 * @param bid
	 */
	public void updateAgreedValues(Object sender, Bid bid) {
		// senderが過去に登場していない場合は初期化
		if (!agreedValueFrequency.containsKey(sender)) {
			agreedValueFrequency.put(sender, new HashMap<Issue, HashMap<Value, Integer>>());

			for (Issue issue : issues) {
				agreedValueFrequency.get(sender).put(issue, new HashMap<Value, Integer>());

				ArrayList<Value> values = getValues(issue);
				for (Value value : values) {
					agreedValueFrequency.get(sender).get(issue).put(value, 0);
				}
			}
		}

		// 各issue毎に個数をカウント
		for (Issue issue : issues) {
			Value value = bid.getValue(issue.getNumber());
			agreedValueFrequency.get(sender).get(issue).put(value,
					agreedValueFrequency.get(sender).get(issue).get(value) + 1);
		}

		if (isPrinting_Stats) {
			System.out.println("[isPrint_Stats] (ACCEPT) " + sender.toString() + ":");
			for (Issue issue : issues) {
				ArrayList<Value> values = getValues(issue);
				for (Value value : values) {
					System.out.print(agreedValueFrequency.get(sender).get(issue).get(value) + " ");
				}
				System.out.println();
			}

			getMostAgreedValues(sender);
		}
	}

	/**
	 * senderが拒絶したValueの頻度を更新
	 * 
	 * @param sender
	 * @param bid
	 */
	public void updateRejectedValues(Object sender, Bid bid) {
		// senderが過去に登場していない場合は初期化
		if (!rejectedValueFrequency.containsKey(sender)) {
			rejectedValueFrequency.put(sender, new HashMap<Issue, HashMap<Value, Integer>>());

			for (Issue issue : issues) {
				rejectedValueFrequency.get(sender).put(issue, new HashMap<Value, Integer>());

				ArrayList<Value> values = getValues(issue);
				for (Value value : values) {
					rejectedValueFrequency.get(sender).get(issue).put(value, 0);
				}
			}
		}

		// 各issue毎に個数をカウント
		for (Issue issue : issues) {
			Value value = bid.getValue(issue.getNumber());
			rejectedValueFrequency.get(sender).get(issue).put(value,
					rejectedValueFrequency.get(sender).get(issue).get(value) + 1);
		}

		if (isPrinting_Stats) {
			System.out.println("[isPrint_Stats] (REJECT) " + sender.toString() + ":");
			for (Issue issue : issues) {
				ArrayList<Value> values = getValues(issue);
				for (Value value : values) {
					System.out.print(rejectedValueFrequency.get(sender).get(issue).get(value) + " ");
				}
				System.out.println();
			}

			getMostRejectedValues(sender);
		}
	}

	public void updateNegoStats(Object sender, Bid offeredBid) throws Exception {
		// current session の提案履歴 への追加
		rivalsBidHist.get(sender).add(offeredBid);
		updateAgreedValues(sender, offeredBid);

		// 交渉相手の統計情報 の更新
		double util = info.getUtilitySpace().getUtility(offeredBid);
		rivalsSum.put(sender, rivalsSum.get(sender) + util); // 和
		rivalsPowSum.put(sender, rivalsPowSum.get(sender) + Math.pow(util, 2)); // 二乗和

		int round_num = rivalsBidHist.get(sender).size();
		rivalsMean.put(sender, rivalsSum.get(sender) / round_num); // 平均
		rivalsVar.put(sender, (rivalsPowSum.get(sender) / round_num) - Math.pow(rivalsMean.get(sender), 2)); // 分散

		if (rivalsVar.get(sender) < 0) {
			rivalsVar.put(sender, 0.0);
		}
		rivalsSD.put(sender, Math.sqrt(rivalsVar.get(sender))); // 標準偏差

		// 最大最小の更新
		if (util > rivalsMax.get(sender)) {
			rivalsMax.put(sender, util);
		} else if (util < rivalsMin.get(sender)) {
			rivalsMin.put(sender, util);
		}

		if (isPrinting_Stats) {
			System.out.println("[isPrint_Stats] Mean: " + getRivalMean(sender) + " (Agent: " + sender.toString() + ")");
		}
	}

	/**
	 * 自身の提案情報の更新
	 * 
	 * @param offerBid
	 */
	public void updateMyBidHist(Bid offerBid) {
		myBidHist.add(offerBid);
	}

	// 交渉における基本情報 の取得
	/**
	 * 論点一覧を返す
	 * 
	 * @return
	 */
	public List<Issue> getIssues() {
		return issues;
	}

	/**
	 * 交渉相手の一覧を返す
	 * 
	 * @return
	 */
	public ArrayList<Object> getRivals() {
		return rivals;
	}

	/**
	 * 交渉者数（自身を含む）を返す
	 * 
	 * @return
	 */
	public int getNegotiatorNum() {
		// + 1: 自分
		return rivals.size() + 1;
	}

	/**
	 * 論点における取り得る値の一覧を返す
	 * 
	 * @param issue
	 * @return
	 */
	public ArrayList<Value> getValues(Issue issue) {
		ArrayList<Value> values = new ArrayList<Value>();

		// 効用情報のtype毎に処理が異なる
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
				throw new Exception(
						"issue type \"" + issue.getType() + "\" not supported by" + info.getAgentID().getName());
			} catch (Exception e) {
				System.out.println("[Exception] 論点の取り得る値の取得に失敗しました");
				e.printStackTrace();
			}
		}

		return values;
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
	 * 相対効用行列を返す
	 * 
	 * @return
	 */
	public HashMap<Issue, HashMap<Value, Double>> getValueRelativeUtility() {
		return valueRelativeUtility;
	}

	// current session の提案履歴 の取得

	/**
	 * エージェントsenderの論点issueにおける最大Agree数となる選択肢valueを取得
	 * 
	 * @param sender
	 * @param issue
	 * @return
	 */
	public Value getMostAgreedValue(Object sender, Issue issue) {
		ArrayList<Value> values = getValues(issue);

		int maxN = 0;
		Value mostAgreedValue = values.get(0);
		for (Value value : values) {
			int tempN = agreedValueFrequency.get(sender).get(issue).get(value);
			// もし最大数が更新されたら
			if (maxN < tempN) {
				maxN = tempN;
				mostAgreedValue = value;
			}
		}

		return mostAgreedValue;
	}

	/**
	 * エージェントsenderの論点issueにおける選択肢valueのAgree率を返す
	 * 
	 * @param sender
	 * @param issue
	 * @param value
	 * @return
	 */
	public double getProbAgreedValue(Object sender, Issue issue, Value value) {
		ArrayList<Value> values = getValues(issue);

		int sum = 0;
		for (Value v : values) {
			sum += agreedValueFrequency.get(sender).get(issue).get(v);
		}

		double prob = agreedValueFrequency.get(sender).get(issue).get(value) * 1.0 / sum;
		return prob;
	}

	/**
	 * エージェントSenderにおける各論点における累積確率(CP)を取得
	 * 
	 * @param sender
	 * @param issue
	 * @return
	 */
	public HashMap<Value, ArrayList<Double>> getCPAgreedValue(Object sender, Issue issue) {
		HashMap<Value, ArrayList<Double>> CPMap = new HashMap<Value, ArrayList<Double>>();

		ArrayList<Value> values = getValues(issue);
		int sum = 0;
		for (Value value : values) {
			sum += agreedValueFrequency.get(sender).get(issue).get(value);
		}

		double tempCP = 0.0;
		for (Value value : values) {
			ArrayList<Double> tempArray = new ArrayList<Double>();
			// 範囲のStartを格納
			tempArray.add(tempCP);

			// 範囲のEndを格納
			tempCP += agreedValueFrequency.get(sender).get(issue).get(value) * 1.0 / sum;
			tempArray.add(tempCP);

			CPMap.put(value, tempArray);
		}

		return CPMap;
	}

	/**
	 * エージェントSenderにおける各論点における最大Agree数となる選択肢valueをArrayListで取得
	 * 
	 * @param sender
	 * @return
	 */
	public ArrayList<Value> getMostAgreedValues(Object sender) {
		ArrayList<Value> values = new ArrayList<Value>();

		// issueの内部的な順番はa-b-c-d-...じゃないので注意
		for (Issue issue : issues) {
			values.add(getMostAgreedValue(sender, issue));
		}

		if (isPrinting_Stats) {
			System.out.print("[isPrint_Stats] ");
			for (int i = 0; i < issues.size(); i++) {
				// System.out.print(issues.get(i).toString() + ":" +
				// values.get(issues.get(i).getNumber()-1) + " ");
				// System.out.print(issues.get(i).toString() + ":" +
				// values.get(i) + "(" +
				// getProbAgreedValue(sender,issues.get(i),values.get(i)) + ")
				// ");

				HashMap<Value, ArrayList<Double>> cp = getCPAgreedValue(sender, issues.get(i));

				System.out.print(issues.get(i).toString() + ":" + values.get(i) + "(" + cp.get(values.get(i)).get(0)
						+ " - " + cp.get(values.get(i)).get(1) + ") ");
			}
			System.out.println();
		}

		return values;
	}

	/**
	 * エージェントsenderの論点issueにおける最大Reject数となる選択肢valueを取得
	 * 
	 * @param sender
	 * @param issue
	 * @return
	 */
	public Value getMostRejectedValue(Object sender, Issue issue) {
		ArrayList<Value> values = getValues(issue);

		int maxN = 0;
		Value mostRejectedValue = values.get(0);
		for (Value value : values) {
			int tempN = rejectedValueFrequency.get(sender).get(issue).get(value);
			// もし最大数が更新されたら
			if (maxN < tempN) {
				maxN = tempN;
				mostRejectedValue = value;
			}
		}

		return mostRejectedValue;
	}

	/**
	 * エージェントsenderの論点issueにおける選択肢valueのReject率を返す
	 * 
	 * @param sender
	 * @param issue
	 * @param value
	 * @return
	 */
	public double getProbRejectedValue(Object sender, Issue issue, Value value) {
		ArrayList<Value> values = getValues(issue);

		int sum = 0;
		for (Value v : values) {
			sum += rejectedValueFrequency.get(sender).get(issue).get(v);
		}

		double prob = rejectedValueFrequency.get(sender).get(issue).get(value) * 1.0 / sum;
		return prob;
	}

	/**
	 * エージェントSenderにおける各論点における累積確率(CP)を取得
	 * 
	 * @param sender
	 * @param issue
	 * @return
	 */
	public HashMap<Value, ArrayList<Double>> getCPRejectedValue(Object sender, Issue issue) {
		HashMap<Value, ArrayList<Double>> CPMap = new HashMap<Value, ArrayList<Double>>();

		ArrayList<Value> values = getValues(issue);
		int sum = 0;
		for (Value value : values) {
			sum += rejectedValueFrequency.get(sender).get(issue).get(value);
		}

		double tempCP = 0.0;
		for (Value value : values) {
			ArrayList<Double> tempArray = new ArrayList<Double>();
			// 範囲のStartを格納
			tempArray.add(tempCP);

			// 範囲のEndを格納
			tempCP += rejectedValueFrequency.get(sender).get(issue).get(value) * 1.0 / sum;
			tempArray.add(tempCP);

			CPMap.put(value, tempArray);
		}

		return CPMap;
	}

	/**
	 * エージェントSenderにおける各論点における最大Reject数となる選択肢valueをArrayListで取得
	 * 
	 * @param sender
	 * @return
	 */
	public ArrayList<Value> getMostRejectedValues(Object sender) {
		ArrayList<Value> values = new ArrayList<Value>();

		// issueの内部的な順番はa-b-c-d-...じゃないので注意
		for (Issue issue : issues) {
			values.add(getMostRejectedValue(sender, issue));
		}

		if (isPrinting_Stats) {
			System.out.print("[isPrint_Stats] ");
			for (int i = 0; i < issues.size(); i++) {
				// System.out.print(issues.get(i).toString() + ":" +
				// values.get(issues.get(i).getNumber()-1) + " ");
				// System.out.print(issues.get(i).toString() + ":" +
				// values.get(i) + "(" +
				// getProbRejectedValue(sender,issues.get(i),values.get(i)) + ")
				// ");

				HashMap<Value, ArrayList<Double>> cp = getCPRejectedValue(sender, issues.get(i));

				System.out.print(issues.get(i).toString() + ":" + values.get(i) + "(" + cp.get(values.get(i)).get(0)
						+ " - " + cp.get(values.get(i)).get(1) + ") ");
			}
			System.out.println();
		}

		return values;
	}

	// 交渉相手の統計情報 の取得
	/**
	 * 平均
	 * 
	 * @param sender
	 * @return
	 */
	public double getRivalMean(Object sender) {
		return rivalsMean.get(sender);
	}

	/**
	 * 分散
	 * 
	 * @param sender
	 * @return
	 */
	public double getRivalVar(Object sender) {
		return rivalsVar.get(sender);
	}

	/**
	 * 標準偏差
	 * 
	 * @param sender
	 * @return
	 */
	public double getRivalSD(Object sender) {
		return rivalsSD.get(sender);
	}

	/**
	 * エージェントSenderにおける今sessionにおける提案の自身の最大効用値
	 * 
	 * @param sender
	 * @return
	 */
	public double getRivalMax(Object sender) {
		return rivalsMax.get(sender);
	}

	/**
	 * エージェントSenderにおける今sessionにおける提案の自身の最小効用値
	 * 
	 * @param sender
	 * @return
	 */
	public double getRivalMin(Object sender) {
		return rivalsMin.get(sender);
	}

}
