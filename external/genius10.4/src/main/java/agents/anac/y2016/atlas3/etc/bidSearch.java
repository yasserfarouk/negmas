package agents.anac.y2016.atlas3.etc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import genius.core.Bid;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.IssueReal;
import genius.core.issue.Value;
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;
import genius.core.utility.AbstractUtilitySpace;

public class bidSearch {
	private AbstractUtilitySpace utilitySpace;
	private negotiationInfo negotiationInfo; // 交渉情報
	private Bid maxBid = null; // 最大効用値Bid

	private boolean isPrinting = false; // デバッグ用

	public bidSearch(AbstractUtilitySpace utilitySpace,
			negotiationInfo negotiationInfo, boolean isPrinting)
			throws Exception {
		this.utilitySpace = utilitySpace;
		this.negotiationInfo = negotiationInfo;
		this.isPrinting = isPrinting;

		initMaxBid(); // 最大効用値Bidの初期探索
		negotiationInfo.setValueRelativeUtility(maxBid); // 相対効用値を導出する

		if (this.isPrinting) {
			System.out.println("bidSearch:success");
		}
	}

	// 最大効用値Bidの初期探索(最初は効用空間のタイプが不明であるため，SAを用いて探索する)
	private void initMaxBid() throws Exception {
		int tryNum = utilitySpace.getDomain().getIssues().size(); // 試行回数
		maxBid = utilitySpace.getDomain().getRandomBid(null);
		for (int i = 0; i < tryNum; i++) {
			try {
				do {
					SimulatedAnnealingSearch(maxBid, 1.0);
				} while (utilitySpace.getUtility(maxBid) < utilitySpace
						.getReservationValue());
				if (utilitySpace.getUtility(maxBid) == 1.0) {
					break;
				}
			} catch (Exception e) {
				System.out.println("最大効用値Bidの初期探索に失敗しました");
				e.printStackTrace();
			}
		}
	}

	// Bidを返す
	public Bid getBid(Bid baseBid, double threshold) {
		// Type:Realに対応（暫定版）
		for (Issue issue : negotiationInfo.getIssues()) {
			switch (issue.getType()) {
			case REAL:
				try {
					return (getRandomBid(threshold));
				} catch (Exception e) {
					System.out.println("Bidのランダム探索に失敗しました(Real)");
					e.printStackTrace();
				}
				break;
			default:
				break;
			}
		}

		// Type:Integer or Discrete
		try {
			Bid bid = getBidbyNeighborhoodSearch(baseBid, threshold); // 近傍探索
			if (utilitySpace.getUtility(bid) < threshold) {
				bid = getBidbyAppropriateSearch(baseBid, threshold);
			} // 閾値以上の効用値を持つ合意案候補を探索
			if (utilitySpace.getUtility(bid) < threshold) {
				bid = new Bid(maxBid);
			} // 探索によって得られたBidがthresholdよりも小さい場合，最大効用値Bidを基準とする
			bid = getConvertBidbyFrequencyList(bid); // FrequencyListに従ってBidのValueを置換する
			return bid;
		} catch (Exception e) {
			System.out.println("Bidの探索に失敗しました(Integer or Discrete)");
			e.printStackTrace();
			return baseBid;
		}
	}

	// Bidの探索
	private static int SA_ITERATION = 1;

	private Bid getBidbyAppropriateSearch(Bid baseBid, double threshold) {
		Bid bid = new Bid(baseBid);
		try {
			// 線形効用空間用の探索
			if (negotiationInfo.isLinerUtilitySpace()) {
				bid = relativeUtilitySearch(threshold);
				if (utilitySpace.getUtility(bid) < threshold) {
					negotiationInfo.utilitySpaceTypeisNonLiner();
				} // 探索に失敗した場合，非線形効用空間用の探索に切り替える
			}

			// 非線形効用空間用の探索
			if (!negotiationInfo.isLinerUtilitySpace()) {
				Bid currentBid = null;
				double currentBidUtil = 0;
				double min = 1.0;
				for (int i = 0; i < SA_ITERATION; i++) {
					currentBid = SimulatedAnnealingSearch(bid, threshold);
					currentBidUtil = utilitySpace.getUtility(currentBid);
					if (currentBidUtil <= min && currentBidUtil >= threshold) {
						bid = new Bid(currentBid);
						min = currentBidUtil;
					}
				}
			}
		} catch (Exception e) {
			System.out.println("SA探索に失敗しました");
			System.out.println("Problem with received bid(SA:last):"
					+ e.getMessage() + ". cancelling bidding");
		}
		return bid;
	}

	// 論点ごとに最適化を行う探索
	private Bid relativeUtilitySearch(double threshold) throws Exception {
		Bid bid = new Bid(maxBid);
		double d = threshold - 1.0; // 最大効用値との差
		double concessionSum = 0.0; // 減らした効用値の和
		double relativeUtility = 0.0;
		HashMap<Issue, HashMap<Value, Double>> valueRelativeUtility = negotiationInfo
				.getValueRelativeUtility();
		List<Issue> randomIssues = negotiationInfo.getIssues();
		Collections.shuffle(randomIssues);
		ArrayList<Value> randomValues = null;
		for (Issue issue : randomIssues) {
			randomValues = negotiationInfo.getValues(issue);
			Collections.shuffle(randomValues);
			for (Value value : randomValues) {
				relativeUtility = valueRelativeUtility.get(issue).get(value); // 最大効用値を基準とした相対効用値
				if (d <= concessionSum + relativeUtility) {
					bid = bid.putValue(issue.getNumber(), value);
					concessionSum += relativeUtility;
					break;
				}
			}
		}
		return bid;
	}

	// ランダム探索
	private Bid getRandomBid(double threshold) throws Exception {
		HashMap<Integer, Value> values = new HashMap<Integer, Value>(); // pairs
																		// <issuenumber,chosen
																		// value
																		// string>
		List<Issue> issues = utilitySpace.getDomain().getIssues();
		Random randomnr = new Random();

		Bid bid = null;
		do {
			for (Issue lIssue : issues) {
				switch (lIssue.getType()) {
				case DISCRETE:
					IssueDiscrete lIssueDiscrete = (IssueDiscrete) lIssue;
					int optionIndex = randomnr.nextInt(lIssueDiscrete
							.getNumberOfValues());
					values.put(lIssue.getNumber(),
							lIssueDiscrete.getValue(optionIndex));
					break;
				case REAL:
					IssueReal lIssueReal = (IssueReal) lIssue;
					int optionInd = randomnr.nextInt(lIssueReal
							.getNumberOfDiscretizationSteps() - 1);
					values.put(
							lIssueReal.getNumber(),
							new ValueReal(lIssueReal.getLowerBound()
									+ (lIssueReal.getUpperBound() - lIssueReal
											.getLowerBound())
									* (double) (optionInd)
									/ (double) (lIssueReal
											.getNumberOfDiscretizationSteps())));
					break;
				case INTEGER:
					IssueInteger lIssueInteger = (IssueInteger) lIssue;
					int optionIndex2 = lIssueInteger.getLowerBound()
							+ randomnr.nextInt(lIssueInteger.getUpperBound()
									- lIssueInteger.getLowerBound());
					values.put(lIssueInteger.getNumber(), new ValueInteger(
							optionIndex2));
					break;
				default:
					throw new Exception("issue type " + lIssue.getType()
							+ " not supported by Atlas3");
				}
			}
			bid = new Bid(utilitySpace.getDomain(), values);
		} while (utilitySpace.getUtility(bid) < threshold);

		return bid;
	}

	// SA
	static double START_TEMPERATURE = 1.0; // 開始温度
	static double END_TEMPERATURE = 0.0001; // 終了温度
	static double COOL = 0.999; // 冷却度
	static int STEP = 1;// 変更する幅
	static int STEP_NUM = 1; // 変更する回数

	private Bid SimulatedAnnealingSearch(Bid baseBid, double threshold)
			throws Exception {
		Bid currentBid = new Bid(baseBid); // 初期解の生成
		double currenBidUtil = utilitySpace.getUtility(baseBid);
		Bid nextBid = null; // 評価Bid
		double nextBidUtil = 0.0;
		ArrayList<Bid> targetBids = new ArrayList<Bid>(); // 最適効用値BidのArrayList
		double targetBidUtil = 0.0;
		double p; // 遷移確率
		Random randomnr = new Random(); // 乱数
		double currentTemperature = START_TEMPERATURE; // 現在の温度
		double newCost = 1.0;
		double currentCost = 1.0;
		List<Issue> issues = negotiationInfo.getIssues();

		while (currentTemperature > END_TEMPERATURE) { // 温度が十分下がるまでループ
			nextBid = new Bid(currentBid); // next_bidを初期化
			for (int i = 0; i < STEP_NUM; i++) { // 近傍のBidを取得する
				int issueIndex = randomnr.nextInt(issues.size()); // 論点をランダムに指定
				Issue issue = issues.get(issueIndex); // 指定したindexのissue
				ArrayList<Value> values = negotiationInfo.getValues(issue);
				int valueIndex = randomnr.nextInt(values.size()); // 取り得る値の範囲でランダムに指定
				nextBid = nextBid.putValue(issue.getNumber(),
						values.get(valueIndex));
				nextBidUtil = utilitySpace.getUtility(nextBid);
				if (maxBid == null
						|| nextBidUtil >= utilitySpace.getUtility(maxBid)) {
					maxBid = new Bid(nextBid);
				} // 最大効用値Bidの更新
			}

			newCost = Math.abs(threshold - nextBidUtil);
			currentCost = Math.abs(threshold - currenBidUtil);
			p = Math.exp(-Math.abs(newCost - currentCost) / currentTemperature);
			if (newCost < currentCost || p > randomnr.nextDouble()) {
				currentBid = new Bid(nextBid); // Bidの更新
				currenBidUtil = nextBidUtil;
			}

			// 更新
			if (currenBidUtil >= threshold) {
				if (targetBids.size() == 0) {
					targetBids.add(new Bid(currentBid));
					targetBidUtil = utilitySpace.getUtility(currentBid);
				} else {
					if (currenBidUtil < targetBidUtil) {
						targetBids.clear(); // 初期化
						targetBids.add(new Bid(currentBid)); // 要素を追加
						targetBidUtil = utilitySpace.getUtility(currentBid);
					} else if (currenBidUtil == targetBidUtil) {
						targetBids.add(new Bid(currentBid)); // 要素を追加
					}
				}
			}
			currentTemperature = currentTemperature * COOL; // 温度を下げる
		}

		if (targetBids.size() == 0) {
			return new Bid(baseBid);
		} // 境界値より大きな効用値を持つBidが見つからなかったときは，baseBidを返す
		else {
			return new Bid(targetBids.get(randomnr.nextInt(targetBids.size())));
		} // 効用値が境界値付近となるBidを返す
	}

	// 近傍探索によるBidの探索
	private static int NEAR_ITERATION = 1;

	private Bid getBidbyNeighborhoodSearch(Bid baseBid, double threshold) {
		Bid bid = new Bid(baseBid);
		try {
			for (int i = 0; i < NEAR_ITERATION; i++) {
				bid = NeighborhoodSearch(bid, threshold);
			}
		} catch (Exception e) {
			System.out.println("近傍探索に失敗しました");
			System.out.println("Problem with received bid(Near:last):"
					+ e.getMessage() + ". cancelling bidding");
		}
		return bid;
	}

	// 近傍探索
	private Bid NeighborhoodSearch(Bid baseBid, double threshold)
			throws Exception {
		Bid currentBid = new Bid(baseBid); // 現在のBid
		double currenBidUtil = utilitySpace.getUtility(baseBid);
		ArrayList<Bid> targetBids = new ArrayList<Bid>(); // 最適効用値BidのArrayList
		double targetBidUtil = 0.0;
		Random randomnr = new Random(); // 乱数
		ArrayList<Value> values = null;
		List<Issue> issues = negotiationInfo.getIssues();

		for (Issue issue : issues) {
			values = negotiationInfo.getValues(issue);
			for (Value value : values) {
				currentBid.putValue(issue.getNumber(), value); // 近傍のBidを求める
				currenBidUtil = utilitySpace.getUtility(currentBid);
				if (maxBid == null
						|| currenBidUtil >= utilitySpace.getUtility(maxBid)) {
					maxBid = new Bid(currentBid);
				} // 最大効用値Bidの更新
					// 更新
				if (currenBidUtil >= threshold) {
					if (targetBids.size() == 0) {
						targetBids.add(new Bid(currentBid));
						targetBidUtil = utilitySpace.getUtility(currentBid);
					} else {
						if (currenBidUtil < targetBidUtil) {
							targetBids.clear(); // 初期化
							targetBids.add(new Bid(currentBid)); // 要素を追加
							targetBidUtil = utilitySpace.getUtility(currentBid);
						} else if (currenBidUtil == targetBidUtil) {
							targetBids.add(new Bid(currentBid)); // 要素を追加
						}
					}
				}
			}
			currentBid = new Bid(baseBid); // base_bidにリセットする
		}

		if (targetBids.size() == 0) {
			return new Bid(baseBid);
		} // 境界値より大きな効用値を持つBidが見つからなかったときは，baseBidを返す
		else {
			return new Bid(targetBids.get(randomnr.nextInt(targetBids.size())));
		} // 効用値が境界値付近となるBidを返す
	}

	// 頻度行列に基づきBidを改良する
	private Bid getConvertBidbyFrequencyList(Bid baseBid) {
		try {
			Bid currentBid = new Bid(baseBid);
			ArrayList<Object> randomOrderOpponents = negotiationInfo
					.getOpponents();
			Collections.shuffle(randomOrderOpponents); // ランダムに並び替える(毎回同じ順番で評価した場合，出現数が同値の場合に返却値が偏るため)

			// for(Object sender:randomOrderOpponents){
			List<Issue> randomOrderIssues = utilitySpace.getDomain()
					.getIssues();
			Collections.shuffle(randomOrderIssues); // ランダムに並び替える
			for (Issue issue : randomOrderIssues) {
				Bid nextBid = new Bid(currentBid);
				nextBid = nextBid.putValue(issue.getNumber(),
						negotiationInfo.getValuebyAllFrequencyList(issue)); // 頻出valueを優先してセットする
				// nextBid.setValue(issue.getNumber(),
				// negotiationInfo.getValuebyFrequencyList(sender, issue)); //
				// 頻出valueを優先してセットする
				if (utilitySpace.getUtility(nextBid) >= utilitySpace
						.getUtility(currentBid)) {
					currentBid = new Bid(nextBid);
				}
			}
			// }
			return currentBid;
		} catch (Exception e) {
			System.out.println("頻度行列に基づくBidの改良に失敗しました");
			e.printStackTrace();
			return baseBid;
		}
	}

	// Bidにおける重要論点一覧を返す
	public ArrayList<Issue> criticalIssue(Bid baseBid) throws Exception {
		Bid currentBid = new Bid(baseBid); // 現在のBid
		ArrayList<Issue> criticalIssues = new ArrayList<Issue>(); // Bidにおける重要論点一覧
		ArrayList<Value> values = null;
		List<Issue> issues = negotiationInfo.getIssues();

		for (Issue issue : issues) {
			values = negotiationInfo.getValues(issue);
			for (Value value : values) {
				currentBid = currentBid.putValue(issue.getNumber(), value); // 近傍のBidを求める
				if (utilitySpace.getUtility(currentBid) != utilitySpace
						.getUtility(baseBid)) {
					criticalIssues.add(issue);
					break;
				}
			}
			// baseBidにリセットする
			currentBid = new Bid(baseBid);
		}
		return criticalIssues;
	}
}