package agents.anac.y2016.terra.etc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import genius.core.Bid;
import genius.core.issue.Issue;
import genius.core.issue.Value;
import genius.core.utility.UtilitySpace;

public class bidSearch {
	private UtilitySpace utilitySpace;
	private negotiationInfo negotiationInfo; // 交渉情報
	private Bid maxBid = null; // 最大効用値Bid

	private boolean isPrinting = false; // デバッグ用

	public bidSearch(UtilitySpace utilitySpace, negotiationInfo negotiationInfo)
			throws Exception {
		this.utilitySpace = utilitySpace;
		this.negotiationInfo = negotiationInfo;

		initMaxBid(); // 最大効用値Bidの初期探索
		negotiationInfo.setValueRelativeUtility(maxBid); // 相対効用値を導出する
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
		try {
			Bid bid = getBidbyAppropriateSearch(baseBid, threshold); // 閾値以上の効用値を持つ合意案候補を探索
			if (utilitySpace.getUtility(bid) < threshold) {
				bid = new Bid(maxBid);
			} // 探索によって得られたBidがthresholdよりも小さい場合，最大効用値Bidを基準とする
			return bid;
		} catch (Exception e) {
			System.out.println("Bidの探索に失敗しました");
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
				bid = relativeUtilitySearch2(threshold);
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

	// 探索1 (ランダムに提案を変え閾値を下回る事のないようにしていく)
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

	// 探索2
	// 相手毎にオファー、合意したBidから論点と値の組み合わせを取り出す(論点に対して値はいくつもとり得る)
	// 同じ組み合わせであれば、合計を数える
	// 以下繰り返し
	// 相手の提案の合計を足し合わせる
	// 相手の提案と現在の自分の提案の共通項と、共通でない項を探す
	// 効用値がある一定範囲内であれば、入れ替え
	// 一定以下であれば、入れ替え前のBidを返す
	private Bid relativeUtilitySearch2(double threshold) throws Exception {

		Bid bid = new Bid(maxBid);
		double d = threshold - 1.0; // 最大効用値との差
		double concessionSum = 0.0; // 減らした効用値の和
		double relativeUtility = 0.0;
		HashMap<Issue, HashMap<Value, Double>> valueRelativeUtility = negotiationInfo
				.getValueRelativeUtility();

		ArrayList<Issue> changedIssue = new ArrayList<>();// 変更した論点を保管

		// 2人の提案の合計を入手
		HashMap<Issue, HashMap<Value, Integer>> offredValueNum = new HashMap<>();
		offredValueNum = negotiationInfo.getOfferedValueNum();

		// value毎に一番提案されているものを求める
		HashMap<Issue, HashMap<Value, Integer>> maxValueMap = new HashMap<>();
		for (Issue issue : negotiationInfo.getIssues()) {
			HashMap<Value, Integer> tHashMap = new HashMap<>();// 一番提案されているものを保管
			int maxNum = -1;

			if (!offredValueNum.containsKey(issue))
				continue;

			for (Value value : offredValueNum.get(issue).keySet()) {
				if (maxNum < offredValueNum.get(issue).get(value)) {
					maxNum = offredValueNum.get(issue).get(value);
					tHashMap.clear();
					tHashMap.put(value, maxNum);
				}
			}
			maxValueMap.put(issue, tHashMap);
		}

		// 提案された回数の上から順に交換していく
		while (!maxValueMap.isEmpty()) {
			int maxNum = -1;
			HashMap<Issue, HashMap<Value, Integer>> tValueMap = new HashMap<>();
			for (Issue issue : maxValueMap.keySet()) {
				for (Value value : maxValueMap.get(issue).keySet()) {
					if (maxNum < maxValueMap.get(issue).get(value)) {
						maxNum = maxValueMap.get(issue).get(value);
						tValueMap.clear();
						tValueMap.put(issue, maxValueMap.get(issue));
					}
				}
			}

			// BIDの入れ替え
			for (Issue issue : tValueMap.keySet()) {
				for (Value value : tValueMap.get(issue).keySet()) {
					relativeUtility = valueRelativeUtility.get(issue)
							.get(value); // 最大効用値を基準とした相対効用値

					if (d <= concessionSum + relativeUtility
							&& (int) (Math.random() * 10 % 3) != 0) {
						bid = bid.putValue(issue.getNumber(), value);
						concessionSum += relativeUtility;
						changedIssue.add(issue);
						break;
					}
				}
			}

			// 取り除く
			for (Issue issue : tValueMap.keySet()) {
				if (maxValueMap.containsKey(issue))
					maxValueMap.remove(issue);
			}
		}

		// 上記の方法だけでは足りない場合ランダムに変更
		List<Issue> randomIssues = negotiationInfo.getIssues();
		Collections.shuffle(randomIssues);
		ArrayList<Value> randomValues = null;
		for (Issue issue : randomIssues) {

			if (changedIssue.contains(issue))
				continue;

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

		// System.out.println(bid);
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
}