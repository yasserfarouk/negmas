package agents.anac.y2016.syagent;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import genius.core.Bid;
import genius.core.issue.Issue;
import genius.core.issue.Value;
import genius.core.utility.AbstractUtilitySpace;
import genius.core.utility.UtilitySpace;

public class bidSearch {
	private UtilitySpace utilitySpace;
	private negotiationInfo negotiationInfo; // 交渉情報
	private Bid maxBid = null; // 最大効用値Bid

	private boolean isPrinting = false; // デバッグ用

	public bidSearch(AbstractUtilitySpace utilitySpace,
			negotiationInfo negotiationInfo, boolean isPrinting)
			throws Exception {
		this.utilitySpace = utilitySpace;
		this.negotiationInfo = negotiationInfo;
		this.isPrinting = isPrinting;
		this.maxBid = utilitySpace.getMaxUtilityBid();// 最大効用値Bidの初期探索

		negotiationInfo.setValueRelativeUtility(maxBid); // 相対効用値を導出する
	}

	// Bidを返す
	public Bid getBid(Bid baseBid, double threshold) {
		try {
			Bid bid = getBidbyAppropriateSearch(baseBid, threshold); // 閾値以上の効用値を持つ合意案候補を探索
			if (utilitySpace.getUtility(bid) < threshold) {
				bid = new Bid(maxBid);
			} // 探索によって得られたBidがthresholdよりも小さい場合，最大効用値Bidを基準とする
			bid = improveBid(bid); // FrequencyListに従ってBidのValueを置換する
			return bid;
		} catch (Exception e) {
			System.out.println("Bidの探索に失敗しました");
			e.printStackTrace();
			return baseBid;
		}
	}

	// Bidを改良する
	private Bid improveBid(Bid bid) {
		try {
			Bid currentBid = new Bid(bid);
			ArrayList<Object> opponents = negotiationInfo.getOpponents();
			List<Issue> issues = utilitySpace.getDomain().getIssues();
			Collections.shuffle(opponents); // 　シャッフル
			Collections.shuffle(issues); // シャッフル
			for (Issue issue : issues) {
				Bid nextBid = new Bid(currentBid);
				nextBid.putValue(issue.getNumber(),
						negotiationInfo.getFrequentValue(issue));
				if (utilitySpace.getUtility(nextBid) >= utilitySpace
						.getUtility(currentBid)) {
					currentBid = new Bid(nextBid);
				}
			}
			return currentBid;
		} catch (Exception e) {
			System.out.println("Bidの改良に失敗しました");
			e.printStackTrace();
			return bid;
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
					System.out.println("utilitySpaceTypeisNonLiner");
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

	// 論点(issue)ごとに最適化を行う探索
	private Bid relativeUtilitySearch(double threshold) throws Exception {
		Bid bid = new Bid(maxBid);
		double d = threshold - 1.0; // 最大効用値との差
		double concessionSum = 0.0; // 減らした効用値の和
		double relativeUtility = 0.0;
		HashMap<Issue, HashMap<Value, Double>> valueRelativeUtility = negotiationInfo
				.getValueRelativeUtility();
		List<Issue> randomIssues = negotiationInfo.getIssues(); // ランダムに論点(issue)を選択
		Collections.shuffle(randomIssues); // 論点(issue)をシャッフルして偏りをなくす
		ArrayList<Value> randomValues = null;
		for (Issue issue : randomIssues) {
			randomValues = negotiationInfo.getValues(issue);
			Collections.shuffle(randomValues);// valueをシャッフルして偏りをなくす
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