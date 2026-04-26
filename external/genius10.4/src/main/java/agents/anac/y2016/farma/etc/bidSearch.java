package agents.anac.y2016.farma.etc;

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

	public boolean isOfferMaxBid(int turn, double df) {
		if (turn <= 3)
			return true;
		if (turn % (int) (75 / df) == 0)
			return true;
		return false;
	}

	// Bidを返す
	public Bid getBid(Bid baseBid, double threshold) {
		int myTurn = negotiationInfo.getRound();
		double df = utilitySpace.getDiscountFactor();

		if (myTurn == 1 && negotiationInfo.getValueNum() < 1000) {

		}

		// もし、第1回目のターンの場合、maxBidを返す
		if (isOfferMaxBid(myTurn, df)) {
			// System.out.println("Offer Max Bid!!!");
			return maxBid;
		}

		try {
			Bid bid = neighborhoodBidSearch(threshold);

			// 閾値以上の効用値を持つ合意案候補を探索
			// 適切なbidが見つからなかった場合
			if (bid == null) {
				bid = getBidbyAppropriateSearch(baseBid, threshold);
				ArrayList<Bid> neighborhoodBid = frequencyBidSearch(bid);
				Bid newBid = priorityShiftBid(neighborhoodBid, threshold);
				if (newBid != null)
					bid = newBid;
			}

			// 探索によって得られたBidがthresholdよりも小さい場合，最大効用値Bidを基準とする
			if (utilitySpace.getUtility(bid) < threshold) {
				bid = new Bid(maxBid);
			}

			System.out.println("Threshold: " + threshold);
			System.out.println("・MyOffer-> " + bid);
			return bid;
		} catch (Exception e) {
			System.out.println("Bidの探索に失敗しました");
			e.printStackTrace();
			return baseBid;
		}
	}

	// ランダム探索
	private Bid getRandomBid(double threshold) throws Exception {
		// pairs <issuenumber,chosen value string>
		HashMap<Integer, Value> values = new HashMap<Integer, Value>();
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

	// Bidの探索
	private static int SA_ITERATION = 1;

	private Bid getBidbyAppropriateSearch(Bid baseBid, double threshold) {
		Bid bid = new Bid(baseBid);
		try {
			// 線形効用空間用の探索
			if (negotiationInfo.isLinerUtilitySpace()) {
				bid = relativeUtilitySearch(threshold);

				// 探索に失敗した場合，非線形効用空間用の探索に切り替える
				if (utilitySpace.getUtility(bid) < threshold) {
					negotiationInfo.utilitySpaceTypeisNonLiner();
				}
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
				// 最大効用値を基準とした相対効用値
				relativeUtility = valueRelativeUtility.get(issue).get(value);

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
	static int STEP = 1; // 変更する幅
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

				// 最大効用値Bidの更新
				if (maxBid == null
						|| nextBidUtil >= utilitySpace.getUtility(maxBid)) {
					maxBid = new Bid(nextBid);
				}
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

		// 境界値より大きな効用値を持つBidが見つからなかったときは，baseBidを返す
		if (targetBids.size() == 0) {
			return new Bid(baseBid);
		} else {
			// 効用値が境界値付近となるBidを返す
			return new Bid(targetBids.get(randomnr.nextInt(targetBids.size())));
		}
	}

	// new function

	/**
	 * baseBidについて、各issueについて変更したBid列をbidInfoに追加
	 * 
	 * @param baseBid
	 * @return
	 */
	public void shiftBidSearch(Bid baseBid) {
		List<Issue> issues = negotiationInfo.getIssues();

		for (Issue issue : issues) {
			ArrayList<Value> values = negotiationInfo.getValues(issue);

			for (Value value : values) {
				Value targetValue = baseBid.getValue(issue.getNumber());

				// shiftして異なるbidになった場合
				if (!targetValue.equals(value)) {
					Bid tempBid = baseBid.putValue(issue.getNumber(), value);
					double utilityValue = utilitySpace.getUtility(tempBid);

					// 今まで登録していないbidであった場合
					double rv = utilitySpace.getReservationValue();
					if (utilityValue >= rv
							&& !negotiationInfo
									.isNeighborhoodBidContain(tempBid)) {
						negotiationInfo.updateNeighborhoodBid(tempBid);
						frequencyBidSearch(tempBid);
					}
				}
			}
		}
		// System.out.println("BidInfo NUM: " +
		// negotiationInfo.getNeighborhoodBidSize());
	}

	/**
	 * 頻度探索
	 * 
	 * @param baseBid
	 */
	public ArrayList<Bid> frequencyBidSearch(Bid baseBid) {
		ArrayList<Bid> freqBids = new ArrayList<Bid>();

		List<Issue> issues = negotiationInfo.getIssues();
		ArrayList<Object> senders = negotiationInfo.getOpponents();
		for (Object sender : senders) {
			for (Issue issue : issues) {
				Value freqValue = negotiationInfo.getHighFrequencyValue(sender,
						issue);
				Bid tempBid = baseBid.putValue(issue.getNumber(), freqValue);
				double utilityValue = utilitySpace.getUtility(tempBid);

				freqBids.add(tempBid);

				// 今まで登録していないbidであった場合
				double rv = utilitySpace.getReservationValue();
				if (utilityValue >= rv
						&& !negotiationInfo.isNeighborhoodBidContain(tempBid)) {
					negotiationInfo.updateNeighborhoodBid(tempBid);
				}
			}
		}
		return freqBids;
	}

	/**
	 * bidInfoの探索において、thresholdを超えるものを返す
	 * 
	 * @param threshold
	 * @return
	 */
	public Bid neighborhoodBidSearch(double threshold) {
		HashMap<Bid, Integer> nextBids = negotiationInfo.getNeighborhoodBid();
		ArrayList<Bid> overThresholdBids = new ArrayList<Bid>();

		for (Bid nowBid : nextBids.keySet()) {
			double util = utilitySpace.getUtility(nowBid);

			// thresholdを超えたbidについて
			if (util >= threshold) {
				overThresholdBids.add(nowBid);
			}
		}

		// 該当するbidが存在しなかった場合、次の探索に移行する
		if (overThresholdBids.isEmpty()) {
			return null;
		}
		// 該当するbidが存在した場合は、どのbidを提案するか評価する
		return priorityShiftBid(overThresholdBids, threshold);
	}

	/**
	 * 優先度の高いbidを返す
	 * 
	 * @param bids
	 * @return
	 */
	public Bid priorityShiftBid(ArrayList<Bid> bids, double threshold) {
		Bid ansBid = null;
		Bid subAnsBid = null;

		ArrayList<Object> opponents = negotiationInfo.getOpponents();
		HashMap<Object, Bid> opponentsFrequencyBids = new HashMap<Object, Bid>();
		for (Object sender : opponents) {
			opponentsFrequencyBids.put(sender,
					negotiationInfo.getHighFrequencyBid(sender, maxBid));
		}

		double similarityMax = 0.0;
		double acceptNumMax = 0.0;
		for (Bid bid : bids) {
			double MyUtil = utilitySpace.getUtility(bid);

			// まずthreshold近辺であるかどうか
			if (isNearThreshold(MyUtil, threshold)) {

				// 重み付き類似度の最大Bidを探索
				double similarity = 0.0;
				for (Object sender : opponents) {
					similarity = negotiationInfo.calImportanceRate(sender)
							* negotiationInfo.cosSimilarity(bid,
									opponentsFrequencyBids.get(sender));
				}
				if (similarity > similarityMax) {
					similarityMax = similarity;
					subAnsBid = bid;
				}

				double acceptNum = negotiationInfo.getAcceptNumByBid(bid, true)
						+ similarity;
				if (acceptNum > acceptNumMax) {
					acceptNumMax = acceptNum;
					ansBid = bid;
				}
			}
		}
		if (ansBid != null) {
			return ansBid;
		} else {
			return subAnsBid;
		}
	}

	private static double EPS = 0.125;

	/**
	 * thresholdとの誤差の許容範囲内かを判別
	 * 
	 * @param util
	 * @param threshold
	 * @return
	 */
	public boolean isNearThreshold(double util, double threshold) {
		double dist = util - threshold;
		if (dist >= 0 && dist < EPS) {
			return true;
		} else {
			return false;
		}
	}

}
