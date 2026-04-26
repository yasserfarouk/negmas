package agents.anac.y2017.agentkn.etc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
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

	public bidSearch(AbstractUtilitySpace utilitySpace, negotiationInfo negotiationInfo, boolean isPrinting)
			throws Exception {
		this.utilitySpace = utilitySpace;
		this.negotiationInfo = negotiationInfo;
		this.isPrinting = isPrinting;

		initMaxBid(); // 最大効用値Bidの初期探索
		negotiationInfo.setValueRelativeUtility(maxBid); // 相対効用値を導出する
	}

	/**
	 * 最大効用値Bidの初期探索(最初は効用空間のタイプが不明であるため，SAを用いて探索する)
	 *
	 * @throws Exception
	 */
	private void initMaxBid() throws Exception {
		int tryNum = utilitySpace.getDomain().getIssues().size(); // 試行回数
		maxBid = this.utilitySpace.getDomain().getRandomBid((Random) null);

		for (int i = 0; i < tryNum; i++) {
			try {
				do {
					SimulatedAnnealingSearch(maxBid, 1.0);
				} while (utilitySpace.getUtility(maxBid) < utilitySpace.getReservationValue());
				if (utilitySpace.getUtility(maxBid) == 1.0) {
					break;
				}
			} catch (Exception e) {
				System.out.println("最大効用値Bidの初期探索に失敗しました");
				e.printStackTrace();
			}
		}
	}

	/**
	 * Bidを返す
	 *
	 * @param baseBid
	 * @param threshold
	 * @return
	 */
	public Bid getBid(Bid baseBid, double threshold) {
		Iterator var5 = negotiationInfo.getIssues().iterator();

		System.out.println("AgentKN's Threshold : " + threshold);
		while (true) {
			threshold = Math.max(emax(), threshold);
			System.out.println("search threshold: " + threshold);
			try {
				Bid e1 = this.getBidbyAppropriateSearch(baseBid, threshold);
				// 自分の最大効用知となるものを取得
				List<Bid> targets = new ArrayList<>();
				for (int i = 0; i < 10; i++) {
					Bid e = this.getBidbyAppropriateSearch(getRandomBid(utilitySpace.getUtility(e1)), threshold);
					System.out.println("search result--------------------------------------");
					System.out.println("result : " + e);
					System.out.println("frequency : " + negotiationInfo.getBidFrequcncy(e));
					System.out.println("searched utility : " + this.utilitySpace.getUtility(e1));
					System.out.println("---------------------------------------------------");
					targets.add(e);
				}

				// 頻度と自分の効用ちでソーと
				// targets.sort((b1, b2) -> (sortFuncUtility(b1) >
				// sortFuncUtility(b2)) ? 1 : 0);
				e1 = targets.get(0);

				if (this.utilitySpace.getUtility(e1) < threshold) {
					e1 = new Bid(this.maxBid);
				}
				return e1;
			} catch (Exception var7) {
				System.out.println("Bidの探索に失敗しました");
				var7.printStackTrace();
				return baseBid;
			}
		}
	}

	private double sortFuncUtility(Bid aBid) {
		int frequency = negotiationInfo.getBidFrequcncy(aBid);
		int k = (int) (Math.log10(frequency) + 1);
		return utilitySpace.getUtility(aBid) + Math.pow(0.1, k) * frequency;
	}

	private boolean frequencyCheck(Bid aSearchedBid) {
		System.out.println("check!!!!!!!!!!!!!!!!!!!!!!!");
		List issues = this.negotiationInfo.getIssues();
		ArrayList senders = this.negotiationInfo.getOpponents();
		Iterator var6 = senders.iterator();
		int point = 0;
		while (var6.hasNext()) {
			Object sender = var6.next();
			Iterator var8 = issues.iterator();
			while (var8.hasNext()) {
				Issue issue = (Issue) var8.next();
				Value freqValue = this.negotiationInfo.getHighFrequencyValue(sender, issue);
				if (aSearchedBid.getValues().get(issue).equals(freqValue))
					point++;
			}
		}
		System.out.println("searched bid frequency point : " + point);
		return point > issues.size() * 0.6;
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
					System.out.println(currentBidUtil);
					bid = currentBid;
					if (currentBidUtil <= min && currentBidUtil >= threshold) { // 効用値は推定値を用いている。
						bid = new Bid(currentBid);
						min = currentBidUtil;
					}
				}
			}
		} catch (Exception e) {
			System.out.println("SA探索に失敗しました");
			System.out.println("Problem with received bid(SA:last):" + e.getMessage() + ". cancelling bidding");
		}
		return bid;
	}

	/**
	 * 論点ごとに最適化を行う探索
	 *
	 * @param threshold
	 * @return
	 * @throws Exception
	 */

	private Bid relativeUtilitySearch(double threshold) throws Exception {
		Bid bid = new Bid(maxBid);
		double d = threshold - 1.0; // 最大効用値との差
		double concessionSum = 0.0; // 減らした効用値の和
		double relativeUtility = 0.0;
		HashMap<Issue, HashMap<Value, Double>> valueRelativeUtility = negotiationInfo.getValueRelativeUtility();
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

	// SA
	static double START_TEMPERATURE = 1.0; // 開始温度
	static double END_TEMPERATURE = 0.0001; // 終了温度
	static double COOL = 0.999; // 冷却度
	static int STEP = 1;// 変更する幅
	static int STEP_NUM = 1; // 変更する回数

	private Bid SimulatedAnnealingSearch(Bid baseBid, double threshold) throws Exception {
		Bid currentBid = new Bid(baseBid);
		double currenBidUtil = this.utilitySpace.getUtility(baseBid);
		Bid nextBid = null;
		double nextBidUtil = 0.0D;
		ArrayList targetBids = new ArrayList();
		double targetBidUtil = 0.0D;
		Random randomnr = new Random();
		double currentTemperature = START_TEMPERATURE;
		double newCost = 1.0D;
		double currentCost = 1.0D;

		for (List issues = this.negotiationInfo
				.getIssues(); currentTemperature > END_TEMPERATURE; currentTemperature *= COOL) {
			nextBid = new Bid(currentBid);

			for (int i = 0; i < STEP_NUM; ++i) {
				int issueIndex = randomnr.nextInt(issues.size());
				Issue issue = (Issue) issues.get(issueIndex);
				ArrayList values = this.negotiationInfo.getValues(issue);
				int valueIndex = randomnr.nextInt(values.size());
				nextBid = nextBid.putValue(issue.getNumber(), (Value) values.get(valueIndex));
				nextBidUtil = this.utilitySpace.getUtility(nextBid);
				if (this.maxBid == null || nextBidUtil >= this.utilitySpace.getUtility(this.maxBid)) {
					this.maxBid = new Bid(nextBid);
				}
			}

			newCost = Math.abs(threshold - nextBidUtil);
			currentCost = Math.abs(threshold - currenBidUtil);
			double p = Math.exp(-Math.abs(newCost - currentCost) / currentTemperature);
			if (newCost < currentCost || p > randomnr.nextDouble()) {
				currentBid = new Bid(nextBid);
				currenBidUtil = nextBidUtil;
			}

			if (currenBidUtil >= threshold) {
				if (targetBids.size() == 0) {
					targetBids.add(new Bid(currentBid));
					targetBidUtil = this.utilitySpace.getUtility(currentBid);
				} else if (currenBidUtil < targetBidUtil) {
					targetBids.clear();
					targetBids.add(new Bid(currentBid));
					targetBidUtil = this.utilitySpace.getUtility(currentBid);
				} else if (currenBidUtil == targetBidUtil) {
					targetBids.add(new Bid(currentBid));
				}
			}
		}

		if (targetBids.size() == 0) {
			return new Bid(baseBid);
		} else {
			return new Bid((Bid) targetBids.get(randomnr.nextInt(targetBids.size())));
		}
	}

	/**
	 * @return a random bid with high enough utility value.
	 * @throws Exception
	 *             if we can't compute the utility (eg no evaluators have been
	 *             set) or when other evaluators than a DiscreteEvaluator are
	 *             present in the util space.
	 */
	private Bid getRandomBid(double minUtil) throws Exception {
		HashMap<Integer, Value> values = new HashMap<Integer, Value>(); // pairs
		// <issuenumber,chosen
		// value
		// string>
		List<Issue> issues = utilitySpace.getDomain().getIssues();
		Random randomnr = new Random();

		// create a random bid with utility>MINIMUM_BID_UTIL.
		// note that this may never succeed if you set MINIMUM too high!!!
		// in that case we will search for a bid till the time is up (3 minutes)
		// but this is just a simple agent.
		Bid bid = null;
		do {
			for (Issue lIssue : issues) {
				switch (lIssue.getType()) {
				case DISCRETE:
					IssueDiscrete lIssueDiscrete = (IssueDiscrete) lIssue;
					int optionIndex = randomnr.nextInt(lIssueDiscrete.getNumberOfValues());
					values.put(lIssue.getNumber(), lIssueDiscrete.getValue(optionIndex));
					break;
				case REAL:
					IssueReal lIssueReal = (IssueReal) lIssue;
					int optionInd = randomnr.nextInt(lIssueReal.getNumberOfDiscretizationSteps() - 1);
					values.put(lIssueReal.getNumber(),
							new ValueReal(lIssueReal.getLowerBound()
									+ (lIssueReal.getUpperBound() - lIssueReal.getLowerBound()) * (double) (optionInd)
											/ (double) (lIssueReal.getNumberOfDiscretizationSteps())));
					break;
				case INTEGER:
					IssueInteger lIssueInteger = (IssueInteger) lIssue;
					int optionIndex2 = lIssueInteger.getLowerBound()
							+ randomnr.nextInt(lIssueInteger.getUpperBound() - lIssueInteger.getLowerBound());
					values.put(lIssueInteger.getNumber(), new ValueInteger(optionIndex2));
					break;
				default:
					throw new Exception("issue type " + lIssue.getType() + " not supported by SimpleAgent2");
				}
			}
			bid = new Bid(utilitySpace.getDomain(), values);
		} while (utilitySpace.getUtility(bid) < minUtil);

		return bid;
	}

	private double emax() {
		double ave = 0.0D;
		double extra = 0.0D;
		ArrayList opponents = negotiationInfo.getOpponents();

		double sd = 0.0D;
		for (Iterator i = opponents.iterator(); i.hasNext(); extra = sd) {
			Object sender = i.next();
			if (negotiationInfo.getPartnerBidNum(sender) % 10 == 0) {
				ave = 0.0D;
				extra = 0.0D;
			}

			double m = negotiationInfo.getAverage(sender);
			sd = negotiationInfo.getStandardDeviation(sender);
			ave = m;
		}

		double d = Math.sqrt(3) * sd / Math.sqrt(ave * (1 - ave));

		return ave + (1 - ave) * d;
	}
}