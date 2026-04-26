package agents.anac.y2015.agenth;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import agents.anac.y2015.agenth.BidHistory.Entry;
import genius.core.Bid;
import genius.core.Domain;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.IssueReal;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;
import genius.core.utility.AbstractUtilitySpace;

public class BidHelper {
	// 探索のパラメータ
	private static final int SA_ITERATION = 1;
	private static final double START_TEMPERATURE = 1.0; // 開始温度
	private static final double END_TEMPERATURE = 0.0001; // 終了温度
	private static final double COOL = 0.999; // 冷却度
	private static final int STEP = 1;// 変更する幅
	private static final int STEP_NUM = 1; // 変更する回数

	/** エージェント */
	private AgentH mAgent;
	/** 乱数 */
	private Random mRandom;
	/** 自身の効用空間における各論点値の相対効用値行列（線形効用空間用) */
	private HashMap<Issue, HashMap<Value, Double>> mValueRelativeUtility;
	/** 効用値 MAX の bid */
	private Bid mMaxBid;

	public BidHelper(AgentH agent) throws Exception {
		mAgent = agent;
		mRandom = new Random();
		mValueRelativeUtility = new HashMap<Issue, HashMap<Value, Double>>();

		initMaxBid();
		initValueRelativeUtility();
		setValueRelativeUtility(mMaxBid);
	}

	/** 相対効用行列の初期化 */
	private void initValueRelativeUtility() throws Exception {
		final List<Issue> issues = getIssues();
		for (Issue issue : issues) {
			// 論点行の初期化
			mValueRelativeUtility.put(issue, new HashMap<Value, Double>());
			// 論点行の要素の初期化
			final ArrayList<Value> values = getValuesForIssue(issue);
			for (Value value : values) {
				mValueRelativeUtility.get(issue).put(value, 0.0);
			}
		}
	}

	/** 最大効用値Bidの初期探索(最初は効用空間のタイプが不明であるため，SAを用いて探索する) */
	private void initMaxBid() throws Exception {
		final AbstractUtilitySpace utilitySpace = mAgent.getUtilitySpace();

		int tryNum = getIssues().size(); // 試行回数
		mMaxBid = mAgent.getUtilitySpace().getDomain().getRandomBid(null);
		for (int i = 0; i < tryNum; i++) {
			do {
				generateFromSimulatedAnnealingSearch(mMaxBid, 1.0);
			} while (utilitySpace.getUtility(mMaxBid) < utilitySpace
					.getReservationValue());

			if (utilitySpace.getUtility(mMaxBid) == 1.0) {
				break;
			}
		}
	}

	/** 相対効用行列の導出 */
	public void setValueRelativeUtility(Bid maxBid) throws Exception {
		final AgentH agent = mAgent;

		Bid currentBid = null;
		final List<Issue> issues = getIssues();
		for (Issue issue : issues) {
			currentBid = new Bid(maxBid);
			final ArrayList<Value> values = getValuesForIssue(issue);
			for (Value value : values) {
				currentBid = currentBid.putValue(issue.getNumber(), value);
				mValueRelativeUtility.get(issue)
						.put(value,
								agent.getUtility(currentBid)
										- agent.getUtility(maxBid));
			}
		}
	}

	public Domain getDomain() {
		return mAgent.getUtilitySpace().getDomain();
	}

	public List<Issue> getIssues() {
		return getDomain().getIssues();
	}

	public ArrayList<Value> getValuesForIssue(Issue issue) {
		final ArrayList<Value> values = new ArrayList<Value>();
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
				// System.out.println("論点の取り得る値の取得に失敗しました");
				// e.printStackTrace();
			}
		}
		return values;
	}

	/** 相対効用値に基づく探索 */
	public Bid generateFromRelativeUtilitySearch(double threshold) {
		Bid bid = new Bid(mMaxBid);
		double d = threshold - 1.0; // 最大効用値との差
		double concessionSum = 0.0; // 減らした効用値の和
		double relativeUtility = 0.0;
		final HashMap<Issue, HashMap<Value, Double>> valueRelativeUtility = mValueRelativeUtility;

		List<Issue> randomIssues = getIssues();
		Collections.shuffle(randomIssues);
		ArrayList<Value> randomValues = null;
		for (Issue issue : randomIssues) {
			randomValues = getValuesForIssue(issue);
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

	/** SA */
	public Bid generateFromSimulatedAnnealingSearch(Bid baseBid,
			double threshold) {
		final AgentH agent = mAgent;
		final List<Issue> issues = getIssues();

		Bid currentBid = new Bid(baseBid); // 初期解の生成
		double currenBidUtil = agent.getUtility(baseBid);
		Bid nextBid = null; // 評価Bid
		double nextBidUtil = 0.0;
		ArrayList<Bid> targetBids = new ArrayList<Bid>(); // 最適効用値BidのArrayList
		double targetBidUtil = 0.0;
		double p; // 遷移確率
		Random randomnr = new Random(); // 乱数
		double currentTemperature = START_TEMPERATURE; // 現在の温度
		double newCost = 1.0;
		double currentCost = 1.0;

		while (currentTemperature > END_TEMPERATURE) { // 温度が十分下がるまでループ
			nextBid = new Bid(currentBid); // next_bidを初期化
			for (int i = 0; i < STEP_NUM; i++) { // 近傍のBidを取得する
				int issueIndex = randomnr.nextInt(issues.size()); // 論点をランダムに指定
				Issue issue = issues.get(issueIndex); // 指定したindexのissue
				ArrayList<Value> values = getValuesForIssue(issue);
				int valueIndex = randomnr.nextInt(values.size()); // 取り得る値の範囲でランダムに指定
				nextBid = nextBid.putValue(issue.getNumber(),
						values.get(valueIndex));
				nextBidUtil = agent.getUtility(nextBid);

				// 最大効用値Bidの更新
				if (mMaxBid == null || nextBidUtil >= agent.getUtility(mMaxBid)) {
					mMaxBid = new Bid(nextBid);
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
					targetBidUtil = agent.getUtility(currentBid);
				} else {
					if (currenBidUtil < targetBidUtil) {
						targetBids.clear(); // 初期化
						targetBids.add(new Bid(currentBid)); // 要素を追加
						targetBidUtil = agent.getUtility(currentBid);
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

	/**
	 * 過去の bid から次の bid を生成
	 * 
	 * @param threshold
	 * @return
	 */
	public Bid generateFromHistory(double threshold) {
		Bid nextBid = null;
		double nextUtility = 0;

		// 過去の bid を効用値の高い順に持ってくる
		final BidHistory bidHistory = mAgent.mBidHistory;
		final List<Entry> entries = bidHistory.getSortedList();
		for (BidHistory.Entry e : entries) {
			nextBid = null;
			nextUtility = e.utility;

			// bid を少し変えたものを次の bid とする
			final List<Issue> issues = e.bid.getIssues();
			for (Issue issue : issues) {
				final int issueNr = issue.getNumber();
				Bid bid = new Bid(e.bid);
				switch (issue.getType()) {
				case DISCRETE: {
					final List<ValueDiscrete> values = ((IssueDiscrete) issue)
							.getValues();
					Collections.shuffle(values);
					bid = bid.putValue(issueNr, values.get(0));
				}
					break;
				case INTEGER: {
					final int upperBound = ((IssueInteger) issue)
							.getUpperBound();
					final int lowerBound = ((IssueInteger) issue)
							.getLowerBound();
					bid = bid.putValue(issueNr, new ValueInteger(lowerBound
							+ mRandom.nextInt(upperBound - lowerBound)));
				}
					break;
				case REAL: {
					final double upperBound = ((IssueReal) issue)
							.getUpperBound();
					final double lowerBound = ((IssueReal) issue)
							.getLowerBound();
					bid = bid
							.putValue(issueNr, new ValueReal(lowerBound
									+ mRandom.nextDouble()
									* (upperBound - lowerBound)));
				}
					break;
				}
				if (nextUtility - mAgent.getUtility(bid) < threshold
						&& !bidHistory.containsBid(bid)) {
					nextBid = bid;
					// System.out.println("OreoreAgent#generateNextBid(): nextBid="+nextBid);
				}
			}

			if (nextBid != null) {
				break;
			}
		}

		return nextBid;
	}
}
