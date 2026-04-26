package agents.anac.y2015.RandomDance;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.EndNegotiation;
import genius.core.actions.Offer;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.issue.ValueInteger;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.utility.AdditiveUtilitySpace;

/**
 * This is your negotiation party.
 */
public class RandomDance extends AbstractNegotiationParty {

	/*
	 * パレート最適最速アタックを目指す(ひとまず)
	 */

	@Override
	public void init(NegotiationInfo info) {
		super.init(info);
	}

	final int NashCountMax = 200;
	final int NumberOfAcceptSafety = 5;
	final int NumberOfRandomTargetCheck = 3;

	private boolean init = false;
	/**
	 * Map with {@link AgentID} as key
	 */
	private Map<String, PlayerDataLib> utilityDatas = new HashMap<String, PlayerDataLib>();
	private PlayerData myData = null;

	private List<String> nash = new LinkedList<String>();
	private Map<String, Bid> olderBidMap = new HashMap<String, Bid>();

	private double discountFactor = 1.0;
	private double reservationValue = 0;

	private double olderTime = 0;

	@Override
	public Action chooseAction(List<Class<? extends Action>> validActions) {

		if (!init) {
			init = true;
			myInit();
		}

		Map<String, PlayerData> utilityMap = new HashMap<String, PlayerData>();
		for (String str : utilityDatas.keySet()) {
			utilityMap.put(str, utilityDatas.get(str).getRandomPlayerData());
		}
		utilityMap.put("my", myData);

		/*
		 * 前回の相手のBidのうち、より他の相手に歩み寄って
		 * いるプレイヤーを探す : Find player that walked maximum
		 * value?
		 */
		double maxval = -999;
		String maxPlayer = null;
		for (String string : olderBidMap.keySet()) {
			double utility = 1.0;
			for (String player : utilityMap.keySet()) {
				if (string.equals(player)) {
					continue;
				}
				utility *= utilityMap.get(player)
						.GetUtility(olderBidMap.get(string));
			}
			if (utility > maxval) {
				maxval = utility;
				maxPlayer = string;
			}
		}
		if (maxPlayer != null) {
			nash.add(maxPlayer);
		}
		while (nash.size() > NashCountMax) {
			nash.remove(0);
		}

		Map<String, Double> playerWeight = getWeights();

		Action action = null;
		Offer offer = null;

		double target = GetTarget(utilityMap);
		double utility = 0;

		if (olderBid != null) {

			try {
				utility = utilitySpace.getUtility(olderBid);
			} catch (Exception e) {
				// TODO 自動生成された catch ブロック
				e.printStackTrace();
			}
		}

		try {
			offer = new Offer(getPartyId(),
					SearchBid(target, utilityMap, playerWeight));
			action = offer;
		} catch (Exception e) {
			// TODO 自動生成された catch ブロック
			e.printStackTrace();
		}

		if (action == null || IsAccept(target, utility)) {
			action = new Accept(getPartyId(), olderBid);
		}
		if (IsEndNegotiation(target)) {
			action = new EndNegotiation(getPartyId());
		}

		return action;
	}

	public Map<String, Double> getWeights() {
		/*
		 * プレイヤーウェイトの計算 : Calculation of player weight
		 */
		Map<String, Double> playerWeight = new HashMap<String, Double>();
		int rand = (int) (Math.random() * 3);

		switch (rand) {
		case 0:
			for (String string : utilityDatas.keySet()) {
				playerWeight.put(string, 0.0001);
			}
			for (String string : nash) {
				playerWeight.put(string, playerWeight.get(string) + 1.0);
			}
			break;

		case 1:
			for (String string : utilityDatas.keySet()) {
				playerWeight.put(string, 1.0);
			}
			break;
		case 2:
			boolean flag = Math.random() < 0.5;
			for (String string : utilityDatas.keySet()) {

				if (string.equals("my")) {
					continue;
				}

				if (flag) {
					playerWeight.put(string, 1.0);
				} else {
					playerWeight.put(string, 0.01);
				}
				flag = !flag;
			}
			break;
		default:
			for (String string : utilityDatas.keySet()) {
				playerWeight.put(string, 1.0);
			}
			break;
		}

		// System.err.println("PlayerWeight : " + playerWeight.toString());

		return playerWeight;
	}

	Bid olderBid = null; // the last received bid

	@Override
	public void receiveMessage(AgentID sender, Action action) {
		super.receiveMessage(sender, action);

		if (sender == null) {
			return;
		}
		if (utilityDatas.containsKey(sender.toString()) == false) {
			utilityDatas.put(sender.toString(),
					new PlayerDataLib(utilitySpace.getDomain().getIssues()));
		}

		if (action.getClass() == Offer.class) {
			Offer offer = (Offer) action;
			Bid bid = offer.getBid();
			olderBid = bid;
		}

		olderBidMap.put(sender.toString(), olderBid);

		try {
			utilityDatas.get(sender.toString()).AddBid(olderBid);
		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	private boolean IsAccept(double target, double utility) {
		double time = timeline.getTime();
		double d = time - olderTime;
		olderTime = time;
		// 時間ギリギリならAccept
		if (time + d * NumberOfAcceptSafety > 1.0) {
			// System.err.println("Accept Time");
			return true;
		}

		if (olderBid == null) {
			return false;
		}
		// targetより大きければAccept : If greater than the target Accept
		if (utility > target) {
			// System.err.println("Accept utility over target! " + target + " "
			// + utility);
			return true;
		}
		return false;
	}

	private boolean IsEndNegotiation(double target) {

		if (target < reservationValue) {
			return true;
		}

		return false;
	}

	private double GetTarget(Map<String, PlayerData> datas) {

		double max = 0;

		Map<String, Double> weights = new HashMap<String, Double>();

		/*
		 * for(String str:datas.keySet()){ weights.put(str, 1.0); } Bid bid =
		 * SearchBidWithWeights(datas, weights); try { max = Math.max(max,
		 * utilitySpace.getUtility(bid)); } catch (Exception e) { // TODO
		 * 自動生成された catch ブロック e.printStackTrace(); }
		 */

		for (int i = 0; i < NumberOfRandomTargetCheck; i++) {

			Map<String, PlayerData> utilityMap = new HashMap<String, PlayerData>();
			for (String str : utilityDatas.keySet()) {
				utilityMap.put(str,
						utilityDatas.get(str).getRandomPlayerData());
				weights.put(str, 1.0);
			}
			utilityMap.put("my", myData);
			weights.put("my", 1.0);

			Bid bid = SearchBidWithWeights(utilityMap, weights);
			try {
				max = Math.max(max, utilitySpace.getUtility(bid));
			} catch (Exception e) {
				// TODO 自動生成された catch ブロック
				e.printStackTrace();
			}

		}

		double target = 1.0
				- (1.0 - max) * (Math.pow(timeline.getTime(), discountFactor));

		if (discountFactor > 0.99) {
			target = 1.0 - (1.0 - max) * (Math.pow(timeline.getTime(), 3));
		}

		// System.err.println("time = " + timeline.getTime()+ "target = " +
		// target +" max = " + max);
		return target;
	}

	private void myInit() {
		PlayerData playerData = new PlayerData(
				utilitySpace.getDomain().getIssues(), 1.0);

		try {
			playerData.SetMyUtility((AdditiveUtilitySpace) utilitySpace);
		} catch (Exception e) {
			// TODO 自動生成された catch ブロック
			e.printStackTrace();
		}
		myData = playerData;

		reservationValue = utilitySpace.getReservationValue();
		discountFactor = utilitySpace.getDiscountFactor();

	}

	/**
	 * 
	 * @param datas
	 * @param weights
	 *            Map with {@link AgentID} as key, and weight as value.
	 * @return bid that has maximum utility, using the given weights.
	 */
	private Bid SearchBidWithWeights(Map<String, PlayerData> datas,
			Map<String, Double> weights) {
		Bid ret = generateRandomBid();
		// there must be some player, we use it to get the issue values.
		PlayerData player0data = datas.get(datas.keySet().iterator().next());
		for (Issue issue : utilitySpace.getDomain().getIssues()) {

			List<Value> values = player0data.getIssueData(issue).getValues();

			double max = -1;
			Value maxValue = null;

			for (Value value : values) {

				double v = 0;

				for (String string : datas.keySet()) {
					PlayerData data = datas.get(string);
					double weight = weights.get(string);
					v += data.GetValue(issue, value) * weight;
				}
				if (v > max) {
					max = v;
					maxValue = value;
				}
			}
			ret = ret.putValue(issue.getNumber(), maxValue);
		}
		return ret;
	}

	/*
	 * target以上で良い感じに
	 */
	private Bid SearchBid(double target, Map<String, PlayerData> datas,
			Map<String, Double> weights) throws Exception {

		/*
		 * 引数に渡すようのMapを作る
		 */
		Map<String, PlayerData> map = new HashMap<String, PlayerData>();
		map.putAll(datas);
		map.put("my", myData);

		Map<String, Double> weightbuf = new HashMap<String, Double>();
		/*
		 * 敵ウェイトを合計1になるようにする: set weights to sum
		 * to 1
		 */
		double sum = 0;
		for (Double d : weights.values()) {
			sum += d;
		}
		for (String key : weights.keySet()) {
			weightbuf.put(key, weights.get(key) / sum);
		}

		for (double w = 0; w < 9.999; w += 0.01) {

			double myweight = w / (1.0 - w);
			weightbuf.put("my", myweight);

			Bid bid = SearchBidWithWeights(map, weightbuf);

			if (utilitySpace.getUtility(bid) > target) {
				return bid;
			}
		}

		return utilitySpace.getMaxUtilityBid();
	}

	@Override
	public String getDescription() {
		return "ANAC2015";
	}

}

/**
 * 3 player datas for each player, one with constant derta, one with decreasing
 * and one with increasing.
 */
class PlayerDataLib {

	ArrayList<PlayerData> playerDatas = new ArrayList<PlayerData>();

	public PlayerDataLib(List<Issue> issues) {
		playerDatas.add(new PlayerData(issues, 1.0));
		playerDatas.add(new PlayerData(issues, 1.05));
		playerDatas.add(new PlayerData(issues, 0.95));
	}

	public PlayerData getRandomPlayerData() {
		int rand = (int) (Math.random() * playerDatas.size());
		return playerDatas.get(rand);
	}

	public void AddBid(Bid bid) {
		for (PlayerData d : playerDatas) {
			try {
				d.AddBid(bid);
			} catch (Exception e) {
				// TODO 自動生成された catch ブロック
				e.printStackTrace();
			}
		}
	}

	public ArrayList<PlayerData> getPlayerDataList() {
		return playerDatas;
	}

}

class PlayerData {
	// IssueData is raw type.
	Map<Issue, IssueData> map = new HashMap<Issue, IssueData>();
	Set<Bid> history = new HashSet<Bid>();
	double derta = 1.00;

	public IssueData<Issue, Value> getIssueData(Issue issue) {
		return map.get(issue);
	}

	/**
	 * Mape a map of IssueData for all issues.
	 * 
	 * @param issues
	 * @param derta
	 *            the relevance change for each next bid. So if eg 0.8, each
	 *            next update has a relevance 0.8 times the previous update
	 */
	public PlayerData(List<Issue> issues, double derta) {
		for (Issue issue : issues) {
			if (issue instanceof IssueDiscrete) {
				map.put(issue,
						new IssueDataDiscrete((IssueDiscrete) issue, derta));
			} else {
				map.put(issue,
						new IssueDataInteger((IssueInteger) issue, derta));
			}
		}
		this.derta = derta;
	}

	public double GetUtility(Bid bid) {
		double ret = 0;
		for (Issue issue : bid.getIssues()) {
			try {
				ret += GetValue(issue, bid.getValue(issue.getNumber()));
			} catch (Exception e) {
				// TODO 自動生成された catch ブロック
				e.printStackTrace();
			}
		}
		return ret;
	}

	public double GetValue(Issue issue, Value value) {
		ValuePut(issue);
		return map.get(issue).GetValueWithWeight(value);
	}

	public void SetMyUtility(AdditiveUtilitySpace utilitySpace)
			throws Exception {

		Bid bid = utilitySpace.getMinUtilityBid();
		List<Issue> issues = utilitySpace.getDomain().getIssues();
		double min = utilitySpace.getUtility(bid);

		for (Issue issue : issues) {
			IssueData issueData = map.get(issue);
			bid = utilitySpace.getMinUtilityBid();
			List<Value> values = issueData.getValues();

			for (Value value : values) {
				bid = bid.putValue(issue.getNumber(), value);
				double v = utilitySpace.getUtility(bid) - min;
				issueData.setValue(value, v);
			}
			issueData.setWeight(1.0 / (1.0 - min));
			issueData.Locked();
		}

		// System.err.println(this.toString());

	}

	/**
	 * Accumulates occurences of issues
	 * 
	 * @param bid
	 * @throws Exception
	 */
	public void AddBid(Bid bid) throws Exception {

		if (history.contains(bid)) {
			return;
		}
		history.add(bid);

		double countsum = 0;
		for (Issue issue : bid.getIssues()) {
			ValuePut(issue);
			map.get(issue).Update(bid.getValue(issue.getNumber()));
			countsum += map.get(issue).getMax();
		}

		for (Issue issue : bid.getIssues()) {
			map.get(issue).setWeight(map.get(issue).getMax() / countsum);
		}
	}

	/*
	 * Mapにキーがない時に追加する関数: put in map if not there
	 * yet
	 */
	private void ValuePut(Issue issue) {
		if (!map.containsKey(issue)) {
			if (issue instanceof IssueDiscrete) {
				map.put(issue,
						new IssueDataDiscrete((IssueDiscrete) issue, derta));
			} else {
				map.put(issue,
						new IssueDataInteger((IssueInteger) issue, derta));
			}
		}
	}

	@Override
	public String toString() {

		String ret = "";
		for (Issue issue : map.keySet()) {
			ret += issue.toString() + ":" + map.get(issue).toString() + "\n";
		}
		return ret;
	}

	/*
	 * 各Issueごとの数え上げデータ: count data for each issue
	 * Refactored W.Pasman to handle other issue types
	 */
	abstract class IssueData<IssueType extends Issue, ValueType extends Value> {
		private boolean locked = false;
		private double weight = 1;
		private double derta = 1.0;
		private double max = 1;
		private Map<Value, Double> map = new HashMap<Value, Double>();
		private double adder = 1.0;
		private IssueType issue;

		/**
		 * 
		 * @param issue
		 *            the issue that this data contains
		 * @param derta
		 *            the relevance change for each next bid. So if eg 0.8, each
		 *            next update has a relevance 0.8 times the previous update.
		 */
		public IssueData(IssueType issue, double derta) {
			this.issue = issue;
			this.derta = derta;

			for (ValueType value : getValues()) {
				setValue(value, 0);
			}
		}

		public Issue getIssue() {
			return issue;
		}

		/**
		 * Returns the possible values for this issue. Similar to
		 * {@link IssueDiscrete#getValues()}
		 */
		public abstract List<ValueType> getValues();

		/**
		 * /* 更新禁止のロックをかける ロックは外せない :
		 * lock the update
		 */
		public void Locked() {
			locked = true;
		}

		public double getWeight() {
			return weight;
		}

		public void setWeight(double weight) {
			this.weight = weight;
		}

		public boolean isLocked() {
			return locked;
		}

		private double getMax() {
			return max;
		}

		double GetValue(ValueType value) {
			ValuePut(value);
			return map.get(value) / max;
		}

		double GetValueWithWeight(ValueType value) {
			return GetValue(value) * getWeight();
		}

		/*
		 * 相手のBidがきた時の更新関数 とりあえず1を足す put
		 * value in the map. Each next update will have relevance changed by
		 * factor derta.
		 */
		public void Update(ValueType value) {
			if (isLocked()) {
				System.err.println("LockedAccess!!");
				return;
			}
			ValuePut(value);
			map.put(value, map.get(value) + adder);
			max = Math.max(max, map.get(value));
			adder *= derta;
		}

		@Override
		public String toString() {
			return "weight:" + getWeight() + ":" + map.toString();
		}

		protected void setValue(ValueType value, double util) {
			if (isLocked()) {
				System.err.println("LockedAccess!!");
			} else {
				map.put(value, util);
			}
		}

		/*
		 * Mapにキーがない時に追加する関数 : put key in map if not
		 * yet there
		 */
		protected void ValuePut(ValueType value) {
			if (!map.containsKey(value)) {
				map.put(value, 0.0);
			}
		}

	}

	class IssueDataDiscrete extends IssueData<IssueDiscrete, ValueDiscrete> {

		public IssueDataDiscrete(IssueDiscrete issue, double derta) {
			super(issue, derta);
		}

		@Override
		public List<ValueDiscrete> getValues() {
			return ((IssueDiscrete) getIssue()).getValues();
		}
	}

	class IssueDataInteger extends IssueData<IssueInteger, ValueInteger> {

		public IssueDataInteger(IssueInteger issue, double derta) {
			super(issue, derta);
		}

		@Override
		public List<ValueInteger> getValues() {
			IssueInteger iss = (IssueInteger) getIssue();
			List<ValueInteger> values = new ArrayList<ValueInteger>();

			for (int v = iss.getLowerBound(); v <= iss.getUpperBound(); v++) {
				values.add(new ValueInteger(v));
			}
			return values;
		}

	}

}
