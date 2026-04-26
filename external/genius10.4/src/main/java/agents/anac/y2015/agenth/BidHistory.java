package agents.anac.y2015.agenth;

import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import genius.core.Bid;
import genius.core.Domain;
import genius.core.issue.Issue;
import genius.core.issue.Value;
import genius.core.utility.AdditiveUtilitySpace;

public class BidHistory extends LinkedList<BidHistory.Entry> {
	/** UtilitySpace */
	private AdditiveUtilitySpace mUtilitySpace;
	/** bid と sender の対応マップ */
	private HashMap<Bid, Object> mBids;
	/** エージェントごとの値 */
	private HashMap<Object, ValueCounter> mValueCounterMap;
	/** 最後に挿入された bid */
	private Bid mLastBid;

	/** 全体の平均 bid */
	private Bid mMeans;
	private ValueCounter mValueCounter;

	public BidHistory(AdditiveUtilitySpace utilitySpace) {
		mUtilitySpace = utilitySpace;
		mBids = new HashMap<Bid, Object>();
		mValueCounterMap = new HashMap<Object, ValueCounter>();
		mValueCounter = new ValueCounter();
	}

	public void offer(Object sender, Bid bid, double utility) {
		add(new Entry(sender, bid, utility));
		mBids.put(bid, sender);
		mLastBid = bid;

		// ValueCounter に入れる
		mValueCounter.add(bid);

		ValueCounter counter = mValueCounterMap.get(sender);
		if (counter == null) {
			counter = new ValueCounter();
			mValueCounterMap.put(sender, counter);
		}
		counter.add(bid); // TODO 重み加えるとかする
	}

	public void accept(Object sender, Bid bid) {
		ValueCounter counter = mValueCounterMap.get(sender);
		if (counter == null) {
			counter = new ValueCounter();
			mValueCounterMap.put(sender, counter);
		}
		counter.add(bid); // TODO 重み加えるとかする
	}

	public boolean containsBid(Bid bid) {
		return mBids.containsKey(bid);
	}

	/**
	 * 自分の効用値が高い順にソートされたリストを返す
	 * 
	 * @return
	 */
	public List<Entry> getSortedList() {
		final LinkedList<Entry> list = new LinkedList<Entry>(this);
		Collections.sort(list, mUtilityComparator);
		return list;
	}

	public double getProbOfValue(Object agent, Issue issue, Value value) {
		ValueCounter counter = mValueCounterMap.get(agent);
		if (counter != null) {
			return (counter.get(issue).get(value) / counter.get(issue).size());
		}
		return 0; // FIXME
	}

	private Comparator<Entry> mUtilityComparator = new Comparator<Entry>() {
		@Override
		public int compare(Entry o1, Entry o2) {
			return (int) Math.round((o1.utility - o2.utility) * 10e5);
		}
	};

	/**
	 * BidHistory が持つエントリ
	 */
	public class Entry {
		public Object sender;
		public Bid bid;
		public double utility;

		public Entry(Object sender, Bid bid, double utility) {
			this.bid = bid;
			this.utility = utility;
		}
	}

	/**
	 * Value の頻度を数えるクラス HashMap<IssueNr, HashMap<Value, Count>>
	 */
	private class ValueCounter extends HashMap<Issue, HashMap<Value, Integer>> {

		public ValueCounter() {
			final Domain domain = mUtilitySpace.getDomain();
			final List<Issue> issues = domain.getIssues();
			for (Issue issue : issues) {
				put(issue, new HashMap<Value, Integer>());
			}
		}

		public void add(Bid bid) {
			try {
				final List<Issue> issues = bid.getIssues();
				for (Issue issue : issues) {
					final Value value = bid.getValue(issue.getNumber());
					increase(issue, value);
				}
				System.out.println("BidHistory.ValueCounter#add(): " + this);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}

		public void increase(Issue issue, Value value) {
			final HashMap<Value, Integer> map = get(issue);
			if (map != null) {
				Integer count = map.get(value);
				if (count == null) {
					count = new Integer(0);
				}
				map.put(value, count + 1);
			}
		}

		public Value getMostLikelyValue(Issue issue) {
			final HashMap<Value, Integer> map = get(issue);
			if (map != null) {
				Value v = null;
				int n = 0;
				final Set<Map.Entry<Value, Integer>> es = map.entrySet();
				for (Map.Entry<Value, Integer> e : es) {
					if (n < e.getValue()) {
						v = e.getKey();
						n = e.getValue();
					}
				}
				return v;
			}
			return null;
		}
	}
}
