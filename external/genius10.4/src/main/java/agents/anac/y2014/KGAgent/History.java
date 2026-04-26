package agents.anac.y2014.KGAgent;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import agents.anac.y2014.kGA_gent.library_genetic.GA_Main;
import agents.anac.y2014.kGA_gent.library_genetic.Gene;
import genius.core.Bid;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.Value;
import genius.core.issue.ValueInteger;

public class History {

	static int pairsize = 2;

	double add = 0.00001;
	double derta = 1.1;
	static int type = 0;

	Map<Bid, Integer> bidhistory = new HashMap<Bid, Integer>(1000);
	// ArrayList<Bid> bidlisthistory = new ArrayList<Bid>(1000);
	List<Issue> issues;

	KGAgent agent = null;

	Map<UtilityPair, Double> utility = new HashMap<UtilityPair, Double>(10000);
	int count = 0;

	History(List<Issue> issuelist, KGAgent agent, int ty) {
		issues = issuelist;
		this.agent = agent;
		type = ty;
	}

	// ç›¸æ‰‹ã�®æ��æ¡ˆã‚’å…¥åŠ›ã�™ã‚‹
	void Input(Bid input) {

		if (bidhistory.containsKey(input)) {
			bidhistory.put(input, bidhistory.get(input) + 1);
		} else {
			count++;
			// System.out.println("Enemy New Offer  sise = " + count);
			bidhistory.put(input, 1);
			// bidlisthistory.add(input);
			AddHistory(input);
		}
	}

	/*
	 * ç›¸æ‰‹ã�®äºˆæƒ³åŠ¹ç�?¨å€¤ã‚’è¿�?ã�™
	 */
	double GetEnemyUtility(Bid input) {

		if (type == 0) {
			return GetUtilNonLin(input);
		} else {
			return GetUtilLin(input);
		}
	}

	double GetUtilNonLin(Bid bid) {

		HashMap<Integer, Value> map = bid.getValues();
		IssueInteger lIssueInteger;
		ValueInteger v1, v2;

		double ret = 0.0;

		UtilityPair pair = new UtilityPair();

		int size = issues.size();

		if (count == 0) {
			return 0;
		}
		for (int i = 0; i < size; i++) {
			lIssueInteger = (IssueInteger) issues.get(i);
			v1 = (ValueInteger) map.get(lIssueInteger.getNumber());
			pair.add(new Pair(lIssueInteger.getNumber(), v1.getValue()));

			for (int j = i + 1; j < size; j++) {
				lIssueInteger = (IssueInteger) issues.get(j);
				v1 = (ValueInteger) map.get(lIssueInteger.getNumber());
				pair.add(new Pair(lIssueInteger.getNumber(), v1.getValue()));

				if (utility.containsKey(pair)) {
					ret += utility.put(pair, utility.get(pair));
				}
				pair.remove(pair.size() - 1);
			}
			pair.remove(pair.size() - 1);

		}
		return ret;
	}

	double GetUtilLin(Bid bid) {

		HashMap<Integer, Value> map = bid.getValues();
		IssueDiscrete ID;

		double ret = 0.0;

		UtilityPair pair = new UtilityPair();

		int size = issues.size();

		if (count == 0) {
			return 0;
		}
		for (int i = 0; i < size; i++) {
			ID = (IssueDiscrete) issues.get(i);

			pair.add(new Pair(ID.getNumber(), ID.getNumberOfValues()));

			if (utility.containsKey(pair)) {
				ret += utility.put(pair, utility.get(pair));
			}
			pair.remove(pair.size() - 1);

		}
		return ret;
	}

	/*
	 * ç›¸æ‰‹ã�®å…¥åŠ›bidã�‹ã‚‰ã�™ã�¹ã�¦ã�®ãƒšã‚¢ã‚’ä½œã�£ã�¦Mapã�«æ ¼ç´�ã�™ã‚‹
	 * ç·šå½¢ã�ªã‚‰ã‚·ãƒ³ã‚°ãƒ«ã€€é�žç·šå½¢ã�ªã‚‰ãƒšã‚¢
	 */
	void AddHistory(Bid bid) {
		if (type == 0) {
			AddHistoryNonLin(bid);

		} else {
			AddHistoryLin(bid);
		}
	}

	void AddHistoryNonLin(Bid bid) {

		HashMap<Integer, Value> map = bid.getValues();
		IssueInteger lIssueInteger;
		ValueInteger v1;

		UtilityPair pair = new UtilityPair();

		add *= derta;

		int size = issues.size();

		for (int i = 0; i < size; i++) {
			lIssueInteger = (IssueInteger) issues.get(i);
			v1 = (ValueInteger) map.get(lIssueInteger.getNumber());
			pair.add(new Pair(lIssueInteger.getNumber(), v1.getValue()));

			for (int j = i + 1; j < size; j++) {
				lIssueInteger = (IssueInteger) issues.get(j);
				v1 = (ValueInteger) map.get(lIssueInteger.getNumber());
				pair.add(new Pair(lIssueInteger.getNumber(), v1.getValue()));

				// System.out.print("add!" + pair.get(0).key
				// +""+pair.get(1).key);

				UtilityPair p = new UtilityPair();
				p.addAll(pair);
				if (utility.containsKey(p)) {
					// utility.put(p, add);
					utility.put(p, utility.get(pair) + add);

				} else {

					utility.put(p, add);
				}
				pair.remove(pair.size() - 1);
			}
			pair.remove(pair.size() - 1);

		}
	}

	void AddHistoryLin(Bid bid) {
		HashMap<Integer, Value> map = bid.getValues();
		UtilityPair pair = new UtilityPair();
		add *= derta;
		IssueDiscrete ID;
		int size = issues.size();

		for (int i = 0; i < size; i++) {

			ID = (IssueDiscrete) issues.get(i);

			pair.add(new Pair(ID.getNumber(), ID.getNumberOfValues()));
			UtilityPair p = new UtilityPair();
			p.addAll(pair);
			if (utility.containsKey(p)) {
				// utility.put(p, add);
				utility.put(p, utility.get(pair) + add);

			} else {

				utility.put(p, add);
			}

			pair.remove(pair.size() - 1);
		}

	}

	List<Gene> list = null;

	double SearchMaxPoint() {

		double ret = 0.0;

		GA_Main gaMain;

		if (list == null) {
			gaMain = new GA_Main(new BidGenerationChange(100),
					new CompMyBidGene(-1));

		} else {
			List<Gene> in = new ArrayList<Gene>(100);
			for (int i = 0; i < 10; i++) {
				in.add(new MyBidGene(((MyBidGene) list.get(i)).bid));
			}
			for (int i = 0; i < 100; i++) {
				in.add(new MyBidGene(agent.GetRandomBid()));
			}
			gaMain = new GA_Main(in, new BidGenerationChange(100, 10),
					new CompMyBidGene(0));

		}
		gaMain.Start();
		ret = ((MyBidGene) gaMain.GetList().get(0)).GetValue(-1);
		list = null;
		list = gaMain.GetList();
		return ret;

	}
}

class UtilityPair extends ArrayList<Pair> {

	public UtilityPair() {
		super();
	}

	Boolean Hit(Bid b) {
		for (Pair p : this) {
			HashMap<Integer, Value> map = b.getValues();
			if (((ValueInteger) map.get(p.key)).getValue() == p.value) {
				return false;
			}
		}
		return true;
	}
}

class Pair {
	int key, value;

	Pair(int k, int v) {
		key = k;
		value = v;
	}

	@Override
	public int hashCode() {
		return ((key << 16) | value);
	}

	@Override
	public boolean equals(Object obj) {
		Pair p = (Pair) obj;
		if (key == p.key && value == p.value) {
			return true;
		}
		return false;
	}
}

class CompPair implements java.util.Comparator<Pair> {
	@Override
	public int compare(Pair o1, Pair o2) {
		double x = o1.key - o2.key;
		if (x > 0) {
			return 1;
		}
		if (x < 0) {
			return -1;
		}
		return 0;
	}

}
