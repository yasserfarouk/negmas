package agents.anac.y2015.pokerface;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import genius.core.Bid;
import genius.core.issue.Issue;
import genius.core.issue.Value;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.Evaluator;

import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

public class OpponentBidLists {

	private List<Object> senders;
	// map: Sender -> ("<issue_id,issue_value>" -> count)
	private Map<Object, Map<Pair<Integer, Value>, Integer>> map;
	private boolean single_list;
	private AdditiveUtilitySpace utilitySpace;

	public OpponentBidLists(AdditiveUtilitySpace utilitySpace, boolean single_list) {
		senders = new ArrayList<Object>();
		map = new HashMap<Object, Map<Pair<Integer, Value>, Integer>>();
		this.single_list = single_list;
		this.utilitySpace = utilitySpace;
		if (single_list) {
			Map<Pair<Integer, Value>, Integer> bid_list = new TreeMap<Pair<Integer, Value>, Integer>(
					new MapComparator());
			map.put(this, bid_list);
		}
	}

	private boolean hasSender(Object sender) {
		return (senders.indexOf(sender) != -1);
	}

	private void addSender(Object sender) {
		if (!single_list) {
			Map<Pair<Integer, Value>, Integer> bid_list = new TreeMap<Pair<Integer, Value>, Integer>(
					new MapComparator());
			map.put(sender, bid_list);
		}
		senders.add(sender);
	}

	public List<Object> getSenders() {
		return senders;
	}

	public void insertBid(Object sender, Bid bid) {
		if (!hasSender(sender)) {
			addSender(sender);
		}
		Map<Pair<Integer, Value>, Integer> issue_value_counts;
		if (single_list) {
			issue_value_counts = map.get(this);
		} else {
			issue_value_counts = map.get(sender);
		}

		List<Issue> issues = bid.getIssues();
		Map<Integer, Value> values = bid.getValues();

		for (int j = 0; j < values.size(); j++) {
			Integer issue_id = issues.get(j).getNumber();
			Value value_id = values.get(j + 1);
			Pair<Integer, Value> pair = new Pair<Integer, Value>(issue_id,
					value_id);
			Integer current_count = issue_value_counts.get(pair);
			if (current_count == null)
				issue_value_counts.put(pair, 1);
			else
				issue_value_counts.put(pair, current_count++);
		}
	}

	public Map<Pair<Integer, Value>, Integer> getBids(Object sender) {
		if (single_list) {
			return map.get(this);
		} else {
			return map.get(sender);
		}
	}

	public List<Entry<Pair<Integer, Value>, Integer>> getMostFrequentIssueValues(
			Object sender) {
		Map<Pair<Integer, Value>, Integer> issue_value_counts = map.get(sender);
		if (sender == null || map.get(sender) == null)
			return new ArrayList<Entry<Pair<Integer, Value>, Integer>>(0);
		Set<Entry<Pair<Integer, Value>, Integer>> pull_through = new TreeSet<Entry<Pair<Integer, Value>, Integer>>(
				new Comparator<Entry<Pair<Integer, Value>, Integer>>() {
					public int compare(Entry<Pair<Integer, Value>, Integer> e1,
							Entry<Pair<Integer, Value>, Integer> e2) {
						return e1.getValue().compareTo(e2.getValue());
					}
				});
		pull_through.addAll(issue_value_counts.entrySet());
		return new ArrayList<Entry<Pair<Integer, Value>, Integer>>(pull_through);
	}

	public List<Entry<Pair<Integer, Value>, Double>> weightIssueValues(
			List<Entry<Pair<Integer, Value>, Integer>> sorted_list) {
		Map<Pair<Integer, Value>, Double> map = new HashMap<Pair<Integer, Value>, Double>();
		for (Entry<Pair<Integer, Value>, Integer> entry : sorted_list) {
			double value = (double) entry.getValue();
			int issue_id = entry.getKey().getInteger();
			Evaluator evaluator = utilitySpace.getEvaluator(issue_id);
			double weight = evaluator.getWeight();
			double evaluation;
			try {
				Bid temp = utilitySpace.getMaxUtilityBid();
				temp = temp.putValue(issue_id, entry.getKey().getValue());
				evaluation = evaluator.getEvaluation(utilitySpace, temp,
						issue_id);
			} catch (Exception e) {
				evaluation = 1.0;
			}
			map.put(entry.getKey(), value * weight * evaluation);
		}
		List<Entry<Pair<Integer, Value>, Double>> weighted_list = new ArrayList<Entry<Pair<Integer, Value>, Double>>(
				((Map<Pair<Integer, Value>, Double>) map).entrySet());
		Collections.sort(weighted_list,
				new Comparator<Entry<Pair<Integer, Value>, Double>>() {
					public int compare(Entry<Pair<Integer, Value>, Double> o1,
							Entry<Pair<Integer, Value>, Double> o2) {
						return o1.getValue().compareTo(o2.getValue());
					}
				});
		return weighted_list;
	}

	private class MapComparator implements Comparator<Pair<Integer, Value>> {
		public MapComparator() {
		}

		public int compare(Pair<Integer, Value> p1, Pair<Integer, Value> p2) {
			return (p1.getInteger() + p1.getValue().toString()).compareTo(p2
					.getInteger() + p2.getValue().toString());
		}
	}
}