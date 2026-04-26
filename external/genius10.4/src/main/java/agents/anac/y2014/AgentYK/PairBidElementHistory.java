package agents.anac.y2014.AgentYK;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;


public class PairBidElementHistory {
	List<PairBidElementDetails> pairBidElements;
	HashMap<String, Integer> countMap;
	HashMap<String, Double> weightedCountMap;
	
	public PairBidElementHistory() {
		this.pairBidElements = new ArrayList<PairBidElementDetails>();
		countMap = new HashMap<String, Integer>();
		weightedCountMap = new HashMap<String, Double>();
	}
			
	public PairBidElementHistory(List<PairBidElementDetails> pairBidElements) {
		for(PairBidElementDetails pbe:pairBidElements) {
			this.pairBidElements.add(pbe);
		}
		countMap = new HashMap<String, Integer>();
		weightedCountMap = new HashMap<String, Double>();
	}
	
	public void add(PairBidElementDetails pbed) {
		this.pairBidElements.add(pbed);
		if(this.countMap.containsKey(pbed.getPairBidElement().toString())) {
			int value = this.countMap.get(pbed.getPairBidElement().toString()) + 1;
			this.countMap.put(pbed.getPairBidElement().toString(), value);
		} else {
			this.countMap.put(pbed.getPairBidElement().toString(), 1);
		}
		
		if(this.weightedCountMap.containsKey(pbed.getPairBidElement().toString())) {
			double value = this.weightedCountMap.get(pbed.getPairBidElement().toString()) + pbed.getTime()*pbed.getTime();
			this.weightedCountMap.put(pbed.getPairBidElement().toString(), value);
		} else {
			this.weightedCountMap.put(pbed.getPairBidElement().toString(), pbed.getTime()*pbed.getTime());
		}
	}
	
	public List<PairBidElementDetails> getHistory() {
		return this.pairBidElements;
	}
	
	public PairBidElementHistory filterBetweenTime(double t1, double t2) {
		PairBidElementHistory pbeh = new PairBidElementHistory();
		for(PairBidElementDetails pbed:this.pairBidElements) {
			if(t1 < pbed.getTime() && pbed.getTime() <= t2) pbeh.add(pbed);
		}
		return pbeh;
	}
	
	public int getAppearanceCount(PairBidElement pairBidElement) {
		if(this.countMap.containsKey(pairBidElement.toString())) {
			return this.countMap.get(pairBidElement.toString());
		}
		return 0;
	}
	
	public int getMaxAppearanceCount() {
		int max = 0;
		for(Integer i:this.countMap.values()) {
			if(max < i) max = i;
		}
		return max;
	}
	
	public double getWeightedAppearanceCount(PairBidElement pairBidElement) {
		if(this.weightedCountMap.containsKey(pairBidElement.toString())) {
			return this.weightedCountMap.get(pairBidElement.toString());
		}
		return 0;
	}
	
	public double getWeightedMaxAppearanceCount() {
		double max = 0.0;
		for(Double i:this.weightedCountMap.values()) {
			if(max < i) max = i;
		}
		return max;
	}
	
	public int size() {
		return pairBidElements.size();
	}
}
