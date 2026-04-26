package agents.anac.y2014.AgentYK;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;


public class BidElementHistory {
	List<BidElementDetails> bidElements;
	HashMap<String, Integer> countMap;
	HashMap<String, Double> weightedCountMap;
	
	public BidElementHistory() {
		this.bidElements = new ArrayList<BidElementDetails>();
		countMap = new HashMap<String, Integer>();
		weightedCountMap = new HashMap<String, Double>();
	}
			
	public BidElementHistory(List<BidElementDetails> bidElements) {
		for(BidElementDetails bed:bidElements) {
			this.bidElements.add(bed);
		}
		countMap = new HashMap<String, Integer>();
		weightedCountMap = new HashMap<String, Double>();
	}
	
	public void add(BidElementDetails bed) {
		this.bidElements.add(bed);
		if(this.countMap.containsKey(bed.getBidElement().toString())) {
			int value = this.countMap.get(bed.getBidElement().toString()) + 1;
			this.countMap.put(bed.getBidElement().toString(), value);
		} else {
			this.countMap.put(bed.getBidElement().toString(), 1);
		}
		
		if(this.weightedCountMap.containsKey(bed.getBidElement().toString())) {
			double value = this.weightedCountMap.get(bed.getBidElement().toString()) + bed.getTime()*bed.getTime();
			this.weightedCountMap.put(bed.getBidElement().toString(), value);
		} else {
			this.weightedCountMap.put(bed.getBidElement().toString(), bed.getTime()*bed.getTime());
		}
	}
	
	public List<BidElementDetails> getHistory() {
		return this.bidElements;
	}
	
	public BidElementHistory filterBetweenTime(double t1, double t2) {
		BidElementHistory beh = new BidElementHistory();
		for(BidElementDetails bed:this.bidElements) {
			if(t1 < bed.getTime() && bed.getTime() <= t2) beh.add(bed);
		}
		return beh;
	}
	
	public int getAppearanceCount(BidElement bidElement) {
		if(this.countMap.containsKey(bidElement.toString())) {
			return this.countMap.get(bidElement.toString());
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
	
	public double getWeightedAppearanceCount(BidElement bidElement) {
		if(this.weightedCountMap.containsKey(bidElement.toString())) {
			return this.weightedCountMap.get(bidElement.toString());
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
		return bidElements.size();
	}
}