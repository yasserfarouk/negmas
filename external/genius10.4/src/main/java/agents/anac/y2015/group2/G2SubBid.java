package agents.anac.y2015.group2;

import java.util.ArrayList;

class G2SubBid implements Comparable<G2SubBid>{
	ArrayList<String> values;
	double ourUtility;
	ArrayList<Double> otherUtilities;
	
	public G2SubBid(String value, double ourUtility, ArrayList<Double> otherUtilities) {
		values = new ArrayList<String>(1);
		values.add(value);
		this.ourUtility = ourUtility;
		this.otherUtilities = otherUtilities;
	}
	
	public G2SubBid(G2SubBid one, G2SubBid two) {
		values = new ArrayList<String>(one.values.size() + two.values.size());
		values.addAll(one.values);
		values.addAll(two.values);
		
		ourUtility = one.ourUtility + two.ourUtility;
		
		otherUtilities = new ArrayList<Double>(one.otherUtilities.size());
		for(int i=0; i<one.otherUtilities.size(); i++) {
			otherUtilities.add(one.otherUtilities.get(i) + two.otherUtilities.get(i));
		}
	}
	
	public String getValue(int i) {
		return values.get(i);
	}
	
	@Override
	public int compareTo(G2SubBid other) {
		if(ourUtility > other.ourUtility)
			return 1;
		else if(ourUtility == other.ourUtility)
			return 0;
		else
			return -1;
	}
	
	public boolean hasHigherOtherUtilities(G2SubBid other) {
		if(other == null)
			return true;
					
		for(int i=otherUtilities.size()-1; i>=0; i--) {
			if(otherUtilities.get(i) > other.otherUtilities.get(i) + 0.00001)
				return true;
		}
		return false;
	}
}