package agents.anac.y2015.group2;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

class G2Issue {
	private Map<String, Double> options;
	private double weight;
	
	public G2Issue() {
		options = new HashMap<String, Double>();
		weight = 0;
	}
	
	public void putValue(String name, double option) {
		options.put(name, option);
	}
	
	public double getValue(String name) {
		return options.get(name);
	}
	
	public double getWeight() {
		return weight;
	}
	
	public void setWeight(double weight) {
		this.weight = weight;
	}
	
	public double getScoreForOption(String name) {
		/*if(options.get(name) == null){
			System.out.println(name);
			System.out.println(toString());
		}*/
		return weight * options.get(name);
	}
	public String toString() {
		return options.keySet().toString() + options.values().toString();
	}
	
	public String weightString() {
		Set<Entry<String, Double>> entrySet = options.entrySet();
		String weightString = "" + weight + ": [";
		for(Entry<String, Double> entry : entrySet) {
			weightString += entry.getKey() + ":" + entry.getValue() + ", ";
		}
		
		if(options.size() > 0)
			return weightString.substring(0, weightString.length()-2) + "]";
		else
			return weightString + "]";
	}
	
	public void SetNeutralPreference() {
		for(String key : options.keySet()) {
			options.put(key, 1.0);
		}
	}
	public void SetZeroPreference() {
		for(String key : options.keySet()) {
			options.put(key, 0.0);
		}
	}
	
	public void normalizeOptions(){
		double maxOption = 0.0;
		for(double option : options.values()){
			if(option > maxOption){
				maxOption = option;
			}
		}
		for(Entry<String, Double> entry : options.entrySet()){
			options.put(entry.getKey(), entry.getValue()/maxOption);
		}
	}
	
	public void increaseOption(String option, int nBids){
		putValue(option, options.get(option)+1.0/nBids);
		normalizeOptions();
	}
	
	public void updateWeight(int nBids, double newWeight){
		weight = (weight*nBids + newWeight) / (nBids+1);
	}
	
	public Set<String> getOtherOptions(String option){
		Set<String> otherOptions =  new HashSet <String>(options.keySet());
		otherOptions.remove(option);
		return otherOptions;
	}
	
	public Set<String> getOptionNames() {
		return options.keySet();
	}
}