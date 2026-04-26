package agents.anac.y2015.group2;

import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import agents.anac.y2015.group2.G2Issue;
import genius.core.issue.Objective;
import genius.core.issue.ValueDiscrete;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.Evaluator;
import genius.core.utility.EvaluatorDiscrete;

class G2UtilitySpace {
	Map<String, G2Issue> utilities;
	G2Bid _lastBid;
	
	Map<String, G2Issue> extractUtilitiesFromUtilitySpace(AdditiveUtilitySpace utilitySpace) {
		HashMap<String, G2Issue> util = new HashMap<String, G2Issue>();
		
		Set<Entry<Objective, Evaluator>> evaluatorSet = utilitySpace.getEvaluators();
		for(Entry<Objective, Evaluator> entry : evaluatorSet) {
			String name = ((Objective) entry.getKey()).getName();
			G2Issue newIssue = new G2Issue();
			newIssue.setWeight(utilitySpace.getWeight(entry.getKey().getNumber()));
			
			EvaluatorDiscrete evaluator = ((EvaluatorDiscrete)entry.getValue());
			Set<ValueDiscrete> valueSet = evaluator.getValues();
			for(ValueDiscrete value : valueSet) {
				try {
					newIssue.putValue(value.getValue(), evaluator.getEvaluation(value));
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
			
			util.put(name, newIssue);
		}
		return util;
	}
	
	public double calculateUtility(G2Bid bid) {
		double score = 0.0;
		
		for(Entry<String, String> entry : bid.getEntrySet()) {
			score += utilities.get(entry.getKey()).getScoreForOption(entry.getValue());
		}
		
		return score;
	}
	public double calculateOptionUtility(String issue, String option) {
		return utilities.get(issue).getScoreForOption(option);
	}
	
	public void ResetWeights() {
		int size = utilities.size();
		for(G2Issue i: utilities.values()) {
			i.setWeight(1.0/size);
		}
	}
	public void SetZeroPreferences() {
		for(G2Issue i: utilities.values()) {
			i.SetZeroPreference();
		}
	}
	public void SetNeutralPreferences() {
		for(G2Issue i: utilities.values()) {
			i.SetNeutralPreference();
		}
	}
	
	G2UtilitySpace(AdditiveUtilitySpace utilitySpace)
	{
		utilities = extractUtilitiesFromUtilitySpace(utilitySpace);
	}
	
	public void resetAll(){
		ResetWeights();
		SetNeutralPreferences();
	}
	
	public String allDataString() {
		Set<Entry<String, G2Issue>> entrySet = utilities.entrySet();
		String dataString = "";
		for(Entry<String, G2Issue> entry : entrySet) {
			dataString += entry.getKey() + " weight:" + entry.getValue().weightString() + "\r\n";
		}
		return dataString;
	}
	
	public void updateIssues(G2Bid bid, int nBids){
		
		int hasChanged = 0;
		
		//@todo: update for initial bid
		if(_lastBid != null){

			//update the value of the options based on the bid
			for(Entry<String, String> entry:bid.getEntrySet() ){
				utilities.get(entry.getKey()).increaseOption(entry.getValue(), nBids);
				//keep track on how many issues the bid has changed since the last time
				if(entry.getValue() != _lastBid.getChoice(entry.getKey())){
					hasChanged++;
				}
			}
			
			if(hasChanged > 0){
				//update the weight of the issue
				for(Entry<String, String> entry:bid.getEntrySet() ){
					if(entry.getValue() != _lastBid.getChoice(entry.getKey())){
						utilities.get(entry.getKey()).updateWeight(nBids, 1.0/hasChanged);
					}else{
						utilities.get(entry.getKey()).updateWeight(nBids, 0);
					}
				}
			}
		}
		_lastBid = bid;
	
	}
	
	public G2Issue getIssue(String issue){
		return utilities.get(issue);
	}
	
	public Set<Entry<String, G2Issue>> getIssues(){
		return utilities.entrySet();
	}
}