package agents.anac.y2011.ValueModelAgent;

import java.util.HashMap;
import java.util.Map;

import genius.core.issue.ISSUETYPE;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.IssueReal;
import genius.core.issue.Value;
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;
import genius.core.utility.AdditiveUtilitySpace;

//a class that translate the issue values of all issue types
//into our standart ValueDecrease.
public class IssuesDecreases{
	
	private Map<String,ValueDecrease> values;
	private ISSUETYPE type; 
	private Issue origin;
	//for use in real
	private boolean goingDown;
	
	IssuesDecreases(AdditiveUtilitySpace space){
		values = new HashMap<String,ValueDecrease>();
	}
	//at start assumes that all issues have the same value.
	//also assumes that all values except the "optimal"
	//have a decrease equal to the weight with a very high varience
	public void initilize(Issue issue,Value maximalValue,int issueCount){
		double initWeight = 1.0/issueCount;
		type = issue.getType();
		origin = issue;
		switch(type){
		case DISCRETE:
			IssueDiscrete issueD = (IssueDiscrete)issue;
			int s = issueD.getNumberOfValues();
			for(int i=0;i<s;i++){
				String key = issueD.getValue(i).toString();
				ValueDecrease val;
				if(key.equals(maximalValue.toString())){
					val = new ValueDecrease(0,0.9,0.01);
				}
				else{
					val = new ValueDecrease(initWeight,0.02,initWeight);
				}
				values.put(key, val);
			}
			break;
		case REAL:
			IssueReal issueR = (IssueReal)issue;
			double lower = issueR.getLowerBound();
			double higher = issueR.getUpperBound();
			double maxValue = ((ValueReal)maximalValue).getValue();
			ValueDecrease worst = new ValueDecrease(initWeight,0.1,initWeight);
			values.put("WORST", worst);
			//likely to be either lower or higher.
			//but even if its not, that the first bid is likely to be
			//closer to the optimal...
			goingDown = maxValue<(lower+higher)/2;
			break;
		case INTEGER:
			IssueInteger issueI = (IssueInteger)issue;
			lower = issueI.getLowerBound();
			higher = issueI.getUpperBound();
			maxValue = ((ValueInteger)maximalValue).getValue();
			worst = new ValueDecrease(initWeight,0.1,initWeight);
			values.put("WORST", worst);
			//likely to be either lower or higher.
			//but even if its not, that the first bid is likely to be
			//closer to the optimal...
			goingDown = maxValue<(lower+higher)/2;
			break;
			
		}
		
	}

	//change all costs so that the new maximal decrease equals the old, AND all
	//others change proportionately.
	//assumes that the normalization process accured because of a faulty
	//scale when evaluating other player's bids
	public void normalize(double newWeight){
		double originalMinVal = 1;
		for(ValueDecrease val: values.values()){
			if(originalMinVal>val.getDecrease()){
				originalMinVal=val.getDecrease();
			}
		}
		double originalMaxVal = weight();
		if(originalMaxVal == originalMinVal){
			for(ValueDecrease val: values.values()){
				val.forceChangeDecrease(newWeight);
			}
		}
		else{
			double changeRatio = newWeight/(originalMaxVal-originalMinVal);
			for(ValueDecrease val: values.values()){
				val.forceChangeDecrease(
						(val.getDecrease()-originalMinVal)*changeRatio);
			}
		}
		
	}
	//maximal decrease
	public double weight(){
		double maximalDecrease = 0;
		for(ValueDecrease val: values.values()){
			if(maximalDecrease<val.getDecrease()){
				maximalDecrease=val.getDecrease();
			}
		}
		return maximalDecrease;
	}
	public ValueDecrease getExpectedDecrease(Value value){
		switch(type){
		case DISCRETE:
			ValueDecrease val = values.get(value.toString());
			if(val!=null) return val;
			break;
		case REAL:
			IssueReal issueR = (IssueReal)origin;
			double lower = issueR.getLowerBound();
			double higher = issueR.getUpperBound();
			double curValue = ((ValueReal)value).getValue();
			double portionOfWorst = (curValue-lower)/(higher-lower);
			if(!goingDown){
				portionOfWorst = 1-portionOfWorst;
			}
			ValueDecrease worst = values.get("WORST");
			if(worst!=null)return new RealValuedecreaseProxy(worst,portionOfWorst);
		case INTEGER:
			IssueInteger issueI = (IssueInteger)origin;
			lower = issueI.getLowerBound();
			higher = issueI.getUpperBound();
			curValue = ((ValueInteger)value).getValue();
			portionOfWorst = (curValue-lower)/(higher-lower);
			if(!goingDown){
				portionOfWorst = 1-portionOfWorst;
			}
			worst = values.get("WORST");
			if(worst!=null)return new RealValuedecreaseProxy(worst,portionOfWorst);
		}
		//should choose something that will make the program recoverable

		return new ValueDecrease(1,0.01,1);
	}
}