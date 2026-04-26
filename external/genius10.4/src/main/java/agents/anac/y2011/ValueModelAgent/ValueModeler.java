package agents.anac.y2011.ValueModelAgent;

import genius.core.Bid;
import genius.core.issue.Value;
import genius.core.utility.AdditiveUtilitySpace;

public class ValueModeler {
	public boolean initialized=false;
	AdditiveUtilitySpace utilitySpace;
	IssuesDecreases[] issues;
	//initializing the opponenent model with the opponentFirstBid,
	//which is assumed to be the best bid possible
	public void initialize(AdditiveUtilitySpace space,Bid firstBid) throws Exception{
		initialized=true;
		utilitySpace  = space;
		int issueCount = utilitySpace.getDomain().getIssues().size();
		issues = new IssuesDecreases[issueCount];
		for(int i =0; i<issueCount;i++) {
			Value value = firstBid.getValue(utilitySpace.getIssue(i).getNumber());
			issues[i] = new IssuesDecreases(utilitySpace);
			issues[i].initilize(utilitySpace.getDomain().getIssues().get(i), value, issueCount);
		}
	}
	private void normalize(){
		
		for(int i=0;i<issues.length;i++){
			//issues[i].normalize(issues[i].weight()/sumWeight);
			//OK so it dosn't really sum up to 1...
			issues[i].normalize(issues[i].weight());
		}
		
		
	}
	//this function gets a reliability measurement of a value and determines
	//how much units of deviation should this reliability level move
	//for each movement of 100% reliability
	private double reliabilityToDevUnits(double reliability){
		if(reliability>1/2){
			return (1/(reliability*reliability))/issues.length;
		}
		if(reliability>1/4){
			return (2/reliability)/issues.length;
		}
		if(reliability>0){
			return (4/Math.sqrt(reliability))/issues.length;
		}
		//this case shouldn't be reached
		return 1000;
	}
	//the bid utility for the player is assumed to be 1-expectedDecrease
	public void assumeBidWorth(Bid bid,double expectedDecrease,double stdDev) throws Exception{
		ValueDecrease[] values = new ValueDecrease[issues.length];
		double maxReliableDecrease=0;
		for(int i=0;i<issues.length;i++){
			Value value = bid.getValue(utilitySpace.getIssue(i).getNumber());
			values[i] = issues[i].getExpectedDecrease(value);
		}
		//double deviationUnit=stdDev;
		double deviationUnit=0;
		for(int i=0;i<issues.length;i++){
			deviationUnit +=
				reliabilityToDevUnits(values[i].getReliabilty())
				*values[i].getDeviance();
			if(maxReliableDecrease<values[i].getDecrease() && values[i].getReliabilty()>0.8){
				maxReliableDecrease=values[i].getDecrease();
			}
		}
		ValueDecrease origEvaluation=utilityLoss(bid);
		double unitsToMove = (expectedDecrease-origEvaluation.getDecrease())/deviationUnit;
		for(int i=0;i<issues.length;i++){
			double newVal = values[i].getDecrease()+
				reliabilityToDevUnits(values[i].getReliabilty())
				*values[i].getDeviance()
				*unitsToMove;
			if(values[i].getMaxReliabilty()>0.7 || maxReliableDecrease<newVal){
				if(newVal>0){
					values[i].updateWithNewValue(newVal, origEvaluation.getReliabilty());
				}
				else values[i].updateWithNewValue(0, origEvaluation.getReliabilty());
			}
			//assumes that new unreliable values costs more than previously seen values.
			//if our opponent selected a bid that costs 10%,
			//that is split between values that costs 4%,6%.
			//than if 4%->6% we will think that 6%->4%.
			//worst this sway also influences the estimate
			//of our opponent's concession, so we may think he
			//Consented to 7% and 6%->1%. 
			//both issues require this failsafe...
			//else values[i].updateWithNewValue(maxReliableDecrease, origEvaluation.getReliabilty());
			else values[i].updateWithNewValue(newVal, origEvaluation.getReliabilty());
			
		}
		normalize();
		
	}
	public ValueDecrease utilityLoss(Bid bid) throws Exception{
		ValueDecrease[] values = new ValueDecrease[issues.length];
		for(int i=0;i<issues.length;i++){
			Value value = bid.getValue(utilitySpace.getIssue(i).getNumber());
			values[i] = issues[i].getExpectedDecrease(value);
		}
		double stdDev = 0;
		for(int i=0;i<issues.length;i++){
			stdDev += values[i].getDeviance();
		}
		double decrease=0;
		for(int i=0;i<issues.length;i++){
			decrease += values[i].getDecrease();
		}
		//the sum square of 1/reliability
		double sumSQ=0;
		for(int i=0;i<issues.length;i++){
			double rel = values[i].getReliabilty();
			rel = rel>0?rel:0.01;
			sumSQ += (1/rel)*(1/rel);
		}
		//added postBG 
		sumSQ/=issues.length;
		double rel = Math.sqrt(1/sumSQ);
		return new ValueDecrease(decrease,rel,stdDev);
	}
	
	public IssuesDecreases getIssue(int index){
		if(index<issues.length && index>=0){
			return issues[index];
		}
		return issues[0];
		
	}
	
}
