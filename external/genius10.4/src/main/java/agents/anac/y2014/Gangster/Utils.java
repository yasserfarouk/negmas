package agents.anac.y2014.Gangster;


import genius.core.Bid;
import genius.core.issue.ValueInteger;

class Utils {
	
	

	public static int calculateManhattanDistance(Bid bid1, Bid bid2) throws Exception{
		
		int numIssues = bid1.getValues().size();
		
		int x = 0;
		for(int i=1; i<=numIssues; i++){
			x += Math.abs(((ValueInteger)bid1.getValue(i)).getValue() - ((ValueInteger)bid2.getValue(i)).getValue());
		}

		return x;
		
	}
	
}
