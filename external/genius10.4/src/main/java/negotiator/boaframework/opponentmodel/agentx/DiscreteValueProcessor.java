package negotiator.boaframework.opponentmodel.agentx;

import java.util.ArrayList;
import java.util.List;

import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;

/**
 * Class for processing discrete values, ranking them based on past bids.
 * 
 * @author E. Jacobs
 *
 */
public class DiscreteValueProcessor {
	
	private ArrayList<ValueDiscrete> valueList = new ArrayList<ValueDiscrete> ();
	private ArrayList<Integer> bidList = new ArrayList<Integer> ();
	private ArrayList<Integer> rankList = new ArrayList<Integer> ();
	
	private int nVals;
	
	/**
	 * Creates a valueProcessor containing ranks of and number of bids on a list of values. Use getValues from the Issue class to get such a list.
	 * @param valList
	 */
	public DiscreteValueProcessor(List<?> valList){
		for (Object o : valList){
			if (o instanceof ValueDiscrete){
				valueList.add((ValueDiscrete) o);
			}
			else if (o instanceof Value){
				Value v = (Value) o;
				valueList.add((ValueDiscrete) v);
			}			
		}
		
		fillLists();
	}
	
	/**
	 * Fills both the bid- and the ranklists
	 */
	private void fillLists(){
		
		nVals = valueList.size();
		
		for (int i = 0;i<nVals;i++){
			bidList.add(0);
			rankList.add(i+1);
		}
		
	}
	
	/**
	 * Returns the index of a certain value in the valuelist.
	 * @param v The value
	 * @return The index of the value in the list, or -1 if it is not in the list.
	 */
	public int getValueIndex(ValueDiscrete v){
		return valueList.indexOf(v);
	}
	
	/**
	 * Gets the rank of a certain value. Higher is more important. Lowest is rank 1, highest is nValues.
	 * @param v The value
	 * @return The rank of the value, or -1 if the value is not in the list
	 */
	public int getValueRank(ValueDiscrete v){
		
		int index = getValueIndex(v);
		
		if (index != -1)
			return rankList.get(index);
		else
			return index;
	}
	
	/**
	 * Gets the normalized rank of a certain value. Higher is more important. Lowest is rank 1/ nValues, highest is rank 1.
	 * @param v The value
	 * @return The rank of the value, or -1 if the value is not in the list
	 */
	public double getNormalizedValueRank(ValueDiscrete v){
		
		double valueRank = (double) getValueRank(v);

		if (valueRank != -1)
			return valueRank / ((double) nVals);
		else
			return valueRank;
	}
	
	/**
	 * Returns the list index of a certain rank
	 * @param rank The required rank
	 * @return index of the rank throughout the different lists, or -1 if the rank does not exist.
	 */
	private int getRankIndex(int rank){
		
		if (rank > nVals) return -1; 
		
		return rankList.indexOf(rank);
		
	}
	
	/**
	 * Returns the highest ranked value within an issue
	 * @return The highest ranked discrete value
	 */
	public ValueDiscrete getHighestRankedValue(){
		return valueList.get(getRankIndex(nVals));
	}
	
	/**
	 * Gives the number of bids for a certain value
	 * @param v The value for which the bids are required
	 * @return The number of bids, or -1 if the value is not in the list.
	 */
	private int getValueBids(ValueDiscrete v){

		int index = getValueIndex(v);
		
		if (index !=  -1)
			return bidList.get(index);
		else
			return index;
	}
	
	/**
	 * Adds 1 to the bidList for a certain value. Use this if a bid has been made containing that value, allowing for a change in rank
	 * @param v The value v on which the bid was done
	 */
	public void addBidForValue(ValueDiscrete v){
		
		int index = getValueIndex(v);
		
		bidList.set(index, bidList.get(index) + 1);
		changeRankByBid(v);
	}
	
	/**
	 * Reranks a value based on its number of bids. Use immediately after changing the bids of a value.
	 * @param v The value for which the number of bids has changed
	 */
	private void changeRankByBid(ValueDiscrete v){
		
		//System.out.println("Changing ranks by bid...");
		//System.out.println("Maximum rank is:" + nVals);
		
		int newRank = getValueRank(v)+1;
		int bids = getValueBids(v);
		int rankChange = 0;
		int rankindex = 0;
		
		while (newRank <= nVals){
			
			 rankindex = getRankIndex(newRank);
			 if (bids > bidList.get(rankindex)){
				 newRank ++;
				 rankChange ++;
			 }
			 else
				 break;
		}
		
		while (rankChange > 0){
			increaseRank(v);
			rankChange --;
		}
	}

	/**
	 * Directly increases the rank of a value. Note that higher ranks are higher numbers, denoting more important values
	 * @param v The value for which the rank has to be changed
	 */
	public void increaseRank(ValueDiscrete v){
		
		int oldRank = getValueRank(v); 
		int newRank = oldRank + 1;
		
		if (oldRank == nVals) return;
		
		int oldIndex = getRankIndex(oldRank);
		int newIndex = getRankIndex(newRank);	
			
		rankList.set(oldIndex, newRank);
		rankList.set(newIndex, oldRank);
	}
	
	/**
	 * Directly decreases the rank of a value. Note that lower ranks are lower numbers, denoting less important values
	 * @param v The value for which the rank has to be changed
	 */
	public void decreaseRank(ValueDiscrete v){
		
		int oldRank = getValueRank(v); 
		int newRank = oldRank - 1;
		
		if (oldRank == 0) return;
		
		int oldIndex = getRankIndex(oldRank);
		int newIndex = getRankIndex(newRank);	
			
		rankList.set(oldIndex, newRank);
		rankList.set(newIndex, oldRank);
	}
	
	@Override
	public String toString(){	
		
		String str = "";
		
		for (int i = 0;i < nVals;i++){
			str += "Value: " + valueList.get(i).getValue() + ", bids: " + bidList.get(i) + ", rank: " + rankList.get(i) +"\n";
		}
		
		return str;
	}
}
