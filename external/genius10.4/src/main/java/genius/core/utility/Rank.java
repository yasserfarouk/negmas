package genius.core.utility;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Set;


public class Rank {
	
	private int indexofIssue;
	private HashMap<String, Integer> ranks; //value --> its rank
	private int maximumRank;
	
	
	public Rank(int indexOfIssue){
		this.setIndexofIssue(indexOfIssue);
		this.ranks=new HashMap<String,Integer>();
	}

	public Rank(int indexOfIssue, HashMap<String,Integer> ranks){
		this.setIndexofIssue(indexOfIssue);
		this.ranks=ranks;
		this.maximumRank=getMaxRank();
	}
	
	public void addRank(String value, Integer rank){
		this.ranks.put(value,rank);
		if (rank>this.maximumRank)
			this.maximumRank=rank;
	}

	public void addRank(String value, String rank){
		addRank(value,Integer.valueOf(rank));
	}
	
	private Integer getMaxRank(){
		
		Integer maximum = 0;
		Set<Integer> rankSet=(Set<Integer>) this.ranks.values();
		
		Iterator<Integer> ratings=rankSet.iterator();
		
		while (ratings.hasNext()){
			int currentValue= ratings.next();
			if (currentValue>maximum)
				maximum=currentValue;
		}
			
		return maximum;
		
	}

	public int getMaximumRank() {
		return maximumRank;
	}

	public void setMaximumRank(int maximumRank) {
		this.maximumRank = maximumRank;
	}
	
	public int getRank(String valueIndex){
		return ranks.get(valueIndex);
	}
	
	public double getNormalizedRank(String value){
		return (double) ((double) ranks.get(value))/maximumRank;
		
	}

	
	public int getIndexofIssue() {
		return indexofIssue;
	}

	public void setIndexofIssue(int indexofIssue) {
		this.indexofIssue = indexofIssue;
	}
}
