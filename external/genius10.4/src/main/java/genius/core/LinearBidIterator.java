package genius.core;

import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;

import genius.core.issue.*;
import genius.core.utility.AdditiveUtilitySpace;

public class LinearBidIterator extends BidIterator {

	private List<Issue> sortedIssues;
	public LinearBidIterator(Domain domain, final AdditiveUtilitySpace space, double maxUtil, double minUtil) {
		super(domain);
		sortedIssues = new LinkedList<Issue>();
		//sort issues wrt their weights
		sortedIssues.addAll(space.getDomain().getIssues());
		Collections.sort(sortedIssues, new Comparator<Issue>(){
			public int compare(Issue o1, Issue o2) {
				if(space.getWeight(o1.getNumber())>space.getWeight(o2.getNumber()))
					return 0;
				else return 1;
			} });
	}
	
	@Override
	public boolean hasNext() {
		return super.hasNext();
	}
	@Override
	public Bid next() {
		return super.next();
	}
	

}
