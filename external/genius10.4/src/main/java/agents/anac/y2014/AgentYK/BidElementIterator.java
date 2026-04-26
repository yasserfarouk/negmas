package agents.anac.y2014.AgentYK;

import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;

import genius.core.Bid;
import genius.core.issue.Issue;

public class BidElementIterator implements java.util.Iterator<BidElement>{
	private Bid bid;
	private List<Integer> issueNrs;
	private int index;
	
	public BidElementIterator(Bid bid) {
		this.bid = new Bid(bid);
		this.issueNrs = new ArrayList<Integer>();
		for(Issue issue:this.bid.getIssues()) {
			this.issueNrs.add(issue.getNumber());
		}
		this.index = 0;
	}
	
	@Override
	public boolean hasNext() {
		if(index <= (this.issueNrs.size() - 1)) {
			return true;
		} else {
			return false;
		}
	}

	@Override
	public BidElement next() throws NoSuchElementException{
		if(this.hasNext()) {
			int IssueNr = this.issueNrs.get(this.index++);
			try {
				return new BidElement(IssueNr, this.bid.getValue(IssueNr));
			} catch(Exception e){}
		} else {
			throw new NoSuchElementException();
		}
		return null;
	}

	@Override
	public void remove() throws UnsupportedOperationException {
		throw new UnsupportedOperationException();
	}
	
}
