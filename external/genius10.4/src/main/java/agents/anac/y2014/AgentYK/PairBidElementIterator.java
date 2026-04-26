package agents.anac.y2014.AgentYK;

import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;

import genius.core.Bid;
import genius.core.issue.Issue;

public class PairBidElementIterator implements java.util.Iterator<PairBidElement>{
	private Bid bid;
	private List<Integer> issueNrs;
	private int index1,index2;
	
	public PairBidElementIterator(Bid bid) {
		this.bid = new Bid(bid);
		this.issueNrs = new ArrayList<Integer>();
		for(Issue issue:this.bid.getIssues()) {
			this.issueNrs.add(issue.getNumber());
		}
		this.index1 = 0;
		this.index2 = 1;
	}
	
	@Override
	public boolean hasNext() {
		if(index1 <= (this.issueNrs.size() - 2) && index2 <= (this.issueNrs.size() - 1)) {
			return true;
		} else {
			return false;
		}
	}

	@Override
	public PairBidElement next() throws NoSuchElementException{
		if(this.hasNext()) {
			int fstIssueNr = this.issueNrs.get(this.index1);
			int sndIssueNr = this.issueNrs.get(this.index2);
			try {
				BidElement fst = new BidElement(fstIssueNr, this.bid.getValue(fstIssueNr));
				BidElement snd = new BidElement(sndIssueNr, this.bid.getValue(sndIssueNr));
				this.index2++;
				if(this.index2 >= this.issueNrs.size()) {
					this.index1++;
					this.index2 = this.index1 + 1;
				}
				return new PairBidElement(fst, snd);
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
