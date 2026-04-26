package agents.anac.y2015.group2;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;

class G2IssueSubSet {
	ArrayList<String> issues;
	LinkedList<G2SubBid> subBids;
	
	G2IssueSubSet (String issue, LinkedList<G2SubBid> subBids) {
		issues = new ArrayList<String>(1);
		issues.add(issue);
		
		this.subBids = subBids;
	}
	
	G2IssueSubSet(G2IssueSubSet one, G2IssueSubSet two) {
		issues = new ArrayList<String>(one.issues.size() + two.issues.size());
		issues.addAll(one.issues);
		issues.addAll(two.issues);
		
		subBids = new LinkedList<G2SubBid>();
		for(G2SubBid bidOne : one.subBids) {
			for(G2SubBid bidTwo : two.subBids) {
				subBids.add(new G2SubBid(bidOne, bidTwo));
			}
		}
	}
	
	ArrayList<G2Bid> generateBids() {
		ArrayList<G2Bid> bids = new ArrayList<G2Bid>();
		for(G2SubBid subBid: subBids) {
			HashMap<String, String> map = new HashMap<String, String>();
			for(int i=issues.size()-1;i>=0; i--) {
				map.put(issues.get(i), subBid.getValue(i));
			}
			bids.add(new G2Bid(map));
		}
		return bids;
	}
	
	void trimSubBids() {
		Collections.sort(subBids);
		Iterator<G2SubBid> iter = subBids.descendingIterator();
		G2SubBid previous = null;
		
		while(iter.hasNext()) {
			G2SubBid curr = iter.next();
			if (!curr.hasHigherOtherUtilities(previous)) {
				//System.out.println("Bid removed");
				iter.remove();
			}
			previous = curr;
		}
	}
}