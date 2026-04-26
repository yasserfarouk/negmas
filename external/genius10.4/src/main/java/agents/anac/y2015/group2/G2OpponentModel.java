package agents.anac.y2015.group2;

import genius.core.utility.AdditiveUtilitySpace;

class G2OpponentModel {
	G2UtilitySpace utilitySpace;
	private int _nBids = 0;
	//private Domain _domain;
	
	G2OpponentModel(AdditiveUtilitySpace domain) {
		utilitySpace = new G2UtilitySpace(domain);
		utilitySpace.resetAll();
		//_domain = domain.getDomain();
	}
	
	public void updateModel(G2Bid bid){
		utilitySpace.updateIssues(bid, _nBids);
		_nBids++;
	}
	
	public double getUtility(G2Bid bid){
		return utilitySpace.calculateUtility(bid);
	}
	
	public String getUtilitySpaceString() {
		return utilitySpace.allDataString();
	}
	
	public G2UtilitySpace getUtilitySpace() {
		return utilitySpace;
	}
}