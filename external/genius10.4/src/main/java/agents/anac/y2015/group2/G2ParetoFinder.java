package agents.anac.y2015.group2;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Map.Entry;
import java.util.Set;

class G2ParetoFinder
{
	ArrayList<G2IssueSubSet> issueSubSets;
	G2ParetoFinder(G2UtilitySpace ourUtilitySpace, ArrayList<G2UtilitySpace> OtherUtilitySpaces)
	{
		issueSubSets = new ArrayList<G2IssueSubSet>();
		Set<Entry<String, G2Issue>> issues = ourUtilitySpace.getIssues();
		for(Entry<String, G2Issue> issueInfo: issues)
		{
			String IssueName = issueInfo.getKey();
			G2Issue issue = issueInfo.getValue();
			LinkedList<G2SubBid> subBids = new LinkedList<G2SubBid>();
			
			Set<String> optionNames = issue.getOptionNames();
			for(String optionName: optionNames)
			{
				ArrayList<Double> OtherUtilities = new ArrayList<Double>(OtherUtilitySpaces.size());
				for(G2UtilitySpace utilitySpace: OtherUtilitySpaces)
				{
					OtherUtilities.add(utilitySpace.calculateOptionUtility(IssueName, optionName));
				}
				double ourUtility = ourUtilitySpace.calculateOptionUtility(IssueName, optionName);
				subBids.add(new G2SubBid(optionName, ourUtility, OtherUtilities));
			}
			issueSubSets.add(new G2IssueSubSet(IssueName, subBids));
		}
	}
	
	private void trimMerge() {
		for(G2IssueSubSet subSet: issueSubSets)
		{
			subSet.trimSubBids();
		}
		
		ArrayList<G2IssueSubSet> newIssueSubSets = new ArrayList<G2IssueSubSet>();
		int i=1;
		while(i<issueSubSets.size())
		{
			newIssueSubSets.add(new G2IssueSubSet(issueSubSets.get(i-1), issueSubSets.get(i)));
			i+=2;
		}
		if(i==issueSubSets.size())
		{
			newIssueSubSets.add(issueSubSets.get(i-1));
		}
		issueSubSets = newIssueSubSets;
	}
	
	ArrayList<G2Bid> findParetoOptimalBids() {
		while (issueSubSets.size()>1) {
			trimMerge();
		}
		issueSubSets.get(0).trimSubBids();
		
		return issueSubSets.get(0).generateBids();
	}
}