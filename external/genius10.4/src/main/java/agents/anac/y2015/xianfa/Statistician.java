package agents.anac.y2015.xianfa;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

import agents.anac.y2015.xianfa.XianFaAgent;
import genius.core.issue.Value;
import genius.core.parties.AbstractNegotiationParty;
 
public class Statistician {
	private XianFaAgent myAgent;
	private String record = "record";
	
	public Statistician(AbstractNegotiationParty myAgent) {
		this.myAgent = (XianFaAgent) myAgent;
	}
	
	public void log() {
		write(analyse());
	}
	
	public String analyse() {
		String c = "";
		String n = System.getProperty("line.separator");
		
		ArrayList<AIssue> issues = myAgent.issuesA;
		ArrayList<AIssue> issuesB = myAgent.issuesB;
		
		for (int i=0; i<issues.size(); i++) {
			c += String.valueOf(issues.get(i).issnr);
			c += n;
			
			for (Value val : issues.get(i).getValues().keySet()) {
				c += val.toString();
				c += "  -  ";
				c += String.valueOf(issues.get(i).getValues().get(val));
				c += " / ";
				c += String.valueOf(issuesB.get(i).getValues().get(val));
				c += n;
			}
		}
		
		c += n;
		c += "Number of offers accepted by opponent A: " + String.valueOf(myAgent.stat_goodBids);
		c += n;
		c += "Number of offers accepted by opponent B: " + String.valueOf(myAgent.stat_BAccepts);
		
		c += n;
		//c += "Failed attempts to find a good enough bid: " + String.valueOf(myAgent.stat_bidsNotInList);
		//c += n;
		//c += "Successful attempts at findding a good enough bid: " + String.valueOf(myAgent.stat_bidsInList);
		//c += n;
		
		c += n;
		c += "Average time per round: " + myAgent.avgRdTime;
		c += n;
		c += "Reservation value is: " + myAgent.resValue;
		c += n;
		c += "Number of times offered history best bid: " + String.valueOf(myAgent.stat_offerHistBest);
		c += n;
		if (myAgent.opponentABest != null) {
			c += "Opponent A's best bid is: " + myAgent.opponentABest.toString();
			c += n;
			try {
				c += "Utility of opponent A's best bid is: " + String.valueOf(myAgent.getUtilitySpace().getUtility(myAgent.opponentABest));
			} catch (Exception e) {
				e.printStackTrace();
			}
			c += n;
		}
		c += "My total unique offers: " + String.valueOf(myAgent.myUniqueOffers);
		c += n;
		c += "Number of unique offers made by A: " + String.valueOf(myAgent.bidSet.size());
		c += n;
		c += "Number of unique offers made by B: " + String.valueOf(myAgent.bidSetB.size());
		
		c += n;
		c += "this is round ";
		c += String.valueOf(myAgent.rounds);
		c += n;
		c += "Normalized time at which session has ended: ";
		c += myAgent.getTimeLine().getTime();
		
		return c;
	}
	
	public void write(String content) {
		try {
 
			//String content = "This is the content to write into file";
 
			File file = new File("C:\\Users\\Kevin\\Research Papers\\" +
			"record.txt");
 
			// if file doesnt exists, then create it
			if (!file.exists()) {
				file.createNewFile();
			}
 
			FileWriter fw = new FileWriter(file.getAbsoluteFile());
			BufferedWriter bw = new BufferedWriter(fw);
			bw.write(content);
			bw.close();
 
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
