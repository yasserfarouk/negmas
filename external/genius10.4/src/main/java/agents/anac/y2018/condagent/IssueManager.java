package agents.anac.y2018.condagent;

import java.util.ArrayList;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.issue.Issue;
import genius.core.timeline.Timeline;
import genius.core.utility.AdditiveUtilitySpace;



public class IssueManager {
	
	AdditiveUtilitySpace US;
	Bid maxBid = null;
	Timeline T;
	double BestOpponBid = 0.0;
	Bid BestEverOpponentBid = null;
	TreeMap<Double, Bid> Bids;
    boolean FirstOfferGiven = false;
    
    BayesianOpponentModel OMone;
	BayesianOpponentModel OMtwo;
   
   //newwwwwww
    Bid LastAgentOneBid;
    Bid LastAgentTwoBid;
    AgentID agentA=null;
	AgentID agentB=null;
	int deltaA = 0;
	int deltaB = 0;
	int roundCount=0;
	public int numOfIssues;
	double NSA = 0.0;
	double NSB = 0.0;
	double NS = 0.0;
	int numOfMyCommitedOffers = 1;
	double selfFactor = 0.0;
	double Eagerness = 0.5;
	double enviromFactor = 0.11;
	double concessionRate = 0.0;
	double myWeight=0.0;
	double maxBidUtil =0.0;
	

    //kainouria
    public void findNegotiationStatus(AgentID opponent, Bid IncomingBid){
    	
    	double incomingUtil = US.getUtility(IncomingBid);
    	
    	if (opponent== agentA && opponent!=null){ //gia ton 1 agent
    		
    		if(incomingUtil< US.getUtility(LastAgentOneBid)){
    			deltaA = deltaA + 1;
    		}else{
    			deltaA = deltaA + 3;
    		}
    		
    		NSA = (deltaA + 2*roundCount)/ 3*roundCount;
        		
        }else if(opponent== agentB && opponent!=null){ //gia ton 2 agent
        	
        	if(incomingUtil< US.getUtility(LastAgentTwoBid)){
    			deltaB = deltaB + 1;
    		}else{
    			deltaB = deltaB + 3;
    		}
        	
        	NSB = (deltaB + 2*roundCount)/ 3*roundCount;
        }
    	
    	NS = (NSA + NSB) / 2;
    }
    
   
	 //an den exw ousiastika discount factor einai san na to pernw dedomeno =1
	 public double GetDiscountFactor() {
        if (this.US.getDiscountFactor() <= 0.001 || this.US.getDiscountFactor() > 1.0) {
            return 1.0;
        }
        return this.US.getDiscountFactor();
    }


	
	 public void ProcessBid(AgentID opponent, Bid IncomingBid) throws Exception{ //elengw an auto einai to kalutero bid p m exei kanei o adipalos
	            
		 
		 roundCount++;
		 
		 if(!Bids.isEmpty()){
			 if(!Bids.containsValue(IncomingBid)){ //an ayto to bid den to eixa upopsin to vazw k auto sti lsita mou
				 Bids.put(US.getUtility(IncomingBid),IncomingBid);
			 }
		 }
	            
		 findNegotiationStatus(opponent, IncomingBid);
	     
	     //SET MY SELF FACTOR
	     double DF = this.GetDiscountFactor();
	     double Time = this.T.getTime();
	     	
	     	if(DF!=1){
	     		Eagerness = 1 - DF;
	     	}
	     	
	     	selfFactor = 0.25*((1/numOfMyCommitedOffers)+ NS + Time + Eagerness);
	     	
	     	//SET CONCESSION RATE
	     	
	     	if(Time <= 0.2){ //start
	     		
	     		concessionRate = 0.0;
	     		
	     	}else if(Time >= 0.9){ //end
	     		
	     		concessionRate = 0.99;
	     		
	     	}else{ //otherwise
	     		
	     		double oppOneWeight = 0;
	     		double oppTwoWeight = 0;

		    	for(int i = 0; i < numOfIssues; i++){                 //ousiastika kanw update twn duo greedy factor se kathe guro
		    		
		        	double tempone = OMone.getExpectedWeight(i);
		        	double temptwo = OMtwo.getExpectedWeight(i);
		        	
		        	oppOneWeight = (oppOneWeight + tempone)/2;
		     		oppTwoWeight = (oppTwoWeight + temptwo)/2;
		    	}
		    	
		    	double overallEnvWeight = (oppOneWeight + oppTwoWeight)/2; //vriskw ton meso oro twn weight tou adipalou
	     		
	     		concessionRate = myWeight * selfFactor + overallEnvWeight*enviromFactor;
	     	}
	 }

	 public void findAllmyBidsPossible() throws Exception{
			
			Random random = new Random();
			int numOfPossibleBids = (int) US.getDomain().getNumberOfPossibleBids();
			
			for(int i =0; i < numOfPossibleBids ; i++){ //prospathw na vrw ola ta pithana bids--> kanw mia arxiki lista
				Bid randomBid = US.getDomain().getRandomBid(random);
				if((!Bids.containsKey(US.getUtility(randomBid))) || (!Bids.containsValue(randomBid))){
					Bids.put(US.getUtility(randomBid), randomBid);
				}
			}
			
		 }
	 

	//kathorismos tou threshold mou me vasi to offer pou tha ekana edw!!!
	 public double CreateThreshold() throws Exception {
		
		 double reservationValue = US.getReservationValue();
		 if(reservationValue <= 0.5){
			 reservationValue = 0.5; //set my own reservation value
		 }
		 
		 
		double offer = maxBidUtil + (reservationValue - maxBidUtil)* concessionRate;
		
		return offer;
		
	    }

	
	 //epistrefei to bid me to amesws mikrotero utillity
	 public Bid GenerateBidWithAtleastUtilityOf(double MinUtility) {
		 Map.Entry<Double, Bid> e = this.Bids.ceilingEntry(MinUtility);
		
		 if (e == null) {
			 return this.maxBid;
		 }
		 
		 return e.getValue();
	 }
	
	 public Bid GetMaxBid(){ 
		 return maxBid;
	 }
	 
	 public IssueManager(AdditiveUtilitySpace US, Timeline T) throws Exception {
			this.T = T;
		        this.US = US;
		       
		        try {
		            double maxBidUtil = US.getUtility(this.maxBid); //pernei to utility tou max bid
		            if (maxBidUtil == 0.0) {
		                this.maxBid = this.US.getMaxUtilityBid();
		            }
		        }
		        catch (Exception e) {
		            try {
		                this.maxBid = this.US.getMaxUtilityBid();
		            }
		            catch (Exception var5_7) {
		                // empty catch block
		            }
		        }
		        this.Bids = new TreeMap<Double, Bid>();
		        
		        //newwwwwwwwwwwwwww
		        maxBidUtil = US.getUtility(maxBid);
		        LastAgentOneBid = US.getMinUtilityBid();
		        LastAgentTwoBid = US.getMinUtilityBid();
		        findAllmyBidsPossible();
		        java.util.List<Issue> AllIssues;
		        AllIssues = new ArrayList<Issue>();
		        AllIssues = US.getDomain().getIssues();
				numOfIssues = AllIssues.size();
				
				for(int i =0; i< numOfIssues ; i++){
					
					myWeight = (myWeight + US.getWeight(i))/2;
					
				}
		        
		   
		   
	}
}
