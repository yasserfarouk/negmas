package agents.anac.y2018.condagent;

import java.util.List;

import genius.core.AgentID; //Unique ID for an agent
import genius.core.Bid; //modified copies using putValue(int, Value)
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.timeline.Timeline; //A time line, running from t = 0 (start) to t = 1 (deadline).
import genius.core.utility.AdditiveUtilitySpace; // couples all objectives to weights and evaluators


public class ConDAgent extends AbstractNegotiationParty {
	
	BayesianOpponentModel OMone;
	BayesianOpponentModel OMtwo;
	IssueManager IM;
	private Bid lastReceivedBid = null;
	AgentID agentOne=null;
	AgentID agentTwo=null;

	
	public void init(NegotiationInfo info) {
		
		//arxikopoihsh olwsn twn metavlitwn mou
		super.init(info);
		
		try{
        
			//kanei intiallize to opponent model me vasi to utilspace kai to timeline
	        this.OMone = new BayesianOpponentModel((AdditiveUtilitySpace) this.utilitySpace);
	        this.OMtwo = new BayesianOpponentModel((AdditiveUtilitySpace) this.utilitySpace);
	        
        //kanei intiallize to issue manager me vasi to utilspace to timeline kai to opponent model
        this.IM = new IssueManager((AdditiveUtilitySpace)this.utilitySpace, (Timeline) this.timeline);

		}catch(Exception e){
			e.printStackTrace();
		}
	}
	
	//epeksergaia twn bids olwn twn paiktwn
	
		public void receiveMessage(AgentID sender, Action action) {
			
			super.receiveMessage(sender, action);
			
			if (lastReceivedBid == null){ 
				agentOne = sender;
				IM.agentA = agentOne;
				}
			
			if (sender!= agentOne){ 
				
				sender = agentTwo;
				IM.agentB = agentTwo;
				}
			
			if (action instanceof Offer){ //exw neo offer
				
				//ta offer pou ginodai
					lastReceivedBid = ((Offer) action).getBid();  //gia kathe bid
					
					
					try {
						IM.ProcessBid(sender, this.lastReceivedBid);
					} catch (Exception var2_2) {}
					
	               //kanw update to poso simadiko einai gia ton adipalo
					try {
						if(sender==agentOne){
							 OMone.updateBeliefs(this.lastReceivedBid);
							 IM.LastAgentOneBid = lastReceivedBid;
						}else{
							OMtwo.updateBeliefs(this.lastReceivedBid);
							IM.LastAgentTwoBid = lastReceivedBid;
						}
						
					} catch (Exception var2_2) {}
				}
			
		}
		
		// dialegw to action mou -> accept h offer. an paizw prwtos mono offer
		
		public Action chooseAction(List<Class<? extends Action>> validActions){

			Bid myBid;
			//an paizw prwtos
			if (lastReceivedBid == null){
			
				return new Offer(this.getPartyId(), IM.GetMaxBid());
				
			}else{ //an den paizw prwtos
				
				try{
				//dexetai an to utility einai panw apo 0,95
		        if (this.utilitySpace.getUtility(this.lastReceivedBid) > 0.95) {
		        	return new Accept(getPartyId(), lastReceivedBid);
		        }
		        
		      //pairnei to threshold 
		      double threshold = this.IM.CreateThreshold();
		      
		      //dexetai an to utillity pou tou proteinete einai megalutero apto threshold tou
		      if (this.lastReceivedBid != null && this.utilitySpace.getUtility(this.lastReceivedBid) >= threshold) {
		    	  return new Accept(this.getPartyId(), lastReceivedBid);
		      }
				}catch(Exception e){
					e.printStackTrace();
				}
		      
				double mythreshold = 1.0;
				try {
					mythreshold = this.IM.CreateThreshold();
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				IM.numOfMyCommitedOffers++;
				myBid = this.IM.GenerateBidWithAtleastUtilityOf(mythreshold);
		    
		      
		      return new Offer(this.getPartyId(), myBid); 
			}
		}


		
	    @Override
	    public String getDescription() {
	        return "ANAC2018";
	    }

	}
