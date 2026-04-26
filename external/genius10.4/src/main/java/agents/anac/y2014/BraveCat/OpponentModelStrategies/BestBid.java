 package agents.anac.y2014.BraveCat.OpponentModelStrategies;
 
 import agents.anac.y2014.BraveCat.OpponentModels.OpponentModel;
 import java.util.HashMap;
 import java.util.List;
 import java.util.Random;
 import agents.anac.y2014.BraveCat.necessaryClasses.NegotiationSession;
import genius.core.bidding.BidDetails;
 
 public class BestBid extends OMStrategy
 {
   double updateThreshold = 1.1D;
 
   public BestBid()
   {
   }
 
   public BestBid(NegotiationSession negotiationSession, OpponentModel model)
   {
     try
     {
       super.init(negotiationSession, model);
     } catch (Exception e) {
     }
   }
 
   @Override
   public void init(NegotiationSession negotiationSession, OpponentModel model, HashMap<String, Double> parameters)
     throws Exception
   {
     super.init(negotiationSession, model);
     if (parameters.get("t") != null)
       this.updateThreshold = ((Double)parameters.get("t")).doubleValue();
     else
       System.out.println("OMStrategy assumed t = 1.1");
   }
 
   @Override
   public BidDetails getBid(List<BidDetails> allBids)
   {
     if (allBids.size() == 1) {
       return (BidDetails)allBids.get(0);
     }
     double bestUtil = -1.0D;
     BidDetails bestBid = (BidDetails)allBids.get(0);
 
     boolean allWereZero = true;
 
     for (BidDetails bid : allBids)
     {
         try
         {
             double evaluation = this.model.getBidEvaluation(bid);
             if (evaluation > 0.0001D)
             {
                 allWereZero = false;
             }
             if (evaluation > bestUtil)
             {
                 bestBid = bid;
                 bestUtil = evaluation;
             }
         } catch (Exception ex)
         {
             System.out.println("Exception occured while executing getBidEvaluation in Best Bid OMStrategy!");
         }
     }
 
     if (allWereZero) {
       Random r = new Random();
       return (BidDetails)allBids.get(r.nextInt(allBids.size()));
     }
     return bestBid;
   }
 
   @Override
   public boolean canUpdateOM()
   {
     return this.negotiationSession.getTime() < this.updateThreshold;
   }
 }