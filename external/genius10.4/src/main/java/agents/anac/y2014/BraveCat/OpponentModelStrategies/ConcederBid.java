 package agents.anac.y2014.BraveCat.OpponentModelStrategies;

 import agents.anac.y2014.BraveCat.OpponentModels.OpponentModel;
 import java.util.ArrayList;
 import java.util.HashMap;
 import java.util.List;
 import agents.anac.y2014.BraveCat.necessaryClasses.NegotiationSession;
import genius.core.bidding.BidDetails;
 
 public class ConcederBid extends OMStrategy
 {
   double updateThreshold = 1.1D;
 
   public ConcederBid()
   {
   }
 
   public ConcederBid(NegotiationSession negotiationSession, OpponentModel model)
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
     //----------------------------------------------------------------
     List<BidDetails> BidsPredictedAsConceder = new ArrayList();
     List<BidDetails> BidsPredictedAsUnfortunate = new ArrayList();
     List<Double> BidsPredictedAsConcederUtility = new ArrayList();
     List<Double> BidsPredictedAsUnfortunateUtility = new ArrayList();
     for (BidDetails bid : allBids)
     {
       try {
           Double evaluation = this.model.getBidEvaluation(bid);
           if (evaluation < 0)
           {
               BidsPredictedAsUnfortunate.add(bid);
               BidsPredictedAsUnfortunateUtility.add(evaluation);
           }
           else if(evaluation >= 0)
           {
               BidsPredictedAsConceder.add(bid);
               BidsPredictedAsConcederUtility.add(evaluation);
           }
       } catch (Exception ex) {
       }
     }
     //System.out.println("Unfortunate: " + BidsPredictedAsUnfortunate.size() + " Conceder: " + BidsPredictedAsConceder.size());
     //----------------------------------------------------------------
     if (BidsPredictedAsConceder.isEmpty()) {
    
       /*double max = Double.NEGATIVE_INFINITY;
       int maxindex = 0;
       for(int i = 0; i < BidsPredictedAsUnfortunateUtility.size(); i++)
           if(max < BidsPredictedAsUnfortunateUtility.get(i))
           {
               max = BidsPredictedAsUnfortunateUtility.get(i);
               maxindex = i;
           }
       return (BidDetails)BidsPredictedAsUnfortunate.get(maxindex);*/
       //Random r = new Random();
       //return (BidDetails)BidsPredictedAsUnfortunate.get(r.nextInt(BidsPredictedAsUnfortunate.size()));
       return this.negotiationSession.getOwnBidHistory().getLastBidDetails();
     }
     else
     {
       double max = Double.NEGATIVE_INFINITY;
       int maxindex = 0;
       for(int i = 0; i < BidsPredictedAsConcederUtility.size(); i++)
           if(max < BidsPredictedAsConcederUtility.get(i))
           {
               max = BidsPredictedAsConcederUtility.get(i);
               maxindex = i;
           }
       return (BidDetails)BidsPredictedAsConceder.get(maxindex);
       //Random r = new Random();
       //return (BidDetails)BidsPredictedAsConceder.get(r.nextInt(BidsPredictedAsConceder.size()));
     }
   }
 
   @Override
   public boolean canUpdateOM()
   {
     return this.negotiationSession.getTime() < this.updateThreshold;
   }
 }