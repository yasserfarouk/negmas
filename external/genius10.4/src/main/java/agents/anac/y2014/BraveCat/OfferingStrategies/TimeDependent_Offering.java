 package agents.anac.y2014.BraveCat.OfferingStrategies;
 
 import agents.anac.y2014.BraveCat.OpponentModelStrategies.OMStrategy;
 import agents.anac.y2014.BraveCat.OpponentModels.NoModel;
 import agents.anac.y2014.BraveCat.OpponentModels.OpponentModel;
 import java.util.HashMap;
 import agents.anac.y2014.BraveCat.necessaryClasses.NegotiationSession;
 import agents.anac.y2014.BraveCat.necessaryClasses.SortedOutcomeSpace;
import genius.core.bidding.BidDetails;
 
 public class TimeDependent_Offering extends OfferingStrategy
 {
   private double k;
   private double Pmax;
   private double Pmin;
   private double e;
   SortedOutcomeSpace outcomespace;
 
   public TimeDependent_Offering()
   {
   }
 
   public TimeDependent_Offering(NegotiationSession negoSession, OpponentModel model, OMStrategy oms, double e, double k, double max, double min)
   {
     this.e = e;
     this.k = k;
     this.Pmax = max;
     this.Pmin = min;
     this.negotiationSession = negoSession;
     this.outcomespace = new SortedOutcomeSpace(this.negotiationSession.getUtilitySpace());
     this.negotiationSession.setOutcomeSpace(this.outcomespace);
     this.opponentModel = model;
     this.omStrategy = oms;
   }
 
     @Override
     public void init(NegotiationSession negoSession, OpponentModel model, OMStrategy oms, HashMap<String, Double> parameters)
     throws Exception
   {
     if (parameters.get("e") != null) {
       this.negotiationSession = negoSession;
 
       this.outcomespace = new SortedOutcomeSpace(this.negotiationSession.getUtilitySpace());
       this.negotiationSession.setOutcomeSpace(this.outcomespace);
 
       this.e = ((Double)parameters.get("e")).doubleValue();
 
       if (parameters.get("k") != null)
         this.k = ((Double)parameters.get("k")).doubleValue();
       else {
         this.k = 0.0D;
       }
       if (parameters.get("min") != null)
         this.Pmin = ((Double)parameters.get("min")).doubleValue();
       else {
         this.Pmin = negoSession.getMinBidinDomain().getMyUndiscountedUtil();
       }
       if (parameters.get("max") != null) {
         this.Pmax = ((Double)parameters.get("max")).doubleValue();
       } else {
         BidDetails maxBid = negoSession.getMaxBidinDomain();
         this.Pmax = maxBid.getMyUndiscountedUtil();
       }
 
       this.opponentModel = model;
       this.omStrategy = oms;
     } else {
       throw new Exception("Constant \"e\" for the concession speed was not set.");
     }
   }
 
   @Override
   public BidDetails determineOpeningBid()
   {
     return determineNextBid();
   }
 
   @Override
   public BidDetails determineNextBid()
   {
     double time = this.negotiationSession.getTime();
 
     double utilityGoal = p(time);
     
     if ((this.opponentModel instanceof NoModel))
       this.nextBid = this.negotiationSession.getOutcomeSpace().getBidNearUtility(utilityGoal);
     else {
       this.nextBid = this.omStrategy.getBid(this.outcomespace, utilityGoal);
     }
     return this.nextBid;
   }
 
   public double f(double t)
   {
     if (this.e == 0.0D)
       return this.k;
     double ft = this.k + (1.0D - this.k) * Math.pow(t, 1.0D / this.e);
     return ft;
   }
 
   public double p(double t)
   {
       return this.Pmin + (this.Pmax - this.Pmin) * (1.0D - f(t));
   }
 
   public NegotiationSession getNegotiationSession()
   {
       return this.negotiationSession;
   }
   @Override
   public String GetName()
   {
       return "TimeDependent Offering!";
   }
 }