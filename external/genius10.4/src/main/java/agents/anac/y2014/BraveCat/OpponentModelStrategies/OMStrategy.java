 package agents.anac.y2014.BraveCat.OpponentModelStrategies;

 import agents.anac.y2014.BraveCat.OpponentModels.OpponentModel;
 import java.util.HashMap;
 import java.util.List;

import agents.anac.y2014.BraveCat.necessaryClasses.NegotiationSession;
 import agents.anac.y2014.BraveCat.necessaryClasses.OutcomeSpace;
 import agents.anac.y2014.BraveCat.necessaryClasses.SortedOutcomeSpace;
import genius.core.bidding.BidDetails;
import genius.core.misc.Range;
 
 public abstract class OMStrategy
 {
   protected NegotiationSession negotiationSession;
   protected OpponentModel model;
   private final double RANGE_INCREMENT = 0.01D;
 
   private final int EXPECTED_BIDS_IN_WINDOW = 100;
 
   private final double INITIAL_WINDOW_RANGE = 0.01D;
 
   public void init(NegotiationSession negotiationSession, OpponentModel model, HashMap<String, Double> parameters)
     throws Exception
   {
     this.negotiationSession = negotiationSession;
     this.model = model;
   }
 
   public void init(NegotiationSession negotiationSession, OpponentModel model)
     throws Exception
   {
     this.negotiationSession = negotiationSession;
     this.model = model;
   }
 
   public abstract BidDetails getBid(List<BidDetails> paramList);
 
   public BidDetails getBid(OutcomeSpace space, Range range)
   {
     List bids = space.getBidsinRange(range);
     if (bids.size() == 0) {
       if (range.getUpperbound() < 1.01D) {
         range.increaseUpperbound(0.01D);
         return getBid(space, range);
       }
       this.negotiationSession.setOutcomeSpace(space);
       return this.negotiationSession.getMaxBidinDomain();
     }
 
     return getBid(bids);
   }
 
   public void setOpponentModel(OpponentModel model) {
     this.model = model;
   }
 
   public BidDetails getBid(SortedOutcomeSpace space, double targetUtility)
   {
     Range range = new Range(targetUtility, targetUtility + 0.01D);
     List bids = space.getBidsinRange(range);
     if (bids.size() < 100) {
       if (range.getUpperbound() < 1.01D) {
         range.increaseUpperbound(0.01D);
         return getBid(space, range);
       }
 
       return getBid(bids);
     }
 
     return getBid(bids);
   }
 
   public abstract boolean canUpdateOM();
 }