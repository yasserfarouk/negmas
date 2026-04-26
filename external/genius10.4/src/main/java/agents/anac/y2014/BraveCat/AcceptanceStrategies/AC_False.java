 package agents.anac.y2014.BraveCat.AcceptanceStrategies;
 
 import genius.core.boaframework.Actions;
 
 public class AC_False extends AcceptanceStrategy
 {
   @Override
   public Actions determineAcceptability()
   {
     return Actions.Reject;
   }
 }