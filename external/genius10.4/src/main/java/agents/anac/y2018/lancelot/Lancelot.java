package agents.anac.y2018.lancelot;

import java.util.List;

import agents.anac.y2018.lancelot.etc.bidSearch;
import agents.anac.y2018.lancelot.etc.strategy;
import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.timeline.TimeLineInfo;

/**
 * This is your negotiation party.
 */
public class Lancelot extends AbstractNegotiationParty {

    private Bid lastReceivedBid = null;
    private double opponent_eval = 0.0;
    private TimeLineInfo timeLineInfo;
    private strategy mStrategy;
    private bidSearch mBidSearch;

    private boolean DEBUG = false;


    @Override
    public void init(NegotiationInfo info) {

        super.init(info);

        if (DEBUG){
            System.out.println("*** MyAgent ***");
        }
        System.out.println("Discount Factor is " + info.getUtilitySpace().getDiscountFactor());
        System.out.println("Reservation Value is " + info.getUtilitySpace().getReservationValueUndiscounted());

        // if you need to initialize some variables, please initialize them
        // below
        mStrategy = new strategy(utilitySpace,timeline,info);
        mBidSearch = new bidSearch(utilitySpace,timeline);
    }


    /**
     * Each round this method gets called and ask you to accept or offer. The
     * first party in the first round is a bit different, it can only propose an
     * offer.
     *
     * @param validActions
     *            Either a list containing both accept and offer or only offer.
     * @return The chosen action.
     */
    public Action chooseAction(List<Class<? extends Action>> validActions) {
        if (lastReceivedBid == null || !validActions.contains(Accept.class) ||!(mStrategy.decideAcceptOrOffer(lastReceivedBid,opponent_eval,utilitySpace.getUtilityWithDiscount(lastReceivedBid,timeline.getTime())))) {
            Bid offer_bid = null;
//            if(timeline.getTime() < 0.6) {
//                offer_bid = mBidSearch.offerOppositeBid();
//            } else if(timeline.getTime() < 0.8){
//                offer_bid = mBidSearch.getRandomBid(timeline.getTime() + 0.1);
//            } else{
//                double min_util = mStrategy.getUtilThreshold2(timeline.getTime(),opponent_eval,utilitySpace.getUtilityWithDiscount(lastReceivedBid,timeline.getTime()));
//                offer_bid = mBidSearch.offerPositiveBid(min_util);
//            }
            if(timeline.getTime() < 0.2){
                offer_bid = mBidSearch.getRandomBid(1-timeline.getTime() - 0.1);
            } else if(timeline.getTime() < 0.98){
                double min_util = mStrategy.getUtilThreshold2(timeline.getTime(), opponent_eval, utilitySpace.getUtilityWithDiscount(lastReceivedBid, timeline.getTime()));
                offer_bid = mBidSearch.offerPositiveBid(min_util);
            } else{
//                double min_util = mStrategy.getUtilThreshold3(timeline.getTime(), opponent_eval, utilitySpace.getUtilityWithDiscount(lastReceivedBid, timeline.getTime()));
                double min_util = mStrategy.getUtilThresholdForOffer();
                offer_bid = mBidSearch.offerPositiveBid(min_util);
            }
            return new Offer(getPartyId(), offer_bid);
        }
        else {
            System.out.println("Accept");
            return new Accept(getPartyId(), lastReceivedBid);
        }
    }

    /**
     * All offers proposed by the other parties will be received as a message.
     * You can use this information to your advantage, for example to predict
     * their utility.
     *
     * @param sender
     *            The party that did the action. Can be null.
     * @param action
     *            The action that party did.
     */
    @Override
    public void receiveMessage(AgentID sender, Action action) {
        super.receiveMessage(sender, action);
        if (action instanceof Offer) {
            lastReceivedBid = ((Offer) action).getBid();
            mBidSearch.updateBidTable(lastReceivedBid);
            opponent_eval = mStrategy.evaluateOpponent(lastReceivedBid);
//            System.out.println("receivedBid = " + lastReceivedBid + "by " + sender.getName());
        }
    }

    @Override
    public String getDescription() {
        return "ANAC2018";
    }

}
