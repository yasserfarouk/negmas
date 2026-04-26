package agents.anac.y2018.ateamagent;

import java.util.List;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Map;
import java.util.TreeMap;
import java.util.TreeSet;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.Domain;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.issue.Issue;
import genius.core.issue.Value;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;

class OpponentBidEvaluator
{
    private Map<String, List<Bid>> bidsPerOpponent;

    public OpponentBidEvaluator()
    {
        bidsPerOpponent = new TreeMap<String, List<Bid>>();
    }

    public void registerBid(AgentID agentId, Bid bid)
    {
        String agentName = agentId.toString();
        if (bidsPerOpponent.containsKey(agentName)==false)
        {
            bidsPerOpponent.put(agentName, new LinkedList<>());
        }
        bidsPerOpponent.get(agentName).add(bid);
    }

    public Double EstimateBidUtility(AgentID agentID, Bid bid){

        List<Double> estimateBidIssues = EstimateBid(agentID,bid);
        int numIssues = estimateBidIssues.size();
        Double totalValue = 0.0;
        for (int issueIndex = 0; issueIndex < numIssues; issueIndex++)
            totalValue += estimateBidIssues.get(issueIndex);

        // return avarage issues value
        return totalValue/numIssues;

    }


    public List<Double> EstimateBid(AgentID agentID, Bid bid)
    {
        String agentName;
        if (agentID == null)
        {
            agentName = (String)bidsPerOpponent.keySet().toArray()[0];
        }
        else
        {
            agentName = agentID.toString();
        }

        List<Bid> previousBids = this.bidsPerOpponent.get(agentName);
        int numPrevBids = previousBids.size();
        int numIssues = bid.getIssues().size();
        List<Double> bidPercentages = new ArrayList<Double>(numIssues);
        for (int issueIndex = 1; issueIndex <= numIssues; ++issueIndex) {
            int count = 0;
            Value bidIssueValue = bid.getValue(issueIndex);
            // Cycle through previous bids and check percentage
            for (int prevBidIndex = 0; prevBidIndex < numPrevBids; ++prevBidIndex) {
                String prevBidValue = previousBids.get(prevBidIndex).getValue(issueIndex).toString();
                if (prevBidValue.equals(bidIssueValue.toString())) {
                    ++count;
                }
            }
            bidPercentages.add(1.0 * count / numPrevBids);
        }
        return bidPercentages;
    }
}
public class ATeamAgent extends AbstractNegotiationParty {
    private final String description = "anac2018/ateamagent";
    private OpponentBidEvaluator opponentBidEvaluator;
    private Bid lastReceivedOffer; // offer on the table
    private Bid previousOffer;
    private Bid myLastOffer;
    private Bid lastBidGeneratedByMe;

    private double timeInterval = 0.0D;
    private double threshold = 1.0;

    private int HISTORY_SAVE = 5;

   private List<Bid> history;  // last K bids

    TreeMap<Bid, Double> allpossbids;

    @Override
    public void init(NegotiationInfo info) {
        super.init(info);

        GenerteAllSortedBids();
        history = new LinkedList<Bid>();
        lastReceivedOffer = null; // offer on the table
        previousOffer = null;
        myLastOffer = null;
        lastBidGeneratedByMe = getMaxUtilityBid();
        opponentBidEvaluator = new OpponentBidEvaluator();

    }

    private void GenerteAllSortedBids(){

        Map<Bid, Double> random = new HashMap<Bid, Double>();
        allpossbids = new TreeMap<Bid, Double>(new BidUtilComparator(random));

        int i;
        //generate all possible bids
        for (i =0 ; i < this.utilitySpace.getDomain().getNumberOfPossibleBids(); i++)
        {
            Bid newbid = null;
            do {
                newbid = generateRandomBid();}
            while (random.containsKey(newbid));
            random.put(newbid, this.getUtility(newbid));
            allpossbids.put(newbid, this.getUtility(newbid));
        }

    }

    private int check_commom_issues(Bid newbid, HashMap<Integer, Value> bidvalues) //returns list of values, with the common
    {
        Domain d = this.utilitySpace.getDomain();
        List<Issue> issues = d.getIssues();

        int commonissues = 0;

        if (!issues.isEmpty() && previousOffer != null) {

            Iterator var5 = issues.iterator();

            while (var5.hasNext()) {
                Issue issue = (Issue) var5.next();

                if (newbid.getValue(issue.getNumber()) == previousOffer.getValue(issue.getNumber())) {
                    commonissues++;
                    bidvalues.put(issue.getNumber(), newbid.getValue(issue.getNumber()));
                }
            }
        }
        return commonissues;
    }


    // generate next bid -
    // what can we learn from agent history? we know which offer it accepted at time t. so we can offer that
    // same offer at time t and it would accept, but before time t we can try bids that are with lower util for him.
    // can we tell the opponents util??

    //if the issue isnt already in the bd, get the best value for it.

    private Bid improve_common_bid(HashMap<Integer, Value> common_issues) {

        TreeSet <Bid> bidset = new TreeSet<Bid>(allpossbids.descendingKeySet());

        Iterator iterator;
        iterator = bidset.iterator();

        while (iterator.hasNext()) {
            Bid newbid = (Bid)iterator.next();
            if(is_common(newbid,common_issues)&& (this.getUtility(newbid) > this.getUtility(lastReceivedOffer)) )
                return newbid;
        }

        return null;
    }

    private boolean is_common(Bid a, HashMap<Integer, Value> common_issues)
    {
        Domain d = this.utilitySpace.getDomain();
        List<Issue> issues = d.getIssues();

        Bid bid = null;

        for (Issue lIssue : issues) {
            if((common_issues.containsKey(lIssue.getNumber())) && a.getValue(lIssue.getNumber())
                    != common_issues.get(lIssue.getNumber()))
                return false;
            }

        return true;
    }

    private Bid generateMyNextBid(HashMap<Integer, Value> bidvalues) {

        Bid bid = null;

        // allow the upper 10% of the sorted bids we have left to see if any gives good estimation for opponent utility
        int max_tries = (int)(0.1 * allpossbids.size());

        if (allpossbids.size() > 1)
        {
            bid = allpossbids.lastKey();
            //walk from best to worst bid, and estimate if the probability that the opponent will accept this bid is bigger than 0.5
            boolean lookForBetter = true;
            Bid testBid  = bid;
            while (lookForBetter & (testBid != null ) && max_tries > 0) {
                // Estimate opponent utility based on past bid
                if (opponentBidEvaluator.EstimateBidUtility(null, testBid) < 0.5f) {
                    testBid = allpossbids.lowerKey(testBid);
                    --max_tries;
                }
                else {
                    // test bid seems good for opponent (>0.5 estimated utility)
                    // suggest this bid
                    bid = testBid;
                    lookForBetter = false;
                }
            }

            lastBidGeneratedByMe = bid;
            allpossbids.remove(bid);

        }
        else if (allpossbids.size() == 1)
        {
            bid = this.getMaxUtilityBid();
            lastBidGeneratedByMe = bid;
        }

        return bid;
    }

    /**
     * When this function is called, it is expected that the Party chooses one of the actions from the possible
     * action list and returns an instance of the chosen action.
     *
     * @param list
     * @return
     */
    @Override
    public Action chooseAction(List<Class<? extends Action>> list) {
        // According to Stacked Alternating Offers Protocol list includes
        // Accept, Offer and EndNegotiation actions only.
        double time = getTimeLine().getTime(); // Gets the time, running from t = 0 (start) to t = 1 (deadline).

        // Update thresold
        threshold = this.getUtilityWithDiscount(getMaxUtilityBid()) * 0.8;
        // Get the time interval so we can estimate next best bid
        if (timeInterval == 0 && time > 0)
        {
            timeInterval = time;
        }

        // for the first 0.5 of the time suggest max bid.
        // we either get what we want or get bids from opponent so we can learn estimated utility
        if (lastReceivedOffer == null || time < 0.5) {

            Bid bestBid = this.getMaxUtilityBid();
            return new Offer(this.getPartyId(), this.getMaxUtilityBid());
        }

        if (lastReceivedOffer != null
                && myLastOffer != null
                && isAcceptable(lastReceivedOffer, time)) {
            // Accepts the bid on the table in this phase,
            // if the utility of the bid is higher than Example Agent's last bid.
            return new Accept(this.getPartyId(), lastReceivedOffer);
        }
        else
            {
            HashMap<Integer, Value> bidcommonvalues = new HashMap<Integer, Value>();
            if (check_commom_issues(lastReceivedOffer, bidcommonvalues) > 0)  //returns list of values, with the common
            {
                //add our values in issues that are not common
                //check bid not in history
                myLastOffer = improve_common_bid(bidcommonvalues);
                if (myLastOffer == null)
                    myLastOffer = generateMyNextBid(bidcommonvalues);
                else if (!isAcceptable(myLastOffer, time))  //the threshold
                    myLastOffer = generateMyNextBid(bidcommonvalues);

                //List<Double> estimate = opponentBidEvaluator.EstimateBid(null, myLastOffer);
                return new Offer(this.getPartyId(), myLastOffer);
            } else {
                //no common issues - check if bid is good and if so accept
                //else generate my new bid
                //this takes into considreration reservation value, discount factor and the time passed
                myLastOffer = generateMyNextBid(bidcommonvalues);
                return new Offer(this.getPartyId(), myLastOffer);
            }
        }

    }

    private boolean isBetterThanNextTurnBest(Bid bid)
    {
        double bidUtility = this.utilitySpace.getUtilityWithDiscount(bid, this.getTimeLine());
        double nextTimeFrame = this.getTimeLine().getTime() + this.timeInterval;
        Bid bestBid = this.getMaxUtilityBid();
        double nextBestUtility = this.utilitySpace.getUtilityWithDiscount(bestBid, nextTimeFrame);
        return bidUtility > nextBestUtility;
    }
    private boolean isAcceptable(Bid bid, double time) {
        double lastOfferUtility = this.utilitySpace.getUtilityWithDiscount(bid, time);
        double minimalReasonableUtility = this.utilitySpace.getReservationValue();
        double myLastOfferUtility = this.utilitySpace.getUtilityWithDiscount(myLastOffer, time);
        if (minimalReasonableUtility < lastOfferUtility && // is rational
                (lastOfferUtility > myLastOfferUtility || // better than my last offer
                        isBetterThanNextTurnBest(lastReceivedOffer))) {
            return true;
        }
        else {
            return false;
        }

    }

    private void addHistory(Bid newbid) {
        if (history.size() <= HISTORY_SAVE) {
            history.add(newbid);
            history.remove(0);
        } else
            history.add(newbid);
    }


    /**
     * This method is called to inform the party that another NegotiationParty chose an Action.
     *
     * @param sender
     * @param act
     */
    @Override
    public void receiveMessage(AgentID sender, Action act) {
        super.receiveMessage(sender, act);

        if (act instanceof Offer) { // sender is making an offer
            Offer offer = (Offer) act;

            previousOffer = lastReceivedOffer;
            // storing last received offer
            lastReceivedOffer = offer.getBid();
            addHistory(offer.getBid());
            opponentBidEvaluator.registerBid(sender, ((Offer) act).getBid());
        }
    }

    /**
     * A human-readable description for this party.
     *
     * @return
     */
    @Override
    public String getDescription() {
        return "ANAC2018";
    }
    
    private Bid getMaxUtilityBid() {
        try {
            return this.utilitySpace.getMaxUtilityBid();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

//end of class

}

class BidUtilComparator implements Comparator<Bid> {

    private Map<Bid, Double> map;

    public BidUtilComparator(Map<Bid, Double> map) {
        this.map = map;
    }

    @Override
    public int compare(Bid a, Bid b) {
        return (map.get(a).compareTo(map.get(b)));
    }
}
